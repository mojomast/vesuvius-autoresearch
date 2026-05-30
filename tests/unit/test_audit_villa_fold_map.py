from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.audit_villa_fold_map import inspect_fold_map


def _write_npz(path: Path, positives: list[tuple[int, int]]) -> None:
    labels = np.zeros((1, 1, 4, 4), dtype=np.float32)
    for y, x in positives:
        labels[0, 0, y, x] = 1.0
    images = np.zeros((1, 3, 4, 4), dtype=np.float32)
    np.savez_compressed(path, images=images, labels=labels)


def test_audit_drops_zero_positive_validation_fold(tmp_path: Path) -> None:
    zero = tmp_path / "zero.npz"
    positive = tmp_path / "positive.npz"
    _write_npz(zero, [])
    _write_npz(positive, [(0, 0)])
    fold_map = tmp_path / "fold_map.json"
    fold_map.write_text(json.dumps({
        "zero": {"train_npz": "train_zero.npz", "val_npz": str(zero)},
        "positive": {"train_npz": "train_positive.npz", "val_npz": str(positive)},
    }))

    result = inspect_fold_map(fold_map)

    assert result["folds_dropped"] == 1
    assert result["dropped_folds"][0]["fold_id"] == "zero"
    assert sorted(result["filtered_fold_map"]) == ["positive"]


def test_audit_respects_min_positive_pixels(tmp_path: Path) -> None:
    one_positive = tmp_path / "one_positive.npz"
    _write_npz(one_positive, [(0, 0)])
    fold_map = tmp_path / "fold_map.json"
    fold_map.write_text(json.dumps({"fold": {"train_npz": "train.npz", "val_npz": str(one_positive)}}))

    assert inspect_fold_map(fold_map, min_positive_pixels=1)["folds_kept"] == 1
    assert inspect_fold_map(fold_map, min_positive_pixels=2)["folds_kept"] == 0


def test_cli_writes_filtered_fold_map(tmp_path: Path) -> None:
    zero = tmp_path / "zero.npz"
    positive = tmp_path / "positive.npz"
    _write_npz(zero, [])
    _write_npz(positive, [(1, 1)])
    fold_map = tmp_path / "fold_map.json"
    fold_map.write_text(json.dumps({
        "zero": {"train_npz": "train_zero.npz", "val_npz": str(zero)},
        "positive": {"train_npz": "train_positive.npz", "val_npz": str(positive)},
    }))
    summary = tmp_path / "summary.json"
    filtered = tmp_path / "filtered.json"

    result = subprocess.run([
        sys.executable,
        "scripts/audit_villa_fold_map.py",
        "--fold-map",
        str(fold_map),
        "--summary-out",
        str(summary),
        "--filtered-fold-map-out",
        str(filtered),
    ], cwd=Path(__file__).resolve().parents[2], check=False, text=True, capture_output=True)

    assert result.returncode == 1
    assert json.loads(summary.read_text())["folds_dropped"] == 1
    assert sorted(json.loads(filtered.read_text())) == ["positive"]

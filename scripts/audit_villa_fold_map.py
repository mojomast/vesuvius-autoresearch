#!/usr/bin/env python3
"""Audit villa-label fold maps for validation folds with positive labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def _labels_from_npz(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "labels" not in data.files:
            raise KeyError(f"{path} does not contain a 'labels' array")
        return np.asarray(data["labels"])


def inspect_fold_map(fold_map_path: str | Path, min_positive_pixels: int = 1) -> dict[str, Any]:
    """Return a fold-map audit and a filtered map containing usable validation folds."""
    if min_positive_pixels < 1:
        raise ValueError("min_positive_pixels must be >= 1")
    resolved_fold_map = _resolve(fold_map_path)
    fold_map = json.loads(resolved_fold_map.read_text())
    rows: list[dict[str, Any]] = []
    filtered: dict[str, Any] = {}
    for fold_id, fold in sorted(fold_map.items()):
        val_npz = _resolve(fold["val_npz"])
        labels = _labels_from_npz(val_npz)
        positive_pixels = int((labels > 0.5).sum())
        total_pixels = int(labels.size)
        usable = positive_pixels >= min_positive_pixels
        row = {
            "fold_id": str(fold_id),
            "val_npz": str(fold["val_npz"]),
            "positive_pixels": positive_pixels,
            "total_pixels": total_pixels,
            "positive_rate": float(positive_pixels / total_pixels) if total_pixels else 0.0,
            "usable_for_promotion_loo": usable,
        }
        rows.append(row)
        if usable:
            filtered[str(fold_id)] = fold
    dropped = [row for row in rows if not row["usable_for_promotion_loo"]]
    return {
        "fold_map": str(resolved_fold_map),
        "min_positive_pixels": min_positive_pixels,
        "folds_total": len(rows),
        "folds_kept": len(filtered),
        "folds_dropped": len(dropped),
        "dropped_folds": dropped,
        "folds": rows,
        "filtered_fold_map": filtered,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit villa fold-map validation positives")
    parser.add_argument("--fold-map", default="data/fold_map_villa_labels.json")
    parser.add_argument("--min-positive-pixels", type=int, default=1)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--filtered-fold-map-out", default=None)
    args = parser.parse_args()
    result = inspect_fold_map(args.fold_map, min_positive_pixels=args.min_positive_pixels)
    if args.summary_out:
        summary_out = _resolve(args.summary_out)
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.filtered_fold_map_out:
        filtered_out = _resolve(args.filtered_fold_map_out)
        filtered_out.parent.mkdir(parents=True, exist_ok=True)
        filtered_out.write_text(json.dumps(result["filtered_fold_map"], indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["folds_dropped"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

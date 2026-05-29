from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.generate_synthetic_data import TRAIN_SEGMENT, VAL_SEGMENT, generate


REQUIRED_METADATA = {
    "segment_id",
    "split",
    "val_stride",
    "spatial_overlap_checked",
    "provenance",
    "patch_size",
    "z_offsets",
    "source_segments",
    "n_train",
    "n_val",
    "val_positive_rate",
}


def test_generate_synthetic_npzs_have_runner_keys_and_shapes(tmp_path: Path) -> None:
    summary = generate(tmp_path / "data", n_train=12, n_val=8, seed=123)

    with np.load(summary["train_path"]) as train, np.load(summary["val_path"]) as val:
        assert set(train.files) == {"images", "labels"}
        assert set(val.files) == {"images", "labels"}
        assert train["images"].shape == (12, 3, 64, 64)
        assert train["labels"].shape == (12, 1, 64, 64)
        assert val["images"].shape == (8, 3, 64, 64)
        assert val["labels"].shape == (8, 1, 64, 64)
        assert train["images"].dtype == np.float32
        assert train["labels"].dtype == np.float32


def test_metadata_has_required_provenance_fields(tmp_path: Path) -> None:
    summary = generate(tmp_path / "data", n_train=12, n_val=8, seed=123)

    for npz_path, segment_id, split in (
        (summary["train_path"], TRAIN_SEGMENT, "train"),
        (summary["val_path"], VAL_SEGMENT, "val"),
    ):
        metadata = json.loads((Path(npz_path).parent / "metadata.json").read_text())
        assert REQUIRED_METADATA <= set(metadata)
        assert metadata["segment_id"] == segment_id
        assert metadata["split"] == split
        assert metadata["spatial_overlap_checked"] is True
        assert metadata["provenance"] == "synthetic"
        assert metadata["patch_size"] == 64
        assert metadata["z_offsets"] == [-4, 0, 4]
        assert metadata["source_segments"] == [TRAIN_SEGMENT, VAL_SEGMENT]
        assert metadata["n_train"] == 12
        assert metadata["n_val"] == 8


def test_positive_rate_is_in_expected_range(tmp_path: Path) -> None:
    summary = generate(tmp_path / "data", n_train=32, n_val=16, seed=456)

    with np.load(summary["train_path"]) as train, np.load(summary["val_path"]) as val:
        assert 0.03 <= float(train["labels"].mean()) <= 0.12
        assert 0.03 <= float(val["labels"].mean()) <= 0.12


def test_fold_map_synthetic_is_valid(tmp_path: Path) -> None:
    summary = generate(tmp_path / "data", n_train=12, n_val=8, seed=123)
    fold_map = json.loads(Path(summary["fold_map_path"]).read_text())

    assert set(fold_map) == {TRAIN_SEGMENT, VAL_SEGMENT}
    for heldout, fold in fold_map.items():
        assert heldout in {TRAIN_SEGMENT, VAL_SEGMENT}
        assert set(fold) == {"train_npz", "val_npz"}
        assert fold["train_npz"].endswith(".npz")
        assert fold["val_npz"].endswith(".npz")

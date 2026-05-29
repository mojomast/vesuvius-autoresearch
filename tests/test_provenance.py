from __future__ import annotations

import json

import numpy as np

from experiments.runner import _validation_setup
from scripts.combine_segment_npzs import combine
from scripts.prepare_vesuvius_segment_npz import _regions_overlap, _spatial_region, _split_regions, _write_npz


REQUIRED_METADATA_KEYS = {
    "segment_id",
    "split",
    "source_segments",
    "val_stride",
    "patch_size",
    "provenance",
    "z_offsets",
    "spatial_overlap_checked",
    "spatial_region",
}


def _synthetic_image_label() -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((3, 96, 160), dtype=np.float32)
    label = np.zeros((96, 160), dtype=np.float32)
    label[16:48, 16:80] = 1.0
    return image, label


def _base_meta(segment_id: str) -> dict:
    return {
        "segment_id": segment_id,
        "source_segments": [segment_id],
        "provenance": "public-directory",
        "patch_size": 32,
        "z_offsets": [-4, 0, 4],
        "spatial_overlap_checked": True,
    }


def test_prepare_metadata_has_required_provenance_keys(tmp_path):
    image, label = _synthetic_image_label()
    out = tmp_path / "segment_a" / "val.npz"

    _write_npz(out, image, label, (96, 160), "val", 32, 2, 0.0, 3, _base_meta("seg-a"), True, 32, 0.001)

    metadata = json.loads(out.with_suffix(".metadata.json").read_text())
    assert REQUIRED_METADATA_KEYS <= metadata.keys()
    assert metadata["split"] == "val"
    assert metadata["source_segments"] == ["seg-a"]
    assert metadata["val_stride"] == 32
    assert metadata["spatial_overlap_checked"] is True


def test_same_segment_cross_region_spatial_overlap_is_zero():
    train_region, val_region = _split_regions(width=200, patch_size=32, mode="cross_region")
    train_spatial = _spatial_region(train_region, (100, 200))
    val_spatial = _spatial_region(val_region, (100, 200))

    assert not _regions_overlap(train_spatial, val_spatial)


def test_cross_segment_source_segments_are_non_overlapping(tmp_path):
    image, label = _synthetic_image_label()
    train_a = tmp_path / "segment_a" / "train.npz"
    train_b = tmp_path / "segment_b" / "train.npz"
    val_c = tmp_path / "segment_c" / "val.npz"

    _write_npz(train_a, image, label, (0, 80), "train", 32, 2, 0.0, 1, _base_meta("seg-a"), False, 32, 0.001)
    _write_npz(train_b, image, label, (0, 80), "train", 32, 2, 0.0, 2, _base_meta("seg-b"), False, 32, 0.001)
    _write_npz(val_c, image, label, (96, 160), "val", 32, 2, 0.0, 3, _base_meta("seg-c"), True, 32, 0.001)

    combined = tmp_path / "leaveout_c" / "train.npz"
    combine([train_a, train_b], combined, split="train")

    train_meta = {"metadata": json.loads(combined.with_suffix(".metadata.json").read_text())}
    val_meta = {"metadata": json.loads(val_c.with_suffix(".metadata.json").read_text())}
    setup = _validation_setup(train_meta, val_meta)

    assert set(train_meta["metadata"]["source_segments"]).isdisjoint(val_meta["metadata"]["source_segments"])
    assert train_meta["metadata"]["spatial_overlap_checked"] is True
    assert val_meta["metadata"]["spatial_overlap_checked"] is True
    assert setup["mode"] == "leave-one-segment-out"
    assert setup["train_segments"] == ["seg-a", "seg-b"]


def test_validation_setup_rejects_overlapping_source_segments(tmp_path):
    train_meta = {"metadata": {"scroll_id": "1", "source_segments": ["seg-a", "seg-b"], "spatial_overlap_checked": True}}
    val_meta = {"metadata": {"scroll_id": "1", "segment_id": "seg-b", "source_segments": ["seg-b"], "spatial_overlap_checked": True}}

    setup = _validation_setup(train_meta, val_meta)

    assert setup["mode"] == "spatial-same-segment"
    assert setup["warning"]

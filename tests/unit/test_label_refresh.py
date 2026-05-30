import numpy as np
import pytest

from scripts.rebuild_npz_with_villa_labels import compute_iou


def test_rebuild_iou_calculation_matches_hardfold_report_counts() -> None:
    ours_positive = 120700
    villa_positive = 117646
    villa_not_ours = 0
    ours_not_villa = 3054
    intersection = ours_positive - ours_not_villa
    union = intersection + villa_not_ours + ours_not_villa
    assert intersection == villa_positive
    assert intersection / union == pytest.approx(0.9747, abs=0.0001)


def test_compute_iou_counts_changed_pixels() -> None:
    ours = np.array([[1, 1], [0, 0]], dtype=np.float32)
    villa = np.array([[1, 0], [1, 0]], dtype=np.float32)
    result = compute_iou(ours, villa)
    assert result["iou"] == pytest.approx(1 / 3)
    assert result["villa_not_ours"] == 1
    assert result["ours_not_villa"] == 1

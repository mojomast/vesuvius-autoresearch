from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from experiments.runner import _pixel_metrics_from_probs, _positive_rate_excess, _resolve_positive_rate_target, _sample_patch_indices, _validation_setup


class RunnerValidationTest(unittest.TestCase):
    def test_sampling_strategy_alias_enables_hard_mining(self) -> None:
        images = np.zeros((8, 3, 4, 4), dtype=np.float32)
        images[:, 1] = np.arange(8, dtype=np.float32).reshape(8, 1, 1)
        labels = np.zeros((8, 4, 4), dtype=np.float32)
        labels[:3, 1:3, 1:3] = 1.0

        _idx, metrics = _sample_patch_indices(images, labels, 4, 123, {"sampling_strategy": "hard_mining"})

        self.assertEqual(metrics["patch_sampling"], "hard_mining")
        self.assertEqual(metrics["selected_patches"], 4)

    def test_leave_one_out_metadata_is_held_out_validation(self) -> None:
        train_meta = {"metadata": {"scroll_id": "1", "train_segments": ["seg-a", "seg-b"], "heldout_segment": "seg-c"}}
        val_meta = {"metadata": {"scroll_id": "1", "segment_id": "seg-c"}}

        setup = _validation_setup(train_meta, val_meta)

        self.assertEqual(setup["mode"], "leave-one-segment-out")
        self.assertIsNone(setup["warning"])
        self.assertEqual(setup["heldout_segment"], "seg-c")

    def test_train_segments_containing_val_segment_warns(self) -> None:
        train_meta = {"metadata": {"scroll_id": "1", "train_segments": ["seg-a", "seg-c"]}}
        val_meta = {"metadata": {"scroll_id": "1", "segment_id": "seg-c"}}

        setup = _validation_setup(train_meta, val_meta)

        self.assertEqual(setup["mode"], "spatial-same-segment")
        self.assertIn("present in train_segments", setup["warning"])

    def test_threshold_selection_can_constrain_positive_rate_ratio(self) -> None:
        labels = np.zeros(100, dtype=np.float32)
        labels[:10] = 1.0
        probs = np.concatenate([
            np.full(10, 0.8, dtype=np.float32),
            np.full(50, 0.7, dtype=np.float32),
            np.full(40, 0.1, dtype=np.float32),
        ])

        with tempfile.TemporaryDirectory() as tmp:
            metrics = _pixel_metrics_from_probs(
                probs,
                labels,
                labels,
                0.5,
                Path(tmp),
                {},
                {"max_pred_positive_rate_ratio": 4.0},
            )

        self.assertEqual(metrics["threshold_selection"], "positive_rate_constrained")
        self.assertLessEqual(metrics["pred_positive_rate"] / metrics["val_positive_rate"], 4.0)

    def test_positive_rate_loss_helpers_resolve_auto_train_target(self) -> None:
        labels = np.zeros((2, 1, 4, 4), dtype=np.float32)
        labels[0, 0, :2, :2] = 1.0

        self.assertAlmostEqual(_resolve_positive_rate_target("auto_train", labels), float(labels.mean()))
        self.assertAlmostEqual(_resolve_positive_rate_target(0.15, labels), 0.15)
        self.assertIsNone(_resolve_positive_rate_target(None, labels))
        self.assertEqual(_positive_rate_excess(0.12, 0.10, 0.03), 0.0)
        self.assertAlmostEqual(_positive_rate_excess(0.20, 0.10, 0.03), 0.07)

    def test_threshold_selection_tracks_target_positive_rate(self) -> None:
        labels = np.asarray([1, 0, 0, 0], dtype=np.float32)
        probs = np.asarray([0.9, 0.8, 0.4, 0.1], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            metrics = _pixel_metrics_from_probs(
                probs,
                labels,
                labels,
                0.5,
                Path(tmp),
                {},
                {"max_pred_positive_rate_ratio": 4.0, "target_pred_positive_rate": "auto_val"},
            )

        self.assertEqual(metrics["target_pred_positive_rate"], metrics["val_positive_rate"])
        self.assertIn("threshold_selection", metrics)


if __name__ == "__main__":
    unittest.main()

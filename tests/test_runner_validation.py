from __future__ import annotations

import unittest

import numpy as np

from experiments.runner import _sample_patch_indices, _validation_setup


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


if __name__ == "__main__":
    unittest.main()

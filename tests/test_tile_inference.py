from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.tile_inference import (
    evaluate_probability_map,
    _promotion_checks,
    self_test,
    stitch_probabilities_from_predictor,
    tile_origins,
    write_outputs,
)


class TileInferenceTest(unittest.TestCase):
    def test_tile_origins_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "patch_size must be positive"):
            tile_origins(16, 16, patch_size=0, stride=8)
        with self.assertRaisesRegex(ValueError, "stride must be positive"):
            tile_origins(16, 16, patch_size=8, stride=0)
        with self.assertRaisesRegex(ValueError, "exceeds image shape"):
            tile_origins(7, 16, patch_size=8, stride=4)
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            stitch_probabilities_from_predictor(
                lambda patches: np.zeros((patches.shape[0], 8, 8), dtype=np.float32),
                np.zeros((1, 16, 16), dtype=np.float32),
                patch_size=8,
                stride=8,
                batch_size=0,
            )

    def test_constant_stitching(self) -> None:
        image = np.zeros((1, 24, 24), dtype=np.float32)

        def predict_batch(patches: np.ndarray) -> np.ndarray:
            return np.full((patches.shape[0], patches.shape[2], patches.shape[3]), 0.25, dtype=np.float32)

        prob_map, meta = stitch_probabilities_from_predictor(predict_batch, image, patch_size=8, stride=4, batch_size=3)

        self.assertEqual(prob_map.shape, (24, 24))
        self.assertTrue(np.allclose(prob_map, 0.25))
        self.assertEqual(meta["patch_size"], 8)
        self.assertEqual(meta["stride"], 4)

    def test_bad_predictor_shape(self) -> None:
        image = np.zeros((1, 16, 16), dtype=np.float32)

        def predict_batch(patches: np.ndarray) -> np.ndarray:
            return np.zeros((patches.shape[0], 1, 8, 8), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "Predictor returned shape"):
            stitch_probabilities_from_predictor(predict_batch, image, patch_size=8, stride=8, batch_size=2)

    def test_metric_aliases_and_shape_validation(self) -> None:
        prob_map = np.asarray([[0.9, 0.1], [0.8, 0.2]], dtype=np.float32)
        label = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

        metrics, _ = evaluate_probability_map(prob_map, label)

        self.assertEqual(metrics["val_f1"], metrics["tile_f1"])
        self.assertEqual(metrics["val_f05"], metrics["tile_f05"])
        self.assertEqual(metrics["val_loss"], metrics["tile_loss"])
        self.assertEqual(metrics["val_positive_rate"], metrics["label_positive_rate"])
        with self.assertRaisesRegex(ValueError, "does not match label shape"):
            evaluate_probability_map(prob_map, label[:1])

    def test_write_outputs_overwrite_protection(self) -> None:
        prob_map = np.zeros((2, 2), dtype=np.float32)
        metrics = {"val_f1": 1.0}
        rows = [{"threshold": 0.5, "precision": 1.0, "recall": 1.0, "f05": 1.0, "f1": 1.0, "pred_positive_rate": 0.5}]

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_outputs(output_dir, prob_map, metrics, rows)
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                write_outputs(output_dir, prob_map, metrics, rows)
            outputs = write_outputs(output_dir, prob_map, metrics, rows, overwrite=True)

        self.assertIn("probability_map", outputs)
        self.assertIn("metrics_json", outputs)
        self.assertIn("threshold_csv", outputs)

    def test_self_test(self) -> None:
        result = self_test()

        self.assertGreaterEqual(result["metrics"]["fixed_threshold_f1"], 0.99)
        self.assertGreater(result["threshold_rows"], 0)

    def test_promotion_checks_warn_on_training_segment_leakage(self) -> None:
        cfg = {
            "validation_setup": {"mode": "spatial-same-segment", "train_segment_id": "abc", "val_segment_id": "abc"}
        }

        checks = _promotion_checks(cfg, "abc")

        self.assertFalse(checks["eligible"])
        self.assertIn("training segment", " ".join(checks["warnings"]))
        self.assertEqual(checks["inference_segment_id"], "abc")


if __name__ == "__main__":
    unittest.main()

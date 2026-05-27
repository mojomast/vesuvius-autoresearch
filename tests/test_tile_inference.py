from __future__ import annotations

import tempfile
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from data.tile_inference import (
    evaluate_probability_map,
    _promotion_checks,
    load_public_segment,
    mine_failure_patches,
    run_full_tile_inference,
    self_test,
    stitch_probabilities_from_predictor,
    tile_origins,
    write_mined_npz,
    write_outputs,
)
from scripts.prepare_vesuvius_segment_npz import _open_layers


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
        self.assertIn("brier_score", metrics)
        self.assertIn("expected_calibration_error", metrics)
        self.assertIn("ap_prevalence_lift", metrics)
        self.assertEqual(metrics["selected_threshold_reason"], metrics["threshold_selection"])
        with self.assertRaisesRegex(ValueError, "does not match label shape"):
            evaluate_probability_map(prob_map, label[:1])

    def test_tile_metrics_respect_positive_rate_constraints(self) -> None:
        label = np.zeros((10, 10), dtype=np.float32)
        label[:2, :5] = 1.0
        prob_map = np.full((10, 10), 0.7, dtype=np.float32)
        prob_map[:2, :5] = 0.9
        prob_map[8:, :] = 0.1

        metrics, _rows = evaluate_probability_map(
            prob_map,
            label,
            fixed_threshold=0.5,
            eval_cfg={"max_pred_positive_rate_ratio": 2.0, "target_pred_positive_rate": "auto_val"},
        )

        self.assertEqual(metrics["threshold_selection"], "positive_rate_constrained")
        self.assertLessEqual(metrics["pred_positive_rate"] / metrics["val_positive_rate"], 2.0)
        self.assertEqual(metrics["target_pred_positive_rate"], metrics["val_positive_rate"])

    def test_tile_calibration_diagnostics_have_expected_values(self) -> None:
        prob_map = np.asarray([[0.75, 0.25], [0.75, 0.25]], dtype=np.float32)
        label = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

        metrics, _rows = evaluate_probability_map(prob_map, label)

        self.assertAlmostEqual(metrics["brier_score"], 0.0625)
        self.assertAlmostEqual(metrics["ap_prevalence_lift"], 2.0)
        self.assertGreaterEqual(metrics["expected_calibration_error"], 0.0)
        self.assertLessEqual(metrics["expected_calibration_error"], 1.0)
        self.assertEqual(metrics["fixed_threshold_status"], "ok")

    def test_tile_metrics_resolve_auto_train_positive_rate_target(self) -> None:
        label = np.zeros((10, 10), dtype=np.float32)
        label[:2, :5] = 1.0
        prob_map = np.full((10, 10), 0.4, dtype=np.float32)
        prob_map[:2, :5] = 0.8

        metrics, _rows = evaluate_probability_map(
            prob_map,
            label,
            fixed_threshold=0.5,
            eval_cfg={"positive_rate_loss_target": "auto_train", "train_positive_rate": 0.12},
        )

        self.assertEqual(metrics["target_pred_positive_rate"], 0.12)

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

    def test_mine_failure_patches_selects_false_positive_negatives(self) -> None:
        image = np.arange(64, dtype=np.float32).reshape(1, 8, 8)
        label = np.zeros((8, 8), dtype=np.float32)
        label[:4, :4] = 1.0
        prob_map = np.zeros((8, 8), dtype=np.float32)
        prob_map[4:, 4:] = 0.9
        prob_map[:4, :4] = 0.95

        images, labels, meta = mine_failure_patches(
            image,
            label,
            prob_map,
            patch_size=4,
            stride=4,
            threshold=0.5,
            max_patches=2,
            max_label_positive_rate=0.0,
        )

        self.assertEqual(images.shape, (1, 1, 4, 4))
        self.assertEqual(labels.shape, (1, 1, 4, 4))
        self.assertEqual(meta["samples"], 1)
        self.assertEqual(meta["origins_yx"], [[4, 4]])
        self.assertTrue(np.array_equal(images[0], image[:, 4:, 4:]))
        self.assertEqual(float(labels.mean()), 0.0)

    def test_write_mined_npz_uses_runner_schema_and_metadata(self) -> None:
        images = np.zeros((2, 1, 4, 4), dtype=np.float32)
        labels = np.zeros((2, 1, 4, 4), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "mined.npz"
            paths = write_mined_npz(output, images, labels, {"samples": 2})
            with np.load(output) as data:
                self.assertEqual(data["images"].shape, images.shape)
                self.assertEqual(data["labels"].shape, labels.shape)
            self.assertTrue(Path(paths["mined_metadata_json"]).exists())
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                write_mined_npz(output, images, labels)

    def test_run_full_tile_inference_optionally_writes_mined_npz(self) -> None:
        image = np.zeros((1, 8, 8), dtype=np.float32)
        label = np.zeros((8, 8), dtype=np.float32)
        prob_map = np.zeros((8, 8), dtype=np.float32)
        prob_map[4:, 4:] = 0.8
        cfg = {"dataset": {"patch_size": 4}, "evaluation": {"threshold": 0.5}, "model": {"name": "tiny_torch_unet"}}

        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch("data.tile_inference.load_public_segment", return_value=(image, label, {"segment_id": "seg"})), \
             mock.patch("data.tile_inference.load_torch_unet_artifact", return_value=([object()], cfg, Path(tmp) / "artifact", {})), \
             mock.patch("data.tile_inference.stitch_probabilities", return_value=(prob_map, {"patches": 4})):
            output_dir = Path(tmp) / "tile"
            mined_path = Path(tmp) / "mined.npz"
            result = run_full_tile_inference(
                artifact=Path(tmp) / "artifact",
                segment_id="seg",
                output_dir=output_dir,
                mine_output=mined_path,
                mine_max_patches=1,
                mine_threshold=0.5,
                mine_stride=4,
            )

            self.assertTrue(mined_path.exists())
            with np.load(mined_path) as data:
                self.assertEqual(data["images"].shape, (1, 1, 4, 4))
                self.assertEqual(data["labels"].shape, (1, 1, 4, 4))
            self.assertIn("mined_npz", result["outputs"])
            self.assertEqual(result["metrics"]["mined_hard_negatives"]["samples"], 1)
            self.assertEqual(result["metrics"]["mined_hard_negatives"]["mined_segment_id"], "seg")
            self.assertEqual(result["metrics"]["mined_hard_negatives"]["forbidden_heldout_segments"], ["seg"])

    def test_public_segment_retry_handles_rate_limit(self) -> None:
        class RateLimitError(Exception):
            status = 429

        image = np.ones((1, 4, 4), dtype=np.float32)
        label = np.zeros((4, 4), dtype=np.float32)

        with mock.patch("scripts.prepare_vesuvius_segment_npz._segment_meta", return_value={"zarr_url": "zarr", "inklabels_url": "labels"}), \
             mock.patch("scripts.prepare_vesuvius_segment_npz._open_layers", side_effect=[RateLimitError("Too Many Requests"), (image, [2])]) as open_layers, \
             mock.patch("scripts.prepare_vesuvius_segment_npz._normalize", side_effect=lambda arr: arr), \
             mock.patch("scripts.prepare_vesuvius_segment_npz._read_label", return_value=label), \
             mock.patch("scripts.prepare_vesuvius_segment_npz._align_label", side_effect=lambda raw, shape: raw), \
             mock.patch("data.tile_inference.time.sleep") as sleep:
            loaded_image, loaded_label, meta = load_public_segment("seg", "1", [0], retry_count=1, retry_delay_sec=0.5)

        self.assertEqual(open_layers.call_count, 2)
        sleep.assert_called_once_with(0.5)
        self.assertTrue(np.array_equal(loaded_image, image))
        self.assertTrue(np.array_equal(loaded_label, label))
        self.assertEqual(meta["z_indices"], [2])

    def test_public_segment_plumbs_chunk_pacing(self) -> None:
        image = np.ones((1, 4, 4), dtype=np.float32)
        label = np.zeros((4, 4), dtype=np.float32)

        with mock.patch("scripts.prepare_vesuvius_segment_npz._segment_meta", return_value={"zarr_url": "zarr", "inklabels_url": "labels"}), \
             mock.patch("scripts.prepare_vesuvius_segment_npz._open_layers", return_value=(image, [2])) as open_layers, \
             mock.patch("scripts.prepare_vesuvius_segment_npz._normalize", side_effect=lambda arr: arr), \
             mock.patch("scripts.prepare_vesuvius_segment_npz._read_label", return_value=label), \
             mock.patch("scripts.prepare_vesuvius_segment_npz._align_label", side_effect=lambda raw, shape: raw):
            _loaded_image, _loaded_label, meta = load_public_segment(
                "seg", "1", [0], public_chunk_delay_sec=0.25, public_chunk_retry_count=3, public_chunk_retry_delay_sec=4.0
            )

        open_layers.assert_called_once_with("zarr", "1", [0], 0.25, 3, 4.0)
        self.assertEqual(meta["public_chunk_delay_sec"], 0.25)
        self.assertEqual(meta["public_chunk_retry_count"], 3)
        self.assertEqual(meta["public_chunk_retry_delay_sec"], 4.0)

    def test_open_layers_reads_chunk_aligned_with_retry_without_network(self) -> None:
        class RateLimitError(Exception):
            status = 429

        class FakeArray:
            shape = (3, 4, 5)
            chunks = (1, 2, 3)

            def __init__(self) -> None:
                self.data = np.arange(60, dtype=np.float32).reshape(self.shape)
                self.keys = []
                self.failed = False

            def __getitem__(self, key):
                self.keys.append(key)
                if not self.failed:
                    self.failed = True
                    raise RateLimitError("Too Many Requests")
                return self.data[key]

        arr = FakeArray()
        fake_fsspec = types.SimpleNamespace(get_mapper=lambda url: f"mapper:{url}")
        fake_zarr = types.SimpleNamespace(open=lambda mapper, mode="r": {"1": arr})

        with mock.patch.dict(sys.modules, {"fsspec": fake_fsspec, "zarr": fake_zarr}), \
             mock.patch("scripts.prepare_vesuvius_segment_npz.time.sleep") as sleep:
            layers, z_indices = _open_layers("url", "1", [0], public_chunk_delay_sec=0.1, public_chunk_retry_count=1, public_chunk_retry_delay_sec=0.5)

        self.assertEqual(z_indices, [1])
        self.assertTrue(np.array_equal(layers[0], arr.data[1]))
        self.assertEqual(arr.keys[0], (slice(1, 2), slice(0, 2), slice(0, 3)))
        self.assertEqual(arr.keys[1], (slice(1, 2), slice(0, 2), slice(0, 3)))
        self.assertIn((slice(1, 2), slice(2, 4), slice(3, 5)), arr.keys)
        sleep.assert_any_call(0.5)
        self.assertEqual(sleep.call_count, 5)

    def test_open_layers_reuses_z_chunk_for_multiple_offsets(self) -> None:
        class FakeArray:
            shape = (4, 4, 4)
            chunks = (4, 2, 2)

            def __init__(self) -> None:
                self.data = np.arange(64, dtype=np.float32).reshape(self.shape)
                self.keys = []

            def __getitem__(self, key):
                self.keys.append(key)
                return self.data[key]

        arr = FakeArray()
        fake_fsspec = types.SimpleNamespace(get_mapper=lambda url: f"mapper:{url}")
        fake_zarr = types.SimpleNamespace(open=lambda mapper, mode="r": {"1": arr})

        with mock.patch.dict(sys.modules, {"fsspec": fake_fsspec, "zarr": fake_zarr}), \
             mock.patch("scripts.prepare_vesuvius_segment_npz.time.sleep"):
            layers, z_indices = _open_layers("url", "1", [-1, 0, 1], public_chunk_delay_sec=0.1)

        self.assertEqual(z_indices, [1, 2, 3])
        self.assertTrue(np.array_equal(layers, arr.data[[1, 2, 3]]))
        self.assertEqual(len(arr.keys), 4)
        self.assertEqual(arr.keys[0], (slice(0, 4), slice(0, 2), slice(0, 2)))

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

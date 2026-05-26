from __future__ import annotations

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class NextBestMovesExamplesTest(unittest.TestCase):
    def test_template_config_covers_all_protocols(self) -> None:
        cfg = yaml.safe_load((ROOT / "configs" / "next_best_moves_robust_template.yaml").read_text())
        protocol = cfg["evaluation_protocol"]

        self.assertEqual(cfg["dataset"]["patch_size"], 64)
        self.assertEqual(cfg["dataset"]["validation_mode"], "leave-one-segment-out")
        self.assertIn("fold_map", cfg["dataset"])
        self.assertGreater(len(cfg["dataset"]["z_offsets"]), 1)
        self.assertEqual(cfg["model"]["planned_name"], "residual_25d_torch_unet")
        self.assertIn("seed_repeat_leave_one_out", protocol)
        self.assertIn("safe_data_expansion", protocol)
        self.assertIn("full_tile_inference", protocol)
        self.assertIn("tta_seed_ensemble", protocol)
        self.assertIn("residual_25d_unet", protocol)

    def test_fold_map_paths_are_repo_relative_npzs(self) -> None:
        fold_map = {
            "20230520175435": {
                "train_npz": "data/example_folds/leaveout_20230520175435/train.npz",
                "val_npz": "data/example_folds/segment_20230520175435/val.npz",
            },
            "20230522181603": {
                "train_npz": "data/example_folds/leaveout_20230522181603/train.npz",
                "val_npz": "data/example_folds/segment_20230522181603/val.npz",
            },
            "20230522215721": {
                "train_npz": "data/example_folds/leaveout_20230522215721/train.npz",
                "val_npz": "data/example_folds/segment_20230522215721/val.npz",
            },
        }

        self.assertGreaterEqual(len(fold_map), 3)
        for heldout, fold in fold_map.items():
            self.assertTrue(heldout.isdigit())
            self.assertTrue(fold["train_npz"].endswith("train.npz"))
            self.assertTrue(fold["val_npz"].endswith("val.npz"))
            self.assertFalse(Path(fold["train_npz"]).is_absolute())
            self.assertFalse(Path(fold["val_npz"]).is_absolute())
            self.assertNotIn(f"segment_{heldout}/train.npz", fold["train_npz"])

    def test_seed_repeat_summary_is_segment_first(self) -> None:
        from scripts.evaluate_leave_one_out import _summarize

        rows = [
            {"returncode": 0, "heldout_segment": "a", "seed": 1, "val_f1": 0.1, "average_precision": 0.2},
            {"returncode": 0, "heldout_segment": "a", "seed": 2, "val_f1": 0.3, "average_precision": 0.4},
            {"returncode": 0, "heldout_segment": "b", "seed": 1, "val_f1": 0.8, "average_precision": 0.6},
            {"returncode": 0, "heldout_segment": "b", "seed": 2, "val_f1": 1.0, "average_precision": 0.8},
        ]
        summary = _summarize(rows)

        self.assertEqual(summary["per_fold_val_f1"], {"a": 0.2, "b": 0.9})
        self.assertAlmostEqual(summary["median_val_f1"], 0.55)

    def test_docs_reference_the_five_next_best_moves(self) -> None:
        text = (ROOT / "docs" / "next_best_moves_may2026.md").read_text()

        for phrase in [
            "Seed-Repeat Leave-One-Out",
            "Safe Data Expansion",
            "Full-Tile Inference",
            "TTA And Seed Ensembling",
            "2.5D Residual U-Net",
        ]:
            self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()

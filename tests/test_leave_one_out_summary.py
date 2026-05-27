from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_leave_one_out import _run_fold_job, _summarize, main


class LeaveOneOutSummaryTest(unittest.TestCase):
    def test_seed_repeat_robust_promotion_fields(self) -> None:
        rows = [
            {"returncode": 0, "heldout_segment": "a", "seed": 1, "val_f1": 0.2, "average_precision": 0.3, "precision": 0.4, "recall": 0.5, "pred_positive_rate": 0.2, "val_positive_rate": 0.1},
            {"returncode": 0, "heldout_segment": "b", "seed": 1, "val_f1": 0.6, "average_precision": 0.5, "precision": 0.4, "recall": 0.5, "pred_positive_rate": 0.2, "val_positive_rate": 0.1},
            {"returncode": 0, "heldout_segment": "a", "seed": 2, "val_f1": 0.4, "average_precision": 0.7, "precision": 0.4, "recall": 0.5, "pred_positive_rate": 0.2, "val_positive_rate": 0.1},
            {"returncode": 0, "heldout_segment": "b", "seed": 2, "val_f1": 0.8, "average_precision": 0.9, "precision": 0.4, "recall": 0.5, "pred_positive_rate": 0.2, "val_positive_rate": 0.1},
        ]

        summary = _summarize(rows, min_seeds_for_promotion=2)

        self.assertAlmostEqual(summary["per_fold_val_f1"]["a"], 0.3)
        self.assertAlmostEqual(summary["per_fold_val_f1"]["b"], 0.7)
        self.assertAlmostEqual(summary["per_seed_median_val_f1"]["1"], 0.4)
        self.assertAlmostEqual(summary["per_seed_median_val_f1"]["2"], 0.6)
        self.assertEqual(summary["per_seed_min_val_f1"], {"1": 0.2, "2": 0.4})
        self.assertEqual(summary["per_seed_mean_average_precision"], {"1": 0.4, "2": 0.8})
        self.assertAlmostEqual(summary["median_over_seeds_median_val_f1"], 0.5)
        self.assertAlmostEqual(summary["worst_seed_median_val_f1"], 0.4)
        self.assertAlmostEqual(summary["worst_fold_val_f1"], 0.3)
        self.assertEqual(summary["worst_fold_id"], "a")
        self.assertAlmostEqual(summary["mean_average_precision"], 0.6)
        self.assertAlmostEqual(summary["median_average_precision"], 0.6)
        self.assertEqual(summary["folds_with_zero_precision_or_recall"], [])
        self.assertEqual(summary["folds_with_positive_rate_alarm"], [])
        self.assertEqual(summary["promotion_warnings"], [])
        self.assertTrue(summary["promotion_ready"])

    def test_single_seed_is_not_promotion_ready(self) -> None:
        rows = [
            {"returncode": 0, "heldout_segment": "a", "seed": 1, "val_f1": 0.2, "average_precision": 0.3, "precision": 0.4, "recall": 0.5, "pred_positive_rate": 0.2, "val_positive_rate": 0.1},
            {"returncode": 0, "heldout_segment": "b", "seed": 1, "val_f1": 0.6, "average_precision": 0.5, "precision": 0.4, "recall": 0.5, "pred_positive_rate": 0.2, "val_positive_rate": 0.1},
        ]

        summary = _summarize(rows)

        self.assertEqual(summary["min_seeds_for_promotion"], 3)
        self.assertEqual(summary["distinct_successful_seeds"], 1)
        self.assertIn("insufficient_seed_repeats:1/3", summary["promotion_warnings"])
        self.assertFalse(summary["promotion_ready"])

    def test_warnings_include_failed_rows_and_metric_alarms(self) -> None:
        jsonl = "\n".join([
            json.dumps({"returncode": 0, "heldout_segment": "high", "seed": 3, "val_f1": 0.5, "average_precision": 0.2, "precision": 0.0, "recall": 0.4, "pred_positive_rate": 0.51, "val_positive_rate": 0.1}),
            json.dumps({"returncode": 0, "heldout_segment": "low", "seed": 3, "val_f1": 0.6, "average_precision": 0.4, "precision": 0.3, "recall": 0.0, "pred_positive_rate": 0.009, "val_positive_rate": 0.1}),
            json.dumps({"returncode": 1, "heldout_segment": "failed", "seed": 3, "error": "RuntimeError('boom')"}),
        ])
        rows = [json.loads(line) for line in jsonl.splitlines()]

        summary = _summarize(rows)

        self.assertEqual(summary["folds_successful"], 2)
        self.assertEqual(summary["folds_failed"], 1)
        self.assertEqual(summary["folds_with_zero_precision_or_recall"], ["high:seed=3", "low:seed=3"])
        self.assertEqual(summary["folds_with_positive_rate_alarm"], ["high:seed=3", "low:seed=3"])
        self.assertIn("failed_fold:failed:seed=3:RuntimeError('boom')", summary["promotion_warnings"])
        self.assertIn("zero_precision_or_recall:high:seed=3", summary["promotion_warnings"])
        self.assertIn("positive_rate_alarm:low:seed=3", summary["promotion_warnings"])
        self.assertFalse(summary["promotion_ready"])

    def test_failed_only_rows_do_not_crash(self) -> None:
        summary = _summarize([{"returncode": 1, "heldout_segment": "x", "error": "bad"}])

        self.assertEqual(summary["folds_successful"], 0)
        self.assertEqual(summary["folds_failed"], 1)
        self.assertEqual(summary["folds_with_zero_precision_or_recall"], [])
        self.assertEqual(summary["folds_with_positive_rate_alarm"], [])
        self.assertIn("failed_fold:x:bad", summary["promotion_warnings"])
        self.assertIn("no_successful_folds", summary["promotion_warnings"])
        self.assertFalse(summary["promotion_ready"])

    def test_missing_val_f1_gets_warning_without_crashing(self) -> None:
        summary = _summarize([
            {"returncode": 0, "heldout_segment": "good", "val_f1": 0.5, "average_precision": 0.4},
            {"returncode": 0, "heldout_segment": "bad", "average_precision": 0.2},
        ])

        self.assertEqual(summary["folds_successful"], 1)
        self.assertEqual(summary["folds_failed"], 1)
        self.assertIn("missing_or_invalid_val_f1:bad", summary["promotion_warnings"])
        self.assertFalse(summary["promotion_ready"])

    def test_dry_run_summary_has_promotion_fields(self) -> None:
        summary = _summarize([{"dry_run": True}, {"dry_run": True}])

        self.assertTrue(summary["dry_run"])
        self.assertEqual(summary["folds_with_zero_precision_or_recall"], [])
        self.assertEqual(summary["folds_with_positive_rate_alarm"], [])
        self.assertEqual(summary["promotion_warnings"], ["dry_run_no_promotion_metrics"])
        self.assertFalse(summary["promotion_ready"])

    def test_dry_run_jobs_preserve_fold_seed_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "base.yaml"
            config.write_text("dataset:\n  patch_size: 64\nmodel:\n  name: tiny_torch_unet\ntraining:\n  seed: 7\n")
            fold_map = root / "fold_map.json"
            fold_map.write_text(json.dumps({
                "b": {"train_npz": "train_b.npz", "val_npz": "val_b.npz"},
                "a": {"train_npz": "train_a.npz", "val_npz": "val_a.npz"},
            }))
            output = root / "loo.jsonl"
            summary = root / "loo.summary.json"

            with patch("sys.argv", ["evaluate_leave_one_out.py", "--base-config", str(config), "--fold-map", str(fold_map), "--output-jsonl", str(output), "--summary-json", str(summary), "--seeds", "2,1", "--jobs", "2", "--dry-run"]):
                self.assertEqual(main(), 0)

            rows = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual([(row["heldout_segment"], row["seed"]) for row in rows], [("a", 2), ("a", 1), ("b", 2), ("b", 1)])

    def test_jobs_must_be_positive(self) -> None:
        with self.assertRaises(SystemExit):
            with patch("sys.argv", ["evaluate_leave_one_out.py", "--base-config", "x", "--fold-map", "y", "--output-jsonl", "z", "--jobs", "0", "--dry-run"]):
                main()

    def test_run_fold_job_returns_metrics_or_error(self) -> None:
        with patch("scripts.evaluate_leave_one_out.run_experiment", return_value={"run_id": "run1", "artifact_dir": "artifact", "metrics": {"val_f1": 0.2, "val_f05": 0.1, "average_precision": 0.3, "precision": 0.4, "recall": 0.5, "best_threshold": 0.6, "val_positive_rate": 0.07, "pred_positive_rate": 0.08}}):
            row = _run_fold_job("cfg.yaml", {"heldout_segment": "a", "seed": 1})

        self.assertEqual(row["returncode"], 0)
        self.assertEqual(row["run_id"], "run1")

        with patch("scripts.evaluate_leave_one_out.run_experiment", side_effect=RuntimeError("boom")):
            failed = _run_fold_job("cfg.yaml", {"heldout_segment": "b", "seed": 2})

        self.assertEqual(failed["returncode"], 1)
        self.assertIn("RuntimeError", failed["error"])


if __name__ == "__main__":
    unittest.main()

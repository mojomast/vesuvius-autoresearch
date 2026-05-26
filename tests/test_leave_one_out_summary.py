from __future__ import annotations

import json
import unittest

from scripts.evaluate_leave_one_out import _summarize


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


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts.compare_loo_summaries import compare_summaries, main, render_markdown


def _summary(**overrides):
    data = {
        "folds_successful": 6,
        "folds_failed": 0,
        "mean_val_f1": 0.20,
        "median_over_seeds_median_val_f1": 0.18,
        "mean_average_precision": 0.14,
        "median_average_precision": 0.13,
        "worst_fold_id": "b",
        "worst_fold_val_f1": 0.06,
        "worst_seed_median_val_f1": 0.17,
        "per_fold_val_f1": {"a": 0.30, "b": 0.06},
        "folds_with_positive_rate_alarm": [],
        "folds_with_zero_precision_or_recall": [],
        "folds_with_fixed_threshold_not_ok": [],
        "folds_with_threshold_edge_case": [],
        "folds_with_weak_ap_prevalence_lift": [],
        "promotion_warnings": [],
    }
    data.update(overrides)
    return data


class CompareLooSummariesTest(unittest.TestCase):
    def test_compare_reports_metric_and_per_fold_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate = root / "candidate.summary.json"
            baseline = root / "baseline.summary.json"
            candidate.write_text(json.dumps(_summary(mean_val_f1=0.22, per_fold_val_f1={"a": 0.33, "b": 0.05})), encoding="utf-8")
            baseline.write_text(json.dumps(_summary(mean_val_f1=0.20, per_fold_val_f1={"a": 0.30, "b": 0.06})), encoding="utf-8")

            report = compare_summaries(
                candidate,
                {"baseline": baseline},
                min_median_over_seeds_f1=0.18,
                min_mean_ap=0.14,
                min_worst_fold_f1=0.05,
            )

        self.assertTrue(report["passed"])
        self.assertAlmostEqual(report["deltas_vs_primary_baseline"]["mean_val_f1"], 0.02)
        self.assertAlmostEqual(report["per_fold_f1_deltas_vs_primary_baseline"]["a"], 0.03)
        self.assertAlmostEqual(report["per_fold_f1_deltas_vs_primary_baseline"]["b"], -0.01)

    def test_paired_comparisons_report_direction_and_sign_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate = root / "candidate.summary.json"
            baseline = root / "baseline.summary.json"
            candidate.write_text(json.dumps(_summary(
                per_fold_val_f1={"a": 0.31, "b": 0.22, "c": 0.15},
                per_fold_average_precision={"a": 0.41, "b": 0.32, "c": 0.25},
                per_seed_median_val_f1={"1": 0.21, "2": 0.25, "3": 0.28},
            )), encoding="utf-8")
            baseline.write_text(json.dumps(_summary(
                per_fold_val_f1={"a": 0.30, "b": 0.20, "c": 0.10},
                per_fold_average_precision={"a": 0.40, "b": 0.31, "c": 0.20},
                per_seed_median_val_f1={"1": 0.20, "2": 0.22, "3": 0.26},
            )), encoding="utf-8")

            report = compare_summaries(candidate, {"baseline": baseline})
            markdown = render_markdown(report)

        paired = report["paired_comparisons_vs_primary_baseline"]["per_fold_val_f1"]
        self.assertEqual(paired["wins"], 3)
        self.assertEqual(paired["losses"], 0)
        self.assertAlmostEqual(paired["mean_delta"], (0.01 + 0.02 + 0.05) / 3)
        self.assertAlmostEqual(paired["sign_test_two_sided_p"], 0.25)
        self.assertIn("Paired Comparisons", markdown)

    def test_gates_fail_on_quality_regression_or_alarms(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate = root / "candidate.summary.json"
            baseline = root / "baseline.summary.json"
            candidate.write_text(
                json.dumps(
                    _summary(
                        median_over_seeds_median_val_f1=0.12,
                        folds_with_positive_rate_alarm=["a:seed=1"],
                        folds_with_fixed_threshold_not_ok=["b:seed=1", "b:seed=2"],
                    )
                ),
                encoding="utf-8",
            )
            baseline.write_text(json.dumps(_summary()), encoding="utf-8")

            report = compare_summaries(
                candidate,
                {"baseline": baseline},
                min_median_over_seeds_f1=0.18,
                max_fixed_threshold_not_ok=1,
            )

        self.assertFalse(report["passed"])
        failed = {gate["label"] for gate in report["gates"] if not gate["passed"]}
        self.assertIn("no positive-rate alarms", failed)
        self.assertIn("median-over-seeds F1 floor", failed)
        self.assertIn("fixed-threshold warning cap", failed)

    def test_markdown_and_fail_on_gate_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate = root / "candidate.summary.json"
            baseline = root / "baseline.summary.json"
            candidate.write_text(json.dumps(_summary(worst_fold_val_f1=0.04)), encoding="utf-8")
            baseline.write_text(json.dumps(_summary()), encoding="utf-8")

            report = compare_summaries(candidate, {"baseline": baseline}, min_worst_fold_f1=0.05)
            markdown = render_markdown(report)
            with redirect_stdout(StringIO()):
                code = main([
                    "--candidate",
                    str(candidate),
                    "--baseline",
                    f"baseline={baseline}",
                    "--min-worst-fold-f1",
                    "0.05",
                    "--fail-on-gate",
                ])

        self.assertIn("# LOO Summary Comparison", markdown)
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()

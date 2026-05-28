from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts.compare_full_tile_metrics import compare_full_tile_metrics, main, render_batch_markdown, render_markdown, summarize_comparisons


def _metrics(**overrides):
    data = {
        "evaluation_region": {"type": "whole_segment", "segment_id": "20230522181603"},
        "segment": "20230522181603",
        "val_f1": 0.14,
        "val_f05": 0.12,
        "average_precision": 0.09,
        "ap_prevalence_lift": 1.7,
        "precision": 0.10,
        "recall": 0.45,
        "val_positive_rate": 0.04,
        "pred_positive_rate": 0.12,
        "best_threshold": 0.30,
        "fixed_threshold_f1": 0.10,
        "fixed_threshold_status": "ok",
        "brier_score": 0.08,
        "expected_calibration_error": 0.03,
        "prob_mean": 0.09,
        "prob_p95": 0.40,
        "prob_max": 0.99,
        "threshold_risk_summary": {
            "selected": {
                "threshold": 0.31,
                "precision": 0.11,
                "recall": 0.42,
                "f05": 0.13,
                "f1": 0.17,
                "pred_positive_rate": 0.115,
                "pred_to_val_ratio": 2.875,
            }
        },
    }
    data.update(overrides)
    return data


class CompareFullTileMetricsTest(unittest.TestCase):
    def test_reports_core_metric_deltas_and_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            baseline.write_text(json.dumps(_metrics()), encoding="utf-8")
            candidate.write_text(json.dumps(_metrics(
                val_f1=0.15,
                val_f05=0.125,
                average_precision=0.095,
                ap_prevalence_lift=1.8,
                brier_score=0.07,
                expected_calibration_error=0.02,
                pred_positive_rate=0.116,
            )), encoding="utf-8")

            report = compare_full_tile_metrics(baseline, candidate, label="weak-fold")

        self.assertEqual(report["label"], "weak-fold")
        self.assertAlmostEqual(report["deltas"]["val_f1"], 0.01)
        self.assertAlmostEqual(report["deltas"]["pred_to_val_ratio"], -0.1)
        self.assertEqual(report["warnings"], [])
        self.assertEqual(report["verdict"], "candidate_improves_tracked_metrics")

    def test_warns_on_segment_mismatch_and_non_whole_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            baseline.write_text(json.dumps(_metrics()), encoding="utf-8")
            candidate.write_text(json.dumps(_metrics(
                segment="20230530212931",
                evaluation_region={"type": "validation_crop", "segment_id": "20230530212931"},
            )), encoding="utf-8")

            report = compare_full_tile_metrics(baseline, candidate, require_whole_segment=True)

        self.assertIn("segment mismatch", "; ".join(report["warnings"]))
        self.assertIn("candidate evaluation_region.type is not whole_segment", "; ".join(report["warnings"]))

    def test_markdown_and_cli_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline = root / "baseline.json"
            ensemble = root / "ensemble.json"
            baseline.write_text(json.dumps(_metrics()), encoding="utf-8")
            ensemble.write_text(json.dumps(_metrics(val_f1=0.15, average_precision=0.095)), encoding="utf-8")

            report = compare_full_tile_metrics(baseline, ensemble)
            markdown = render_markdown(report)
            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(["--baseline", str(baseline), "--ensemble", str(ensemble), "--markdown"])

        self.assertEqual(code, 0)
        self.assertIn("# Full-Tile Metrics Comparison", markdown)
        self.assertIn("average_precision", stdout.getvalue())

    def test_cli_pair_returns_nonzero_for_required_whole_segment_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            baseline.write_text(json.dumps(_metrics(evaluation_region={"type": "crop", "segment_id": "20230522181603"})), encoding="utf-8")
            candidate.write_text(json.dumps(_metrics()), encoding="utf-8")

            with redirect_stdout(StringIO()):
                code = main(["--pair", f"fold={baseline},{candidate}", "--require-whole-segment"])

        self.assertEqual(code, 1)

    def test_batch_summary_packages_pair_status_and_mean_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline_a = root / "baseline_a.json"
            candidate_a = root / "candidate_a.json"
            baseline_b = root / "baseline_b.json"
            candidate_b = root / "candidate_b.json"
            baseline_a.write_text(json.dumps(_metrics()), encoding="utf-8")
            candidate_a.write_text(json.dumps(_metrics(val_f1=0.15, val_f05=0.13, average_precision=0.095, pred_positive_rate=0.12)), encoding="utf-8")
            baseline_b.write_text(json.dumps(_metrics(segment="seg-b", evaluation_region={"type": "whole_segment", "segment_id": "seg-b"})), encoding="utf-8")
            candidate_b.write_text(json.dumps(_metrics(segment="seg-b", evaluation_region={"type": "whole_segment", "segment_id": "seg-b"}, val_f1=0.13, average_precision=0.085)), encoding="utf-8")

            reports = [
                compare_full_tile_metrics(baseline_a, candidate_a, label="a"),
                compare_full_tile_metrics(baseline_b, candidate_b, label="b"),
            ]
            summary = summarize_comparisons(reports)
            markdown = render_batch_markdown({"summary": summary, "comparisons": reports})

        self.assertEqual(summary["status_counts"]["core_improved"], 1)
        self.assertEqual(summary["status_counts"]["core_regressed"], 1)
        self.assertEqual(summary["verdict"], "mixed_core_evidence")
        self.assertAlmostEqual(summary["mean_deltas"]["val_f1"], 0.0)
        self.assertIn("Full-Tile Evidence Package", markdown)
        self.assertIn("core_regressed", markdown)

    def test_cli_multi_pair_can_fail_on_core_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            baseline.write_text(json.dumps(_metrics()), encoding="utf-8")
            candidate.write_text(json.dumps(_metrics(val_f1=0.13, average_precision=0.08)), encoding="utf-8")

            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(["--pair", f"fold={baseline},{candidate}", "--fail-on-core-regression", "--markdown"])

        self.assertEqual(code, 1)
        self.assertIn("core_regressed", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()

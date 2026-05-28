from __future__ import annotations

import csv
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts.compare_threshold_caps import compare_threshold_caps, compare_threshold_caps_batch, main, render_batch_markdown, render_markdown


def _write_metrics_artifact(root: Path, name: str, selected_f1: float = 0.30) -> Path:
    artifact = root / name
    artifact.mkdir(parents=True)
    metrics = artifact / "metrics.json"
    csv_path = artifact / "metrics_by_threshold.csv"
    metrics.write_text(json.dumps({"val_positive_rate": 0.1, "threshold_risk_summary": {"selected": {"threshold": 0.3, "precision": 0.2, "recall": 0.6, "f05": 0.23, "f1": selected_f1, "pred_positive_rate": 0.30, "pred_to_val_ratio": 3.0}}}), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["threshold", "precision", "recall", "f05", "f1", "pred_positive_rate"])
        writer.writeheader()
        writer.writerow({"threshold": 0.50, "precision": 0.30, "recall": 0.18, "f05": 0.26, "f1": 0.225, "pred_positive_rate": 0.20})
        writer.writerow({"threshold": 0.40, "precision": 0.25, "recall": 0.35, "f05": 0.26, "f1": 0.2917, "pred_positive_rate": 0.25})
        writer.writerow({"threshold": 0.30, "precision": 0.20, "recall": 0.60, "f05": 0.23, "f1": 0.30, "pred_positive_rate": 0.30})
    return metrics


class CompareThresholdCapsTest(unittest.TestCase):
    def test_recommends_strictest_cap_that_retains_f1(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metrics = root / "metrics.json"
            csv_path = root / "metrics_by_threshold.csv"
            metrics.write_text(json.dumps({"val_positive_rate": 0.1, "threshold_risk_summary": {"selected": {"threshold": 0.3, "precision": 0.2, "recall": 0.6, "f05": 0.23, "f1": 0.30, "pred_positive_rate": 0.30, "pred_to_val_ratio": 3.0}}}), encoding="utf-8")
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["threshold", "precision", "recall", "f05", "f1", "pred_positive_rate"])
                writer.writeheader()
                writer.writerow({"threshold": 0.40, "precision": 0.25, "recall": 0.35, "f05": 0.26, "f1": 0.29, "pred_positive_rate": 0.25})
                writer.writerow({"threshold": 0.50, "precision": 0.30, "recall": 0.25, "f05": 0.29, "f1": 0.27, "pred_positive_rate": 0.20})
                writer.writerow({"threshold": 0.30, "precision": 0.20, "recall": 0.60, "f05": 0.23, "f1": 0.30, "pred_positive_rate": 0.30})

            report = compare_threshold_caps(metrics, caps=[2.0, 2.5, 3.0], min_retained_f1=0.90, baseline_cap=2.0)

        self.assertEqual(report["recommended_cap"]["cap"], 2.0)
        self.assertAlmostEqual(report["recommended_cap"]["best"]["f1"], 0.27)
        self.assertIn("f1_delta", report["improvement_vs_baseline_cap"])

    def test_reports_improvement_for_intermediate_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metrics = root / "metrics.json"
            csv_path = root / "metrics_by_threshold.csv"
            metrics.write_text(json.dumps({"val_positive_rate": 0.1, "threshold_risk_summary": {"selected": {"threshold": 0.3, "precision": 0.2, "recall": 0.6, "f05": 0.23, "f1": 0.30, "pred_positive_rate": 0.30, "pred_to_val_ratio": 3.0}}}), encoding="utf-8")
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["threshold", "precision", "recall", "f05", "f1", "pred_positive_rate"])
                writer.writeheader()
                writer.writerow({"threshold": 0.50, "precision": 0.30, "recall": 0.18, "f05": 0.26, "f1": 0.225, "pred_positive_rate": 0.20})
                writer.writerow({"threshold": 0.40, "precision": 0.25, "recall": 0.35, "f05": 0.26, "f1": 0.2917, "pred_positive_rate": 0.25})
                writer.writerow({"threshold": 0.30, "precision": 0.20, "recall": 0.60, "f05": 0.23, "f1": 0.30, "pred_positive_rate": 0.30})

            report = compare_threshold_caps(metrics, caps=[2.0, 2.5, 3.0], min_retained_f1=0.90, baseline_cap=2.0)
            markdown = render_markdown(report)
            with redirect_stdout(StringIO()):
                code = main(["--metrics", str(metrics), "--caps", "2.0,2.5,3.0", "--markdown"])

        self.assertEqual(code, 0)
        self.assertEqual(report["recommended_cap"]["cap"], 2.5)
        self.assertGreater(report["improvement_vs_baseline_cap"]["f1_delta"], 0.0)
        self.assertIn("Threshold Cap Comparison", markdown)
        self.assertIn("F0.5 retained", markdown)

    def test_recommendation_requires_f05_retention(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metrics = root / "metrics.json"
            csv_path = root / "metrics_by_threshold.csv"
            metrics.write_text(json.dumps({"val_positive_rate": 0.1, "threshold_risk_summary": {"selected": {"threshold": 0.3, "precision": 0.2, "recall": 0.6, "f05": 0.30, "f1": 0.30, "pred_positive_rate": 0.30, "pred_to_val_ratio": 3.0}}}), encoding="utf-8")
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["threshold", "precision", "recall", "f05", "f1", "pred_positive_rate"])
                writer.writeheader()
                writer.writerow({"threshold": 0.50, "precision": 0.30, "recall": 0.20, "f05": 0.10, "f1": 0.29, "pred_positive_rate": 0.20})
                writer.writerow({"threshold": 0.40, "precision": 0.25, "recall": 0.40, "f05": 0.29, "f1": 0.291, "pred_positive_rate": 0.25})
                writer.writerow({"threshold": 0.30, "precision": 0.20, "recall": 0.60, "f05": 0.30, "f1": 0.30, "pred_positive_rate": 0.30})

            report = compare_threshold_caps(metrics, caps=[2.0, 2.5, 3.0], min_retained_f1=0.95, min_retained_f05=0.95)

        self.assertEqual(report["recommended_cap"]["cap"], 2.5)
        self.assertGreaterEqual(report["caps"][0]["f1_retained_vs_selected"], 0.95)
        self.assertLess(report["caps"][0]["f05_retained_vs_selected"], 0.95)

    def test_batch_comparison_aggregates_multiple_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = _write_metrics_artifact(root, "run1")
            second = _write_metrics_artifact(root, "run2")

            report = compare_threshold_caps_batch([first, second], caps=[2.0, 2.5, 3.0], min_retained_f1=0.95)
            markdown = render_batch_markdown(report)

        self.assertEqual(report["metrics_count"], 2)
        self.assertEqual(report["aggregate"]["recommended_cap"]["cap"], 2.5)
        self.assertAlmostEqual(report["aggregate"]["per_cap"]["2.5"]["mean_f1"], 0.2917)
        self.assertIn("min_f05_retained_vs_selected", report["aggregate"]["per_cap"]["2.5"])
        self.assertIn("Threshold Cap Batch Comparison", markdown)
        self.assertIn("min F0.5 retained", markdown)

    def test_cli_batch_from_loo_jsonl_resolves_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = _write_metrics_artifact(root, "run1")
            second = _write_metrics_artifact(root, "run2")
            loo = root / "loo.jsonl"
            loo.write_text("\n".join([
                json.dumps({"returncode": 0, "artifact_dir": str(first.parent)}),
                json.dumps({"returncode": 0, "artifact_dir": str(second.parent)}),
            ]) + "\n", encoding="utf-8")

            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(["--loo-jsonl", str(loo), "--caps", "2.0,2.5,3.0", "--markdown"])

        self.assertEqual(code, 0)
        self.assertIn("Metrics files: 2", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()

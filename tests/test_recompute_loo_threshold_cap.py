from __future__ import annotations

import csv
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts.recompute_loo_threshold_cap import main, recompute_loo_threshold_cap, recompute_loo_threshold_caps, render_aggregate_markdown


def _write_artifact(root: Path, name: str, val_positive_rate: float = 0.1) -> Path:
    artifact = root / name
    artifact.mkdir(parents=True)
    (artifact / "metrics.json").write_text(json.dumps({"val_positive_rate": val_positive_rate, "threshold_risk_summary": {"selected": {"threshold": 0.3, "precision": 0.2, "recall": 0.6, "f05": 0.23, "f1": 0.30, "pred_positive_rate": 0.30, "pred_to_val_ratio": 3.0}}}), encoding="utf-8")
    with (artifact / "metrics_by_threshold.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["threshold", "precision", "recall", "f05", "f1", "pred_positive_rate"])
        writer.writeheader()
        writer.writerow({"threshold": 0.5, "precision": 0.3, "recall": 0.2, "f05": 0.27, "f1": 0.24, "pred_positive_rate": 0.2})
        writer.writerow({"threshold": 0.4, "precision": 0.25, "recall": 0.4, "f05": 0.27, "f1": 0.31, "pred_positive_rate": 0.25})
        writer.writerow({"threshold": 0.3, "precision": 0.2, "recall": 0.6, "f05": 0.23, "f1": 0.30, "pred_positive_rate": 0.3})
    return artifact


class RecomputeLooThresholdCapTest(unittest.TestCase):
    def test_recomputes_successful_rows_from_artifact_threshold_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = _write_artifact(root, "run1")
            input_jsonl = root / "input.jsonl"
            input_jsonl.write_text(json.dumps({"returncode": 0, "artifact_dir": str(artifact), "heldout_segment": "a", "seed": 1, "val_f1": 0.30, "val_f05": 0.23, "precision": 0.2, "recall": 0.6, "best_threshold": 0.3, "pred_positive_rate": 0.3, "average_precision": 0.5, "fixed_threshold_status": "ok"}) + "\n", encoding="utf-8")

            rows = recompute_loo_threshold_cap(input_jsonl, 2.5, repo_root=root)

        self.assertEqual(rows[0]["returncode"], 0)
        self.assertAlmostEqual(rows[0]["val_f1"], 0.31)
        self.assertAlmostEqual(rows[0]["recomputed_pred_to_val_ratio"], 2.5)
        self.assertAlmostEqual(rows[0]["val_positive_rate"], 0.1)
        self.assertAlmostEqual(rows[0]["original_val_f1"], 0.30)
        self.assertAlmostEqual(rows[0]["original_val_f05"], 0.23)
        self.assertAlmostEqual(rows[0]["f1_retained_vs_original"], 0.31 / 0.30)
        self.assertAlmostEqual(rows[0]["f05_retained_vs_original"], 0.27 / 0.23)

    def test_main_writes_jsonl_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = _write_artifact(root, "run1")
            input_jsonl = root / "input.jsonl"
            output_jsonl = root / "output.jsonl"
            summary_json = root / "summary.json"
            input_jsonl.write_text(json.dumps({"returncode": 0, "artifact_dir": str(artifact), "heldout_segment": "a", "seed": 1, "val_f1": 0.30, "val_f05": 0.23, "average_precision": 0.5, "precision": 0.2, "recall": 0.3, "val_positive_rate": 0.1}) + "\n", encoding="utf-8")

            with redirect_stdout(StringIO()):
                code = main([
                    "--input-jsonl",
                    str(input_jsonl),
                    "--output-jsonl",
                    str(output_jsonl),
                    "--summary-json",
                    str(summary_json),
                    "--cap",
                    "2.5",
                    "--min-seeds-for-promotion",
                    "1",
                ])

            summary = json.loads(summary_json.read_text())
            output_exists = output_jsonl.exists()

        self.assertEqual(code, 0)
        self.assertTrue(output_exists)
        self.assertAlmostEqual(summary["recomputed_max_pred_positive_rate_ratio"], 2.5)
        self.assertEqual(summary["cap_policy"], "recomputed_positive_rate_cap_2p5")
        self.assertAlmostEqual(summary["mean_val_f1"], 0.31)
        self.assertAlmostEqual(summary["mean_recomputed_pred_to_val_ratio"], 2.5)
        self.assertAlmostEqual(summary["mean_f1_retained_vs_original"], 0.31 / 0.30)
        self.assertAlmostEqual(summary["min_f05_retained_vs_original"], 0.27 / 0.23)
        self.assertEqual(summary["threshold_selection_counts"], {"recomputed_positive_rate_cap_2p5": 1})

    def test_recomputes_multiple_caps_in_one_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = _write_artifact(root, "run1")
            input_jsonl = root / "input.jsonl"
            input_jsonl.write_text(json.dumps({"returncode": 0, "artifact_dir": str(artifact), "heldout_segment": "a", "seed": 1, "val_f1": 0.30, "val_f05": 0.23, "average_precision": 0.5, "precision": 0.2, "recall": 0.3, "val_positive_rate": 0.1}) + "\n", encoding="utf-8")

            rows_by_cap = recompute_loo_threshold_caps(input_jsonl, [2.0, 2.5, 3.0], repo_root=root)

        self.assertEqual(sorted(rows_by_cap), [2.0, 2.5, 3.0])
        self.assertAlmostEqual(rows_by_cap[2.0][0]["val_f1"], 0.24)
        self.assertAlmostEqual(rows_by_cap[2.5][0]["val_f1"], 0.31)
        self.assertAlmostEqual(rows_by_cap[3.0][0]["val_f1"], 0.31)

    def test_main_batch_mode_writes_outputs_and_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = _write_artifact(root, "run1")
            input_jsonl = root / "input.jsonl"
            output_dir = root / "out"
            aggregate_json = root / "aggregate.json"
            aggregate_markdown = root / "aggregate.md"
            input_jsonl.write_text(json.dumps({"returncode": 0, "artifact_dir": str(artifact), "heldout_segment": "a", "seed": 1, "val_f1": 0.30, "val_f05": 0.23, "average_precision": 0.5, "precision": 0.2, "recall": 0.3, "val_positive_rate": 0.1}) + "\n", encoding="utf-8")

            with redirect_stdout(StringIO()):
                code = main([
                    "--input-jsonl",
                    str(input_jsonl),
                    "--caps",
                    "2.0,2.5",
                    "--output-dir",
                    str(output_dir),
                    "--output-prefix",
                    "candidate",
                    "--aggregate-json",
                    str(aggregate_json),
                    "--aggregate-markdown",
                    str(aggregate_markdown),
                    "--min-seeds-for-promotion",
                    "1",
                ])

            aggregate = json.loads(aggregate_json.read_text())
            markdown = aggregate_markdown.read_text()
            cap20_exists = (output_dir / "candidate_prratio2_recomputed.summary.json").exists()
            cap25_exists = (output_dir / "candidate_prratio2p5_recomputed.summary.json").exists()

        self.assertEqual(code, 0)
        self.assertTrue(cap20_exists)
        self.assertTrue(cap25_exists)
        self.assertEqual(aggregate["best_cap_by_median_over_seeds_f1"], 2.5)
        self.assertAlmostEqual(aggregate["cap_summaries"]["2.5"]["min_f1_retained_vs_original"], 0.31 / 0.30)
        self.assertAlmostEqual(aggregate["cap_summaries"]["2.5"]["min_f05_retained_vs_original"], 0.27 / 0.23)
        self.assertIn("# Recomputed LOO Cap Sweep", markdown)
        self.assertIn("best_cap_by_median_over_seeds_f1", markdown)
        self.assertIn("min F0.5 retained", markdown)

    def test_render_aggregate_markdown_marks_best_caps(self) -> None:
        markdown = render_aggregate_markdown({
            "caps": [2.0, 2.5],
            "best_cap_by_median_over_seeds_f1": 2.5,
            "best_cap_by_worst_fold_f1": 2.0,
            "cap_summaries": {
                "2.0": {"mean_val_f1": 0.1, "median_over_seeds_median_val_f1": 0.1, "worst_fold_val_f1": 0.05, "min_f1_retained_vs_original": 0.8, "min_f05_retained_vs_original": 0.9, "folds_failed": 0},
                "2.5": {"mean_val_f1": 0.2, "median_over_seeds_median_val_f1": 0.2, "worst_fold_val_f1": 0.04, "min_f1_retained_vs_original": 1.0, "min_f05_retained_vs_original": 1.0, "folds_failed": 0},
            },
        })

        self.assertIn("2.5 best-median", markdown)
        self.assertIn("2 best-worst", markdown)
        self.assertIn("min F1 retained", markdown)


if __name__ == "__main__":
    unittest.main()

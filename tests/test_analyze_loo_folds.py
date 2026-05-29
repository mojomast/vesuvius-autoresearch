from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_loo_folds import load_diagnostics, render_markdown


class AnalyzeLooFoldsTest(unittest.TestCase):
    def test_basic_summary_fallback_parses_pooled_folds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = Path(tmpdir) / "tiny.summary.json"
            summary.write_text(json.dumps({
                "per_fold_val_f1": {"fold_a": 0.5},
                "per_fold_average_precision": {"fold_a": 0.25},
                "folds_with_positive_rate_alarm": [],
                "folds_with_zero_precision_or_recall": [],
            }))

            rows = load_diagnostics(summary)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["fold_id"], "fold_a")
        self.assertEqual(rows[0]["seed"], "pooled")
        self.assertAlmostEqual(rows[0]["val_f1"], 0.5)

    def test_detects_low_f1_folds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = Path(tmpdir) / "tiny.summary.json"
            summary.write_text(json.dumps({"per_fold_val_f1": {"weak": 0.039}, "per_fold_average_precision": {}}))

            rows = load_diagnostics(summary)

        self.assertTrue(rows[0]["low_f1"])
        self.assertIn("low_f1", render_markdown(rows))

    def test_detects_over_cap_from_jsonl_companion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "config.json"
            config.write_text(json.dumps({"evaluation": {"max_pred_positive_rate_ratio": 3.0}}))
            summary = root / "run.summary.json"
            jsonl = root / "run.jsonl"
            summary.write_text(json.dumps({"base_config": str(config), "folds_with_positive_rate_alarm": ["fold_a:seed=1"]}))
            jsonl.write_text(json.dumps({
                "heldout_segment": "fold_a",
                "seed": 1,
                "val_f1": 0.2,
                "average_precision": 0.1,
                "pred_positive_rate": 0.5,
                "val_positive_rate": 0.1,
                "precision": 0.2,
                "recall": 0.3,
                "fixed_threshold_status": "ok",
            }) + "\n")

            rows = load_diagnostics(summary)

        self.assertTrue(rows[0]["over_cap"])
        self.assertTrue(rows[0]["positive_rate_alarm"])
        self.assertAlmostEqual(rows[0]["pred_val_ratio"], 5.0)

    def test_cli_accepts_multiple_summary_json_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first.summary.json"
            second = root / "second.summary.json"
            first.write_text(json.dumps({"per_fold_val_f1": {"a": 0.1}, "per_fold_average_precision": {"a": 0.2}}))
            second.write_text(json.dumps({"per_fold_val_f1": {"b": 0.2}, "per_fold_average_precision": {"b": 0.3}}))

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/analyze_loo_folds.py",
                    "--summary-json",
                    str(first),
                    "--summary-json",
                    str(second),
                    "--markdown",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                text=True,
                capture_output=True,
            )

        self.assertIn("| first.summary.json | a | pooled |", result.stdout)
        self.assertIn("| second.summary.json | b | pooled |", result.stdout)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.audit_research_state import audit_research_state, render_markdown


class AuditResearchStateTest(unittest.TestCase):
    def test_audit_counts_local_state_and_snapshot_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "configs" / "auto_20260526T000000Z_1.yaml").write_text("model: {}\n")
            (root / "configs" / "baseline.yaml").write_text("model: {}\n")

            run_dir = root / "experiments" / "runs" / "run1"
            run_dir.mkdir(parents=True)
            (run_dir / "metrics.json").write_text(json.dumps({"val_f1": 0.5}))
            (root / "experiments" / "runs" / "run2").mkdir()

            db = root / "experiments" / "experiments.db"
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE experiments (run_id TEXT PRIMARY KEY)")
                conn.executemany("INSERT INTO experiments VALUES (?)", [("run1",), ("run2",)])
                conn.commit()
            finally:
                conn.close()

            (root / "logs").mkdir()
            (root / "logs" / "loo.summary.json").write_text("{}\n")

            snapshot = {
                "schema_version": "vesuvius-dashboard/v1",
                "research_summary": {
                    "champions": {"peak_score": {"run_id": "run1"}},
                    "decision": {"next_action": "Run seed-repeat LOO.", "promotion_gate": {"ready": False, "criteria": [{"id": "seed_repeat_loo", "label": "Seed-repeat LOO", "state": "warning", "detail": "missing"}]}, "blocker_counts": {"missing_val_f1": 2, "zero_precision_or_recall": 1}},
                },
            }
            with mock.patch("research_dashboard.snapshot.build_snapshot", return_value=snapshot):
                report = audit_research_state(root)

        self.assertEqual(report["generated_auto_config_count"], 1)
        self.assertEqual(report["experiment_db"]["run_count"], 2)
        self.assertEqual(report["artifacts"]["run_directory_count"], 2)
        self.assertGreater(report["artifacts"]["total_bytes"], 0)
        self.assertEqual(report["logs"]["summary_json_count"], 1)
        self.assertTrue(report["dashboard"]["snapshot_contract_available"])
        self.assertEqual(report["dashboard"]["champions"]["peak_score"]["run_id"], "run1")
        self.assertFalse(report["dashboard"]["promotion_gate_ready"])
        self.assertEqual(report["dashboard"]["next_action"], "Run seed-repeat LOO.")
        self.assertEqual(report["dashboard"]["top_promotion_blockers"][0], {"code": "missing_val_f1", "count": 2})

    def test_markdown_renders_when_snapshot_unavailable(self) -> None:
        report = {
            "project_root": "/tmp/repo",
            "generated_auto_config_count": 0,
            "experiment_db": {"exists": False, "run_count": None},
            "artifacts": {"run_directory_count": 0, "total_bytes": 0},
            "logs": {"summary_json_count": 0},
            "dashboard": {"snapshot_contract_available": False},
        }

        markdown = render_markdown(report)

        self.assertIn("# Research State Audit", markdown)
        self.assertIn("Experiment DB runs: DB missing", markdown)
        self.assertIn("Dashboard snapshot contract available: False", markdown)

    def test_markdown_renders_candidate_evidence(self) -> None:
        report = {
            "project_root": "/tmp/repo",
            "generated_auto_config_count": 0,
            "experiment_db": {"exists": True, "run_count": 1},
            "artifacts": {"run_directory_count": 1, "total_bytes": 10},
            "logs": {"summary_json_count": 1},
            "dashboard": {
                "snapshot_contract_available": True,
                "next_action": "Run full-tile on weak fold weakseg before promotion review.",
                "candidate_evidence": {
                    "candidate_run_id": "candidate",
                    "loo": {"worst_fold_id": "weakseg", "worst_fold_val_f1": 0.04},
                    "full_tile": {"segments_covered": ["goodseg"]},
                    "loo_full_tile": {"segments_covered": ["goodseg", "weakseg"], "coverage_count": 2},
                    "risk_summary": {"risk_level": "warning", "warnings": ["2 LOO full-tile segment(s) have pred/val positive-rate ratio >= 3.5x"]},
                    "weak_fold_full_tile": {"status": "missing"},
                    "promotion_actions": [{"label": "Run full-tile on weak fold weakseg", "command_text": ".venv/bin/python scripts/infer_full_tile.py --segment-id weakseg"}],
                },
                "promotion_actions": [],
            },
        }

        markdown = render_markdown(report)

        self.assertIn("## Candidate Evidence", markdown)
        self.assertIn("Next action: Run full-tile on weak fold weakseg before promotion review.", markdown)
        self.assertIn("LOO full-tile diagnostics covered: goodseg, weakseg", markdown)
        self.assertIn("Positive-rate risk: warning", markdown)
        self.assertIn("2 LOO full-tile segment(s)", markdown)
        self.assertIn("`weakseg`", markdown)
        self.assertIn("Run full-tile on weak fold weakseg", markdown)
        self.assertIn("scripts/infer_full_tile.py", markdown)

    def test_audit_surfaces_top_cap_tightening_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = {
                "schema_version": "vesuvius-dashboard/v1",
                "research_summary": {"decision": {}},
                "mining": {
                    "calibration_mining_decisions": [
                        {"action": "mine_hard_negatives", "segment_id": "seg-b", "reason": "flooding"},
                        {"action": "tighten_positive_rate_cap", "segment_id": "seg-a", "target_max_pred_positive_rate_ratio": 2.5, "reason": "prratio2p5_preserves_selected_f1_f05", "selected_pred_to_val_ratio": 2.9, "f1_retained_fraction": 0.96, "f05_retained_fraction": 0.97},
                    ],
                    "cap_comparison_commands": [{"segment_id": "seg-a", "command_text": ".venv/bin/python scripts/compare_threshold_caps.py --metrics metrics.json --markdown"}],
                },
            }
            with mock.patch("research_dashboard.snapshot.build_snapshot", return_value=snapshot):
                report = audit_research_state(root)

        action = report["dashboard"]["top_calibration_mining_action"]
        markdown = render_markdown(report)

        self.assertEqual(action["action"], "tighten_positive_rate_cap")
        self.assertEqual(action["target_max_pred_positive_rate_ratio"], 2.5)
        self.assertIn("Top Calibration/Mining/Cap Action", markdown)
        self.assertIn("Target cap: 2.5x", markdown)
        self.assertIn("F0.5 retained: 0.97", markdown)
        self.assertIn("scripts/compare_threshold_caps.py", markdown)

    def test_audit_surfaces_mining_command_for_top_mine_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = {
                "schema_version": "vesuvius-dashboard/v1",
                "research_summary": {"decision": {}},
                "mining": {
                    "calibration_mining_decisions": [{"action": "mine_hard_negatives", "segment_id": "seg-a", "reason": "lower_ratio_caps_reduce_selected_f1"}],
                    "mine_commands": [{"segment_id": "seg-a", "command_text": ".venv/bin/python scripts/infer_full_tile.py --mine-output data/mined/seg-a.npz", "mine_output": "data/mined/seg-a.npz"}],
                },
            }
            with mock.patch("research_dashboard.snapshot.build_snapshot", return_value=snapshot):
                report = audit_research_state(root)

        action = report["dashboard"]["top_calibration_mining_action"]
        markdown = render_markdown(report)

        self.assertEqual(action["action"], "mine_hard_negatives")
        self.assertIn("--mine-output", action["command_text"])
        self.assertIn("scripts/infer_full_tile.py", markdown)


if __name__ == "__main__":
    unittest.main()

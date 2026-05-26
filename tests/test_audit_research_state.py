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
                    "decision": {"blocker_counts": {"missing_val_f1": 2, "zero_precision_or_recall": 1}},
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


if __name__ == "__main__":
    unittest.main()

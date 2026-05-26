from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from research_dashboard.artifacts import preview_artifact
from research_dashboard.app import make_handler
from research_dashboard.inventory import build_inventory
from research_dashboard.snapshot import build_snapshot


class ResearchDashboardTest(unittest.TestCase):
    def test_snapshot_contract_from_temp_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "configs" / "robust_multisegment_dice035_expanded.yaml").write_text("model:\n  name: tiny_torch_unet\n")
            runs_dir = root / "experiments" / "runs" / "run1"
            runs_dir.mkdir(parents=True)
            (runs_dir / "metrics.json").write_text(json.dumps({"val_f1": 0.4}))
            db = root / "experiments" / "experiments.db"
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE experiments (run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, config_json TEXT NOT NULL, main_metric REAL NOT NULL, secondary_metrics_json TEXT NOT NULL, artifact_dir TEXT NOT NULL)")
                conn.execute(
                    "INSERT INTO experiments VALUES (?,?,?,?,?,?)",
                    (
                        "run1",
                        "2026-05-26T00:00:00Z",
                        json.dumps({"model": {"name": "tiny_torch_unet"}, "evaluation": {"main_metric": "val_f1"}, "validation_setup": {"mode": "cross-segment", "train_segment_id": "a", "val_segment_id": "b"}}),
                        0.4,
                        json.dumps({"val_f1": 0.4, "average_precision": 0.2}),
                        str(runs_dir),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            with mock.patch("research_dashboard.datasets.dataset_summary", return_value={"source": "test", "scrolls": [], "splits": {}}):
                snapshot = build_snapshot(root)

        self.assertEqual(snapshot["schema_version"], "vesuvius-dashboard/v1")
        self.assertTrue(snapshot["project"]["exists"])
        self.assertEqual(snapshot["experiments"]["count"], 1)
        self.assertEqual(snapshot["experiments"]["latest"]["run_id"], "run1")
        self.assertIn("inventory", snapshot)
        self.assertIn("progress", snapshot)
        self.assertFalse(snapshot["capabilities"]["enable_runs"])

    def test_artifact_preview_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "experiments" / "runs" / "run1"
            runs.mkdir(parents=True)
            safe = runs / "metrics.json"
            safe.write_text('{"val_f1": 1.0}')
            outside = root / "README.md"
            outside.write_text("outside")

            preview = preview_artifact(root, str(safe))
            self.assertEqual(preview["preview"]["val_f1"], 1.0)
            with self.assertRaisesRegex(ValueError, "experiments/runs"):
                preview_artifact(root, str(outside))

    def test_inventory_uses_copyable_safe_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inventory = build_inventory(root)

        self.assertGreaterEqual(len(inventory["features"]), 5)
        for feature in inventory["features"]:
            self.assertIsInstance(feature["command"], list)
            self.assertFalse(feature["safe_to_execute_from_dashboard"])
            self.assertNotIn("/home/mojo/.hermes", feature["command_text"])

    def test_research_dashboard_package_has_no_hermes_specific_terms(self) -> None:
        package_root = Path(__file__).resolve().parents[1] / "research_dashboard"
        forbidden = [".hermes", "HERMES_", "Becomussy", "self-improvement", "ScrollPrize"]
        for path in package_root.glob("*.py"):
            text = path.read_text()
            for term in forbidden:
                self.assertNotIn(term, text, f"{term} leaked into {path}")

    def test_standalone_dashboard_token_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(root, "secret-token"))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(base + "/api/research", timeout=5)
                self.assertEqual(ctx.exception.code, 401)
                resp = urllib.request.urlopen(base + "/api/research?token=secret-token", timeout=5)
                data = json.loads(resp.read().decode())
                self.assertEqual(data["schema_version"], "vesuvius-dashboard/v1")
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()

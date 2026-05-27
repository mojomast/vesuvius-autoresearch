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
        self.assertIn("next_action", snapshot["research_summary"])
        self.assertIn("blocker_counts", snapshot["research_summary"])
        self.assertIn("inventory", snapshot)
        self.assertIn("progress", snapshot)
        self.assertFalse(snapshot["capabilities"]["enable_runs"])

    def test_snapshot_separates_peak_robust_and_promotable_champions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            db = root / "experiments" / "experiments.db"
            db.parent.mkdir(parents=True)
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE experiments (run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, config_json TEXT NOT NULL, main_metric REAL NOT NULL, secondary_metrics_json TEXT NOT NULL, artifact_dir TEXT NOT NULL)")

                def insert(run_id: str, timestamp: str, cfg: dict, metrics: dict, main_metric: float) -> None:
                    artifact_dir = root / "experiments" / "runs" / run_id
                    artifact_dir.mkdir(parents=True)
                    conn.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?)", (run_id, timestamp, json.dumps(cfg), main_metric, json.dumps(metrics), str(artifact_dir)))

                base_eval = {"evaluation": {"main_metric": "val_f1"}, "validation_setup": {"mode": "cross-segment", "train_segment_id": "a", "val_segment_id": "b"}}
                insert(
                    "peak_only",
                    "2026-05-26T00:03:00Z",
                    {**base_eval, "model": {"name": "tiny_numpy"}, "dataset": {"research_scope": "focused_single_segment"}},
                    {"val_f1": 0.90, "average_precision": 0.50, "precision": 0.8, "recall": 0.8, "pred_positive_rate": 0.2, "val_positive_rate": 0.1},
                    0.90,
                )
                insert(
                    "robust_blocked",
                    "2026-05-26T00:02:00Z",
                    {**base_eval, "model": {"name": "tiny_torch_unet"}, "dataset": {"research_scope": "multi_segment_robust_expanded"}},
                    {"val_f1": 0.70, "average_precision": 0.40, "precision": 0.7, "recall": 0.7, "pred_positive_rate": 0.9, "val_positive_rate": 0.1},
                    0.70,
                )
                insert(
                    "promotable_lower",
                    "2026-05-26T00:01:00Z",
                    {**base_eval, "model": {"name": "tiny_torch_unet"}, "dataset": {"research_scope": "multi_segment_robust_expanded"}},
                    {"val_f1": 0.60, "average_precision": 0.35, "precision": 0.7, "recall": 0.7, "pred_positive_rate": 0.2, "val_positive_rate": 0.1, "loo_promotion_ready": True, "full_tile_promotion_ready": True, "fixed_threshold_f1": 0.55},
                    0.60,
                )
                conn.commit()
            finally:
                conn.close()

            with mock.patch("research_dashboard.datasets.dataset_summary", return_value={"source": "test", "scrolls": [], "splits": {}}):
                snapshot = build_snapshot(root)

        champions = snapshot["experiments"]["champions"]
        self.assertEqual(champions["peak_score"]["run_id"], "peak_only")
        self.assertEqual(champions["robust_candidate"]["run_id"], "robust_blocked")
        self.assertEqual(champions["promotion_eligible"]["run_id"], "promotable_lower")
        self.assertIn("peak_score", next(run for run in snapshot["experiments"]["recent"] if run["run_id"] == "peak_only")["champion_classes"])
        self.assertIn("robust_candidate", next(run for run in snapshot["experiments"]["recent"] if run["run_id"] == "robust_blocked")["champion_classes"])
        self.assertIn("promotion_eligible", next(run for run in snapshot["experiments"]["recent"] if run["run_id"] == "promotable_lower")["champion_classes"])
        self.assertTrue(snapshot["research_summary"]["decision"]["promotion_gate"]["ready"])

    def test_snapshot_classifies_promotion_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "experiments" / "experiments.db"
            db.parent.mkdir(parents=True)
            run_dir = root / "experiments" / "runs" / "blocked"
            run_dir.mkdir(parents=True)
            cfg = {"model": {"name": "tiny_torch_unet"}, "evaluation": {"main_metric": "val_f1"}, "dataset": {"research_scope": "multi_segment_robust"}, "validation_setup": {"mode": "cross-segment", "train_segment_id": "a", "val_segment_id": "b"}}
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE experiments (run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, config_json TEXT NOT NULL, main_metric REAL NOT NULL, secondary_metrics_json TEXT NOT NULL, artifact_dir TEXT NOT NULL)")
                conn.execute(
                    "INSERT INTO experiments VALUES (?,?,?,?,?,?)",
                    (
                        "blocked",
                        "2026-05-26T00:00:00Z",
                        json.dumps(cfg),
                        0.4,
                        json.dumps({"val_f1": 0.4, "average_precision": 0.2, "precision": 0.0, "recall": 0.8, "pred_positive_rate": 0.8, "val_positive_rate": 0.1}),
                        str(run_dir),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            with mock.patch("research_dashboard.datasets.dataset_summary", return_value={"source": "test", "scrolls": [], "splits": {}}):
                snapshot = build_snapshot(root)

        run = snapshot["experiments"]["recent"][0]
        codes = {blocker["code"] for blocker in run["promotion_blockers"]}
        self.assertEqual(run["promotion_status"], "blocked")
        self.assertIn("zero_precision_or_recall", codes)
        self.assertIn("pred_positive_rate_ratio_suspicious", codes)
        self.assertEqual(snapshot["research_summary"]["decision"]["blocker_counts"]["zero_precision_or_recall"], 1)
        self.assertEqual(snapshot["research_summary"]["blocker_counts"]["zero_precision_or_recall"], 1)
        self.assertIn("next_action", snapshot["research_summary"]["decision"])
        self.assertIsInstance(snapshot["research_summary"]["next_action"], str)

    def test_robust_candidate_prefers_held_out_over_same_segment_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "experiments" / "experiments.db"
            db.parent.mkdir(parents=True)
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE experiments (run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, config_json TEXT NOT NULL, main_metric REAL NOT NULL, secondary_metrics_json TEXT NOT NULL, artifact_dir TEXT NOT NULL)")

                def insert(run_id: str, mode: str, f1: float) -> None:
                    artifact_dir = root / "experiments" / "runs" / run_id
                    artifact_dir.mkdir(parents=True)
                    cfg = {"model": {"name": "tiny_torch_unet"}, "evaluation": {"main_metric": "val_f1"}, "dataset": {"research_scope": "multi_segment_robust_expanded"}, "validation_setup": {"mode": mode, "train_segment_id": "a", "val_segment_id": "b"}}
                    metrics = {"val_f1": f1, "average_precision": 0.2, "precision": 0.4, "recall": 0.6, "pred_positive_rate": 0.2, "val_positive_rate": 0.1}
                    conn.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?)", (run_id, "2026-05-26T00:00:00Z", json.dumps(cfg), f1, json.dumps(metrics), str(artifact_dir)))

                insert("same_segment_high", "spatial-same-segment", 0.8)
                insert("heldout_lower", "cross-segment", 0.5)
                conn.commit()
            finally:
                conn.close()

            with mock.patch("research_dashboard.datasets.dataset_summary", return_value={"source": "test", "scrolls": [], "splits": {}}):
                snapshot = build_snapshot(root)

        self.assertEqual(snapshot["experiments"]["champions"]["robust_candidate"]["run_id"], "heldout_lower")

    def test_dashboard_gate_surfaces_missing_loo_and_full_tile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "experiments" / "experiments.db"
            db.parent.mkdir(parents=True)
            run_dir = root / "experiments" / "runs" / "robust"
            run_dir.mkdir(parents=True)
            cfg = {"model": {"name": "tiny_torch_unet"}, "evaluation": {"main_metric": "val_f1"}, "dataset": {"research_scope": "multi_segment_robust_expanded"}, "validation_setup": {"mode": "cross-segment", "train_segment_id": "a", "val_segment_id": "b"}}
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE experiments (run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, config_json TEXT NOT NULL, main_metric REAL NOT NULL, secondary_metrics_json TEXT NOT NULL, artifact_dir TEXT NOT NULL)")
                conn.execute(
                    "INSERT INTO experiments VALUES (?,?,?,?,?,?)",
                    (
                        "robust",
                        "2026-05-26T00:00:00Z",
                        json.dumps(cfg),
                        0.4,
                        json.dumps({"val_f1": 0.4, "average_precision": 0.2, "precision": 0.4, "recall": 0.6, "pred_positive_rate": 0.2, "val_positive_rate": 0.1, "fixed_threshold_f1": 0.3}),
                        str(run_dir),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

            with mock.patch("research_dashboard.datasets.dataset_summary", return_value={"source": "test", "scrolls": [], "splits": {}}):
                snapshot = build_snapshot(root)

        codes = {blocker["code"] for blocker in snapshot["experiments"]["recent"][0]["promotion_blockers"]}
        self.assertIn("missing_seed_repeat_loo", codes)
        self.assertIn("missing_full_tile_evidence", codes)
        self.assertFalse(snapshot["research_summary"]["decision"]["promotion_gate"]["ready"])
        self.assertLess(snapshot["progress"]["summary"]["foundation_readiness"], 100)

    def test_dashboard_reads_loo_summary_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            logs.mkdir()
            (logs / "robust.summary.json").write_text(json.dumps({"promotion_ready": True, "median_over_seeds_median_val_f1": 0.2, "worst_fold_val_f1": 0.08, "mean_average_precision": 0.11, "promotion_warnings": []}))

            with mock.patch("research_dashboard.datasets.dataset_summary", return_value={"source": "test", "scrolls": [], "splits": {}}):
                snapshot = build_snapshot(root)

        self.assertEqual(len(snapshot["experiments"]["loo_summaries"]), 1)
        gate = snapshot["research_summary"]["decision"]["promotion_gate"]
        loo_item = next(item for item in gate["criteria"] if item["id"] == "seed_repeat_loo")
        self.assertEqual(loo_item["state"], "warning")

    def test_dashboard_does_not_use_unrelated_loo_summary_for_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            logs.mkdir()
            (logs / "other.summary.json").write_text(json.dumps({"promotion_ready": True, "run_ids": ["other_run"], "promotion_warnings": []}))
            db = root / "experiments" / "experiments.db"
            db.parent.mkdir(parents=True)
            run_dir = root / "experiments" / "runs" / "candidate"
            run_dir.mkdir(parents=True)
            cfg = {"model": {"name": "tiny_torch_unet"}, "evaluation": {"main_metric": "val_f1"}, "dataset": {"research_scope": "multi_segment_robust_expanded"}, "validation_setup": {"mode": "cross-segment", "train_segment_id": "a", "val_segment_id": "b"}}
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE experiments (run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, config_json TEXT NOT NULL, main_metric REAL NOT NULL, secondary_metrics_json TEXT NOT NULL, artifact_dir TEXT NOT NULL)")
                conn.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?)", ("candidate", "2026-05-26T00:00:00Z", json.dumps(cfg), 0.4, json.dumps({"val_f1": 0.4, "average_precision": 0.2, "precision": 0.4, "recall": 0.6, "pred_positive_rate": 0.2, "val_positive_rate": 0.1}), str(run_dir)))
                conn.commit()
            finally:
                conn.close()

            with mock.patch("research_dashboard.datasets.dataset_summary", return_value={"source": "test", "scrolls": [], "splits": {}}):
                snapshot = build_snapshot(root)

        run = snapshot["experiments"]["recent"][0]
        self.assertIn("missing_seed_repeat_loo", {blocker["code"] for blocker in run["promotion_blockers"]})
        loo_item = next(item for item in snapshot["research_summary"]["decision"]["promotion_gate"]["criteria"] if item["id"] == "seed_repeat_loo")
        self.assertEqual(loo_item["state"], "warning")

    def test_dashboard_links_loo_summary_by_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            logs.mkdir()
            (logs / "candidate.summary.json").write_text(json.dumps({"promotion_ready": True, "run_ids": ["candidate"], "promotion_warnings": []}))
            db = root / "experiments" / "experiments.db"
            db.parent.mkdir(parents=True)
            run_dir = root / "experiments" / "runs" / "candidate"
            full_tile_dir = run_dir / "full_tile_b"
            full_tile_dir.mkdir(parents=True)
            (full_tile_dir / "metrics.json").write_text(json.dumps({"promotion_checks": {"eligible": True}}))
            cfg = {"model": {"name": "tiny_torch_unet"}, "evaluation": {"main_metric": "val_f1"}, "dataset": {"research_scope": "multi_segment_robust_expanded"}, "validation_setup": {"mode": "cross-segment", "train_segment_id": "a", "val_segment_id": "b"}}
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE experiments (run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, config_json TEXT NOT NULL, main_metric REAL NOT NULL, secondary_metrics_json TEXT NOT NULL, artifact_dir TEXT NOT NULL)")
                conn.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?)", ("candidate", "2026-05-26T00:00:00Z", json.dumps(cfg), 0.4, json.dumps({"val_f1": 0.4, "average_precision": 0.2, "precision": 0.4, "recall": 0.6, "pred_positive_rate": 0.2, "val_positive_rate": 0.1}), str(run_dir)))
                conn.commit()
            finally:
                conn.close()

            with mock.patch("research_dashboard.datasets.dataset_summary", return_value={"source": "test", "scrolls": [], "splits": {}}):
                snapshot = build_snapshot(root)

        run = snapshot["experiments"]["recent"][0]
        codes = {blocker["code"] for blocker in run["promotion_blockers"]}
        self.assertNotIn("missing_seed_repeat_loo", codes)
        self.assertNotIn("missing_full_tile_evidence", codes)
        self.assertEqual(snapshot["research_summary"]["decision"]["promotion_gate"]["ready"], True)

    def test_dashboard_surfaces_weak_fold_full_tile_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            logs.mkdir()
            (logs / "candidate.summary.json").write_text(json.dumps({
                "promotion_ready": True,
                "run_ids": ["candidate"],
                "seeds": [1, 2, 3],
                "worst_fold_id": "weakseg",
                "worst_fold_val_f1": 0.04,
                "per_fold_average_precision": {"weakseg": 0.03},
                "promotion_warnings": [],
            }))
            db = root / "experiments" / "experiments.db"
            db.parent.mkdir(parents=True)
            run_dir = root / "experiments" / "runs" / "candidate"
            full_tile_dir = run_dir / "full_tile_goodseg"
            full_tile_dir.mkdir(parents=True)
            (full_tile_dir / "metrics.json").write_text(json.dumps({"evaluation_region": {"type": "whole_segment", "segment_id": "goodseg"}, "promotion_checks": {"eligible": True}, "val_f1": 0.2, "average_precision": 0.1}))
            cfg = {"model": {"name": "tiny_torch_unet"}, "evaluation": {"main_metric": "val_f1"}, "dataset": {"research_scope": "multi_segment_robust_expanded"}, "validation_setup": {"mode": "leave-one-segment-out", "train_segment_id": "?", "val_segment_id": "goodseg"}}
            metrics = {"val_f1": 0.4, "average_precision": 0.2, "precision": 0.4, "recall": 0.6, "pred_positive_rate": 0.2, "val_positive_rate": 0.1}
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE experiments (run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, config_json TEXT NOT NULL, main_metric REAL NOT NULL, secondary_metrics_json TEXT NOT NULL, artifact_dir TEXT NOT NULL)")
                conn.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?)", ("candidate", "2026-05-26T00:00:00Z", json.dumps(cfg), 0.4, json.dumps(metrics), str(run_dir)))
                conn.commit()
            finally:
                conn.close()

            with mock.patch("research_dashboard.datasets.dataset_summary", return_value={"source": "test", "scrolls": [], "splits": {}}):
                snapshot = build_snapshot(root)

        evidence = snapshot["research_summary"]["candidate_evidence"]
        self.assertEqual(evidence["candidate_run_id"], "candidate")
        self.assertEqual(evidence["loo"]["worst_fold_id"], "weakseg")
        self.assertEqual(evidence["weak_fold_full_tile"]["status"], "missing")
        action = evidence["promotion_actions"][0]
        self.assertEqual(action["id"], "weak_fold_full_tile")
        self.assertIn("--segment-id weakseg", action["command_text"])
        self.assertFalse(action["safe_to_execute_from_dashboard"])

    def test_dashboard_marks_weak_fold_full_tile_done_when_metrics_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            logs.mkdir()
            (logs / "candidate.summary.json").write_text(json.dumps({"promotion_ready": True, "run_ids": ["candidate"], "worst_fold_id": "weakseg", "worst_fold_val_f1": 0.04, "promotion_warnings": []}))
            db = root / "experiments" / "experiments.db"
            db.parent.mkdir(parents=True)
            run_dir = root / "experiments" / "runs" / "candidate"
            full_tile_dir = run_dir / "full_tile_weak_fold_weakseg"
            full_tile_dir.mkdir(parents=True)
            (full_tile_dir / "metrics.json").write_text(json.dumps({"evaluation_region": {"type": "whole_segment", "segment_id": "weakseg"}, "promotion_checks": {"eligible": True}, "val_f1": 0.05, "average_precision": 0.03}))
            cfg = {"model": {"name": "tiny_torch_unet"}, "evaluation": {"main_metric": "val_f1"}, "dataset": {"research_scope": "multi_segment_robust_expanded"}, "validation_setup": {"mode": "leave-one-segment-out", "train_segment_id": "?", "val_segment_id": "weakseg"}}
            metrics = {"val_f1": 0.4, "average_precision": 0.2, "precision": 0.4, "recall": 0.6, "pred_positive_rate": 0.2, "val_positive_rate": 0.1}
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE experiments (run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, config_json TEXT NOT NULL, main_metric REAL NOT NULL, secondary_metrics_json TEXT NOT NULL, artifact_dir TEXT NOT NULL)")
                conn.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?)", ("candidate", "2026-05-26T00:00:00Z", json.dumps(cfg), 0.4, json.dumps(metrics), str(run_dir)))
                conn.commit()
            finally:
                conn.close()

            with mock.patch("research_dashboard.datasets.dataset_summary", return_value={"source": "test", "scrolls": [], "splits": {}}):
                snapshot = build_snapshot(root)

        evidence = snapshot["research_summary"]["candidate_evidence"]
        self.assertEqual(evidence["weak_fold_full_tile"]["status"], "done")
        self.assertNotEqual(evidence["promotion_actions"][0]["id"], "weak_fold_full_tile")

    def test_dashboard_detects_nested_full_tile_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "experiments" / "experiments.db"
            db.parent.mkdir(parents=True)
            run_dir = root / "experiments" / "runs" / "fulltile"
            full_tile_dir = run_dir / "full_tile_abc"
            full_tile_dir.mkdir(parents=True)
            (full_tile_dir / "metrics.json").write_text(json.dumps({"promotion_checks": {"eligible": True}, "evaluation_region": {"type": "whole_segment"}}))
            cfg = {"model": {"name": "tiny_torch_unet"}, "evaluation": {"main_metric": "val_f1"}, "dataset": {"research_scope": "multi_segment_robust_expanded"}, "validation_setup": {"mode": "leave-one-segment-out", "train_segment_id": "?", "val_segment_id": "abc"}}
            metrics = {"val_f1": 0.3, "average_precision": 0.2, "precision": 0.3, "recall": 0.6, "pred_positive_rate": 0.2, "val_positive_rate": 0.1, "loo_promotion_ready": True}
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE experiments (run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, config_json TEXT NOT NULL, main_metric REAL NOT NULL, secondary_metrics_json TEXT NOT NULL, artifact_dir TEXT NOT NULL)")
                conn.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?)", ("fulltile", "2026-05-26T00:00:00Z", json.dumps(cfg), 0.3, json.dumps(metrics), str(run_dir)))
                conn.commit()
            finally:
                conn.close()

            with mock.patch("research_dashboard.datasets.dataset_summary", return_value={"source": "test", "scrolls": [], "splits": {}}):
                snapshot = build_snapshot(root)

        codes = {blocker["code"] for blocker in snapshot["experiments"]["recent"][0]["promotion_blockers"]}
        self.assertNotIn("missing_full_tile_evidence", codes)
        full_tile_gate = next(item for item in snapshot["research_summary"]["decision"]["promotion_gate"]["criteria"] if item["id"] == "full_tile")
        self.assertEqual(full_tile_gate["state"], "done")

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

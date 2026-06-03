from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import numpy as np

from research_dashboard.artifacts import list_artifact_files, preview_artifact
from research_dashboard.app import HTML, _agent_chat, _visual_artifact_analysis, make_handler
from research_dashboard.configs import load_configs
from research_dashboard.experiments import _full_tile_ready, _hard_fold_profile, _positive_rate_risk_summary
from research_dashboard.inventory import build_inventory
from research_dashboard.operations import classify_no_progress_line, operations_snapshot
from research_dashboard.quality import decoded_output_quality
from research_dashboard.snapshot import build_snapshot, reset_snapshot_cache
from research_dashboard.settings import load_dashboard_settings


class ResearchDashboardTest(unittest.TestCase):
    def test_dashboard_has_decoded_output_gallery(self) -> None:
        self.assertIn("Decoded Output Gallery", HTML)
        self.assertIn("decoded-output-gallery-panel", HTML)
        self.assertIn("decoded-output-gallery", HTML)
        self.assertIn("decodeVisibleOutputs", HTML)
        self.assertIn("decodeAllOutputs", HTML)
        self.assertIn("decodedPreviewCache", HTML)
        self.assertIn("Download full artifact", HTML)
        self.assertIn("/api/artifact/download", HTML)
        self.assertIn("Research Usefulness Leaderboard", HTML)
        self.assertIn("quality-leaderboard-panel", HTML)
        self.assertIn("leaderboard-search", HTML)
        self.assertIn("quality-leaderboard-container", HTML)
        self.assertIn("Quality next action", HTML)
        self.assertIn("Leaderboard next action", HTML)
        self.assertIn("Validation fold matrix actions", HTML)
        self.assertIn("matrixCommandCard('seed_repeat_loo_dry_run')", HTML)
        self.assertIn("selectBestMatrixRun", HTML)
        self.assertIn("jsArg", HTML)
        self.assertIn("ledger-search", HTML)
        self.assertIn("ledger-status-filter", HTML)
        self.assertIn("ledger-scroll-container", HTML)
        self.assertIn("Text filter...", HTML)
        self.assertIn("Mining & Calibration Plan", HTML)
        self.assertIn("mining-calibration-panel", HTML)
        self.assertIn("Cap comparison commands", HTML)
        self.assertIn("Copy read-only cap comparison", HTML)
        self.assertIn("Top calibration action", HTML)
        self.assertIn("ops-no-progress-detail", HTML)
        self.assertIn("Evidence Packages", HTML)
        self.assertIn("evidence-packages-panel", HTML)
        self.assertIn("Fold-safe summary", HTML)
        self.assertIn("Inventory status counts", HTML)
        self.assertIn("Config preview valid", HTML)
        self.assertIn("image-rendering: auto", HTML)
        self.assertNotIn("image-rendering: pixelated", HTML)

    def test_decoded_output_quality_passes_coherent_structure(self) -> None:
        probs = np.full((16, 16), 0.05, dtype=np.float32)
        probs[4:12, 3:13] = 0.82
        probs[5:11, 4:12] = 0.92

        quality = decoded_output_quality(probs, threshold=0.5)

        self.assertEqual(quality["verdict"], "pass")
        self.assertGreaterEqual(quality["score"], 0.70)
        self.assertGreaterEqual(quality["supported_positive_rate"], 0.95)
        self.assertEqual(quality["component_count"], 1)
        self.assertEqual(quality["largest_component_positive_fraction"], 1.0)
        self.assertEqual(quality["small_component_positive_fraction"], 0.0)
        self.assertLess(quality["transition_density"], 0.15)

    def test_decoded_output_quality_rejects_noise_failure_modes(self) -> None:
        blank = decoded_output_quality(np.full((8, 8), 0.02, dtype=np.float32), threshold=0.5)
        flood = decoded_output_quality(np.full((8, 8), 0.90, dtype=np.float32), threshold=0.5)
        speckles = np.full((16, 16), 0.05, dtype=np.float32)
        speckles[1::3, 1::3] = 0.90
        speckle = decoded_output_quality(speckles, threshold=0.5)
        checker = (np.indices((8, 8)).sum(axis=0) % 2).astype(np.float32) * 0.90 + 0.05
        noisy = decoded_output_quality(checker, threshold=0.5)

        self.assertEqual(blank["verdict"], "fail")
        self.assertIn("blank_or_no_positive_mask", blank["flags"])
        self.assertEqual(flood["verdict"], "fail")
        self.assertIn("flooding", flood["flags"])
        self.assertEqual(speckle["verdict"], "fail")
        self.assertIn("speckle_or_isolated_positives", speckle["flags"])
        self.assertGreater(speckle["component_count"], 1)
        self.assertLess(speckle["largest_component_positive_fraction"], 0.10)
        self.assertEqual(speckle["small_component_positive_fraction"], 1.0)
        self.assertEqual(noisy["verdict"], "fail")
        self.assertIn("fragmented_threshold_mask", noisy["flags"])

    def test_hard_fold_profile_uses_summary_weak_ap_flags_without_jsonl(self) -> None:
        profile = _hard_fold_profile(
            {
                "promotion_ready": True,
                "worst_fold_id": "20230530172803",
                "worst_fold_val_f1": 0.06,
                "per_fold_val_f1": {"20230530172803": 0.06},
                "per_fold_average_precision": {"20230530172803": 0.08},
                "folds_with_weak_ap_prevalence_lift": ["20230530172803:seed=11001"],
            },
            {"evidence": []},
        )

        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile["failure_mode"], "low_ap_near_prevalence")
        self.assertEqual(profile["severity"], "blocker")

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
        self.assertIn("mining", snapshot)
        self.assertIn("mine_commands", snapshot["mining"])
        self.assertIn("cap_comparison_commands", snapshot["mining"])
        self.assertIn("blocker_counts", snapshot["research_summary"])
        self.assertIn("inventory", snapshot)
        self.assertIn("progress", snapshot)
        self.assertFalse(snapshot["capabilities"]["enable_runs"])

    def test_no_progress_cause_parser_and_snapshot(self) -> None:
        self.assertEqual(classify_no_progress_line("SKIP load_guard load1=99")["code"], "load_guard")
        self.assertEqual(classify_no_progress_line("Run linked seed-repeat LOO before promotion")["code"], "promotion_evidence_required")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "logs").mkdir()
            (root / "logs" / "autoresearch_guard.log").write_text("SKIP mem_guard mem_available_gib=1\n")

            ops = operations_snapshot(root)

        self.assertEqual(ops["no_progress"]["latest"]["code"], "mem_guard")
        self.assertEqual(ops["no_progress"]["counts"]["mem_guard"], 1)

    def test_config_inventory_skips_disappearing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            (cfg_dir / "good.yaml").write_text("model:\n  name: tiny_torch_unet\n")
            (cfg_dir / "vanished.yaml").symlink_to("missing.yaml")

            configs = load_configs(root)

        self.assertEqual([cfg["name"] for cfg in configs], ["good.yaml"])

    def test_snapshot_cache_reuses_and_can_bypass_or_reset(self) -> None:
        reset_snapshot_cache()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with mock.patch("research_dashboard.snapshot.build_inventory", side_effect=[{"stamp": 1}, {"stamp": 2}, {"stamp": 3}]) as inventory:
                    first = build_snapshot(root)
                    second = build_snapshot(root)
                    second["inventory"]["stamp"] = 99
                    third = build_snapshot(root)
                    bypassed = build_snapshot(root, use_cache=False)
                    reset_snapshot_cache()
                    after_reset = build_snapshot(root)

            self.assertEqual(inventory.call_count, 3)
            self.assertEqual(first["inventory"], {"stamp": 1})
            self.assertEqual(third["inventory"], {"stamp": 1})
            self.assertEqual(bypassed["inventory"], {"stamp": 2})
            self.assertEqual(after_reset["inventory"], {"stamp": 3})
        finally:
            reset_snapshot_cache()

    def test_snapshot_mining_plan_uses_dataset_fold_maps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            mined_dir = root / "data" / "mined"
            mined_dir.mkdir(parents=True)
            np.savez_compressed(mined_dir / "seg-a.npz", images=np.zeros((1, 1, 4, 4), dtype=np.float32), labels=np.zeros((1, 1, 4, 4), dtype=np.float32))
            (mined_dir / "seg-a.metadata.json").write_text(json.dumps({"mined_segment_id": "seg-a", "samples": 1, "forbidden_heldout_segments": ["seg-a"]}))
            db = root / "experiments" / "experiments.db"
            db.parent.mkdir(parents=True)
            run_dir = root / "experiments" / "runs" / "run1"
            run_dir.mkdir(parents=True)
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE experiments (run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, config_json TEXT NOT NULL, main_metric REAL NOT NULL, secondary_metrics_json TEXT NOT NULL, artifact_dir TEXT NOT NULL)")
                conn.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?)", ("run1", "2026-05-26T00:00:00Z", json.dumps({"model": {"name": "tiny_torch_unet"}, "dataset": {"research_scope": "multi_segment_robust_expanded"}, "validation_setup": {"mode": "leave-one-segment-out", "val_segment_id": "seg-b"}}), 0.4, json.dumps({"val_f1": 0.4, "average_precision": 0.2, "precision": 0.4, "recall": 0.6, "pred_positive_rate": 0.2, "val_positive_rate": 0.1}), str(run_dir)))
                conn.commit()
            finally:
                conn.close()

            with mock.patch("research_dashboard.datasets.dataset_summary", return_value={"source": "test", "scrolls": [], "splits": {}}), mock.patch("research_dashboard.snapshot.fold_maps", return_value=[{"heldout_segments": ["seg-a", "seg-b"]}]):
                snapshot = build_snapshot(root)

        by_fold = snapshot["mining"]["fold_safe_extra_train_npzs_by_heldout"]
        self.assertIn("seg-a", by_fold)
        self.assertIn("seg-b", by_fold)
        self.assertEqual(by_fold["seg-b"]["eligible_extra_train_npzs"], ["data/mined/seg-a.npz"])

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
        actions = snapshot["research_summary"]["candidate_evidence"]["promotion_actions"]
        loo_action = next(action for action in actions if action["id"] == "seed_repeat_loo")
        self.assertIn("scripts/evaluate_leave_one_out.py", loo_action["command_text"])
        self.assertIn("--base-config experiments/runs/robust/config.json", loo_action["command_text"])
        self.assertIn("--output-jsonl logs/robust_seedrepeat_loo.jsonl", loo_action["command_text"])
        self.assertFalse(loo_action["safe_to_execute_from_dashboard"])

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
            (full_tile_dir / "metrics.json").write_text(json.dumps({"promotion_checks": {"eligible": True}, "evaluation_region": {"type": "whole_segment", "segment_id": "b"}}))
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

    def test_dashboard_reports_linked_failed_loo_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            logs.mkdir()
            (logs / "candidate.summary.json").write_text(json.dumps({"promotion_ready": False, "run_ids": ["candidate"], "worst_fold_id": "weak", "worst_fold_val_f1": 0.04, "folds_requested": 6, "folds_successful": 4, "folds_failed": 2, "expected_folds": ["weak", "strong"], "planned_folds": ["weak"], "missing_expected_folds": ["strong"], "missing_expected_seed_repeats": ["weak:2"], "expected_seeds": [1, 2], "expected_task_count": 4, "planned_task_count": 2, "task_limit_applied": True, "min_seeds_for_promotion": 2, "distinct_successful_seeds": 1, "per_fold_successful_seeds": {"weak": 1}, "folds_with_zero_positive_validation": ["weak:seed=1"], "folds_with_weak_ap_prevalence_lift": ["weak:seed=1"], "promotion_warnings": []}))
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

        loo = snapshot["research_summary"]["decision"]["candidate_evidence"]["loo"]
        self.assertEqual(loo["linked"], True)
        self.assertEqual(loo["ready"], False)
        self.assertEqual(loo["worst_fold_id"], "weak")
        self.assertEqual(loo["folds_requested"], 6)
        self.assertEqual(loo["missing_expected_folds"], ["strong"])
        self.assertEqual(loo["missing_expected_seed_repeats"], ["weak:2"])
        self.assertEqual(loo["folds_with_zero_positive_validation"], ["weak:seed=1"])
        self.assertEqual(loo["folds_with_weak_ap_prevalence_lift"], ["weak:seed=1"])
        self.assertEqual(loo["per_fold_successful_seeds"], {"weak": 1})
        self.assertIn("missing_seed_repeat_loo", {blocker["code"] for blocker in snapshot["experiments"]["recent"][0]["promotion_blockers"]})

    def test_dashboard_adds_lift_and_zero_positive_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "experiments" / "experiments.db"
            db.parent.mkdir(parents=True)
            run_dir = root / "experiments" / "runs" / "candidate"
            run_dir.mkdir(parents=True)
            cfg = {"model": {"name": "tiny_torch_unet"}, "evaluation": {"main_metric": "val_f1"}, "dataset": {"research_scope": "multi_segment_robust_expanded"}, "validation_setup": {"mode": "cross-segment", "train_segment_id": "a", "val_segment_id": "b"}}
            metrics = {"val_f1": 0.2, "average_precision": 0.0, "precision": 0.1, "recall": 0.1, "pred_positive_rate": 0.0, "val_positive_rate": 0.0, "ap_prevalence_lift": 1.0}
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE experiments (run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, config_json TEXT NOT NULL, main_metric REAL NOT NULL, secondary_metrics_json TEXT NOT NULL, artifact_dir TEXT NOT NULL)")
                conn.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?)", ("candidate", "2026-05-26T00:00:00Z", json.dumps(cfg), 0.2, json.dumps(metrics), str(run_dir)))
                conn.commit()
            finally:
                conn.close()

            with mock.patch("research_dashboard.datasets.dataset_summary", return_value={"source": "test", "scrolls": [], "splits": {}}):
                snapshot = build_snapshot(root)

        codes = {blocker["code"] for blocker in snapshot["experiments"]["recent"][0]["promotion_blockers"]}
        self.assertIn("zero_positive_validation", codes)
        self.assertIn("weak_ap_prevalence_lift", codes)

    def test_dashboard_blocks_validation_segment_leakage_despite_heldout_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            logs.mkdir()
            (logs / "leaky.summary.json").write_text(json.dumps({"promotion_ready": True, "run_ids": ["leaky"], "promotion_warnings": []}))
            db = root / "experiments" / "experiments.db"
            db.parent.mkdir(parents=True)
            run_dir = root / "experiments" / "runs" / "leaky"
            full_tile_dir = run_dir / "full_tile_def"
            full_tile_dir.mkdir(parents=True)
            (full_tile_dir / "metrics.json").write_text(json.dumps({"promotion_checks": {"eligible": True}, "evaluation_region": {"type": "whole_segment", "segment_id": "def"}}))
            cfg = {"model": {"name": "tiny_torch_unet"}, "evaluation": {"main_metric": "val_f1"}, "dataset": {"research_scope": "multi_segment_robust_expanded"}, "validation_setup": {"mode": "leave-one-segment-out", "train_segment_id": "abc", "train_segments": ["abc", "def"], "val_segment_id": "def"}}
            metrics = {"val_f1": 0.40, "average_precision": 0.30, "precision": 0.5, "recall": 0.6, "pred_positive_rate": 0.20, "val_positive_rate": 0.10, "fixed_threshold_f1": 0.30}
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE experiments (run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, config_json TEXT NOT NULL, main_metric REAL NOT NULL, secondary_metrics_json TEXT NOT NULL, artifact_dir TEXT NOT NULL)")
                conn.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?)", ("leaky", "2026-05-26T00:00:00Z", json.dumps(cfg), 0.4, json.dumps(metrics), str(run_dir)))
                conn.commit()
            finally:
                conn.close()

            with mock.patch("research_dashboard.datasets.dataset_summary", return_value={"source": "test", "scrolls": [], "splits": {}}):
                snapshot = build_snapshot(root)

        run = snapshot["experiments"]["recent"][0]
        self.assertEqual(run["validation_setup"]["train_segments"], ["abc", "def"])
        self.assertEqual(run["promotion_status"], "blocked")
        self.assertIn("validation_segment_leakage", {blocker["code"] for blocker in run["promotion_blockers"]})
        self.assertFalse(snapshot["research_summary"]["decision"]["promotion_gate"]["ready"])

    def test_full_tile_ready_rejects_probability_map_artifact_alone(self) -> None:
        run = {"artifacts": [{"name": "full_tile_abc/probability_map.npy"}]}

        self.assertFalse(_full_tile_ready(run))

    def test_full_tile_ready_accepts_eligible_whole_segment_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            full_tile_dir = run_dir / "full_tile_abc"
            full_tile_dir.mkdir()
            (full_tile_dir / "metrics.json").write_text(json.dumps({"promotion_checks": {"eligible": True}, "evaluation_region": {"type": "whole_segment", "segment_id": "abc"}}))

            self.assertTrue(_full_tile_ready({"artifact_dir": str(run_dir)}))

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
        self.assertIn("--public-retry-count 5", action["command_text"])
        self.assertIn("--public-retry-delay-sec 180", action["command_text"])
        self.assertIn("--public-chunk-delay-sec 0.5", action["command_text"])
        self.assertIn("--public-chunk-retry-count 5", action["command_text"])
        self.assertIn("--public-chunk-retry-delay-sec 180", action["command_text"])
        self.assertFalse(action["safe_to_execute_from_dashboard"])
        self.assertIn("Run full-tile on weak fold weakseg", snapshot["research_summary"]["decision"]["next_action"])
        self.assertIn("before promotion review", snapshot["research_summary"]["decision"]["next_action"])
        self.assertIn("Run full-tile on weak fold weakseg", snapshot["progress"]["summary"]["next_action"])

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

    def test_dashboard_marks_weak_fold_full_tile_done_from_candidate_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            logs.mkdir()
            (logs / "candidate.summary.json").write_text(json.dumps({"promotion_ready": True, "run_ids": ["candidate"], "worst_fold_id": "weakseg", "worst_fold_val_f1": 0.04, "promotion_warnings": []}))
            (logs / "candidate.jsonl").write_text(json.dumps({"run_id": "loo_weak", "artifact_dir": str(root / "experiments" / "runs" / "loo_weak"), "heldout_segment": "weakseg", "seed": 15050, "val_f1": 0.04, "returncode": 0}) + "\n")
            db = root / "experiments" / "experiments.db"
            db.parent.mkdir(parents=True)
            run_dir = root / "experiments" / "runs" / "candidate"
            full_tile_dir = run_dir / "full_tile_weakseg"
            full_tile_dir.mkdir(parents=True)
            (root / "experiments" / "runs" / "loo_weak").mkdir(parents=True)
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

        weak = snapshot["research_summary"]["candidate_evidence"]["weak_fold_full_tile"]
        self.assertEqual(weak["status"], "done")
        self.assertIn("experiments/runs/candidate/full_tile_weakseg/metrics.json", weak["metrics"]["relative_path"])
        self.assertIn("experiments/runs/loo_weak", weak["command_text"])
        action_ids = {item["id"] for item in snapshot["research_summary"]["candidate_evidence"]["promotion_actions"]}
        self.assertNotIn("weak_fold_full_tile", action_ids)

    def test_dashboard_uses_loo_heldout_artifact_for_weak_fold_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            logs.mkdir()
            summary = logs / "candidate.summary.json"
            summary.write_text(json.dumps({"promotion_ready": True, "run_ids": ["candidate"], "worst_fold_id": "weakseg", "worst_fold_val_f1": 0.047, "promotion_warnings": []}))
            (logs / "candidate.jsonl").write_text(json.dumps({"run_id": "loo_weak", "artifact_dir": str(root / "experiments" / "runs" / "loo_weak"), "heldout_segment": "weakseg", "seed": 15050, "val_f1": 0.047, "average_precision": 0.03, "best_threshold": 0.29, "pred_positive_rate": 0.40, "val_positive_rate": 0.10, "returncode": 0}) + "\n")
            db = root / "experiments" / "experiments.db"
            db.parent.mkdir(parents=True)
            candidate_dir = root / "experiments" / "runs" / "candidate"
            candidate_tile_dir = candidate_dir / "full_tile_goodseg"
            candidate_tile_dir.mkdir(parents=True)
            (candidate_tile_dir / "metrics.json").write_text(json.dumps({"evaluation_region": {"type": "whole_segment", "segment_id": "goodseg"}, "promotion_checks": {"eligible": True}, "pred_positive_rate": 0.36, "val_positive_rate": 0.09}))
            loo_tile_dir = root / "experiments" / "runs" / "loo_weak" / "full_tile_weakseg"
            loo_tile_dir.mkdir(parents=True)
            (loo_tile_dir / "metrics.json").write_text(json.dumps({"evaluation_region": {"type": "whole_segment", "segment_id": "weakseg"}, "promotion_checks": {"eligible": True}, "val_f1": 0.12, "average_precision": 0.08, "best_threshold": 0.31, "pred_positive_rate": 0.20, "val_positive_rate": 0.05}))
            cfg = {"model": {"name": "tiny_torch_unet"}, "evaluation": {"main_metric": "val_f1"}, "dataset": {"research_scope": "multi_segment_robust_expanded"}, "validation_setup": {"mode": "leave-one-segment-out", "train_segment_id": "?", "val_segment_id": "goodseg"}}
            metrics = {"val_f1": 0.4, "average_precision": 0.2, "precision": 0.4, "recall": 0.6, "pred_positive_rate": 0.2, "val_positive_rate": 0.1}
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE experiments (run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, config_json TEXT NOT NULL, main_metric REAL NOT NULL, secondary_metrics_json TEXT NOT NULL, artifact_dir TEXT NOT NULL)")
                conn.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?)", ("candidate", "2026-05-26T00:00:00Z", json.dumps(cfg), 0.4, json.dumps(metrics), str(candidate_dir)))
                conn.commit()
            finally:
                conn.close()

            with mock.patch("research_dashboard.datasets.dataset_summary", return_value={"source": "test", "scrolls": [], "splits": {}}):
                snapshot = build_snapshot(root)

        weak = snapshot["research_summary"]["candidate_evidence"]["weak_fold_full_tile"]
        self.assertEqual(weak["loo_run"]["run_id"], "loo_weak")
        self.assertIn("experiments/runs/loo_weak", weak["command_text"])
        self.assertNotIn("--artifact experiments/runs/candidate", weak["command_text"])
        loo_full = snapshot["research_summary"]["candidate_evidence"]["loo_full_tile"]
        self.assertEqual(loo_full["coverage_count"], 1)
        self.assertEqual(loo_full["segments_covered"], ["weakseg"])
        self.assertEqual(loo_full["evidence"][0]["loo_run_id"], "loo_weak")
        self.assertEqual(loo_full["evidence"][0]["loo_pred_positive_rate"], 0.40)
        risk = snapshot["research_summary"]["candidate_evidence"]["risk_summary"]
        self.assertEqual(risk["risk_level"], "warning")
        self.assertAlmostEqual(risk["loo_sampled_vs_full_tile"][0]["sampled_to_full_pred_positive_rate_ratio"], 2.0)
        self.assertTrue(any("LOO full-tile" in warning for warning in risk["warnings"]))

    def test_positive_rate_risk_summary_is_strict_json_safe(self) -> None:
        risk = _positive_rate_risk_summary(
            {"metrics": {"pred_positive_rate": float("nan"), "val_positive_rate": 0.1}},
            [{"segment_id": "seg", "pred_positive_rate": float("inf"), "val_positive_rate": 0.1}],
            {"evidence": [{"segment_id": "seg", "loo_pred_positive_rate": float("nan"), "pred_positive_rate": 0.2, "loo_best_threshold": float("inf"), "best_threshold": 0.3}]},
        )

        json.dumps(risk, allow_nan=False)
        self.assertIsNone(risk["candidate_pred_to_val_ratio"])
        self.assertEqual(risk["loo_sampled_vs_full_tile"][0]["sampled_pred_positive_rate"], None)
        self.assertEqual(risk["loo_sampled_vs_full_tile"][0]["sampled_best_threshold"], None)

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

    def test_dashboard_surfaces_full_tile_quality_and_leaderboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "experiments" / "experiments.db"
            db.parent.mkdir(parents=True)
            run_dir = root / "experiments" / "runs" / "quality_run"
            full_tile_dir = run_dir / "full_tile_abc"
            full_tile_dir.mkdir(parents=True)
            (full_tile_dir / "metrics.json").write_text(json.dumps({"promotion_checks": {"eligible": True}, "evaluation_region": {"type": "whole_segment", "segment_id": "abc"}, "val_f1": 0.22, "average_precision": 0.2, "val_positive_rate": 0.08, "pred_positive_rate": 0.14, "fixed_threshold_f1": 0.12}))
            cfg = {"model": {"name": "tiny_torch_unet"}, "evaluation": {"main_metric": "val_f1"}, "dataset": {"research_scope": "multi_segment_robust_expanded"}, "autoresearch": {"scope_policy": "recent_robust_torch_winner"}, "validation_setup": {"mode": "leave-one-segment-out", "train_segment_id": "?", "val_segment_id": "abc"}}
            metrics = {"val_f1": 0.22, "average_precision": 0.2, "precision": 0.3, "recall": 0.6, "pred_positive_rate": 0.14, "val_positive_rate": 0.08, "loo_promotion_ready": True, "fixed_threshold_f1": 0.12, "extra_train_npz_count": 1, "extra_train_samples": 128}
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE experiments (run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, config_json TEXT NOT NULL, main_metric REAL NOT NULL, secondary_metrics_json TEXT NOT NULL, artifact_dir TEXT NOT NULL)")
                conn.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?)", ("quality_run", "2026-05-26T00:00:00Z", json.dumps(cfg), 0.22, json.dumps(metrics), str(run_dir)))
                conn.commit()
            finally:
                conn.close()

            with mock.patch("research_dashboard.datasets.dataset_summary", return_value={"source": "test", "scrolls": [], "splits": {}}):
                snapshot = build_snapshot(root)

        evidence = snapshot["research_summary"]["candidate_evidence"]["full_tile"]["evidence"][0]
        self.assertIn(evidence["quality_verdict"]["verdict"], {"pass", "review"})
        self.assertEqual(snapshot["experiments"]["leaderboard"][0]["run_id"], "quality_run")
        self.assertIn("quality_verdict", snapshot["experiments"]["leaderboard"][0])
        self.assertIn("quality_next_actions", evidence)
        self.assertIn("quality_next_actions", snapshot["research_summary"]["candidate_evidence"]["full_tile"])
        self.assertIn("quality_next_action", snapshot["experiments"]["leaderboard"][0])
        self.assertEqual(snapshot["experiments"]["leaderboard"][0]["research_scope"], "multi_segment_robust_expanded")
        self.assertEqual(snapshot["experiments"]["leaderboard"][0]["scope_policy"], "recent_robust_torch_winner")
        self.assertEqual(snapshot["experiments"]["leaderboard"][0]["extra_train_npz_count"], 1)
        self.assertEqual(snapshot["research_summary"]["candidate_evidence"]["research_scope"], "multi_segment_robust_expanded")

    def test_quality_action_uses_fixed_threshold_failure_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "experiments" / "experiments.db"
            db.parent.mkdir(parents=True)
            run_dir = root / "experiments" / "runs" / "fixed_fail"
            full_tile_dir = run_dir / "full_tile_abc"
            full_tile_dir.mkdir(parents=True)
            (full_tile_dir / "metrics.json").write_text(json.dumps({"promotion_checks": {"eligible": True}, "evaluation_region": {"type": "whole_segment", "segment_id": "abc"}, "val_f1": 0.30, "average_precision": 0.20, "val_positive_rate": 0.10, "pred_positive_rate": 0.25, "fixed_threshold_f1": 0.0, "fixed_threshold_failure_reason": "no_fixed_positive_predictions"}))
            cfg = {"model": {"name": "tiny_torch_unet"}, "evaluation": {"main_metric": "val_f1"}, "dataset": {"research_scope": "multi_segment_robust_expanded"}, "validation_setup": {"mode": "leave-one-segment-out", "train_segment_id": "?", "val_segment_id": "abc"}}
            metrics = {"val_f1": 0.30, "average_precision": 0.20, "precision": 0.3, "recall": 0.6, "pred_positive_rate": 0.25, "val_positive_rate": 0.10, "loo_promotion_ready": True, "fixed_threshold_f1": 0.0, "fixed_threshold_failure_reason": "no_fixed_positive_predictions"}
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE experiments (run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, config_json TEXT NOT NULL, main_metric REAL NOT NULL, secondary_metrics_json TEXT NOT NULL, artifact_dir TEXT NOT NULL)")
                conn.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?)", ("fixed_fail", "2026-05-26T00:00:00Z", json.dumps(cfg), 0.30, json.dumps(metrics), str(run_dir)))
                conn.commit()
            finally:
                conn.close()

            with mock.patch("research_dashboard.datasets.dataset_summary", return_value={"source": "test", "scrolls": [], "splits": {}}):
                snapshot = build_snapshot(root)

        evidence = snapshot["research_summary"]["candidate_evidence"]["full_tile"]["evidence"][0]
        reasons = {action["reason"] for action in evidence["quality_next_actions"]}
        self.assertIn("no_fixed_positive_predictions", reasons)
        self.assertIn("fixed threshold predicts no positives", evidence["quality_next_action"])

    def test_quality_review_action_precedes_promotion_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            logs.mkdir()
            (logs / "quality_review.summary.json").write_text(json.dumps({"promotion_ready": True, "run_ids": ["quality_review"], "promotion_warnings": []}))
            db = root / "experiments" / "experiments.db"
            db.parent.mkdir(parents=True)
            run_dir = root / "experiments" / "runs" / "quality_review"
            full_tile_dir = run_dir / "full_tile_abc"
            full_tile_dir.mkdir(parents=True)
            (full_tile_dir / "metrics.json").write_text(json.dumps({"promotion_checks": {"eligible": True}, "evaluation_region": {"type": "whole_segment", "segment_id": "abc"}, "val_f1": 0.50, "average_precision": 0.30, "val_positive_rate": 0.10, "pred_positive_rate": 0.25, "fixed_threshold_f1": 0.30}))
            cfg = {"model": {"name": "tiny_torch_unet"}, "evaluation": {"main_metric": "val_f1"}, "dataset": {"research_scope": "multi_segment_robust_expanded"}, "validation_setup": {"mode": "leave-one-segment-out", "train_segment_id": "?", "val_segment_id": "abc"}}
            metrics = {"val_f1": 0.50, "average_precision": 0.30, "precision": 0.5, "recall": 0.6, "pred_positive_rate": 0.25, "val_positive_rate": 0.10, "loo_promotion_ready": True, "fixed_threshold_f1": 0.30}
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE experiments (run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, config_json TEXT NOT NULL, main_metric REAL NOT NULL, secondary_metrics_json TEXT NOT NULL, artifact_dir TEXT NOT NULL)")
                conn.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?)", ("quality_review", "2026-05-26T00:00:00Z", json.dumps(cfg), 0.5, json.dumps(metrics), str(run_dir)))
                conn.commit()
            finally:
                conn.close()

            with mock.patch("research_dashboard.datasets.dataset_summary", return_value={"source": "test", "scrolls": [], "splits": {}}):
                snapshot = build_snapshot(root)

        actions = snapshot["research_summary"]["candidate_evidence"]["promotion_actions"]
        self.assertEqual(snapshot["experiments"]["recent"][0]["promotion_status"], "blocked")
        self.assertEqual(actions[0]["id"], "review_positive_rate")
        self.assertIn("Review positive-rate ratio", snapshot["research_summary"]["decision"]["next_action"])

    def test_quality_fail_blocks_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            logs.mkdir()
            (logs / "quality_fail.summary.json").write_text(json.dumps({"promotion_ready": True, "run_ids": ["quality_fail"], "promotion_warnings": []}))
            db = root / "experiments" / "experiments.db"
            db.parent.mkdir(parents=True)
            run_dir = root / "experiments" / "runs" / "quality_fail"
            full_tile_dir = run_dir / "full_tile_abc"
            full_tile_dir.mkdir(parents=True)
            (full_tile_dir / "metrics.json").write_text(json.dumps({"promotion_checks": {"eligible": True}, "evaluation_region": {"type": "whole_segment", "segment_id": "abc"}, "val_f1": 0.02, "average_precision": 0.02, "val_positive_rate": 0.10, "pred_positive_rate": 0.90, "fixed_threshold_f1": 0.0}))
            cfg = {"model": {"name": "tiny_torch_unet"}, "evaluation": {"main_metric": "val_f1"}, "dataset": {"research_scope": "multi_segment_robust_expanded"}, "validation_setup": {"mode": "leave-one-segment-out", "train_segment_id": "?", "val_segment_id": "abc"}}
            metrics = {"val_f1": 0.50, "average_precision": 0.30, "precision": 0.5, "recall": 0.6, "pred_positive_rate": 0.20, "val_positive_rate": 0.10, "loo_promotion_ready": True, "fixed_threshold_f1": 0.30}
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE experiments (run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, config_json TEXT NOT NULL, main_metric REAL NOT NULL, secondary_metrics_json TEXT NOT NULL, artifact_dir TEXT NOT NULL)")
                conn.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?)", ("quality_fail", "2026-05-26T00:00:00Z", json.dumps(cfg), 0.5, json.dumps(metrics), str(run_dir)))
                conn.commit()
            finally:
                conn.close()

            with mock.patch("research_dashboard.datasets.dataset_summary", return_value={"source": "test", "scrolls": [], "splits": {}}):
                snapshot = build_snapshot(root)

        run = snapshot["experiments"]["recent"][0]
        actions = snapshot["research_summary"]["candidate_evidence"]["promotion_actions"]
        self.assertEqual(run["promotion_status"], "blocked")
        self.assertIn("full_tile_quality_fail", {blocker["code"] for blocker in run["promotion_blockers"]})
        self.assertEqual(actions[0]["id"], "calibrate_positive_rate")
        self.assertNotIn("promotion_review", {action["id"] for action in actions})

    def test_quality_review_blocks_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            logs.mkdir()
            (logs / "quality_review.summary.json").write_text(json.dumps({"promotion_ready": True, "run_ids": ["quality_review"], "promotion_warnings": []}))
            db = root / "experiments" / "experiments.db"
            db.parent.mkdir(parents=True)
            run_dir = root / "experiments" / "runs" / "quality_review"
            full_tile_dir = run_dir / "full_tile_abc"
            full_tile_dir.mkdir(parents=True)
            (full_tile_dir / "metrics.json").write_text(json.dumps({"promotion_checks": {"eligible": True}, "evaluation_region": {"type": "whole_segment", "segment_id": "abc"}, "val_f1": 0.25, "average_precision": 0.20, "val_positive_rate": 0.10, "pred_positive_rate": 0.27, "fixed_threshold_f1": 0.20, "fixed_threshold_status": "ok"}))
            cfg = {"model": {"name": "tiny_torch_unet"}, "evaluation": {"main_metric": "val_f1"}, "dataset": {"research_scope": "multi_segment_robust_expanded"}, "validation_setup": {"mode": "leave-one-segment-out", "train_segment_id": "?", "val_segment_id": "abc"}}
            metrics = {"val_f1": 0.50, "average_precision": 0.30, "precision": 0.5, "recall": 0.6, "pred_positive_rate": 0.20, "val_positive_rate": 0.10, "loo_promotion_ready": True, "fixed_threshold_f1": 0.30}
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE experiments (run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, config_json TEXT NOT NULL, main_metric REAL NOT NULL, secondary_metrics_json TEXT NOT NULL, artifact_dir TEXT NOT NULL)")
                conn.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?)", ("quality_review", "2026-05-26T00:00:00Z", json.dumps(cfg), 0.5, json.dumps(metrics), str(run_dir)))
                conn.commit()
            finally:
                conn.close()

            with mock.patch("research_dashboard.datasets.dataset_summary", return_value={"source": "test", "scrolls": [], "splits": {}}):
                snapshot = build_snapshot(root)

        run = snapshot["experiments"]["recent"][0]
        self.assertEqual(run["promotion_status"], "blocked")
        self.assertIn("full_tile_quality_fail", {blocker["code"] for blocker in run["promotion_blockers"]})
        self.assertFalse(snapshot["research_summary"]["decision"]["promotion_gate"]["ready"])

    def test_positive_rate_ratio_above_3p5_blocks_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            logs.mkdir()
            (logs / "ratio.summary.json").write_text(json.dumps({"promotion_ready": True, "run_ids": ["ratio"], "promotion_warnings": []}))
            db = root / "experiments" / "experiments.db"
            db.parent.mkdir(parents=True)
            run_dir = root / "experiments" / "runs" / "ratio"
            full_tile_dir = run_dir / "full_tile_abc"
            full_tile_dir.mkdir(parents=True)
            (full_tile_dir / "metrics.json").write_text(json.dumps({"promotion_checks": {"eligible": True}, "evaluation_region": {"type": "whole_segment", "segment_id": "abc"}}))
            cfg = {"model": {"name": "tiny_torch_unet"}, "evaluation": {"main_metric": "val_f1"}, "dataset": {"research_scope": "multi_segment_robust_expanded"}, "validation_setup": {"mode": "leave-one-segment-out", "train_segment_id": "?", "val_segment_id": "abc"}}
            metrics = {"val_f1": 0.40, "average_precision": 0.30, "precision": 0.5, "recall": 0.6, "pred_positive_rate": 0.351, "val_positive_rate": 0.10, "loo_promotion_ready": True, "fixed_threshold_f1": 0.30}
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE experiments (run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, config_json TEXT NOT NULL, main_metric REAL NOT NULL, secondary_metrics_json TEXT NOT NULL, artifact_dir TEXT NOT NULL)")
                conn.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?)", ("ratio", "2026-05-26T00:00:00Z", json.dumps(cfg), 0.4, json.dumps(metrics), str(run_dir)))
                conn.commit()
            finally:
                conn.close()

            with mock.patch("research_dashboard.datasets.dataset_summary", return_value={"source": "test", "scrolls": [], "splits": {}}):
                snapshot = build_snapshot(root)

        run = snapshot["experiments"]["recent"][0]
        self.assertEqual(run["promotion_status"], "blocked")
        self.assertIn("pred_positive_rate_ratio_suspicious", {blocker["code"] for blocker in run["promotion_blockers"]})
        self.assertFalse(snapshot["research_summary"]["decision"]["promotion_gate"]["ready"])

    def test_fixed_threshold_status_weak_blocks_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            logs.mkdir()
            (logs / "fixed_status.summary.json").write_text(json.dumps({"promotion_ready": True, "run_ids": ["fixed_status"], "promotion_warnings": []}))
            db = root / "experiments" / "experiments.db"
            db.parent.mkdir(parents=True)
            run_dir = root / "experiments" / "runs" / "fixed_status"
            full_tile_dir = run_dir / "full_tile_abc"
            full_tile_dir.mkdir(parents=True)
            (full_tile_dir / "metrics.json").write_text(json.dumps({"promotion_checks": {"eligible": True}, "evaluation_region": {"type": "whole_segment", "segment_id": "abc"}}))
            cfg = {"model": {"name": "tiny_torch_unet"}, "evaluation": {"main_metric": "val_f1"}, "dataset": {"research_scope": "multi_segment_robust_expanded"}, "validation_setup": {"mode": "leave-one-segment-out", "train_segment_id": "?", "val_segment_id": "abc"}}
            metrics = {"val_f1": 0.40, "average_precision": 0.30, "precision": 0.5, "recall": 0.6, "pred_positive_rate": 0.20, "val_positive_rate": 0.10, "loo_promotion_ready": True, "fixed_threshold_f1": 0.30, "fixed_threshold_status": "weak"}
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE experiments (run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, config_json TEXT NOT NULL, main_metric REAL NOT NULL, secondary_metrics_json TEXT NOT NULL, artifact_dir TEXT NOT NULL)")
                conn.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?)", ("fixed_status", "2026-05-26T00:00:00Z", json.dumps(cfg), 0.4, json.dumps(metrics), str(run_dir)))
                conn.commit()
            finally:
                conn.close()

            with mock.patch("research_dashboard.datasets.dataset_summary", return_value={"source": "test", "scrolls": [], "splits": {}}):
                snapshot = build_snapshot(root)

        run = snapshot["experiments"]["recent"][0]
        self.assertEqual(run["promotion_status"], "blocked")
        self.assertIn("fixed_threshold_status_weak", {blocker["code"] for blocker in run["promotion_blockers"]})

    def test_hard_fold_low_ap_profile_blocks_and_prioritizes_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            logs.mkdir()
            summary = {
                "promotion_ready": True,
                "run_ids": ["hard_fold"],
                "worst_fold_id": "20230530172803",
                "worst_fold_val_f1": 0.032,
                "per_fold_val_f1": {"20230530172803": 0.032},
                "per_fold_average_precision": {"20230530172803": 0.025},
                "promotion_warnings": [],
            }
            (logs / "hard_fold.summary.json").write_text(json.dumps(summary))
            rows = [
                {"run_id": "loo_a", "artifact_dir": str(root / "experiments" / "runs" / "loo_a"), "heldout_segment": "20230530172803", "seed": 1, "val_f1": 0.032, "average_precision": 0.025, "ap_prevalence_lift": 1.4, "val_positive_rate": 0.018, "pred_positive_rate": 0.045, "fixed_threshold_status": "ok", "returncode": 0},
                {"run_id": "loo_b", "artifact_dir": str(root / "experiments" / "runs" / "loo_b"), "heldout_segment": "20230530172803", "seed": 2, "val_f1": 0.022, "average_precision": 0.024, "ap_prevalence_lift": 1.3, "val_positive_rate": 0.018, "pred_positive_rate": 0.047, "fixed_threshold_status": "ok", "returncode": 0},
            ]
            (logs / "hard_fold.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            db = root / "experiments" / "experiments.db"
            db.parent.mkdir(parents=True)
            run_dir = root / "experiments" / "runs" / "hard_fold"
            full_tile_dir = run_dir / "full_tile_20230530172803"
            full_tile_dir.mkdir(parents=True)
            (full_tile_dir / "metrics.json").write_text(json.dumps({"promotion_checks": {"eligible": True}, "evaluation_region": {"type": "whole_segment", "segment_id": "20230530172803"}, "val_f1": 0.08, "average_precision": 0.027, "val_positive_rate": 0.018, "pred_positive_rate": 0.045, "fixed_threshold_f1": 0.04, "fixed_threshold_status": "ok"}))
            cfg = {"model": {"name": "tiny_torch_unet"}, "evaluation": {"main_metric": "val_f1"}, "dataset": {"research_scope": "multi_segment_robust_expanded"}, "validation_setup": {"mode": "leave-one-segment-out", "train_segment_id": "?", "val_segment_id": "20230530172803"}}
            metrics = {"val_f1": 0.40, "average_precision": 0.30, "precision": 0.5, "recall": 0.6, "pred_positive_rate": 0.20, "val_positive_rate": 0.10, "fixed_threshold_f1": 0.30, "fixed_threshold_status": "ok"}
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE experiments (run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, config_json TEXT NOT NULL, main_metric REAL NOT NULL, secondary_metrics_json TEXT NOT NULL, artifact_dir TEXT NOT NULL)")
                conn.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?)", ("hard_fold", "2026-05-26T00:00:00Z", json.dumps(cfg), 0.4, json.dumps(metrics), str(run_dir)))
                conn.commit()
            finally:
                conn.close()

            with mock.patch("research_dashboard.datasets.dataset_summary", return_value={"source": "test", "scrolls": [], "splits": {}}):
                snapshot = build_snapshot(root)

        run = snapshot["experiments"]["recent"][0]
        evidence = snapshot["research_summary"]["candidate_evidence"]
        profile = evidence["hard_fold_profile"]
        self.assertEqual(run["promotion_status"], "blocked")
        self.assertIn("hard_fold_low_ap", {blocker["code"] for blocker in run["promotion_blockers"]})
        self.assertEqual(profile["failure_mode"], "low_ap_near_prevalence")
        self.assertEqual(profile["recommended_action"], "audit_hard_fold_labels_and_sampling")
        self.assertEqual(evidence["promotion_actions"][0]["id"], "audit_hard_fold_labels")
        self.assertIn("scripts/analyze_loo_folds.py", evidence["promotion_actions"][0]["command_text"])

    def test_loo_full_tile_quality_fail_blocks_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = root / "logs"
            logs.mkdir()
            (logs / "loo_quality.summary.json").write_text(json.dumps({"promotion_ready": True, "run_ids": ["loo_quality"], "promotion_warnings": []}))
            (logs / "loo_quality.jsonl").write_text(json.dumps({"run_id": "loo_bad", "artifact_dir": str(root / "experiments" / "runs" / "loo_bad"), "heldout_segment": "abc", "seed": 1, "val_f1": 0.30, "average_precision": 0.20, "pred_positive_rate": 0.20, "val_positive_rate": 0.10, "returncode": 0}) + "\n")
            db = root / "experiments" / "experiments.db"
            db.parent.mkdir(parents=True)
            run_dir = root / "experiments" / "runs" / "loo_quality"
            full_tile_dir = run_dir / "full_tile_abc"
            full_tile_dir.mkdir(parents=True)
            (full_tile_dir / "metrics.json").write_text(json.dumps({"promotion_checks": {"eligible": True}, "evaluation_region": {"type": "whole_segment", "segment_id": "abc"}}))
            loo_tile_dir = root / "experiments" / "runs" / "loo_bad" / "full_tile_abc"
            loo_tile_dir.mkdir(parents=True)
            (loo_tile_dir / "metrics.json").write_text(json.dumps({"promotion_checks": {"eligible": True}, "evaluation_region": {"type": "whole_segment", "segment_id": "abc"}, "val_f1": 0.02, "average_precision": 0.02, "pred_positive_rate": 0.90, "val_positive_rate": 0.10, "fixed_threshold_f1": 0.0}))
            cfg = {"model": {"name": "tiny_torch_unet"}, "evaluation": {"main_metric": "val_f1"}, "dataset": {"research_scope": "multi_segment_robust_expanded"}, "validation_setup": {"mode": "leave-one-segment-out", "train_segment_id": "?", "val_segment_id": "abc"}}
            metrics = {"val_f1": 0.40, "average_precision": 0.30, "precision": 0.5, "recall": 0.6, "pred_positive_rate": 0.20, "val_positive_rate": 0.10, "fixed_threshold_f1": 0.30}
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE experiments (run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, config_json TEXT NOT NULL, main_metric REAL NOT NULL, secondary_metrics_json TEXT NOT NULL, artifact_dir TEXT NOT NULL)")
                conn.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?)", ("loo_quality", "2026-05-26T00:00:00Z", json.dumps(cfg), 0.4, json.dumps(metrics), str(run_dir)))
                conn.commit()
            finally:
                conn.close()

            with mock.patch("research_dashboard.datasets.dataset_summary", return_value={"source": "test", "scrolls": [], "splits": {}}):
                snapshot = build_snapshot(root)

        run = snapshot["experiments"]["recent"][0]
        evidence = snapshot["research_summary"]["candidate_evidence"]
        self.assertEqual(run["promotion_status"], "blocked")
        self.assertIn("loo_full_tile_quality_fail", {blocker["code"] for blocker in run["promotion_blockers"]})
        self.assertEqual(evidence["loo_full_tile"]["coverage_count"], 1)
        self.assertEqual(evidence["loo_full_tile"]["evidence"][0]["quality_verdict"]["verdict"], "fail")
        self.assertFalse(snapshot["research_summary"]["decision"]["promotion_gate"]["ready"])
        self.assertNotIn("promotion_review", {action["id"] for action in evidence["promotion_actions"]})

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

    def test_artifact_download_serves_only_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "experiments" / "runs" / "run1"
            runs.mkdir(parents=True)
            safe = runs / "metrics.json"
            safe.write_text('{"val_f1": 1.0}')
            outside = root / "README.md"
            outside.write_text("outside")
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(root, "secret-token"))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                url = base + "/api/artifact/download?token=secret-token&path=" + urllib.parse.quote(str(safe))
                resp = urllib.request.urlopen(url, timeout=5)
                self.assertEqual(resp.read(), b'{"val_f1": 1.0}')
                self.assertIn("attachment", resp.headers.get("Content-Disposition", ""))

                bad_url = base + "/api/artifact/download?token=secret-token&path=" + urllib.parse.quote(str(outside))
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(bad_url, timeout=5)
                self.assertEqual(ctx.exception.code, 400)
            finally:
                server.shutdown()
                server.server_close()

    def test_artifact_preview_decodes_probability_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full_tile = root / "experiments" / "runs" / "run1" / "full_tile_seg"
            full_tile.mkdir(parents=True)
            prob_path = full_tile / "probability_map.npy"
            np.save(prob_path, np.asarray([[0.1, 0.8], [0.3, 0.9]], dtype=np.float32))
            (full_tile / "metrics.json").write_text(json.dumps({"best_threshold": 0.7, "val_f1": 0.2}))

            files = list_artifact_files(str(root / "experiments" / "runs" / "run1"))
            preview = preview_artifact(root, str(prob_path))

        self.assertEqual(files[0]["relative_path"], "full_tile_seg/probability_map.npy")
        self.assertEqual(preview["kind"], "npy")
        self.assertEqual(preview["preview"]["shape"], [2, 2])
        self.assertAlmostEqual(preview["preview"]["threshold"], 0.7)
        self.assertTrue(preview["preview"]["heatmap_data_url"].startswith("data:image/png;base64,"))
        self.assertTrue(preview["preview"]["mask_data_url"].startswith("data:image/png;base64,"))
        self.assertIn("map_quality", preview["preview"])
        self.assertIn("quality_verdict", preview["preview"])

    def test_artifact_preview_area_downsamples_full_resolution_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full_tile = root / "experiments" / "runs" / "run1" / "full_tile_seg"
            full_tile.mkdir(parents=True)
            prob_path = full_tile / "probability_map.npy"
            probs = np.full((768, 768), 0.10, dtype=np.float32)
            probs[::2, ::2] = 0.90
            np.save(prob_path, probs)
            (full_tile / "metrics.json").write_text(json.dumps({"best_threshold": 0.5, "val_f1": 0.2}))

            preview = preview_artifact(root, str(prob_path))["preview"]

        self.assertEqual(preview["rendered_shape"], [384, 384])
        self.assertEqual(preview["downsample"]["method"], "area_mean")
        self.assertIn("preview_downsampled_area_mean_not_full_resolution", preview["preview_warnings"])
        self.assertAlmostEqual(preview["pred_positive_rate"], 0.25)
        self.assertAlmostEqual(preview["mask_positive_fraction_preview_mean"], 0.25)

    def test_inventory_uses_copyable_safe_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inventory = build_inventory(root)

        self.assertGreaterEqual(len(inventory["features"]), 5)
        safe_ids = {feature["id"] for feature in inventory["features"] if feature["safe_to_execute_from_dashboard"]}
        self.assertIn("seed_repeat_loo_dry_run", safe_ids)
        self.assertIn("full_tile_self_test", safe_ids)
        for feature in inventory["features"]:
            self.assertIsInstance(feature["command"], list)
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

    def test_dashboard_post_controls_require_auth_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(root, None))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                req = urllib.request.Request(base + "/api/agent/chat", data=json.dumps({"message": "hi"}).encode(), headers={"Content-Type": "application/json"}, method="POST")
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(req, timeout=5)
                self.assertEqual(ctx.exception.code, 403)
            finally:
                server.shutdown()
                server.server_close()

    def test_dashboard_post_controls_can_be_private_mode_tokenless(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("os.environ", {"VESUVIUS_DASHBOARD_ALLOW_UNAUTHENTICATED_POSTS": "1"}, clear=False):
            root = Path(tmp)
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(root, None))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                req = urllib.request.Request(base + "/api/settings/snapshot", data=json.dumps({"reason": "tokenless private mode test"}).encode(), headers={"Content-Type": "application/json"}, method="POST")
                data = json.loads(urllib.request.urlopen(req, timeout=5).read().decode())
                self.assertTrue(data["ok"])
                self.assertEqual(data["snapshot"]["reason"], "tokenless private mode test")
            finally:
                server.shutdown()
                server.server_close()

    def test_dashboard_safe_command_endpoint_is_allowlisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("os.environ", {"VESUVIUS_DASHBOARD_ENABLE_RUNS": "1"}, clear=False):
            root = Path(tmp)
            (root / "scripts").mkdir()
            (root / "configs").mkdir()
            (root / "scripts" / "evaluate_leave_one_out.py").write_text("")
            (root / "configs" / "robust_multisegment_dice035_expanded.yaml").write_text("")
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(root, "secret-token"))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with mock.patch("research_dashboard.app.subprocess.run") as run_mock:
                    run_mock.return_value = mock.Mock(returncode=0, stdout="planned", stderr="")
                    body = json.dumps({"id": "seed_repeat_loo_dry_run"}).encode()
                    req = urllib.request.Request(base + "/api/run-command?token=secret-token", data=body, headers={"Content-Type": "application/json"}, method="POST")
                    data = json.loads(urllib.request.urlopen(req, timeout=5).read().decode())
                self.assertTrue(data["ok"])
                self.assertIn("planned", data["stdout"])
                self.assertEqual(run_mock.call_args.kwargs["cwd"], root)
                self.assertFalse(run_mock.call_args.kwargs["check"])
            finally:
                server.shutdown()
                server.server_close()

    def test_dashboard_agent_chat_proxy_does_not_expose_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("os.environ", {"VESUVIUS_DASHBOARD_AGENT_ENABLED": "1", "VESUVIUS_DASHBOARD_AGENT_API_KEY": "server-secret"}, clear=False):
            root = Path(tmp)
            with mock.patch("research_dashboard.app.urlrequest.urlopen") as open_mock:
                open_mock.return_value.__enter__.return_value.read.return_value = json.dumps({"reply": "ok"}).encode()
                data = _agent_chat(root, {"message": "summarize blockers", "base_url": "http://agent.local/chat"})
            self.assertTrue(data["ok"])
            self.assertEqual(data["reply"], "ok")
            self.assertNotIn("server-secret", json.dumps(data))
            outbound = open_mock.call_args.args[0]
            self.assertEqual(outbound.headers.get("Authorization"), "Bearer server-secret")

    def test_dashboard_settings_apply_snapshots_and_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(root, "secret-token"))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                before = load_dashboard_settings(root)["version"]
                body = json.dumps({"patch": {"decoded_gallery_limit": 7}, "base_version": before, "reason": "test update"}).encode()
                req = urllib.request.Request(base + "/api/settings/apply?token=secret-token", data=body, headers={"Content-Type": "application/json"}, method="POST")
                data = json.loads(urllib.request.urlopen(req, timeout=5).read().decode())
                self.assertTrue(data["ok"])
                self.assertEqual(data["settings"]["values"]["decoded_gallery_limit"], 7)
                pre_snapshot = data["pre_change_snapshot"]["id"]

                rollback = json.dumps({"snapshot_id": pre_snapshot, "reason": "test rollback"}).encode()
                req = urllib.request.Request(base + "/api/settings/rollback?token=secret-token", data=rollback, headers={"Content-Type": "application/json"}, method="POST")
                rolled = json.loads(urllib.request.urlopen(req, timeout=5).read().decode())
                self.assertTrue(rolled["ok"])
                self.assertEqual(rolled["settings"]["values"]["decoded_gallery_limit"], 12)
                self.assertNotEqual(rolled["settings"]["version"], before)
            finally:
                server.shutdown()
                server.server_close()

    def test_agent_settings_apply_requires_explicit_enablement_and_rejects_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(root, "secret-token"))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                body = json.dumps({"actor": "agent", "patch": {"decoded_gallery_limit": 8}}).encode()
                req = urllib.request.Request(base + "/api/settings/apply?token=secret-token", data=body, headers={"Content-Type": "application/json"}, method="POST")
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(req, timeout=5)
                self.assertEqual(ctx.exception.code, 403)

                with mock.patch.dict("os.environ", {"VESUVIUS_DASHBOARD_AGENT_SETTINGS_WRITE": "1"}, clear=False):
                    secret_body = json.dumps({"actor": "agent", "patch": {"agent_base_url": "http://agent.local/chat?token=leaked"}}).encode()
                    req = urllib.request.Request(base + "/api/settings/apply?token=secret-token", data=secret_body, headers={"Content-Type": "application/json"}, method="POST")
                    with self.assertRaises(urllib.error.HTTPError) as ctx2:
                        urllib.request.urlopen(req, timeout=5)
                    self.assertEqual(ctx2.exception.code, 400)
            finally:
                server.shutdown()
                server.server_close()

    def test_agent_chat_can_return_validated_settings_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("os.environ", {"VESUVIUS_DASHBOARD_AGENT_ENABLED": "1", "VESUVIUS_DASHBOARD_AGENT_SETTINGS_WRITE": "1"}, clear=False):
            root = Path(tmp)
            agent_reply = {"reply": json.dumps({"reply": "Use a smaller gallery for slower browsers.", "settings_patch": {"decoded_gallery_limit": 6}, "reason": "reduce dashboard load"})}
            with mock.patch("research_dashboard.app.urlrequest.urlopen") as open_mock:
                open_mock.return_value.__enter__.return_value.read.return_value = json.dumps(agent_reply).encode()
                data = _agent_chat(root, {"message": "fix dashboard settings", "action_mode": True, "base_url": "http://agent.local/chat"})
            self.assertTrue(data["ok"])
            self.assertTrue(data["settings_action_mode"])
            self.assertEqual(data["settings_proposal"]["patch"], {"decoded_gallery_limit": 6})
            self.assertTrue(data["settings_proposal"]["valid"])
            self.assertEqual(load_dashboard_settings(root)["values"]["decoded_gallery_limit"], 12)

    def test_visual_artifact_analysis_is_opt_in_and_secret_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("os.environ", {"VESUVIUS_DASHBOARD_VISUAL_ANALYSIS_ENABLED": "1", "VESUVIUS_DASHBOARD_VISUAL_ANALYSIS_API_KEY": "vision-secret"}, clear=False):
            root = Path(tmp)
            run_dir = root / "experiments" / "runs" / "run1"
            run_dir.mkdir(parents=True)
            arr = np.zeros((16, 16), dtype=np.float32)
            arr[4:12, 4:12] = 0.9
            np.save(run_dir / "probability_map.npy", arr)
            with mock.patch("research_dashboard.app.urlrequest.urlopen") as open_mock:
                open_mock.return_value.__enter__.return_value.read.return_value = json.dumps({"reply": "coherent structure"}).encode()
                data = _visual_artifact_analysis(root, {"path": str(run_dir / "probability_map.npy"), "base_url": "http://agent.local/chat"})
            self.assertTrue(data["ok"])
            self.assertEqual(data["image_count"], 2)
            self.assertNotIn("vision-secret", json.dumps(data))
            outbound = open_mock.call_args.args[0]
            self.assertEqual(outbound.headers.get("Authorization"), "Bearer vision-secret")
            sent = json.loads(outbound.data.decode())
            self.assertEqual({img["label"] for img in sent["images"]}, {"probability_heatmap", "threshold_mask"})


    def test_fixed_threshold_f1_low_detects_zero_value(self) -> None:
        """Regression: fixed_threshold_f1=0.0 is falsy, so the old 'or' fallback masked it."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "experiments" / "experiments.db"
            db.parent.mkdir(parents=True)
            logs = root / "logs"
            logs.mkdir()
            (logs / "fixed_zero.summary.json").write_text(json.dumps({"promotion_ready": True, "run_ids": ["fixed_zero"], "promotion_warnings": []}))
            run_dir = root / "experiments" / "runs" / "fixed_zero"
            run_dir.mkdir(parents=True)
            full_tile_dir = run_dir / "full_tile_abc"
            full_tile_dir.mkdir(parents=True)
            (full_tile_dir / "metrics.json").write_text(json.dumps({"promotion_checks": {"eligible": True}, "evaluation_region": {"type": "whole_segment", "segment_id": "abc"}}))
            cfg = {"model": {"name": "tiny_torch_unet"}, "evaluation": {"main_metric": "val_f1"}, "dataset": {"research_scope": "multi_segment_robust_expanded"}, "validation_setup": {"mode": "leave-one-segment-out", "train_segment_id": "?", "val_segment_id": "abc"}}
            metrics = {"val_f1": 0.30, "average_precision": 0.20, "precision": 0.3, "recall": 0.6, "pred_positive_rate": 0.20, "val_positive_rate": 0.10, "loo_promotion_ready": True, "fixed_threshold_f1": 0.0}
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE experiments (run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, config_json TEXT NOT NULL, main_metric REAL NOT NULL, secondary_metrics_json TEXT NOT NULL, artifact_dir TEXT NOT NULL)")
                conn.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?)", ("fixed_zero", "2026-05-26T00:00:00Z", json.dumps(cfg), 0.30, json.dumps(metrics), str(run_dir)))
                conn.commit()
            finally:
                conn.close()

            with mock.patch("research_dashboard.datasets.dataset_summary", return_value={"source": "test", "scrolls": [], "splits": {}}):
                snapshot = build_snapshot(root)

        run = snapshot["experiments"]["recent"][0]
        blockers = run["promotion_blockers"]
        codes = {b["code"] for b in blockers}
        self.assertIn("fixed_threshold_f1_low", codes)
        # It should be a warning, not a blocker
        fixed_blocker = next(b for b in blockers if b["code"] == "fixed_threshold_f1_low")
        self.assertEqual(fixed_blocker["severity"], "warning")
        # With only warnings, promotion_status should be eligible
        self.assertEqual(run["promotion_status"], "eligible")

    def test_promotion_status_blocked_only_by_real_blockers(self) -> None:
        """Warnings should not block promotion_status."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "experiments" / "experiments.db"
            db.parent.mkdir(parents=True)
            logs = root / "logs"
            logs.mkdir()
            (logs / "warning_only.summary.json").write_text(json.dumps({"promotion_ready": True, "run_ids": ["warning_only"], "promotion_warnings": []}))
            run_dir = root / "experiments" / "runs" / "warning_only"
            run_dir.mkdir(parents=True)
            full_tile_dir = run_dir / "full_tile_abc"
            full_tile_dir.mkdir(parents=True)
            (full_tile_dir / "metrics.json").write_text(json.dumps({"promotion_checks": {"eligible": True}, "evaluation_region": {"type": "whole_segment", "segment_id": "abc"}}))
            cfg = {"model": {"name": "tiny_torch_unet"}, "evaluation": {"main_metric": "val_f1"}, "dataset": {"research_scope": "multi_segment_robust_expanded"}, "validation_setup": {"mode": "leave-one-segment-out", "train_segment_id": "?", "val_segment_id": "abc"}}
            metrics = {"val_f1": 0.30, "average_precision": 0.20, "precision": 0.3, "recall": 0.6, "pred_positive_rate": 0.20, "val_positive_rate": 0.10, "loo_promotion_ready": True, "fixed_threshold_f1": 0.0, "best_threshold": 0.94}
            conn = sqlite3.connect(db)
            try:
                conn.execute("CREATE TABLE experiments (run_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, config_json TEXT NOT NULL, main_metric REAL NOT NULL, secondary_metrics_json TEXT NOT NULL, artifact_dir TEXT NOT NULL)")
                conn.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?)", ("warning_only", "2026-05-26T00:00:00Z", json.dumps(cfg), 0.30, json.dumps(metrics), str(run_dir)))
                conn.commit()
            finally:
                conn.close()

            with mock.patch("research_dashboard.datasets.dataset_summary", return_value={"source": "test", "scrolls": [], "splits": {}}):
                snapshot = build_snapshot(root)

        run = snapshot["experiments"]["recent"][0]
        # Should have two warnings but no blockers
        self.assertEqual(run["promotion_status"], "eligible")
        warnings = [b for b in run["promotion_blockers"] if b.get("severity") == "warning"]
        self.assertEqual(len(warnings), 2)
        self.assertIn("fixed_threshold_f1_low", {w["code"] for w in warnings})
        self.assertIn("best_threshold_at_sweep_edge", {w["code"] for w in warnings})


if __name__ == "__main__":
    unittest.main()

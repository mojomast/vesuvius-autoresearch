from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research_dashboard.mining import build_fold_safe_config_preview, build_hard_negative_plan, mined_npz_inventory, resolve_safe_config_path


class HardNegativePlannerTest(unittest.TestCase):
    def test_inventory_reads_mined_metadata_and_forbidden_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mined_dir = root / "data" / "mined"
            mined_dir.mkdir(parents=True)
            np.savez_compressed(mined_dir / "seg-a.npz", images=np.zeros((1, 1, 4, 4), dtype=np.float32), labels=np.zeros((1, 1, 4, 4), dtype=np.float32))
            (mined_dir / "seg-a.metadata.json").write_text(json.dumps({"source": "full_tile_failure_mining", "mined_segment_id": "seg-a", "samples": 1, "forbidden_heldout_segments": ["seg-a"]}))

            inventory = mined_npz_inventory(root)

        self.assertEqual(inventory["count"], 1)
        self.assertEqual(inventory["mined_npzs"][0]["relative_path"], "data/mined/seg-a.npz")
        self.assertEqual(inventory["mined_npzs"][0]["forbidden_heldout_segments"], ["seg-a"])
        self.assertEqual(inventory["mined_npzs"][0]["eligibility_status"], "eligible")

    def test_plan_emits_mining_command_and_rejects_heldout_mined_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "experiments" / "runs" / "run1"
            full_tile = artifact / "full_tile_seg-a"
            full_tile.mkdir(parents=True)
            metrics = full_tile / "metrics.json"
            metrics.write_text("{}")
            mined_dir = root / "data" / "mined"
            mined_dir.mkdir(parents=True)
            np.savez_compressed(mined_dir / "seg-a.npz", images=np.zeros((1, 1, 4, 4), dtype=np.float32), labels=np.zeros((1, 1, 4, 4), dtype=np.float32))
            (mined_dir / "seg-a.metadata.json").write_text(json.dumps({"mined_segment_id": "seg-a", "samples": 1, "forbidden_heldout_segments": ["seg-a"]}))
            np.savez_compressed(mined_dir / "parent-only.npz", images=np.zeros((1, 1, 4, 4), dtype=np.float32), labels=np.zeros((1, 1, 4, 4), dtype=np.float32))
            (mined_dir / "parent-only.metadata.json").write_text(json.dumps({"samples": 1, "parent_full_tile": {"segment_id": "seg-a"}}))
            snapshot = {
                "research_summary": {
                    "candidate_evidence": {
                        "candidate_run_id": "run1abcdef",
                        "candidate_artifact_dir": str(artifact),
                        "loo": {"worst_fold_id": "seg-a"},
                        "full_tile": {"evidence": [{"path": str(metrics), "segment_id": "seg-a", "pred_positive_rate": 0.3, "val_positive_rate": 0.1, "fixed_threshold_status": "weak"}]},
                    }
                }
            }

            plan = build_hard_negative_plan(root, snapshot)

        self.assertEqual(plan["mine_commands"][0]["segment_id"], "seg-a")
        self.assertIn("--mine-output", plan["mine_commands"][0]["command"])
        self.assertNotIn("--overwrite", plan["mine_commands"][0]["command"])
        output_dir = plan["mine_commands"][0]["command"][plan["mine_commands"][0]["command"].index("--output-dir") + 1]
        self.assertNotEqual(output_dir, "experiments/runs/run1/full_tile_seg-a")
        self.assertEqual(output_dir, "experiments/runs/run1/mining_refresh_seg-a")
        self.assertEqual(plan["eligible_extra_train_npzs"], [])
        self.assertEqual({item["relative_path"] for item in plan["rejected_extra_train_npzs"]}, {"data/mined/seg-a.npz", "data/mined/parent-only.npz"})
        self.assertTrue(plan["dry_run"])

    def test_fold_eligibility_map_is_per_heldout_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mined_dir = root / "data" / "mined"
            mined_dir.mkdir(parents=True)
            for segment in ("seg-a", "seg-b"):
                np.savez_compressed(mined_dir / f"{segment}.npz", images=np.zeros((1, 1, 4, 4), dtype=np.float32), labels=np.zeros((1, 1, 4, 4), dtype=np.float32))
                (mined_dir / f"{segment}.metadata.json").write_text(json.dumps({"mined_segment_id": segment, "samples": 1, "forbidden_heldout_segments": [segment]}))
            snapshot = {"research_summary": {"candidate_evidence": {"loo": {"worst_fold_id": "seg-a"}, "loo_full_tile": {"segments_covered": ["seg-a", "seg-b", "seg-c"]}}}}

            plan = build_hard_negative_plan(root, snapshot)

        by_fold = plan["fold_safe_extra_train_npzs_by_heldout"]
        self.assertEqual(by_fold["seg-a"]["eligible_extra_train_npzs"], ["data/mined/seg-b.npz"])
        self.assertEqual(set(by_fold["seg-c"]["eligible_extra_train_npzs"]), {"data/mined/seg-a.npz", "data/mined/seg-b.npz"})
        self.assertEqual(plan["eligible_extra_train_npzs"], ["data/mined/seg-b.npz"])

    def test_inventory_rejects_empty_or_missing_provenance_and_reviews_contamination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mined_dir = root / "data" / "mined"
            mined_dir.mkdir(parents=True)
            np.savez_compressed(mined_dir / "empty.npz", images=np.zeros((0, 1, 4, 4), dtype=np.float32), labels=np.zeros((0, 1, 4, 4), dtype=np.float32))
            (mined_dir / "empty.metadata.json").write_text(json.dumps({"mined_segment_id": "seg-a", "samples": 0, "forbidden_heldout_segments": ["seg-a"]}))
            np.savez_compressed(mined_dir / "unknown.npz", images=np.zeros((1, 1, 4, 4), dtype=np.float32), labels=np.zeros((1, 1, 4, 4), dtype=np.float32))
            (mined_dir / "unknown.metadata.json").write_text(json.dumps({"samples": 1}))
            labels = np.ones((1, 1, 4, 4), dtype=np.float32)
            np.savez_compressed(mined_dir / "contaminated.npz", images=np.zeros((1, 1, 4, 4), dtype=np.float32), labels=labels)
            (mined_dir / "contaminated.metadata.json").write_text(json.dumps({"mined_segment_id": "seg-b", "samples": 1, "forbidden_heldout_segments": ["seg-b"]}))

            inventory = mined_npz_inventory(root)

        statuses = {item["relative_path"]: item["eligibility_status"] for item in inventory["mined_npzs"]}
        self.assertEqual(statuses["data/mined/empty.npz"], "reject")
        self.assertEqual(statuses["data/mined/unknown.npz"], "reject")
        self.assertEqual(statuses["data/mined/contaminated.npz"], "review")

    def test_config_preview_is_invalid_when_heldout_paths_do_not_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "base.yaml"
            cfg.write_text("dataset:\n  patch_size: 64\n  train_npz: data/leaveout_old/train.npz\n  val_npz: data/old/val.npz\nautoresearch:\n  heldout_segment: old\n")

            preview = build_fold_safe_config_preview(cfg, root, "seg-c", ["data/mined/seg-a.npz"])

        self.assertFalse(preview["mutation"])
        self.assertFalse(preview["valid"])
        self.assertIn("heldout_segment_override_does_not_rewrite_train_val_npz_paths", preview["warnings"])
        self.assertIsNone(preview["patched_config_yaml"])

    def test_config_preview_is_stdout_safe_and_sets_extra_train_npzs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "base.yaml"
            cfg.write_text("dataset:\n  patch_size: 64\n  train_npz: data/leaveout_seg-c/train.npz\n  val_npz: data/seg-c/val.npz\nautoresearch:\n  heldout_segment: seg-c\n")

            preview = build_fold_safe_config_preview(cfg, root, "seg-c", ["data/mined/seg-a.npz"])

        self.assertTrue(preview["valid"])
        self.assertEqual(preview["warnings"], [])
        self.assertIn("extra_train_npzs", preview["patched_config_yaml"])
        self.assertIn("seg-c", preview["patched_config_yaml"])

    def test_threshold_risk_prefers_cap_tightening_when_low_cap_preserves_f1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "experiments" / "runs" / "run1"
            full_tile = artifact / "full_tile_seg-a"
            full_tile.mkdir(parents=True)
            metrics = full_tile / "metrics.json"
            metrics.write_text("{}")
            snapshot = {"research_summary": {"candidate_evidence": {"loo": {"worst_fold_id": "seg-b"}, "full_tile": {"evidence": [{"path": str(metrics), "segment_id": "seg-a", "pred_positive_rate": 0.3, "val_positive_rate": 0.1, "fixed_threshold_f1": 0.0, "threshold_risk_summary": {"selected": {"f1": 0.30, "pred_to_val_ratio": 3.0}, "best_under_prratio2p0": {"f1": 0.29, "pred_to_val_ratio": 1.9}, "best_under_prratio3p0": {"f1": 0.30, "pred_to_val_ratio": 2.9}}}]}}}}

            plan = build_hard_negative_plan(root, snapshot)

        self.assertEqual(plan["calibration_mining_decisions"][0]["action"], "tighten_positive_rate_cap")
        self.assertEqual(plan["mine_commands"], [])

    def test_plan_emits_read_only_cap_comparison_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "experiments" / "runs" / "run1"
            full_tile = artifact / "full_tile_seg-a"
            full_tile.mkdir(parents=True)
            metrics = full_tile / "metrics.json"
            metrics.write_text("{}")
            (full_tile / "metrics_by_threshold.csv").write_text("threshold,precision,recall,f05,f1,pred_positive_rate\n")
            snapshot = {"research_summary": {"candidate_evidence": {"loo": {"worst_fold_id": "seg-b"}, "full_tile": {"evidence": [{"path": str(metrics), "segment_id": "seg-a", "pred_positive_rate": 0.3, "val_positive_rate": 0.1, "fixed_threshold_f1": 0.0, "threshold_risk_summary": {"selected": {"f1": 0.30, "f05": 0.20, "pred_to_val_ratio": 3.0}, "best_under_prratio3p0": {"f1": 0.30, "f05": 0.20, "pred_to_val_ratio": 2.9}}}]}}}}

            plan = build_hard_negative_plan(root, snapshot)

        self.assertEqual(plan["cap_comparison_commands"][0]["segment_id"], "seg-a")
        self.assertFalse(plan["cap_comparison_commands"][0]["writes_artifacts"])
        self.assertIn("scripts/compare_threshold_caps.py", plan["cap_comparison_commands"][0]["command_text"])
        self.assertIn("--min-retained-f05", plan["calibration_mining_decisions"][0]["cap_comparison_command_text"])

    def test_missing_threshold_risk_requests_cap_comparison_before_mining(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "experiments" / "runs" / "run1"
            full_tile = artifact / "full_tile_seg-a"
            full_tile.mkdir(parents=True)
            metrics = full_tile / "metrics.json"
            metrics.write_text("{}")
            (full_tile / "metrics_by_threshold.csv").write_text("threshold,precision,recall,f05,f1,pred_positive_rate\n")
            snapshot = {"research_summary": {"candidate_evidence": {"loo": {"worst_fold_id": "seg-b"}, "full_tile": {"evidence": [{"path": str(metrics), "segment_id": "seg-a", "pred_positive_rate": 0.3, "val_positive_rate": 0.1, "fixed_threshold_f1": 0.0}]}}}}

            plan = build_hard_negative_plan(root, snapshot)

        self.assertEqual(plan["calibration_mining_decisions"][0]["action"], "run_cap_comparison")
        self.assertEqual(plan["calibration_mining_decisions"][0]["reason"], "missing_threshold_risk_summary")
        self.assertEqual(plan["mine_commands"], [])
        self.assertIn("scripts/compare_threshold_caps.py", plan["cap_comparison_commands"][0]["command_text"])

    def test_partial_threshold_risk_requests_cap_comparison_before_mining(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "experiments" / "runs" / "run1"
            full_tile = artifact / "full_tile_seg-a"
            full_tile.mkdir(parents=True)
            metrics = full_tile / "metrics.json"
            metrics.write_text("{}")
            (full_tile / "metrics_by_threshold.csv").write_text("threshold,precision,recall,f05,f1,pred_positive_rate\n")
            snapshot = {"research_summary": {"candidate_evidence": {"loo": {"worst_fold_id": "seg-b"}, "full_tile": {"evidence": [{"path": str(metrics), "segment_id": "seg-a", "val_f1": 0.3, "pred_positive_rate": 0.3, "val_positive_rate": 0.1, "fixed_threshold_f1": 0.0, "threshold_risk_summary": {"cap_binding": True}}]}}}}

            plan = build_hard_negative_plan(root, snapshot)

        self.assertEqual(plan["calibration_mining_decisions"][0]["action"], "run_cap_comparison")
        self.assertEqual(plan["calibration_mining_decisions"][0]["reason"], "partial_threshold_risk_summary")
        self.assertEqual(plan["mine_commands"], [])

    def test_refreshed_threshold_risk_supersedes_stale_same_artifact_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "experiments" / "runs" / "run1"
            stale_dir = artifact / "full_tile_seg-a"
            refresh_dir = artifact / "full_tile_seg-a_riskrefresh"
            stale_dir.mkdir(parents=True)
            refresh_dir.mkdir(parents=True)
            stale_metrics = stale_dir / "metrics.json"
            refresh_metrics = refresh_dir / "metrics.json"
            stale_metrics.write_text("{}")
            refresh_metrics.write_text("{}")
            stale_tile = {"path": str(stale_metrics), "segment_id": "seg-a", "pred_positive_rate": 0.3, "val_positive_rate": 0.1, "fixed_threshold_f1": 0.0}
            refresh_tile = {"path": str(refresh_metrics), "segment_id": "seg-a", "pred_positive_rate": 0.3, "val_positive_rate": 0.1, "fixed_threshold_f1": 0.0, "threshold_risk_summary": {"selected": {"f1": 0.30, "pred_to_val_ratio": 3.0}, "best_under_prratio3p0": {"f1": 0.30}}}
            snapshot = {"research_summary": {"candidate_evidence": {"loo": {"worst_fold_id": "seg-a"}, "loo_full_tile": {"evidence": [stale_tile, refresh_tile]}}}}

            plan = build_hard_negative_plan(root, snapshot)

        self.assertEqual([item["action"] for item in plan["calibration_mining_decisions"]], ["tighten_positive_rate_cap"])
        self.assertEqual(plan["mine_commands"], [])

    def test_threshold_risk_uses_intermediate_cap_before_mining(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "experiments" / "runs" / "run1"
            full_tile = artifact / "full_tile_seg-a"
            full_tile.mkdir(parents=True)
            metrics = full_tile / "metrics.json"
            metrics.write_text("{}")
            snapshot = {"research_summary": {"candidate_evidence": {"loo": {"worst_fold_id": "seg-b"}, "full_tile": {"evidence": [{"path": str(metrics), "segment_id": "seg-a", "pred_positive_rate": 0.3, "val_positive_rate": 0.1, "fixed_threshold_f1": 0.0, "threshold_risk_summary": {"selected": {"f1": 0.30, "pred_to_val_ratio": 3.0}, "best_under_prratio2p0": {"f1": 0.20, "pred_to_val_ratio": 1.9}, "best_under_prratio2p5": {"f1": 0.286, "pred_to_val_ratio": 2.4}, "best_under_prratio3p0": {"f1": 0.30, "pred_to_val_ratio": 2.9}}}]}}}}

            plan = build_hard_negative_plan(root, snapshot)

        decision = plan["calibration_mining_decisions"][0]
        self.assertEqual(decision["action"], "tighten_positive_rate_cap")
        self.assertEqual(decision["target_max_pred_positive_rate_ratio"], 2.5)
        self.assertEqual(plan["mine_commands"], [])

    def test_threshold_risk_requires_f05_retention_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "experiments" / "runs" / "run1"
            full_tile = artifact / "full_tile_seg-a"
            full_tile.mkdir(parents=True)
            metrics = full_tile / "metrics.json"
            metrics.write_text("{}")
            snapshot = {"research_summary": {"candidate_evidence": {"loo": {"worst_fold_id": "seg-b"}, "full_tile": {"evidence": [{"path": str(metrics), "segment_id": "seg-a", "pred_positive_rate": 0.3, "val_positive_rate": 0.1, "fixed_threshold_f1": 0.0, "threshold_risk_summary": {"selected": {"f1": 0.30, "f05": 0.30, "pred_to_val_ratio": 3.0}, "best_under_prratio2p5": {"f1": 0.29, "f05": 0.20, "pred_to_val_ratio": 2.4}, "best_under_prratio3p0": {"f1": 0.30, "f05": 0.29, "pred_to_val_ratio": 2.9}}}]}}}}

            plan = build_hard_negative_plan(root, snapshot)

        decision = plan["calibration_mining_decisions"][0]
        self.assertEqual(decision["action"], "tighten_positive_rate_cap")
        self.assertEqual(decision["target_max_pred_positive_rate_ratio"], 3.0)
        self.assertLess(decision["f05_retained_fraction"], 1.0)
        self.assertEqual(plan["mine_commands"], [])

    def test_explicit_mine_action_still_emits_command_when_cap_tightening_is_primary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "experiments" / "runs" / "run1"
            full_tile = artifact / "full_tile_seg-a"
            full_tile.mkdir(parents=True)
            metrics = full_tile / "metrics.json"
            metrics.write_text("{}")
            tile = {
                "path": str(metrics),
                "segment_id": "seg-a",
                "pred_positive_rate": 0.3,
                "val_positive_rate": 0.1,
                "fixed_threshold_f1": 0.0,
                "quality_next_actions": [{"id": "mine_hard_negatives"}],
                "threshold_risk_summary": {"selected": {"f1": 0.30, "pred_to_val_ratio": 3.0}, "best_under_prratio2p0": {"f1": 0.29}},
            }
            snapshot = {"research_summary": {"candidate_evidence": {"loo": {"worst_fold_id": "seg-b"}, "full_tile": {"evidence": [tile]}}}}

            plan = build_hard_negative_plan(root, snapshot)

        self.assertEqual(plan["calibration_mining_decisions"][0]["action"], "tighten_positive_rate_cap")
        self.assertEqual(plan["mine_commands"][0]["segment_id"], "seg-a")

    def test_threshold_risk_prefers_mining_when_lower_caps_collapse_f1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "experiments" / "runs" / "run1"
            full_tile = artifact / "full_tile_seg-a"
            full_tile.mkdir(parents=True)
            metrics = full_tile / "metrics.json"
            metrics.write_text("{}")
            snapshot = {"research_summary": {"candidate_evidence": {"loo": {"worst_fold_id": "seg-b"}, "full_tile": {"evidence": [{"path": str(metrics), "segment_id": "seg-a", "pred_positive_rate": 0.35, "val_positive_rate": 0.1, "fixed_threshold_f1": 0.0, "threshold_risk_summary": {"selected": {"f1": 0.30, "pred_to_val_ratio": 3.5}, "best_under_prratio2p0": {"f1": 0.10, "pred_to_val_ratio": 1.9}, "best_under_prratio3p0": {"f1": 0.15, "pred_to_val_ratio": 2.9}, "cap_binding": True}}]}}}}

            plan = build_hard_negative_plan(root, snapshot)

        self.assertEqual(plan["calibration_mining_decisions"][0]["action"], "mine_hard_negatives")
        self.assertEqual(plan["mine_commands"][0]["threshold_risk_decision"]["action"], "mine_hard_negatives")

    def test_resolve_safe_config_path_rejects_escape_and_non_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            cfg = cfg_dir / "base.yaml"
            cfg.write_text("model: {}\n")
            outside = Path(tmp) / "outside.yaml"
            outside.write_text("model: {}\n")
            text = cfg_dir / "base.txt"
            text.write_text("model: {}\n")

            self.assertEqual(resolve_safe_config_path(root, "configs/base.yaml"), cfg.resolve())
            self.assertEqual(resolve_safe_config_path(root, cfg.resolve()), cfg.resolve())
            with self.assertRaisesRegex(ValueError, "inside project root"):
                resolve_safe_config_path(root, outside)
            with self.assertRaisesRegex(ValueError, "YAML"):
                resolve_safe_config_path(root, text)
            with self.assertRaises(FileNotFoundError):
                resolve_safe_config_path(root, "configs/missing.yaml")


if __name__ == "__main__":
    unittest.main()

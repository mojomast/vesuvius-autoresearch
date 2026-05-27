from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research_dashboard.mining import build_hard_negative_plan, mined_npz_inventory


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
        self.assertEqual(plan["eligible_extra_train_npzs"], [])
        self.assertEqual({item["relative_path"] for item in plan["rejected_extra_train_npzs"]}, {"data/mined/seg-a.npz", "data/mined/parent-only.npz"})
        self.assertTrue(plan["dry_run"])


if __name__ == "__main__":
    unittest.main()

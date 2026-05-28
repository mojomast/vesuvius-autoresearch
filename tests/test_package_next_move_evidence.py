from __future__ import annotations

import csv
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts.package_next_move_evidence import build_next_move_evidence_package, main, render_markdown


def _write_threshold_artifact(root: Path, run_id: str = "run1", segment_id: str = "seg-a") -> Path:
    artifact = root / "experiments" / "runs" / run_id / f"full_tile_{segment_id}"
    artifact.mkdir(parents=True)
    metrics = artifact / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "segment": segment_id,
                "evaluation_region": {"type": "whole_segment", "segment_id": segment_id},
                "val_positive_rate": 0.1,
                "pred_positive_rate": 0.3,
                "val_f1": 0.30,
                "val_f05": 0.20,
                "average_precision": 0.15,
                "threshold_risk_summary": {
                    "selected": {"threshold": 0.3, "precision": 0.2, "recall": 0.6, "f05": 0.20, "f1": 0.30, "pred_positive_rate": 0.30, "pred_to_val_ratio": 3.0},
                    "best_under_prratio2p5": {"f1": 0.29, "f05": 0.20, "pred_to_val_ratio": 2.4},
                },
            }
        ),
        encoding="utf-8",
    )
    with (artifact / "metrics_by_threshold.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["threshold", "precision", "recall", "f05", "f1", "pred_positive_rate"])
        writer.writeheader()
        writer.writerow({"threshold": 0.5, "precision": 0.3, "recall": 0.2, "f05": 0.27, "f1": 0.24, "pred_positive_rate": 0.2})
        writer.writerow({"threshold": 0.4, "precision": 0.25, "recall": 0.35, "f05": 0.26, "f1": 0.291, "pred_positive_rate": 0.25})
        writer.writerow({"threshold": 0.3, "precision": 0.2, "recall": 0.6, "f05": 0.20, "f1": 0.30, "pred_positive_rate": 0.30})
    return metrics


def _write_snapshot(root: Path, metrics: Path, segment_id: str = "seg-a") -> Path:
    snapshot = {
        "schema_version": "vesuvius-dashboard/v1",
        "research_summary": {
            "candidate_evidence": {
                "candidate_run_id": "run1",
                "loo": {"worst_fold_id": segment_id},
                "full_tile": {
                    "evidence": [
                        {
                            "path": str(metrics),
                            "segment_id": segment_id,
                            "pred_positive_rate": 0.3,
                            "val_positive_rate": 0.1,
                            "fixed_threshold_f1": 0.0,
                            "threshold_risk_summary": {
                                "selected": {"f1": 0.30, "f05": 0.20, "pred_to_val_ratio": 3.0},
                                "best_under_prratio2p5": {"f1": 0.29, "f05": 0.20, "pred_to_val_ratio": 2.4},
                            },
                        }
                    ]
                },
            },
            "decision": {"next_action": "Review cap evidence", "promotion_gate": {"ready": True}},
        },
    }
    path = root / "snapshot.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    return path


class PackageNextMoveEvidenceTest(unittest.TestCase):
    def test_package_runs_planner_and_cap_comparison_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metrics = _write_threshold_artifact(root)
            snapshot = _write_snapshot(root, metrics)
            before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))

            package = build_next_move_evidence_package(root, snapshot_json=snapshot, caps=[2.0, 2.5, 3.0])
            after = sorted(str(path.relative_to(root)) for path in root.rglob("*"))

        self.assertEqual(before, after)
        self.assertFalse(package["writes_artifacts"])
        self.assertEqual(package["next_action"], "Review cap evidence")
        self.assertEqual(package["hard_negative_plan"]["calibration_mining_decisions"][0]["action"], "tighten_positive_rate_cap")
        self.assertEqual(len(package["cap_comparisons"]), 1)
        self.assertEqual(package["cap_comparisons"][0]["recommended_cap"]["cap"], 2.5)
        self.assertEqual(package["cap_recommendation"]["target_max_pred_positive_rate_ratio"], 2.5)
        self.assertEqual(package["recommended_next_move"]["action"], "tighten_positive_rate_cap")
        self.assertFalse(package["recommended_next_move"]["writes_artifacts"])

    def test_package_includes_optional_full_tile_pair_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metrics = _write_threshold_artifact(root)
            snapshot = _write_snapshot(root, metrics)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            baseline.write_text(json.dumps({"segment": "seg-a", "evaluation_region": {"type": "whole_segment", "segment_id": "seg-a"}, "val_f1": 0.2, "val_f05": 0.18, "average_precision": 0.1, "ap_prevalence_lift": 1.5, "val_positive_rate": 0.1, "pred_positive_rate": 0.2}), encoding="utf-8")
            candidate.write_text(json.dumps({"segment": "seg-a", "evaluation_region": {"type": "whole_segment", "segment_id": "seg-a"}, "val_f1": 0.22, "val_f05": 0.19, "average_precision": 0.11, "ap_prevalence_lift": 1.6, "val_positive_rate": 0.1, "pred_positive_rate": 0.2}), encoding="utf-8")

            package = build_next_move_evidence_package(root, snapshot_json=snapshot, pairs=[("seg-a", baseline, candidate)])
            markdown = render_markdown(package)

        self.assertEqual(package["full_tile_comparison_summary"]["status_counts"]["core_improved"], 1)
        self.assertIn("Next-Move Evidence Package", markdown)
        self.assertIn("Recommended Next Move", markdown)
        self.assertIn("Full-Tile Evidence Package", markdown)

    def test_full_tile_regression_blocks_cap_tightening_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metrics = _write_threshold_artifact(root)
            snapshot = _write_snapshot(root, metrics)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            baseline.write_text(json.dumps({"segment": "seg-a", "evaluation_region": {"type": "whole_segment", "segment_id": "seg-a"}, "val_f1": 0.2, "val_f05": 0.18, "average_precision": 0.1, "ap_prevalence_lift": 1.5, "val_positive_rate": 0.1, "pred_positive_rate": 0.2}), encoding="utf-8")
            candidate.write_text(json.dumps({"segment": "seg-a", "evaluation_region": {"type": "whole_segment", "segment_id": "seg-a"}, "val_f1": 0.19, "val_f05": 0.17, "average_precision": 0.09, "ap_prevalence_lift": 1.4, "val_positive_rate": 0.1, "pred_positive_rate": 0.2}), encoding="utf-8")

            package = build_next_move_evidence_package(root, snapshot_json=snapshot, pairs=[("seg-a", baseline, candidate)])
            markdown = render_markdown(package)

        self.assertEqual(package["full_tile_comparison_summary"]["status_counts"]["core_regressed"], 1)
        self.assertEqual(package["recommended_next_move"]["action"], "review_full_tile_regression")
        self.assertIn("review_full_tile_regression", markdown)

    def test_mining_decision_is_review_only_when_no_cap_recommendation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = root / "experiments" / "runs" / "run1" / "full_tile_seg-a"
            artifact.mkdir(parents=True)
            metrics = artifact / "metrics.json"
            metrics.write_text(json.dumps({"val_positive_rate": 0.1, "threshold_risk_summary": {"selected": {"threshold": 0.3, "precision": 0.2, "recall": 0.6, "f05": 0.20, "f1": 0.30, "pred_positive_rate": 0.35, "pred_to_val_ratio": 3.5}}}), encoding="utf-8")
            with (artifact / "metrics_by_threshold.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["threshold", "precision", "recall", "f05", "f1", "pred_positive_rate"])
                writer.writeheader()
                writer.writerow({"threshold": 0.5, "precision": 0.2, "recall": 0.1, "f05": 0.1, "f1": 0.13, "pred_positive_rate": 0.2})
                writer.writerow({"threshold": 0.4, "precision": 0.2, "recall": 0.2, "f05": 0.2, "f1": 0.2, "pred_positive_rate": 0.3})
            snapshot = {
                "research_summary": {
                    "candidate_evidence": {
                        "candidate_run_id": "run1",
                        "loo": {"worst_fold_id": "seg-a"},
                        "full_tile": {
                            "evidence": [
                                {
                                    "path": str(metrics),
                                    "segment_id": "seg-a",
                                    "pred_positive_rate": 0.35,
                                    "val_positive_rate": 0.1,
                                    "fixed_threshold_f1": 0.0,
                                    "threshold_risk_summary": {
                                        "selected": {"f1": 0.30, "f05": 0.20, "pred_to_val_ratio": 3.5},
                                        "best_under_prratio2p0": {"f1": 0.10, "f05": 0.1, "pred_to_val_ratio": 1.9},
                                        "best_under_prratio3p0": {"f1": 0.15, "f05": 0.1, "pred_to_val_ratio": 2.9},
                                        "cap_binding": True,
                                    },
                                }
                            ]
                        },
                    },
                    "decision": {"next_action": "Review mining", "promotion_gate": {"ready": False}},
                }
            }
            snapshot_path = root / "snapshot.json"
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

            package = build_next_move_evidence_package(root, snapshot_json=snapshot_path, caps=[2.0, 3.0])

        self.assertIsNone(package["cap_recommendation"])
        self.assertEqual(package["cap_recommendation_blockers"][0]["reason"], "no_cap_preserves_retention")
        self.assertEqual(package["recommended_next_move"]["action"], "review_fold_safe_hard_negative_mining")
        self.assertFalse(package["recommended_next_move"]["writes_artifacts"])
        self.assertTrue(package["recommended_next_move"]["requires_review"])
        self.assertTrue(package["recommended_next_move"]["command_writes_artifacts"])
        self.assertFalse(package["recommended_next_move"]["safe_to_execute_from_dashboard"])
        self.assertIn("do not commit generated", package["recommended_next_move"]["artifact_policy"])

    def test_cap_comparison_error_blocks_cap_tightening(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = root / "experiments" / "runs" / "run1" / "full_tile_seg-a"
            artifact.mkdir(parents=True)
            metrics = artifact / "metrics.json"
            metrics.write_text(json.dumps({"val_positive_rate": 0.1, "threshold_risk_summary": {"selected": {"threshold": 0.3, "precision": 0.2, "recall": 0.6, "f05": 0.20, "f1": 0.30, "pred_positive_rate": 0.30, "pred_to_val_ratio": 3.0}}}), encoding="utf-8")
            (artifact / "metrics_by_threshold.csv").write_text("threshold,precision\n0.3,0.2\n", encoding="utf-8")
            snapshot = {
                "research_summary": {
                    "candidate_evidence": {
                        "candidate_run_id": "run1",
                        "loo": {"worst_fold_id": "seg-a"},
                        "full_tile": {
                            "evidence": [
                                {
                                    "path": str(metrics),
                                    "segment_id": "seg-a",
                                    "pred_positive_rate": 0.3,
                                    "val_positive_rate": 0.1,
                                    "fixed_threshold_f1": 0.0,
                                    "threshold_risk_summary": {
                                        "selected": {"f1": 0.30, "f05": 0.20, "pred_to_val_ratio": 3.0},
                                        "best_under_prratio2p5": {"f1": 0.29, "f05": 0.20, "pred_to_val_ratio": 2.4},
                                    },
                                }
                            ]
                        },
                    },
                    "decision": {"next_action": "Review cap evidence", "promotion_gate": {"ready": False}},
                }
            }
            snapshot_path = root / "snapshot.json"
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

            package = build_next_move_evidence_package(root, snapshot_json=snapshot_path, caps=[2.0, 2.5])
            markdown = render_markdown(package)

        self.assertIsNone(package["cap_recommendation"])
        self.assertEqual(package["cap_recommendation_blockers"][0]["reason"], "cap_comparison_error")
        self.assertEqual(package["recommended_next_move"]["action"], "follow_dashboard_next_action")
        self.assertIn("Cap Recommendation Blockers", markdown)
        self.assertIn("cap_comparison_error", markdown)

    def test_cli_smoke_outputs_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metrics = _write_threshold_artifact(root)
            snapshot = _write_snapshot(root, metrics)
            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(["--repo-root", str(root), "--snapshot-json", str(snapshot), "--markdown"])

        self.assertEqual(code, 0)
        self.assertIn("# Next-Move Evidence Package", stdout.getvalue())
        self.assertIn("Threshold Cap Comparison", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()

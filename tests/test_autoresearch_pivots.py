from __future__ import annotations

import copy
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import autoresearch
from autoresearch import _mutation_family, _pivot_bases, _prepare_autoresearch_base, _proposal_candidates, _proposal_plan, _promotion_gate, _promotion_next_action, _promotion_or_fallback_proposals, _promotion_phase_manual_action, _promotion_ready_payload, _propose_best_path, _propose_configs, _propose_from_recent_winners, _reserved_signatures, _search_signature, _set_nested, _strategy_phase
from experiments.runner import load_config


class AutoResearchPivotTest(unittest.TestCase):
    def test_torch_bases_get_torch_specific_candidates(self) -> None:
        cfg = load_config("configs/residual_25d_torch_unet_cpu.yaml")

        paths = [path for path, _value, _reason in _proposal_candidates(cfg)]

        self.assertIn(("training", "dice_loss_weight"), paths)
        self.assertIn(("evaluation", "tta_flips"), paths)
        self.assertIn(("model", "base_channels"), paths)
        self.assertIn(("training", "max_train_samples"), paths)

    def test_torch_candidates_include_positive_rate_loss_weight(self) -> None:
        cfg = load_config("configs/robust_calibrated_prloss_w0p03_lr0012_prratio3_seed11018.yaml")
        paths = [path for path, _value, _reason in _proposal_candidates(cfg)]

        self.assertIn(("training", "positive_rate_loss_weight"), paths)

    def test_mutation_family_classifies_existing_candidate_paths(self) -> None:
        self.assertEqual(_mutation_family(("training", "learning_rate")), "optimizer")
        self.assertEqual(_mutation_family(("training", "dice_loss_weight")), "loss_calibration")
        self.assertEqual(_mutation_family(("training", "sampling_strategy")), "data_sampling")
        self.assertEqual(_mutation_family(("model", "base_channels")), "model_family")
        self.assertEqual(_mutation_family(("evaluation", "tta_flips")), "inference_calibration")
        self.assertEqual(_mutation_family(("training", "seed")), "replication")

    def test_exhausted_focused_pair_pivots_to_curated_torch_base(self) -> None:
        base = load_config("configs/baseline.yaml")
        runs = []
        for path, value, _reason in _proposal_candidates(base):
            cfg = copy.deepcopy(base)
            _set_nested(cfg, path, value)
            runs.append({"config": cfg})

        with patch("autoresearch._reserved_signatures", side_effect=lambda current_runs: {_search_signature(run.get("config", {})) for run in current_runs}):
            proposals = autoresearch._propose_with_pivots(base, runs, count=2)

        self.assertTrue(proposals)
        for _name, cfg, _reason in proposals:
            self.assertIn("torch", cfg["model"]["name"])
            self.assertNotEqual(cfg["dataset"].get("research_scope"), base["dataset"].get("research_scope"))
            self.assertNotIn(_search_signature(cfg), {_search_signature(run["config"]) for run in runs})

    def test_best_path_prioritizes_robust_expanded_before_residual_smoke(self) -> None:
        bases = _pivot_bases()

        self.assertGreaterEqual(len(bases), 2)
        self.assertEqual(bases[0][0], "robust_multisegment_dice035_expanded.yaml")
        self.assertIn("robust_calibrated_prloss_w0p03_lr0012_prratio3_seed11018.yaml", {item[0] for item in bases})
        self.assertIn("multi_segment_robust", bases[0][1]["dataset"].get("research_scope", ""))

    def test_unattended_torch_bases_are_cpu_bounded(self) -> None:
        cfg = load_config("configs/robust_multisegment_dice035_expanded.yaml")

        prepared = _prepare_autoresearch_base(cfg)

        self.assertEqual(prepared["training"]["max_train_samples"], 1024)
        self.assertIn("cron_safety", prepared["autoresearch"])

    def test_best_path_does_not_start_with_focused_numpy_when_robust_available(self) -> None:
        base = load_config("configs/baseline.yaml")

        with patch("autoresearch._reserved_signatures", return_value=set()):
            proposals = _propose_best_path(base, [], count=2)

        self.assertTrue(proposals)
        for _name, cfg, _reason in proposals:
            self.assertIn("torch", cfg["model"]["name"])
            self.assertNotEqual(cfg["dataset"].get("research_scope"), base["dataset"].get("research_scope"))

    def test_recent_winner_followups_create_second_order_torch_moves(self) -> None:
        winner = _prepare_autoresearch_base(load_config("configs/robust_multisegment_dice035_expanded.yaml"))
        _set_nested(winner, ("training", "learning_rate"), 0.003)
        runs = [{"run_id": "winner", "config": winner, "main_metric": 0.39, "metrics": {"val_f1": 0.39, "average_precision": 0.24, "precision": 0.25, "recall": 0.7, "pred_positive_rate": 0.2, "val_positive_rate": 0.1}}]

        proposals = _propose_from_recent_winners(runs, count=2)

        self.assertTrue(proposals)
        for _name, cfg, reason in proposals:
            self.assertIn("torch", cfg["model"]["name"])
            self.assertEqual(cfg["dataset"].get("research_scope"), "multi_segment_robust_expanded")
            self.assertNotIn(_search_signature(cfg), {_search_signature(winner)})
            self.assertIn("recent base", reason)

    def test_strategy_phase_detects_plateau_and_forces_diversity(self) -> None:
        cfg = _prepare_autoresearch_base(load_config("configs/robust_multisegment_dice035_expanded.yaml"))
        runs = []
        for idx in range(16):
            run_cfg = copy.deepcopy(cfg)
            _set_nested(run_cfg, ("training", "seed"), 11000 + idx)
            runs.append({
                "run_id": f"run{idx}",
                "config": run_cfg,
                "main_metric": 0.30,
                "metrics": {"val_f1": 0.30, "average_precision": 0.10, "precision": 0.3, "recall": 0.4, "pred_positive_rate": 0.2, "val_positive_rate": 0.1},
            })

        with patch.dict("os.environ", {"AUTORESEARCH_PLATEAU_WINDOW": "8"}):
            strategy = _strategy_phase(runs)

        self.assertTrue(strategy["plateau"])
        self.assertIn(strategy["phase"], {"diversify", "promote"})
        self.assertTrue(strategy["required_families"])
        if strategy["phase"] == "promote":
            self.assertIn("loss_calibration", strategy["required_families"])

    def test_plateau_proposals_use_distinct_mutation_families(self) -> None:
        cfg = _prepare_autoresearch_base(load_config("configs/robust_multisegment_dice035_expanded.yaml"))

        with patch("autoresearch._reserved_signatures", return_value=set()):
            proposals = _propose_configs(
                cfg,
                [],
                count=4,
                lock_to_baseline_scope=False,
                required_families={"loss_calibration", "data_sampling", "model_family", "inference_calibration"},
                strategy_phase="diversify",
            )

        families = [proposal[1]["autoresearch"]["mutation_family"] for proposal in proposals]
        self.assertGreaterEqual(len(families), 3)
        self.assertEqual(len(families), len(set(families)))
        self.assertTrue(all(proposal[1]["autoresearch"]["strategy_phase"] == "diversify" for proposal in proposals))

    def test_promotion_next_action_prefers_loo_then_full_tile(self) -> None:
        cfg = load_config("configs/robust_multisegment_dice035_expanded.yaml")
        cfg["validation_setup"] = {"mode": "leave-one-segment-out"}
        run = {
            "config": cfg,
            "metrics": {"val_f1": 0.4, "average_precision": 0.2, "precision": 0.3, "recall": 0.7, "pred_positive_rate": 0.2, "val_positive_rate": 0.1, "fixed_threshold_status": "ok"},
        }

        self.assertEqual(_promotion_next_action(run), "run_seed_repeat_leave_one_out")
        run["metrics"]["loo_promotion_ready"] = True
        self.assertEqual(_promotion_next_action(run), "run_full_tile_validation")

    def test_proposal_plan_exposes_safe_metadata(self) -> None:
        cfg = _prepare_autoresearch_base(load_config("configs/robust_multisegment_dice035_expanded.yaml"))

        with patch("autoresearch._reserved_signatures", return_value=set()):
            proposals = _propose_configs(cfg, [], count=1, lock_to_baseline_scope=False, strategy_phase="exploit")

        plan = _proposal_plan(proposals)

        self.assertEqual(len(plan), 1)
        self.assertFalse(plan[0]["promotable"])
        self.assertIn("seed_repeat_leave_one_out", plan[0]["promotion_required"])
        self.assertIn("mutation_family", plan[0])

    def test_plan_json_mode_does_not_write_configs_or_run_experiments(self) -> None:
        cfg = _prepare_autoresearch_base(load_config("configs/robust_multisegment_dice035_expanded.yaml"))
        run = {"run_id": "run1", "config": cfg, "main_metric": 0.2, "metrics": {"val_f1": 0.2, "average_precision": 0.1, "precision": 0.2, "recall": 0.4, "pred_positive_rate": 0.2, "val_positive_rate": 0.1}}
        with tempfile.TemporaryDirectory() as tmpdir:
            old_configs = autoresearch.CONFIGS
            autoresearch.CONFIGS = Path(tmpdir)
            stdout = io.StringIO()
            stderr = io.StringIO()
            try:
                with patch("sys.argv", ["autoresearch.py", "--plan", "--json"]), patch("sys.stdout", stdout), patch("sys.stderr", stderr), patch("autoresearch._recent_runs", return_value=[run]), patch("autoresearch._promotion_ready_payload", return_value=None), patch("autoresearch.subprocess.run") as run_mock:
                    self.assertEqual(autoresearch.main(), 0)
            finally:
                autoresearch.CONFIGS = old_configs

        run_mock.assert_not_called()
        self.assertEqual(list(Path(tmpdir).glob("auto_*.yaml")), [])
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "planned")

    def test_plateau_promotion_phase_emits_loo_action_instead_of_proposals(self) -> None:
        cfg = load_config("configs/robust_multisegment_dice035_expanded.yaml")
        cfg["validation_setup"] = {"mode": "leave-one-segment-out"}
        runs = [
            {"run_id": "candidate", "artifact_dir": str(Path("experiments/runs/candidate")), "config": cfg, "main_metric": 0.39, "metrics": {"val_f1": 0.39, "average_precision": 0.24, "precision": 0.25, "recall": 0.7, "pred_positive_rate": 0.2, "val_positive_rate": 0.1, "fixed_threshold_status": "ok"}},
            {"run_id": "other", "artifact_dir": str(Path("experiments/runs/other")), "config": cfg, "main_metric": 0.38, "metrics": {"val_f1": 0.38, "average_precision": 0.23, "precision": 0.25, "recall": 0.7, "pred_positive_rate": 0.2, "val_positive_rate": 0.1, "fixed_threshold_status": "ok"}},
        ]

        with patch.dict("os.environ", {"AUTORESEARCH_PLATEAU_WINDOW": "2"}, clear=False):
            payload = _promotion_phase_manual_action(runs)

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["status"], "manual_promotion_action")
        self.assertEqual(payload["next_action"], "run_seed_repeat_leave_one_out")
        self.assertEqual(payload["candidate_run_id"], "candidate")
        self.assertIn("scripts/evaluate_leave_one_out.py", payload["command"])
        self.assertIn("--seeds 11001,11018,15050", payload["command"])
        self.assertEqual(payload["proposals"], [])

    def test_promotion_phase_auto_promote_runs_bounded_loo_command(self) -> None:
        cfg = load_config("configs/robust_multisegment_dice035_expanded.yaml")
        cfg["validation_setup"] = {"mode": "leave-one-segment-out"}
        runs = [
            {"run_id": "candidate", "artifact_dir": str(Path("experiments/runs/candidate")), "config": cfg, "main_metric": 0.39, "metrics": {"val_f1": 0.39, "average_precision": 0.24, "precision": 0.25, "recall": 0.7, "pred_positive_rate": 0.2, "val_positive_rate": 0.1, "fixed_threshold_status": "ok"}},
            {"run_id": "other", "artifact_dir": str(Path("experiments/runs/other")), "config": cfg, "main_metric": 0.38, "metrics": {"val_f1": 0.38, "average_precision": 0.23, "precision": 0.25, "recall": 0.7, "pred_positive_rate": 0.2, "val_positive_rate": 0.1, "fixed_threshold_status": "ok"}},
        ]

        with patch.dict("os.environ", {"AUTORESEARCH_PLATEAU_WINDOW": "2", "AUTORESEARCH_AUTO_PROMOTE": "1", "AUTORESEARCH_PROMOTION_TIMEOUT_SECONDS": "123"}, clear=False), \
            patch("autoresearch._run_automated_promotion", return_value={"automation_status": "SUCCEEDED"}) as promote_mock:
            payload = _promotion_phase_manual_action(runs)

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["status"], "manual_promotion_action")
        self.assertEqual(payload["next_action"], "run_seed_repeat_leave_one_out")
        self.assertEqual(payload["automation_status"], "SUCCEEDED")
        command_args = promote_mock.call_args.args[0]
        self.assertIn("scripts/evaluate_leave_one_out.py", command_args)
        self.assertIn("--seeds", command_args)
        self.assertIn("11001,11018,15050", command_args)
        self.assertIn("--jobs", command_args)
        self.assertEqual(promote_mock.call_args.args[3], 123)

    def test_promotion_phase_action_has_explicit_override(self) -> None:
        cfg = load_config("configs/robust_multisegment_dice035_expanded.yaml")
        runs = [
            {"run_id": "candidate", "artifact_dir": str(Path("experiments/runs/candidate")), "config": cfg, "main_metric": 0.39, "metrics": {"val_f1": 0.39, "average_precision": 0.24, "precision": 0.25, "recall": 0.7, "pred_positive_rate": 0.2, "val_positive_rate": 0.1, "fixed_threshold_status": "ok"}},
            {"run_id": "other", "artifact_dir": str(Path("experiments/runs/other")), "config": cfg, "main_metric": 0.38, "metrics": {"val_f1": 0.38, "average_precision": 0.23, "precision": 0.25, "recall": 0.7, "pred_positive_rate": 0.2, "val_positive_rate": 0.1, "fixed_threshold_status": "ok"}},
        ]

        with patch.dict("os.environ", {"AUTORESEARCH_PLATEAU_WINDOW": "2", "AUTORESEARCH_CONTINUE_AFTER_PROMOTION_ACTION": "1"}, clear=False):
            self.assertIsNone(_promotion_phase_manual_action(runs))

    def test_promotion_phase_action_uses_calibrated_candidate_when_recent_runs_are_stale(self) -> None:
        cfg = load_config("configs/robust_multisegment_dice035_expanded.yaml")
        cfg["validation_setup"] = {"mode": "leave-one-segment-out"}
        runs = [
            {"run_id": "new_stale", "artifact_dir": str(Path("experiments/runs/new_stale")), "config": cfg, "main_metric": 0.41, "metrics": {"val_f1": 0.41, "average_precision": 0.25, "precision": 0.25, "recall": 0.8, "pred_positive_rate": 0.2, "val_positive_rate": 0.1}},
            {"run_id": "older_stale", "artifact_dir": str(Path("experiments/runs/older_stale")), "config": cfg, "main_metric": 0.40, "metrics": {"val_f1": 0.40, "average_precision": 0.24, "precision": 0.25, "recall": 0.8, "pred_positive_rate": 0.2, "val_positive_rate": 0.1}},
            {"run_id": "calibrated", "artifact_dir": str(Path("experiments/runs/calibrated")), "config": cfg, "main_metric": 0.39, "metrics": {"val_f1": 0.39, "average_precision": 0.24, "precision": 0.25, "recall": 0.8, "pred_positive_rate": 0.2, "val_positive_rate": 0.1, "fixed_threshold_status": "ok"}},
        ]

        with patch.dict("os.environ", {"AUTORESEARCH_PLATEAU_WINDOW": "2"}, clear=False):
            payload = _promotion_phase_manual_action(runs)

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["candidate_run_id"], "calibrated")
        self.assertEqual(payload["next_action"], "run_seed_repeat_leave_one_out")

    def test_promotion_phase_action_advances_after_linked_loo_summary(self) -> None:
        cfg = load_config("configs/robust_multisegment_dice035_expanded.yaml")
        cfg["validation_setup"] = {"mode": "leave-one-segment-out"}
        run = {"run_id": "candidate", "artifact_dir": str(Path("experiments/runs/candidate")), "config": cfg, "main_metric": 0.39, "metrics": {"val_f1": 0.39, "average_precision": 0.24, "precision": 0.25, "recall": 0.7, "pred_positive_rate": 0.2, "val_positive_rate": 0.1, "fixed_threshold_status": "ok"}}
        other = {"run_id": "other", "artifact_dir": str(Path("experiments/runs/other")), "config": cfg, "main_metric": 0.38, "metrics": {"val_f1": 0.38, "average_precision": 0.23, "precision": 0.25, "recall": 0.7, "pred_positive_rate": 0.2, "val_positive_rate": 0.1, "fixed_threshold_status": "ok"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            old_logs = autoresearch.LOGS
            autoresearch.LOGS = Path(tmpdir)
            try:
                (autoresearch.LOGS / "candidate_seedrepeat.summary.json").write_text(json.dumps({"promotion_ready": True, "run_ids": ["candidate"]}))
                with patch.dict("os.environ", {"AUTORESEARCH_PLATEAU_WINDOW": "2"}, clear=False):
                    payload = _promotion_phase_manual_action([run, other])
            finally:
                autoresearch.LOGS = old_logs

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["candidate_run_id"], "candidate")
        self.assertEqual(payload["next_action"], "run_full_tile_validation")
        self.assertIsNone(payload["command"])

    def test_main_plan_json_returns_manual_promotion_action_without_writes(self) -> None:
        cfg = load_config("configs/robust_multisegment_dice035_expanded.yaml")
        cfg["validation_setup"] = {"mode": "leave-one-segment-out"}
        runs = [
            {"run_id": "candidate", "artifact_dir": str(Path("experiments/runs/candidate")), "config": cfg, "main_metric": 0.39, "metrics": {"val_f1": 0.39, "average_precision": 0.24, "precision": 0.25, "recall": 0.7, "pred_positive_rate": 0.2, "val_positive_rate": 0.1, "fixed_threshold_status": "ok"}},
            {"run_id": "other", "artifact_dir": str(Path("experiments/runs/other")), "config": cfg, "main_metric": 0.38, "metrics": {"val_f1": 0.38, "average_precision": 0.23, "precision": 0.25, "recall": 0.7, "pred_positive_rate": 0.2, "val_positive_rate": 0.1, "fixed_threshold_status": "ok"}},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            old_configs = autoresearch.CONFIGS
            autoresearch.CONFIGS = Path(tmpdir)
            stdout = io.StringIO()
            try:
                with patch.dict("os.environ", {"AUTORESEARCH_PLATEAU_WINDOW": "2"}, clear=False), patch("sys.argv", ["autoresearch.py", "--plan", "--json"]), patch("sys.stdout", stdout), patch("autoresearch._recent_runs", return_value=runs), patch("autoresearch._promotion_ready_payload", return_value=None), patch("autoresearch.subprocess.run") as run_mock:
                    self.assertEqual(autoresearch.main(), 0)
            finally:
                autoresearch.CONFIGS = old_configs

        run_mock.assert_not_called()
        self.assertEqual(list(Path(tmpdir).glob("auto_*.yaml")), [])
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "manual_promotion_action")
        self.assertEqual(payload["proposal_count"] if "proposal_count" in payload else len(payload["proposals"]), 0)

    def test_main_pauses_when_promotion_gate_is_ready(self) -> None:
        cfg = _prepare_autoresearch_base(load_config("configs/robust_multisegment_dice035_expanded.yaml"))
        run = {"run_id": "run1", "config": cfg, "main_metric": 0.2, "metrics": {"val_f1": 0.2}}
        with patch("sys.argv", ["autoresearch.py"]), patch("autoresearch._recent_runs", return_value=[run]), patch("autoresearch._promotion_ready_payload", return_value={"status": "promotion_ready", "next_action": "Promote candidate", "action_id": "promotion_review", "proposals": []}), patch("autoresearch.subprocess.run") as run_mock:
            self.assertEqual(autoresearch.main(), 0)

        run_mock.assert_not_called()

    def test_main_auto_executes_safe_promotion_ready_command(self) -> None:
        cfg = _prepare_autoresearch_base(load_config("configs/robust_multisegment_dice035_expanded.yaml"))
        run = {"run_id": "run1", "config": cfg, "main_metric": 0.2, "metrics": {"val_f1": 0.2}}
        payload = {
            "status": "promotion_ready",
            "next_action": "Run full-tile evidence",
            "candidate_run_id": "run1",
            "action_id": "weak_fold_full_tile",
            "command": ".venv/bin/python scripts/infer_full_tile.py --artifact experiments/runs/run1 --segment-id 20230530172803",
            "safe_to_execute_from_dashboard": False,
            "proposals": [],
        }

        with patch.dict("os.environ", {"AUTORESEARCH_AUTO_PROMOTE": "1", "AUTORESEARCH_PROMOTION_TIMEOUT_SECONDS": "222"}, clear=False), \
            patch("sys.argv", ["autoresearch.py"]), \
            patch("autoresearch._recent_runs", return_value=[run]), \
            patch("autoresearch._promotion_ready_payload", return_value=payload), \
            patch("autoresearch._run_automated_promotion", return_value={"automation_status": "SUCCEEDED"}) as promote_mock:
            self.assertEqual(autoresearch.main(), 0)

        command_args = promote_mock.call_args.args[0]
        self.assertEqual(command_args[:2], [".venv/bin/python", "scripts/infer_full_tile.py"])
        self.assertIn("--segment-id", command_args)
        self.assertEqual(promote_mock.call_args.args[1], "run1")
        self.assertEqual(promote_mock.call_args.args[3], 222)

    def test_main_skips_unknown_unsafe_promotion_ready_command(self) -> None:
        cfg = _prepare_autoresearch_base(load_config("configs/robust_multisegment_dice035_expanded.yaml"))
        run = {"run_id": "run1", "config": cfg, "main_metric": 0.2, "metrics": {"val_f1": 0.2}}
        payload = {
            "status": "promotion_ready",
            "next_action": "Run unsafe command",
            "candidate_run_id": "run1",
            "action_id": "unknown",
            "command": "sh -c 'touch should_not_happen'",
            "safe_to_execute_from_dashboard": False,
            "proposals": [],
        }

        with patch.dict("os.environ", {"AUTORESEARCH_AUTO_PROMOTE": "1"}, clear=False), \
            patch("sys.argv", ["autoresearch.py"]), \
            patch("autoresearch._recent_runs", return_value=[run]), \
            patch("autoresearch._promotion_ready_payload", return_value=payload), \
            patch("autoresearch._run_automated_promotion") as promote_mock:
            self.assertEqual(autoresearch.main(), 0)

        promote_mock.assert_not_called()

    def test_main_generates_proposals_for_calibrate_probability_scale(self) -> None:
        cfg = _prepare_autoresearch_base(load_config("configs/robust_multisegment_dice035_expanded.yaml"))
        run = {"run_id": "candidate", "config": cfg, "main_metric": 0.2, "metrics": {"val_f1": 0.2}}
        with tempfile.TemporaryDirectory() as tmpdir:
            old_configs = autoresearch.CONFIGS
            autoresearch.CONFIGS = Path(tmpdir)
            try:
                with patch("sys.argv", ["autoresearch.py"]), patch("autoresearch._recent_runs", return_value=[run]), patch("autoresearch._promotion_ready_payload", return_value={"status": "promotion_ready", "next_action": "Calibrate probability scale", "action_id": "calibrate_probability_scale", "candidate_run_id": "candidate", "proposals": []}), patch("autoresearch.subprocess.run") as run_mock:
                    self.assertEqual(autoresearch.main(), 0)

                run_mock.assert_called()
                call_args = run_mock.call_args_list[0][0][0]
                self.assertIn("run_experiment.py", call_args)
                # Verify the generated config has the threshold changed
                generated_paths = list(autoresearch.CONFIGS.glob("auto_*promotion*threshold*.yaml"))
                self.assertTrue(generated_paths)
                for path in generated_paths:
                    generated_cfg = load_config(path)
                    self.assertIn(generated_cfg["evaluation"]["threshold"], [0.31, 0.33, 0.35, 0.37])
                    self.assertEqual(generated_cfg["autoresearch"]["promotion_action_id"], "calibrate_probability_scale")
                    self.assertEqual(generated_cfg["autoresearch"]["intent"], "promotion_action")
            finally:
                autoresearch.CONFIGS = old_configs

    def test_promotion_ready_payload_prefers_candidate_evidence_action(self) -> None:
        snapshot = {
            "research_summary": {
                "decision": {
                    "next_action": "Promote candidate",
                    "promotion_gate": {"ready": True},
                    "candidate_evidence": {
                        "candidate_run_id": "candidate",
                        "weak_fold_full_tile": {"command_text": "fallback command"},
                    },
                    "promotion_actions": [{
                        "id": "weak_fold_full_tile",
                        "label": "Run full-tile on weak fold weakseg",
                        "command_text": ".venv/bin/python scripts/infer_full_tile.py --public-chunk-delay-sec 0.5",
                        "safe_to_execute_from_dashboard": False,
                        "writes_artifacts": True,
                    }],
                }
            }
        }

        with patch("research_dashboard.snapshot.build_snapshot", return_value=snapshot):
            payload = _promotion_ready_payload()

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["next_action"], "Run full-tile on weak fold weakseg")
        self.assertEqual(payload["candidate_run_id"], "candidate")
        self.assertIn("--public-chunk-delay-sec 0.5", payload["command"])
        self.assertIn("use_public_directory_backoff_and_chunk_pacing", payload["reasoning"])

    def test_promotion_ready_payload_skips_pause_for_expected_compressed_probs(self) -> None:
        snapshot = {
            "research_summary": {
                "decision": {
                    "next_action": "Calibrate probability scale",
                    "promotion_gate": {"ready": True},
                    "candidate_evidence": {
                        "candidate_run_id": "candidate",
                        "full_tile": {
                            "evidence": [
                                {"segment_id": "seg1", "fixed_threshold_failure_reason": "no_fixed_positive_predictions"}
                            ]
                        },
                    },
                    "promotion_actions": [{"id": "calibrate_probability_scale", "label": "Calibrate probability scale", "severity": "warning"}],
                }
            }
        }

        with patch("research_dashboard.snapshot.build_snapshot", return_value=snapshot):
            payload = _promotion_ready_payload()

        self.assertIsNone(payload)

    def test_promotion_ready_payload_omits_completed_weak_fold_command_for_review(self) -> None:
        snapshot = {
            "research_summary": {
                "decision": {
                    "next_action": "Review promotion candidate candidate",
                    "promotion_gate": {"ready": True},
                    "candidate_evidence": {
                        "candidate_run_id": "candidate",
                        "weak_fold_full_tile": {"status": "done", "command_text": "completed command"},
                    },
                    "promotion_actions": [{"id": "promotion_review", "label": "Review promotion candidate candidate", "kind": "review", "writes_artifacts": False}],
                }
            }
        }

        with patch("research_dashboard.snapshot.build_snapshot", return_value=snapshot):
            payload = _promotion_ready_payload()

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["action_id"], "promotion_review")
        self.assertIsNone(payload["command"])
        self.assertNotIn("use_public_directory_backoff_and_chunk_pacing", payload["reasoning"])

    def test_recent_winner_followups_prefer_expanded_robust_over_focused_residual_score(self) -> None:
        robust = _prepare_autoresearch_base(load_config("configs/robust_multisegment_dice035_expanded.yaml"))
        _set_nested(robust, ("training", "learning_rate"), 0.003)
        residual = _prepare_autoresearch_base(load_config("configs/residual_25d_torch_unet_cpu.yaml"))
        _set_nested(residual, ("training", "seed"), 1354)
        runs = [
            {"run_id": "residual", "config": residual, "main_metric": 0.45, "metrics": {"val_f1": 0.45, "average_precision": 0.3, "precision": 0.3, "recall": 0.8, "pred_positive_rate": 0.2, "val_positive_rate": 0.1}},
            {"run_id": "robust", "config": robust, "main_metric": 0.39, "metrics": {"val_f1": 0.39, "average_precision": 0.24, "precision": 0.25, "recall": 0.7, "pred_positive_rate": 0.2, "val_positive_rate": 0.1}},
        ]

        proposals = _propose_from_recent_winners(runs, count=1)

        self.assertEqual(proposals[0][1]["dataset"].get("research_scope"), "multi_segment_robust_expanded")
        self.assertIn("robust", proposals[0][2])

    def test_generated_auto_configs_are_reserved_signatures(self) -> None:
        cfg = _prepare_autoresearch_base(load_config("configs/robust_multisegment_dice035_expanded.yaml"))
        candidate_path, candidate_value, _reason = _proposal_candidates(cfg)[0]
        reserved_cfg = copy.deepcopy(cfg)
        _set_nested(reserved_cfg, candidate_path, candidate_value)
        with tempfile.TemporaryDirectory() as tmpdir:
            old_configs = autoresearch.CONFIGS
            autoresearch.CONFIGS = Path(tmpdir)
            try:
                (autoresearch.CONFIGS / "auto_reserved.yaml").write_text(autoresearch.yaml.safe_dump(reserved_cfg, sort_keys=False))
                reserved = _reserved_signatures([])
                proposals = _propose_configs(cfg, [], count=1, lock_to_baseline_scope=False)
            finally:
                autoresearch.CONFIGS = old_configs

        self.assertIn(_search_signature(reserved_cfg), reserved)
        self.assertTrue(proposals)
        self.assertNotEqual(_search_signature(proposals[0][1]), _search_signature(reserved_cfg))

    def test_stale_generated_auto_configs_do_not_reserve_signatures_forever(self) -> None:
        cfg = _prepare_autoresearch_base(load_config("configs/robust_multisegment_dice035_expanded.yaml"))
        candidate_path, candidate_value, _reason = _proposal_candidates(cfg)[0]
        reserved_cfg = copy.deepcopy(cfg)
        _set_nested(reserved_cfg, candidate_path, candidate_value)
        with tempfile.TemporaryDirectory() as tmpdir:
            old_configs = autoresearch.CONFIGS
            autoresearch.CONFIGS = Path(tmpdir)
            try:
                path = autoresearch.CONFIGS / "auto_stale.yaml"
                path.write_text(autoresearch.yaml.safe_dump(reserved_cfg, sort_keys=False))
                old_time = time.time() - 48 * 3600
                os.utime(path, (old_time, old_time))
                with patch.dict("os.environ", {"AUTORESEARCH_PENDING_CONFIG_TTL_HOURS": "1"}):
                    reserved = _reserved_signatures([])
            finally:
                autoresearch.CONFIGS = old_configs

        self.assertNotIn(_search_signature(reserved_cfg), reserved)

    def test_pending_config_ttl_can_be_disabled(self) -> None:
        cfg = _prepare_autoresearch_base(load_config("configs/robust_multisegment_dice035_expanded.yaml"))
        candidate_path, candidate_value, _reason = _proposal_candidates(cfg)[0]
        reserved_cfg = copy.deepcopy(cfg)
        _set_nested(reserved_cfg, candidate_path, candidate_value)
        with tempfile.TemporaryDirectory() as tmpdir:
            old_configs = autoresearch.CONFIGS
            autoresearch.CONFIGS = Path(tmpdir)
            try:
                path = autoresearch.CONFIGS / "auto_stale.yaml"
                path.write_text(autoresearch.yaml.safe_dump(reserved_cfg, sort_keys=False))
                old_time = time.time() - 48 * 3600
                os.utime(path, (old_time, old_time))
                with patch.dict("os.environ", {"AUTORESEARCH_PENDING_CONFIG_TTL_HOURS": "0"}):
                    reserved = _reserved_signatures([])
            finally:
                autoresearch.CONFIGS = old_configs

        self.assertIn(_search_signature(reserved_cfg), reserved)

    def test_promotion_gate_flags_bad_positive_rate(self) -> None:
        cfg = load_config("configs/robust_multisegment_dice035_expanded.yaml")
        cfg["validation_setup"] = {"mode": "leave-one-segment-out"}
        eligible, warnings = _promotion_gate({"config": cfg, "metrics": {"val_f1": 0.4, "average_precision": 0.2, "precision": 0.1, "recall": 0.9, "pred_positive_rate": 0.8, "val_positive_rate": 0.1}})

        self.assertFalse(eligible)
        self.assertIn("pred_positive_rate_ratio_suspicious", warnings)

    def test_promotion_gate_allows_held_out_validation_modes(self) -> None:
        metrics = {"val_f1": 0.4, "average_precision": 0.2, "precision": 0.2, "recall": 0.8, "pred_positive_rate": 0.2, "val_positive_rate": 0.1, "fixed_threshold_status": "ok"}
        for mode in ("cross-segment", "cross-scroll", "leave-one-segment-out"):
            cfg = load_config("configs/robust_multisegment_dice035_expanded.yaml")
            cfg["validation_setup"] = {"mode": mode}

            eligible, warnings = _promotion_gate({"config": cfg, "metrics": metrics})

            self.assertTrue(eligible, mode)
            self.assertNotIn("validation_not_held_out", warnings)

    def test_promotion_gate_allows_good_calibrated_held_out_candidate(self) -> None:
        cfg = load_config("configs/robust_multisegment_dice035_expanded.yaml")
        cfg["validation_setup"] = {"mode": "leave-one-segment-out"}
        metrics = {"val_f1": 0.4, "average_precision": 0.2, "precision": 0.2, "recall": 0.8, "pred_positive_rate": 0.2, "val_positive_rate": 0.1, "fixed_threshold_status": "ok", "best_threshold": 0.5, "ap_prevalence_lift": 2.0}

        eligible, warnings = _promotion_gate({"config": cfg, "metrics": metrics})

        self.assertTrue(eligible)
        self.assertNotIn("best_threshold_at_sweep_edge", warnings)
        self.assertNotIn("weak_ap_lift", warnings)

    def test_promotion_gate_blocks_best_threshold_at_sweep_edge(self) -> None:
        metrics = {"val_f1": 0.4, "average_precision": 0.2, "precision": 0.2, "recall": 0.8, "pred_positive_rate": 0.2, "val_positive_rate": 0.1, "fixed_threshold_status": "ok", "ap_prevalence_lift": 2.0}
        for best_threshold in (0.03, 0.94):
            cfg = load_config("configs/robust_multisegment_dice035_expanded.yaml")
            cfg["validation_setup"] = {"mode": "leave-one-segment-out"}

            eligible, warnings = _promotion_gate({"config": cfg, "metrics": {**metrics, "best_threshold": best_threshold}})

            self.assertFalse(eligible, best_threshold)
            self.assertIn("best_threshold_at_sweep_edge", warnings)

    def test_promotion_gate_blocks_weak_ap_lift_when_present(self) -> None:
        cfg = load_config("configs/robust_multisegment_dice035_expanded.yaml")
        cfg["validation_setup"] = {"mode": "leave-one-segment-out"}
        metrics = {"val_f1": 0.4, "average_precision": 0.2, "precision": 0.2, "recall": 0.8, "pred_positive_rate": 0.2, "val_positive_rate": 0.1, "fixed_threshold_status": "ok", "best_threshold": 0.5, "ap_prevalence_lift": 1.24}

        eligible, warnings = _promotion_gate({"config": cfg, "metrics": metrics})

        self.assertFalse(eligible)
        self.assertIn("weak_ap_lift", warnings)

    def test_promotion_gate_allows_patch_size_at_scroll_prize_guidance(self) -> None:
        cfg = load_config("configs/robust_multisegment_dice035_expanded.yaml")
        cfg["validation_setup"] = {"mode": "leave-one-segment-out"}
        cfg.setdefault("dataset", {})["patch_size"] = 64
        metrics = {"val_f1": 0.4, "average_precision": 0.2, "precision": 0.2, "recall": 0.8, "pred_positive_rate": 0.2, "val_positive_rate": 0.1, "fixed_threshold_status": "ok"}

        eligible, warnings = _promotion_gate({"config": cfg, "metrics": metrics})

        self.assertTrue(eligible)
        self.assertNotIn("patch_size_exceeds_scroll_prize_guidance", warnings)

    def test_promotion_gate_blocks_patch_size_above_scroll_prize_guidance(self) -> None:
        cfg = load_config("configs/robust_multisegment_dice035_expanded.yaml")
        cfg["validation_setup"] = {"mode": "leave-one-segment-out"}
        cfg.setdefault("dataset", {})["patch_size"] = 65
        metrics = {"val_f1": 0.4, "average_precision": 0.2, "precision": 0.2, "recall": 0.8, "pred_positive_rate": 0.2, "val_positive_rate": 0.1, "fixed_threshold_status": "ok"}

        eligible, warnings = _promotion_gate({"config": cfg, "metrics": metrics})

        self.assertFalse(eligible)
        self.assertIn("patch_size_exceeds_scroll_prize_guidance", warnings)

    def test_promotion_gate_blocks_nested_window_size_above_scroll_prize_guidance(self) -> None:
        cfg = load_config("configs/robust_multisegment_dice035_expanded.yaml")
        cfg["validation_setup"] = {"mode": "leave-one-segment-out"}
        cfg.setdefault("model", {})["window_size"] = [64, 80]
        metrics = {"val_f1": 0.4, "average_precision": 0.2, "precision": 0.2, "recall": 0.8, "pred_positive_rate": 0.2, "val_positive_rate": 0.1, "fixed_threshold_status": "ok"}

        eligible, warnings = _promotion_gate({"config": cfg, "metrics": metrics})

        self.assertFalse(eligible)
        self.assertIn("patch_size_exceeds_scroll_prize_guidance", warnings)

    def test_promotion_gate_blocks_same_segment_and_unknown_validation(self) -> None:
        metrics = {"val_f1": 0.4, "average_precision": 0.2, "precision": 0.2, "recall": 0.8, "pred_positive_rate": 0.2, "val_positive_rate": 0.1, "fixed_threshold_status": "ok"}
        for mode in ("spatial-same-segment", "unknown"):
            cfg = load_config("configs/robust_multisegment_dice035_expanded.yaml")
            cfg["validation_setup"] = {"mode": mode}

            eligible, warnings = _promotion_gate({"config": cfg, "metrics": metrics})

            self.assertFalse(eligible, mode)
            self.assertIn("validation_not_held_out", warnings)

    def test_promotion_gate_blocks_weak_fixed_threshold_status(self) -> None:
        cfg = load_config("configs/robust_multisegment_dice035_expanded.yaml")
        cfg["validation_setup"] = {"mode": "leave-one-segment-out"}
        run = {"config": cfg, "metrics": {"val_f1": 0.4, "average_precision": 0.2, "precision": 0.2, "recall": 0.8, "pred_positive_rate": 0.2, "val_positive_rate": 0.1, "fixed_threshold_status": "weak"}}

        eligible, warnings = _promotion_gate(run)

        self.assertFalse(eligible)
        self.assertIn("fixed_threshold_status_weak", warnings)
        self.assertEqual(_promotion_next_action(run), "calibrate_probability_scale")

    def test_promotion_gate_blocks_missing_fixed_threshold_status(self) -> None:
        cfg = load_config("configs/robust_multisegment_dice035_expanded.yaml")
        cfg["validation_setup"] = {"mode": "leave-one-segment-out"}
        run = {"config": cfg, "metrics": {"val_f1": 0.4, "average_precision": 0.2, "precision": 0.2, "recall": 0.8, "pred_positive_rate": 0.2, "val_positive_rate": 0.1}}

        eligible, warnings = _promotion_gate(run)

        self.assertFalse(eligible)
        self.assertIn("missing_fixed_threshold_status", warnings)
        self.assertEqual(_promotion_next_action(run), "calibrate_probability_scale")

    def test_signature_distinguishes_seed_ensemble_and_threshold(self) -> None:
        cfg = load_config("configs/robust_multisegment_dice035_expanded.yaml")
        ensemble = copy.deepcopy(cfg)
        ensemble["training"]["seeds"] = [1, 2, 3]
        thresholded = copy.deepcopy(cfg)
        thresholded.setdefault("evaluation", {})["threshold"] = 0.35

        self.assertNotEqual(_search_signature(cfg), _search_signature(ensemble))
        self.assertNotEqual(_search_signature(cfg), _search_signature(thresholded))

    def test_best_path_uses_recent_winners_after_static_bases_are_exhausted(self) -> None:
        base = load_config("configs/baseline.yaml")
        runs = []
        for _name, pivot, _scope_policy in _pivot_bases():
            for path, value, _reason in _proposal_candidates(pivot):
                cfg = copy.deepcopy(pivot)
                _set_nested(cfg, path, value)
                runs.append({"run_id": "tested", "config": cfg, "main_metric": 0.1, "metrics": {"val_f1": 0.1}})
        winner = copy.deepcopy(runs[0]["config"])
        winner["training"]["learning_rate"] = 0.003
        runs.insert(0, {"run_id": "winner", "config": winner, "main_metric": 0.39, "metrics": {"val_f1": 0.39, "average_precision": 0.24, "precision": 0.25, "recall": 0.7, "pred_positive_rate": 0.2, "val_positive_rate": 0.1}})

        with patch("autoresearch._pivot_bases", return_value=[]):
            proposals = _propose_best_path(base, runs, count=2)

        self.assertTrue(proposals)

    def test_exhausted_auto_promotion_action_falls_back_to_normal_exploration(self) -> None:
        base = load_config("configs/robust_calibrated_prloss_w0p03_lr0012_prratio3_seed11018.yaml")
        fallback = [("fallback.yaml", copy.deepcopy(base), "normal fallback")]

        with patch("autoresearch._generate_promotion_action_proposals", return_value=[]), patch("autoresearch._propose_best_path", return_value=fallback):
            proposals, source_action = _promotion_or_fallback_proposals([], base, {"action_id": "calibrate_positive_rate"}, 1)

        self.assertEqual(proposals, fallback)
        self.assertIsNone(source_action)


if __name__ == "__main__":
    unittest.main()

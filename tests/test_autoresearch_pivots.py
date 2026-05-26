from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import autoresearch
from autoresearch import _pivot_bases, _prepare_autoresearch_base, _proposal_candidates, _promotion_gate, _propose_best_path, _propose_configs, _propose_from_recent_winners, _reserved_signatures, _search_signature, _set_nested
from experiments.runner import load_config


class AutoResearchPivotTest(unittest.TestCase):
    def test_torch_bases_get_torch_specific_candidates(self) -> None:
        cfg = load_config("configs/residual_25d_torch_unet_cpu.yaml")

        paths = [path for path, _value, _reason in _proposal_candidates(cfg)]

        self.assertIn(("training", "dice_loss_weight"), paths)
        self.assertIn(("evaluation", "tta_flips"), paths)
        self.assertIn(("model", "base_channels"), paths)
        self.assertIn(("training", "max_train_samples"), paths)

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

    def test_promotion_gate_flags_bad_positive_rate(self) -> None:
        cfg = load_config("configs/robust_multisegment_dice035_expanded.yaml")
        eligible, warnings = _promotion_gate({"config": cfg, "metrics": {"val_f1": 0.4, "average_precision": 0.2, "precision": 0.1, "recall": 0.9, "pred_positive_rate": 0.8, "val_positive_rate": 0.1}})

        self.assertFalse(eligible)
        self.assertIn("pred_positive_rate_ratio_suspicious", warnings)

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
        for _name, cfg, reason in proposals:
            self.assertIn("torch", cfg["model"]["name"])
            self.assertIn("recent base", reason)


if __name__ == "__main__":
    unittest.main()

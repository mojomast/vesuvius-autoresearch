from __future__ import annotations

import copy
import unittest

from autoresearch import _proposal_candidates, _propose_with_pivots, _search_signature, _set_nested
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

        proposals = _propose_with_pivots(base, runs, count=2)

        self.assertTrue(proposals)
        for _name, cfg, _reason in proposals:
            self.assertIn("torch", cfg["model"]["name"])
            self.assertNotEqual(cfg["dataset"].get("research_scope"), base["dataset"].get("research_scope"))
            self.assertNotIn(_search_signature(cfg), {_search_signature(run["config"]) for run in runs})


if __name__ == "__main__":
    unittest.main()

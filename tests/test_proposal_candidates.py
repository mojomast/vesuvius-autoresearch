from __future__ import annotations

import autoresearch
from autoresearch import PARAM_BOUNDS, _apply_candidate, _config_cost_tier, _generate_promotion_action_proposals, _mutation_family, _proposal_candidates, _proposal_plan, _proposal_value_slug, _search_signature


def _assert_candidates_within_bounds(config):
    for path, value, _reason in _proposal_candidates(config):
        if path not in PARAM_BOUNDS or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        low, high = PARAM_BOUNDS[path]
        assert low <= float(value) <= high, (path, value, low, high)


def test_numpy_candidates_obey_param_bounds():
    _assert_candidates_within_bounds({
        "model": {"name": "tiny_numpy_ink_logreg", "depth": 3},
        "training": {
            "pos_weight": 100.0,
            "learning_rate": 0.0001,
            "weight_decay": 1.0,
            "epochs": 50,
            "seed": 1,
        },
        "evaluation": {"threshold": 0.5},
    })


def test_torch_candidates_obey_param_bounds():
    _assert_candidates_within_bounds({
        "model": {"name": "tiny_torch_unet", "base_channels": 128},
        "training": {
            "learning_rate": 0.0001,
            "epochs": 1,
            "batch_size": 1,
            "max_train_samples": 999999,
            "dice_loss_weight": 2.0,
            "positive_rate_loss_weight": 2.0,
            "seed": 1,
        },
        "evaluation": {"threshold": 0.5},
    })


def test_torch_candidates_include_evidence_backed_positive_rate_controls():
    raw_candidates = _proposal_candidates({
        "model": {"name": "tiny_torch_unet", "base_channels": 8},
        "training": {
            "learning_rate": 0.0012,
            "epochs": 5,
            "batch_size": 8,
            "max_train_samples": 1024,
            "dice_loss_weight": 0.35,
            "positive_rate_loss_weight": 0.08,
            "positive_rate_loss_tolerance": 0.01,
            "seed": 11018,
        },
        "evaluation": {"threshold": 0.5, "max_pred_positive_rate_ratio": 3.0},
    })
    paths = [path for path, _value, _reason in raw_candidates]
    cap_values = [value for path, value, _reason in raw_candidates if path == ("evaluation", "max_pred_positive_rate_ratio")]
    tol_values = [value for path, value, _reason in raw_candidates if path == ("training", "positive_rate_loss_tolerance")]

    assert 2.5 in cap_values
    assert 2.75 in cap_values
    assert 3.5 in cap_values
    assert 0.008 in tol_values
    assert ("training", "positive_rate_loss_tolerance") in paths


def test_torch_candidates_include_combined_balanced_calibration():
    raw_candidates = _proposal_candidates({
        "model": {"name": "tiny_torch_unet", "base_channels": 8},
        "training": {"positive_rate_loss_tolerance": 0.01},
        "evaluation": {"max_pred_positive_rate_ratio": 3.0},
    })

    combined = [value for path, value, _reason in raw_candidates if path == ("balanced_calibration",)]

    assert {"max_pred_positive_rate_ratio": 2.75, "positive_rate_loss_tolerance": 0.008} in combined
    assert {"max_pred_positive_rate_ratio": 3.0, "positive_rate_loss_weight": 0.10} in combined


def test_balanced_calibration_candidate_applies_both_fields():
    cfg = {"training": {"positive_rate_loss_tolerance": 0.01}, "evaluation": {"max_pred_positive_rate_ratio": 3.0}}
    value = {"max_pred_positive_rate_ratio": 2.75, "positive_rate_loss_tolerance": 0.008}

    _apply_candidate(cfg, ("balanced_calibration",), value)

    assert cfg["evaluation"]["max_pred_positive_rate_ratio"] == 2.75
    assert cfg["training"]["positive_rate_loss_tolerance"] == 0.008
    assert _mutation_family(("balanced_calibration",)) == "balanced_calibration"


def test_balanced_calibration_candidate_can_apply_loss_weight():
    cfg = {"training": {"positive_rate_loss_weight": 0.08}, "evaluation": {"max_pred_positive_rate_ratio": 2.75}}
    value = {"max_pred_positive_rate_ratio": 3.0, "positive_rate_loss_weight": 0.10}

    _apply_candidate(cfg, ("balanced_calibration",), value)

    assert cfg["evaluation"]["max_pred_positive_rate_ratio"] == 3.0
    assert cfg["training"]["positive_rate_loss_weight"] == 0.10
    assert _mutation_family(("balanced_calibration",)) == "balanced_calibration"


def test_torch_candidates_can_propose_4096_samples_when_budget_allows(monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_TORCH_MAX_TRAIN_SAMPLES", "4096")

    sample_values = [value for path, value, _reason in _proposal_candidates({
        "model": {"name": "residual_25d_torch_unet", "base_channels": 8},
        "training": {"max_train_samples": 1024, "learning_rate": 0.0012, "epochs": 5, "batch_size": 8},
        "evaluation": {"threshold": 0.5},
    }) if path == ("training", "max_train_samples")]

    assert 4096 in sample_values


def test_required_hard_bounds_are_enforced():
    candidates = dict((path, value) for path, value, _reason in _proposal_candidates({
        "model": {"name": "tiny_numpy_ink_logreg"},
        "training": {"pos_weight": 1000.0, "learning_rate": 0.00001, "weight_decay": 999.0, "epochs": 5},
        "evaluation": {"threshold": 0.5},
    }))

    assert candidates[("training", "pos_weight")] <= 25.0
    assert candidates[("training", "weight_decay")] <= 0.05
    assert candidates[("training", "learning_rate")] >= 0.001


def test_positive_rate_cap_and_tolerance_participate_in_search_signature():
    base = {
        "model": {"name": "tiny_torch_unet"},
        "training": {"positive_rate_loss_tolerance": 0.01},
        "evaluation": {"max_pred_positive_rate_ratio": 2.5},
    }
    changed_cap = {
        "model": {"name": "tiny_torch_unet"},
        "training": {"positive_rate_loss_tolerance": 0.01},
        "evaluation": {"max_pred_positive_rate_ratio": 3.0},
    }
    changed_tolerance = {
        "model": {"name": "tiny_torch_unet"},
        "training": {"positive_rate_loss_tolerance": 0.005},
        "evaluation": {"max_pred_positive_rate_ratio": 2.5},
    }

    assert _search_signature(base) != _search_signature(changed_cap)
    assert _search_signature(base) != _search_signature(changed_tolerance)


def test_promotion_action_proposals_obey_param_bounds(monkeypatch):
    run = {
        "run_id": "candidate",
        "config": {
            "model": {"name": "tiny_torch_unet"},
            "training": {"learning_rate": 0.0001, "epochs": 5},
            "evaluation": {"threshold": 0.5},
        },
    }
    monkeypatch.setattr(autoresearch, "_reserved_signatures", lambda runs: set())

    proposals = _generate_promotion_action_proposals(
        [run],
        {"action_id": "improve_ranking_signal", "candidate_run_id": "candidate"},
        count=2,
    )

    lr_proposals = [cfg for _name, cfg, _reason in proposals if cfg["autoresearch"]["changed_path"] == "training.learning_rate"]
    assert lr_proposals
    assert lr_proposals[0]["training"]["learning_rate"] == PARAM_BOUNDS[("training", "learning_rate")][0]


def test_proposal_value_slug_is_filesystem_safe_for_lists():
    slug = _proposal_value_slug(["data/mined/seg-a.npz", "data/mined/seg-b.npz"])

    assert slug == "2items"
    assert "/" not in slug
    assert "[" not in slug


def test_config_cost_tier_classifies_simple_and_heavy_configs():
    assert _config_cost_tier({"model": {"name": "tiny_numpy_ink_logreg"}, "training": {"max_train_pixels": 200000}}) == "cheap"
    assert _config_cost_tier({"model": {"name": "tiny_torch_unet"}, "training": {"epochs": 5, "max_train_samples": 1024}}) == "normal"
    assert _config_cost_tier({"model": {"name": "tiny_torch_unet"}, "training": {"epochs": 5, "max_train_samples": 4096}}) == "expensive"
    assert _config_cost_tier({"model": {"name": "tiny_torch_unet"}, "training": {"seeds": [1, 2], "max_train_samples": 1024}}) == "expensive"


def test_promotion_action_proposals_include_cost_tier_metadata(monkeypatch):
    run = {
        "run_id": "candidate",
        "config": {
            "model": {"name": "tiny_torch_unet"},
            "training": {"learning_rate": 0.001, "epochs": 5, "max_train_samples": 1024},
            "evaluation": {"threshold": 0.5},
        },
    }
    monkeypatch.setattr(autoresearch, "_reserved_signatures", lambda runs: set())

    proposals = _generate_promotion_action_proposals([run], {"action_id": "calibrate_probability_scale", "candidate_run_id": "candidate"}, count=1)

    assert proposals
    cfg = proposals[0][1]
    assert cfg["autoresearch"]["cost_tier"] == "normal"
    assert cfg["autoresearch"]["run_profile"] == "exploration"
    assert _proposal_plan(proposals)[0]["cost_tier"] == "normal"

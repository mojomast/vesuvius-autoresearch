from __future__ import annotations

from autoresearch import PARAM_BOUNDS, _proposal_candidates


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


def test_required_hard_bounds_are_enforced():
    candidates = dict((path, value) for path, value, _reason in _proposal_candidates({
        "model": {"name": "tiny_numpy_ink_logreg"},
        "training": {"pos_weight": 1000.0, "learning_rate": 0.00001, "weight_decay": 999.0, "epochs": 5},
        "evaluation": {"threshold": 0.5},
    }))

    assert candidates[("training", "pos_weight")] <= 25.0
    assert candidates[("training", "weight_decay")] <= 0.05
    assert candidates[("training", "learning_rate")] >= 0.001

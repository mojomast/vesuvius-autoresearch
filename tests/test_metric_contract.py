import pytest

from autoresearch import validate_metric_contract


def _valid_run():
    return {
        "run_id": "run-1",
        "timestamp": "2026-05-28T00:00:00+00:00",
        "config": {"evaluation": {"main_metric": "val_f1"}},
        "main_metric": 0.2,
        "artifact_dir": "experiments/runs/run-1",
        "metrics": {
            "val_loss": 0.9,
            "val_f1": 0.2,
            "val_f05": 0.18,
            "best_threshold": 0.35,
            "precision": 0.3,
            "recall": 0.15,
            "average_precision": 0.25,
            "ap_prevalence_lift": 2.0,
            "val_positive_rate": 0.02,
            "pred_positive_rate": 0.03,
            "fixed_threshold_status": "ok",
        },
    }


def test_metric_contract_adds_optional_promotion_defaults():
    run = validate_metric_contract(_valid_run())

    assert run["metrics"]["promotion_checks"] == {}
    assert run["metrics"]["loo_promotion_ready"] is False
    assert run["metrics"]["full_tile_promotion_ready"] is False


def test_metric_contract_rejects_missing_required_metric():
    run = _valid_run()
    del run["metrics"]["average_precision"]

    with pytest.raises(ValueError, match="average_precision"):
        validate_metric_contract(run)


def test_metric_contract_rejects_bad_metric_type():
    run = _valid_run()
    run["metrics"]["val_f1"] = "0.2"

    with pytest.raises(ValueError, match="val_f1"):
        validate_metric_contract(run)

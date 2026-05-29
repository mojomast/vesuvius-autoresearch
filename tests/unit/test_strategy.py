from __future__ import annotations

from autoresearch import _run_quality_score


def test_run_quality_score_penalizes_calibration_error() -> None:
    base = {
        "config": {"dataset": {"research_scope": "multi_segment_test"}},
        "main_metric": 0.4,
        "metrics": {
            "val_f1": 0.4,
            "val_f05": 0.35,
            "average_precision": 0.3,
            "precision": 0.3,
            "recall": 0.5,
            "fixed_threshold_status": "ok",
            "ap_prevalence_lift": 2.0,
            "best_threshold": 0.35,
            "pred_positive_rate": 0.1,
            "val_positive_rate": 0.1,
        },
    }
    over = {**base, "metrics": {**base["metrics"], "pred_positive_rate": 0.3}}

    assert _run_quality_score(base) > _run_quality_score(over)

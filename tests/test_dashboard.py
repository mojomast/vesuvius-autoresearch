from __future__ import annotations

import json
import sqlite3

from experiments.runner import init_db
from research_dashboard.snapshot import build_snapshot


def test_dashboard_snapshot_renders_harness_and_promotion_state(tmp_path):
    db_path = tmp_path / "experiments" / "experiments.db"
    init_db(db_path)
    metrics = {
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
    }
    config = {
        "dataset": {"research_scope": "multi_segment_robust_expanded", "validation_mode": "leave-one-segment-out"},
        "model": {"name": "tiny_torch_unet"},
        "training": {"learning_rate": 0.0012, "weight_decay": 0.0001, "epochs": 5, "pos_weight": "auto"},
        "evaluation": {"main_metric": "val_f1", "threshold": 0.5},
    }
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO experiments(run_id,timestamp,config_json,main_metric,secondary_metrics_json,artifact_dir,config_signature) VALUES (?,?,?,?,?,?,?)", ("run-1", "2026-05-28T00:00:00+00:00", json.dumps(config), 0.2, json.dumps(metrics), str(tmp_path / "runs" / "run-1"), "sig"))
        conn.execute("CREATE TABLE promotion_results(run_id TEXT, timestamp TEXT, status TEXT, payload_json TEXT)")
        conn.execute("INSERT INTO promotion_results VALUES (?,?,?,?)", ("run-1", "2026-05-28T00:01:00+00:00", "FAILED", json.dumps({"log_file": "logs/promotion.log", "error": "mock"})))

    snapshot = build_snapshot(tmp_path, use_cache=False)

    assert snapshot["harness"]["type"] == "vesuvius"
    assert "val_f1" in snapshot["harness"]["metric_contract_keys"]
    assert "training.learning_rate" in snapshot["param_drift"]["bounds"]
    assert snapshot["experiments"]["promotion_results"][0]["log_file"] == "logs/promotion.log"

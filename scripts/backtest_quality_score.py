#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def _ratio(metrics: dict[str, object]) -> float | None:
    pred = metrics.get("pred_positive_rate")
    val = metrics.get("val_positive_rate")
    if pred is None or val is None:
        return None
    return float(pred) / max(float(val), 1e-6)


def _promotion_eligible_proxy(metrics: dict[str, object]) -> bool:
    ratio = _ratio(metrics)
    return (
        float(metrics.get("val_f1") or 0.0) > 0.0
        and float(metrics.get("average_precision") or 0.0) > 0.0
        and str(metrics.get("fixed_threshold_status") or "").lower() == "ok"
        and ratio is not None
        and 0.1 <= ratio <= 3.5
        and float(metrics.get("precision") or 0.0) > 0.0
        and float(metrics.get("recall") or 0.0) > 0.0
    )


def _score(metrics: dict[str, object], ap_weight: float, f05_weight: float, calibration_penalty_weight: float) -> float:
    score = float(metrics.get("val_f1") or 0.0) + ap_weight * float(metrics.get("average_precision") or 0.0) + f05_weight * float(metrics.get("val_f05") or 0.0)
    ratio = _ratio(metrics)
    if ratio is not None:
        score -= calibration_penalty_weight * min(abs(ratio - 1.0), 0.5)
    return score


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest AutoResearch quality score weights against the experiment DB")
    parser.add_argument("--db", default="experiments/experiments.db")
    parser.add_argument("--ap-weights", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--f05-weights", default="0,0.05,0.1,0.2")
    parser.add_argument("--calibration-penalty-weights", default="0,0.03,0.05,0.1")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    db_path = Path(args.db)
    rows = []
    with sqlite3.connect(db_path) as conn:
        for run_id, timestamp, metric_json in conn.execute("SELECT run_id,timestamp,secondary_metrics_json FROM experiments ORDER BY timestamp ASC"):
            try:
                metrics = json.loads(metric_json)
            except json.JSONDecodeError:
                continue
            if isinstance(metrics, dict):
                rows.append({"run_id": run_id, "timestamp": timestamp, "metrics": metrics})

    ap_weights = [float(item) for item in args.ap_weights.split(",") if item]
    f05_weights = [float(item) for item in args.f05_weights.split(",") if item]
    calibration_penalty_weights = [float(item) for item in args.calibration_penalty_weights.split(",") if item]
    results = []
    for ap_weight in ap_weights:
        for f05_weight in f05_weights:
            for calibration_penalty_weight in calibration_penalty_weights:
                ranked = sorted(rows, key=lambda row: _score(row["metrics"], ap_weight, f05_weight, calibration_penalty_weight), reverse=True)
                top = ranked[: args.top_k]
                mean_f1 = sum(float(row["metrics"].get("val_f1") or 0.0) for row in top) / max(len(top), 1)
                eligible_top5 = sum(1 for row in ranked[:5] if _promotion_eligible_proxy(row["metrics"]))
                gt3_top5 = sum(1 for row in ranked[:5] if (_ratio(row["metrics"]) or 0.0) > 3.0)
                results.append({
                    "ap_weight": ap_weight,
                    "f05_weight": f05_weight,
                    "calibration_penalty_weight": calibration_penalty_weight,
                    "top_k": len(top),
                    "mean_top_val_f1": mean_f1,
                    "eligible_proxy_top5": eligible_top5,
                    "pred_ratio_gt3_top5": gt3_top5,
                    "top_run_ids": [row["run_id"] for row in top],
                })

    print(json.dumps({"db": str(db_path), "row_count": len(rows), "results": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

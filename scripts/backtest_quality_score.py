#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def _score(metrics: dict[str, object], ap_weight: float, f05_weight: float) -> float:
    return float(metrics.get("val_f1") or 0.0) + ap_weight * float(metrics.get("average_precision") or 0.0) + f05_weight * float(metrics.get("val_f05") or 0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest AutoResearch quality score weights against the experiment DB")
    parser.add_argument("--db", default="experiments/experiments.db")
    parser.add_argument("--ap-weights", default="0,0.1,0.25,0.5")
    parser.add_argument("--f05-weights", default="0,0.05,0.1,0.2")
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
    results = []
    for ap_weight in ap_weights:
        for f05_weight in f05_weights:
            ranked = sorted(rows, key=lambda row: _score(row["metrics"], ap_weight, f05_weight), reverse=True)
            top = ranked[: args.top_k]
            mean_f1 = sum(float(row["metrics"].get("val_f1") or 0.0) for row in top) / max(len(top), 1)
            results.append({"ap_weight": ap_weight, "f05_weight": f05_weight, "top_k": len(top), "mean_top_val_f1": mean_f1, "top_run_ids": [row["run_id"] for row in top]})

    print(json.dumps({"db": str(db_path), "row_count": len(rows), "results": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

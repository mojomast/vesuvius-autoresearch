from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .artifacts import list_artifact_files


def _get_nested(obj: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    cur: Any = obj
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _metric_direction(run: dict[str, Any]) -> int:
    metric = str(_get_nested(run.get("config", {}), ("evaluation", "main_metric"), "val_loss")).lower()
    return 1 if "loss" in metric else -1


def _validation_setup(run: dict[str, Any]) -> dict[str, Any]:
    cfg = run.get("config", {}) if isinstance(run.get("config"), dict) else {}
    setup = cfg.get("validation_setup") if isinstance(cfg.get("validation_setup"), dict) else {}
    resolved = cfg.get("resolved_data", {}) if isinstance(cfg.get("resolved_data"), dict) else {}
    train_meta = resolved.get("train", {}).get("metadata", {}) if isinstance(resolved.get("train"), dict) else {}
    val_meta = resolved.get("val", {}).get("metadata", {}) if isinstance(resolved.get("val"), dict) else {}
    train_segment = str(setup.get("train_segment_id") or train_meta.get("segment_id") or "?")
    val_segment = str(setup.get("val_segment_id") or val_meta.get("segment_id") or "?")
    mode = str(setup.get("mode") or _get_nested(cfg, ("dataset", "validation_mode"), "unknown"))
    if mode == "unknown" and train_segment != "?" and val_segment != "?":
        mode = "cross-segment" if train_segment != val_segment else "spatial-same-segment"
    warning = setup.get("warning")
    if not warning and mode not in {"cross-segment", "cross-scroll", "leave-one-segment-out"}:
        warning = "Validation is not cross-segment/cross-scroll/leave-one-segment-out."
    return {"mode": mode, "warning": warning, "train_segment_id": train_segment, "val_segment_id": val_segment}


def _config_diff(before: dict[str, Any] | None, after: dict[str, Any] | None, limit: int = 24) -> list[dict[str, Any]]:
    def flatten(obj: Any, prefix: str = "", out: dict[str, Any] | None = None) -> dict[str, Any]:
        out = {} if out is None else out
        if isinstance(obj, dict):
            for key, value in obj.items():
                flatten(value, f"{prefix}.{key}" if prefix else str(key), out)
        elif prefix:
            out[prefix] = obj
        return out

    left = flatten(before or {})
    right = flatten(after or {})
    diffs = []
    for key in sorted(set(left) | set(right)):
        if json.dumps(left.get(key), sort_keys=True, default=str) != json.dumps(right.get(key), sort_keys=True, default=str):
            diffs.append({"path": key, "before": left.get(key), "after": right.get(key)})
            if len(diffs) >= limit:
                break
    return diffs


def load_experiments(project_root: Path, limit: int = 500) -> dict[str, Any]:
    db_path = project_root / "experiments" / "experiments.db"
    empty = {"count": 0, "best": None, "latest": None, "recent": [], "metric_trends": [], "validation_matrix": [], "config_diffs": {"latest_vs_previous": [], "latest_vs_best": [], "latest_vs_baseline": []}, "hypotheses": []}
    if not db_path.exists():
        return empty
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            count = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
            rows = conn.execute("SELECT run_id,timestamp,config_json,main_metric,secondary_metrics_json,artifact_dir FROM experiments ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
        finally:
            conn.close()
    except Exception as exc:
        out = dict(empty)
        out["error"] = str(exc)
        return out

    def row_to_run(row: sqlite3.Row) -> dict[str, Any]:
        run = {"run_id": row["run_id"], "timestamp": row["timestamp"], "main_metric": float(row["main_metric"]), "metrics": json.loads(row["secondary_metrics_json"] or "{}"), "config": json.loads(row["config_json"] or "{}"), "artifact_dir": row["artifact_dir"]}
        run["validation_setup"] = _validation_setup(run)
        run["artifacts"] = list_artifact_files(row["artifact_dir"])
        return run

    runs = [row_to_run(row) for row in rows]
    best = min(runs, key=lambda run: _metric_direction(run) * float(run.get("main_metric", 0.0)), default=None)
    latest = runs[0] if runs else None
    previous = runs[1] if len(runs) > 1 else None
    baseline = runs[-1] if runs else None
    chronological = list(reversed(runs))[-40:]
    matrix: dict[tuple[str, str], dict[str, Any]] = {}
    for run in runs:
        setup = run.get("validation_setup", {})
        key = (str(setup.get("train_segment_id") or "?"), str(setup.get("val_segment_id") or "?"))
        prev = matrix.get(key)
        candidate = {"train_segment_id": key[0], "val_segment_id": key[1], "validation_mode": setup.get("mode"), "best_run_id": run["run_id"], "best_main_metric": run["main_metric"], "best_val_f1": run.get("metrics", {}).get("val_f1"), "run_count": 1}
        if prev is None:
            matrix[key] = candidate
        else:
            prev["run_count"] += 1
            if _metric_direction(run) * run["main_metric"] < _metric_direction(run) * prev["best_main_metric"]:
                candidate["run_count"] = prev["run_count"]
                matrix[key] = candidate
    return {
        "count": int(count),
        "best": best,
        "latest": latest,
        "recent": runs,
        "metric_trends": [{"run_id": r["run_id"], "timestamp": r["timestamp"], "main_metric": r["main_metric"], "val_f1": r.get("metrics", {}).get("val_f1"), "average_precision": r.get("metrics", {}).get("average_precision")} for r in chronological],
        "validation_matrix": sorted(matrix.values(), key=lambda item: (item["train_segment_id"], item["val_segment_id"])),
        "config_diffs": {"latest_vs_previous": _config_diff(previous.get("config") if previous else None, latest.get("config") if latest else None), "latest_vs_best": _config_diff(best.get("config") if best else None, latest.get("config") if latest else None), "latest_vs_baseline": _config_diff(baseline.get("config") if baseline else None, latest.get("config") if latest else None)},
        "hypotheses": [],
    }

from __future__ import annotations

from typing import Any


def build_progress(experiments: dict[str, Any], datasets: dict[str, Any], operations: dict[str, Any]) -> dict[str, Any]:
    recent = experiments.get("recent") or []
    latest = experiments.get("latest")
    best = experiments.get("best")
    best_metrics = best.get("metrics", {}) if isinstance(best, dict) else {}
    latest_metrics = latest.get("metrics", {}) if isinstance(latest, dict) else {}
    decision = experiments.get("decision", {}) if isinstance(experiments.get("decision"), dict) else {}
    promotion_gate = decision.get("promotion_gate", {}) if isinstance(decision.get("promotion_gate"), dict) else {}
    gate_criteria = promotion_gate.get("criteria", []) if isinstance(promotion_gate.get("criteria"), list) else []
    best_f1 = _float(best_metrics.get("val_f1"))
    latest_f1 = _float(latest_metrics.get("val_f1"))
    baseline = recent[-1] if recent else None
    baseline_f1 = _float((baseline or {}).get("metrics", {}).get("val_f1")) if isinstance(baseline, dict) else None
    gain = best_f1 - baseline_f1 if best_f1 is not None and baseline_f1 is not None else None
    mode = (latest or {}).get("validation_setup", {}).get("mode") if isinstance(latest, dict) else "unknown"
    validation_ok = mode in {"cross-segment", "cross-scroll", "leave-one-segment-out"}
    sanity_warnings = []
    pred_pos = latest_metrics.get("pred_positive_rate")
    val_pos = latest_metrics.get("val_positive_rate")
    try:
        ratio = float(pred_pos) / max(float(val_pos), 1e-12)
        if ratio > 4 or ratio < 0.25:
            sanity_warnings.append(f"prediction rate is {ratio:.1f}x validation ink rate")
    except Exception:
        pass
    milestones = [
        {"id": "data", "label": "Prepared data", "state": "done" if datasets.get("prepared") else "warning", "detail": f"{len(datasets.get('prepared') or [])} metadata records"},
        {"id": "validation", "label": "Held-out validation", "state": "done" if validation_ok else "warning", "detail": str(mode or "unknown")},
        {"id": "experiments", "label": "Experiment ledger", "state": "done" if recent else "pending", "detail": f"{experiments.get('count', 0)} runs"},
        {"id": "artifacts", "label": "Artifacts", "state": "done" if latest and latest.get("artifacts") else "warning", "detail": "latest run artifacts visible" if latest and latest.get("artifacts") else "no latest artifacts"},
        *gate_criteria,
        {"id": "lock", "label": "Run lock", "state": "active" if operations.get("lock_active") else "done", "detail": "active" if operations.get("lock_active") else "idle"},
    ]
    percent = round(max(0.0, min(1.0, (gain or 0.0) / 0.02)) * 100)
    decision_status = decision.get("status")
    status = "running" if operations.get("lock_active") else ("warning" if not validation_ok or decision_status == "blocked" or promotion_gate.get("ready") is False else "idle")
    return {
        "summary": {"status": status, "label": "AutoResearch running" if status == "running" else "Ready for review", "percent": percent, "foundation_readiness": round(100 * sum(1 for m in milestones if m["state"] == "done") / max(len(milestones), 1)), "current_step": f"Best F1 {best_f1 if best_f1 is not None else 'n/a'}; latest F1 {latest_f1 if latest_f1 is not None else 'n/a'}.", "next_action": decision.get("next_action") or "Inspect validation health and run seed-repeat LOO before promotion.", "blocker_counts": decision.get("blocker_counts", {}), "plateau": decision.get("plateau", {}), "staleness": decision.get("staleness", {})},
        "scorecard": {"metric": "val_f1", "best": best_f1, "baseline": baseline_f1, "latest": latest_f1, "gain_vs_baseline": gain, "best_run_id": best.get("run_id") if isinstance(best, dict) else None, "latest_run_id": latest.get("run_id") if isinstance(latest, dict) else None},
        "sanity": {"status": "warning" if sanity_warnings else "ok", "warnings": sanity_warnings, "pred_positive_rate": pred_pos, "val_positive_rate": val_pos},
        "milestones": milestones,
    }


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None

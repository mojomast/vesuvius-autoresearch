from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import yaml

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


def _metric_value(run: dict[str, Any]) -> float | None:
    metrics = run.get("metrics", {}) if isinstance(run.get("metrics"), dict) else {}
    metric_name = str(_get_nested(run.get("config", {}), ("evaluation", "main_metric"), "val_loss"))
    value = metrics.get(metric_name, run.get("main_metric"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _run_quality_score(run: dict[str, Any]) -> float:
    metrics = run.get("metrics", {}) if isinstance(run.get("metrics"), dict) else {}
    score = float(metrics.get("val_f1") or run.get("main_metric") or 0.0)
    score += 0.25 * float(metrics.get("average_precision") or 0.0)
    score += 0.10 * float(metrics.get("val_f05") or 0.0)
    pred_rate = metrics.get("pred_positive_rate")
    val_rate = metrics.get("val_positive_rate")
    if pred_rate is not None and val_rate is not None:
        ratio = float(pred_rate) / max(float(val_rate), 1e-6)
        if ratio > 3.0:
            score -= min(0.25, 0.03 * (ratio - 3.0))
        elif ratio < 0.25:
            score -= min(0.25, 0.03 * (0.25 / max(ratio, 1e-6)))
    score -= 0.01 * len(run.get("promotion_blockers") or [])
    return score


def _scope_priority(scope: str, scope_policy: str) -> int:
    label = f"{scope} {scope_policy}"
    if "multi_segment_robust_expanded_tta_ensemble" in label:
        return 0
    if "multi_segment_robust_expanded" in label:
        return 1
    if "expanded_multi_segment" in label or "expanded_leave_one_out" in label:
        return 2
    if "multi_segment" in label:
        return 3
    if "focused_pair_residual_25d_cpu" in label:
        return 4
    return 5


def _validation_priority(run: dict[str, Any]) -> int:
    mode = str((run.get("validation_setup") or {}).get("mode") or "")
    if mode == "cross-scroll":
        return 0
    if mode == "leave-one-segment-out":
        return 1
    if mode == "cross-segment":
        return 2
    return 3


def _is_robust_candidate(run: dict[str, Any]) -> bool:
    cfg = run.get("config", {}) if isinstance(run.get("config"), dict) else {}
    dataset = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
    autoresearch = cfg.get("autoresearch", {}) if isinstance(cfg.get("autoresearch"), dict) else {}
    model_name = str(_get_nested(cfg, ("model", "name"), ""))
    scope = str(dataset.get("research_scope") or "")
    scope_policy = str(autoresearch.get("scope_policy") or scope)
    label = f"{model_name} {scope} {scope_policy}".lower()
    return "torch" in label and any(token in label for token in ("multi_segment_robust", "expanded_multi_segment", "expanded_leave_one_out", "focused_pair_residual_25d_cpu"))


def _loo_ready(metrics: dict[str, Any]) -> bool:
    summary = metrics.get("loo_summary") if isinstance(metrics.get("loo_summary"), dict) else {}
    return bool(metrics.get("loo_promotion_ready") or summary.get("promotion_ready"))


def _path_tail(value: Any) -> str:
    return str(value or "").replace("\\", "/").lstrip("./")


def _same_path_tail(left: Any, right: Any) -> bool:
    left_tail = _path_tail(left)
    right_tail = _path_tail(right)
    return bool(left_tail and right_tail and (left_tail.endswith(right_tail) or right_tail.endswith(left_tail)))


def _normalize_loo_config(cfg: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(cfg, sort_keys=True, default=str))
    normalized.pop("resolved_data", None)
    normalized.pop("validation_setup", None)
    dataset = normalized.get("dataset") if isinstance(normalized.get("dataset"), dict) else {}
    for key in ("train_npz", "val_npz", "validation_mode"):
        dataset.pop(key, None)
    training = normalized.get("training") if isinstance(normalized.get("training"), dict) else {}
    training.pop("seed", None)
    autoresearch = normalized.get("autoresearch") if isinstance(normalized.get("autoresearch"), dict) else {}
    for key in ("heldout_segment", "seed_repeat", "fold_map"):
        autoresearch.pop(key, None)
    return normalized


def _summary_matches_run(summary: dict[str, Any], run: dict[str, Any], project_root: Path | None = None) -> bool:
    if not summary.get("promotion_ready"):
        return False
    run_id = str(run.get("run_id") or "")
    run_ids = {str(item) for item in summary.get("run_ids") or []}
    if run_id and run_id in run_ids:
        return True

    cfg = run.get("config", {}) if isinstance(run.get("config"), dict) else {}
    autoresearch = cfg.get("autoresearch", {}) if isinstance(cfg.get("autoresearch"), dict) else {}
    if summary.get("fold_map") and autoresearch.get("fold_map") and not _same_path_tail(summary.get("fold_map"), autoresearch.get("fold_map")):
        return False

    base_config = summary.get("base_config")
    if not base_config or project_root is None:
        return False
    base_path = Path(str(base_config)).expanduser()
    if not base_path.is_absolute():
        base_path = project_root / base_path
    try:
        base_cfg = yaml.safe_load(base_path.read_text()) or {}
    except Exception:
        return False
    if not isinstance(base_cfg, dict):
        return False
    return _normalize_loo_config(base_cfg) == _normalize_loo_config(cfg)


def _linked_loo_ready(run: dict[str, Any], loo_summaries: list[dict[str, Any]], project_root: Path | None = None) -> bool:
    metrics = run.get("metrics", {}) if isinstance(run.get("metrics"), dict) else {}
    if _loo_ready(metrics):
        return True
    return any(_summary_matches_run(summary, run, project_root) for summary in loo_summaries)


def _full_tile_ready(run: dict[str, Any]) -> bool:
    metrics = run.get("metrics", {}) if isinstance(run.get("metrics"), dict) else {}
    checks = metrics.get("promotion_checks") if isinstance(metrics.get("promotion_checks"), dict) else {}
    if metrics.get("full_tile_promotion_ready") or checks.get("full_tile_evidence"):
        return True
    artifact_dir = run.get("artifact_dir")
    if artifact_dir:
        for path in Path(str(artifact_dir)).glob("full_tile*/metrics.json"):
            try:
                full_tile_metrics = json.loads(path.read_text())
            except Exception:
                continue
            full_tile_checks = full_tile_metrics.get("promotion_checks") if isinstance(full_tile_metrics, dict) else {}
            if isinstance(full_tile_checks, dict) and full_tile_checks.get("eligible") is True:
                return True
    names = {str(item.get("name") or "") for item in run.get("artifacts") or [] if isinstance(item, dict)}
    return any("full_tile" in name or "probability_map" in name for name in names)


def _rel_path(path: Path, project_root: Path | None = None) -> str:
    if project_root is not None:
        try:
            return str(path.resolve().relative_to(project_root.resolve()))
        except Exception:
            pass
    return str(path)


def _linked_loo_summary(run: dict[str, Any], loo_summaries: list[dict[str, Any]], project_root: Path | None = None) -> dict[str, Any] | None:
    for summary in loo_summaries:
        if _summary_matches_run(summary, run, project_root):
            return summary
    return None


def _full_tile_metrics(run: dict[str, Any], project_root: Path | None = None) -> list[dict[str, Any]]:
    artifact_dir = run.get("artifact_dir")
    if not artifact_dir:
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(Path(str(artifact_dir)).glob("full_tile*/metrics.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        checks = data.get("promotion_checks") if isinstance(data.get("promotion_checks"), dict) else {}
        region = data.get("evaluation_region") if isinstance(data.get("evaluation_region"), dict) else {}
        segment_id = str(region.get("segment_id") or checks.get("inference_segment_id") or "")
        out.append({
            "path": str(path),
            "relative_path": _rel_path(path, project_root),
            "segment_id": segment_id or None,
            "eligible": checks.get("eligible"),
            "warnings": checks.get("warnings", []),
            "evaluation_region_type": region.get("type"),
            "val_f1": data.get("val_f1", data.get("tile_f1")),
            "average_precision": data.get("average_precision"),
            "best_threshold": data.get("best_threshold"),
            "pred_positive_rate": data.get("pred_positive_rate"),
            "val_positive_rate": data.get("val_positive_rate"),
        })
    return out


def _candidate_evidence(run: dict[str, Any] | None, loo_summaries: list[dict[str, Any]], project_root: Path | None = None) -> dict[str, Any]:
    if not isinstance(run, dict):
        return {"candidate_run_id": None, "promotion_actions": []}
    summary = _linked_loo_summary(run, loo_summaries, project_root)
    full_tiles = _full_tile_metrics(run, project_root)
    artifact_dir = Path(str(run.get("artifact_dir") or ""))
    weak_fold_id = str((summary or {}).get("worst_fold_id") or "")
    weak_tile = next((item for item in full_tiles if item.get("segment_id") == weak_fold_id), None) if weak_fold_id else None
    weak_status = "not_applicable"
    if weak_fold_id:
        if not weak_tile:
            weak_status = "missing"
        elif weak_tile.get("evaluation_region_type") == "whole_segment" and weak_tile.get("eligible") is True:
            weak_status = "done"
        else:
            weak_status = "warning"

    command: list[str] | None = None
    command_text = None
    if weak_fold_id and artifact_dir:
        artifact_rel = _rel_path(artifact_dir, project_root)
        output_rel = f"{artifact_rel}/full_tile_weak_fold_{weak_fold_id}"
        command = [
            ".venv/bin/python", "scripts/infer_full_tile.py",
            "--artifact", artifact_rel,
            "--segment-id", weak_fold_id,
            "--output-dir", output_rel,
            "--catalog-source", "public-directory",
            "--level", "1",
            "--z-offsets=-4,0,4",
            "--patch-size", "64",
            "--stride", "32",
            "--batch-size", "8",
            "--device", "cpu",
            "--public-retry-count", "5",
            "--public-retry-delay-sec", "180",
            "--public-chunk-delay-sec", "0.5",
            "--public-chunk-retry-count", "5",
            "--public-chunk-retry-delay-sec", "180",
        ]
        command_text = " ".join(command)

    actions: list[dict[str, Any]] = []
    if not summary:
        actions.append({"id": "seed_repeat_loo", "label": "Run linked seed-repeat LOO", "kind": "validation", "writes_artifacts": True})
    elif weak_fold_id and weak_status != "done":
        actions.append({"id": "weak_fold_full_tile", "label": f"Run full-tile on weak fold {weak_fold_id}", "kind": "inference", "command": command, "command_text": command_text, "writes_artifacts": True, "safe_to_execute_from_dashboard": False})
    if not full_tiles:
        actions.append({"id": "full_tile_candidate", "label": "Run candidate full-tile evidence", "kind": "inference", "writes_artifacts": True, "safe_to_execute_from_dashboard": False})
    if run.get("promotion_status") == "eligible":
        actions.append({"id": "promotion_review", "label": f"Review promotion candidate {run.get('run_id')}", "kind": "review", "writes_artifacts": False})

    return {
        "candidate_run_id": run.get("run_id"),
        "candidate_artifact_dir": run.get("artifact_dir"),
        "loo": {
            "ready": bool(summary and summary.get("promotion_ready")),
            "summary_path": summary.get("path") if summary else None,
            "seeds": summary.get("seeds") if summary else None,
            "worst_fold_id": weak_fold_id or None,
            "worst_fold_val_f1": summary.get("worst_fold_val_f1") if summary else None,
            "worst_fold_average_precision": (summary.get("per_fold_average_precision") or {}).get(weak_fold_id) if summary and weak_fold_id else None,
            "median_over_seeds_median_val_f1": summary.get("median_over_seeds_median_val_f1") if summary else None,
            "warnings": summary.get("warnings", []) if summary else [],
        },
        "full_tile": {
            "segments_covered": [item.get("segment_id") for item in full_tiles if item.get("segment_id")],
            "evidence": full_tiles,
            "coverage_count": len(full_tiles),
        },
        "weak_fold_full_tile": {
            "weak_fold_id": weak_fold_id or None,
            "status": weak_status,
            "metrics": weak_tile,
            "command": command,
            "command_text": command_text,
            "safe_to_execute_from_dashboard": False,
            "writes_artifacts": True,
        },
        "promotion_actions": actions,
    }


def _promotion_blockers(run: dict[str, Any], loo_summaries: list[dict[str, Any]] | None = None, project_root: Path | None = None) -> list[dict[str, str]]:
    cfg = run.get("config", {}) if isinstance(run.get("config"), dict) else {}
    metrics = run.get("metrics", {}) if isinstance(run.get("metrics"), dict) else {}
    blockers: list[dict[str, str]] = []

    def add(code: str, category: str, severity: str = "blocker") -> None:
        blockers.append({"code": code, "category": category, "severity": severity})

    if metrics.get("val_f1") is None:
        add("missing_val_f1", "metrics")
    if metrics.get("average_precision") is None:
        add("missing_average_precision", "metrics")
    if float(metrics.get("precision") or 0.0) <= 0.0 or float(metrics.get("recall") or 0.0) <= 0.0:
        add("zero_precision_or_recall", "calibration")
    pred_rate = metrics.get("pred_positive_rate")
    val_rate = metrics.get("val_positive_rate")
    if pred_rate is not None and val_rate is not None:
        ratio = float(pred_rate) / max(float(val_rate), 1e-6)
        if ratio > 4.0 or ratio < 0.1:
            add("pred_positive_rate_ratio_suspicious", "calibration")
    checks = metrics.get("promotion_checks") or {}
    if isinstance(checks, dict) and checks.get("eligible") is False:
        add("promotion_checks_ineligible", "promotion_checks")
    if cfg.get("autoresearch", {}).get("cron_safety"):
        add("cron_safety_run_not_promotable", "policy")
    scope = str(cfg.get("dataset", {}).get("research_scope") or cfg.get("autoresearch", {}).get("scope_policy") or "")
    if "multi_segment" not in scope and "leave_one_out" not in scope:
        add("not_multisegment_scope", "validation")
    mode = str(run.get("validation_setup", {}).get("mode") or "unknown")
    if mode not in {"cross-segment", "cross-scroll", "leave-one-segment-out"}:
        add("validation_not_held_out", "validation")
    if _is_robust_candidate(run):
        if not _linked_loo_ready(run, loo_summaries or [], project_root):
            add("missing_seed_repeat_loo", "validation")
        if not _full_tile_ready(run):
            add("missing_full_tile_evidence", "inference")
    best_threshold = metrics.get("best_threshold")
    if best_threshold is not None:
        threshold = float(best_threshold)
        if threshold <= 0.03 or threshold >= 0.94:
            add("best_threshold_at_sweep_edge", "calibration", "warning")
    if float(metrics.get("val_f1") or 0.0) >= 0.2 and float(metrics.get("fixed_threshold_f1") or metrics.get("val_f1") or 0.0) < 0.05:
        add("fixed_threshold_f1_low", "calibration", "warning")
    return blockers


def _compact_run(run: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(run, dict):
        return None
    return {
        "run_id": run.get("run_id"),
        "timestamp": run.get("timestamp"),
        "main_metric": run.get("main_metric"),
        "metrics": run.get("metrics", {}),
        "validation_setup": run.get("validation_setup", {}),
        "promotion_status": run.get("promotion_status"),
        "promotion_blockers": run.get("promotion_blockers", []),
        "champion_class": run.get("champion_class"),
        "champion_classes": run.get("champion_classes", []),
    }


def _promotion_gate(runs: list[dict[str, Any]], loo_summaries: list[dict[str, Any]], candidate_run: dict[str, Any] | None = None, project_root: Path | None = None) -> dict[str, Any]:
    robust_runs = [run for run in runs if _is_robust_candidate(run)]
    target = candidate_run if candidate_run and _is_robust_candidate(candidate_run) else (robust_runs[0] if robust_runs else None)
    target_id = target.get("run_id") if isinstance(target, dict) else None
    has_loo = bool(target and _linked_loo_ready(target, loo_summaries, project_root))
    has_full_tile = bool(target and _full_tile_ready(target))
    has_heldout = bool(target and (target.get("validation_setup", {}) or {}).get("mode") in {"cross-segment", "cross-scroll", "leave-one-segment-out"})
    no_blocked_promotable = bool(target and target.get("promotion_status") == "eligible")
    target_detail = f" for candidate {target_id}" if target_id else ""
    criteria = [
        {"id": "robust_candidate", "label": "Robust candidate", "state": "done" if robust_runs else "pending", "detail": f"{len(robust_runs)} robust-scope runs"},
        {"id": "heldout_validation", "label": "Held-out validation", "state": "done" if has_heldout else "warning", "detail": "cross-segment/scroll or LOO" if has_heldout else "missing robust held-out run"},
        {"id": "seed_repeat_loo", "label": "Seed-repeat LOO", "state": "done" if has_loo else "warning", "detail": f"promotion-ready summary linked{target_detail}" if has_loo else f"run evaluate_leave_one_out.py --seeds for candidate {target_id or '?'}"},
        {"id": "full_tile", "label": "Full-tile evidence", "state": "done" if has_full_tile else "warning", "detail": f"full-tile artifact/check linked{target_detail}" if has_full_tile else f"run infer_full_tile.py for candidate {target_id or '?'} before promotion"},
        {"id": "promotion_clear", "label": "No promotion blockers", "state": "done" if no_blocked_promotable else "warning", "detail": f"candidate {target_id} is eligible" if no_blocked_promotable else "blockers remain"},
    ]
    return {"criteria": criteria, "ready": all(item["state"] == "done" for item in criteria)}


def _decision_snapshot(runs: list[dict[str, Any]], peak: dict[str, Any] | None, robust: dict[str, Any] | None, promotable: dict[str, Any] | None, loo_summaries: list[dict[str, Any]] | None = None, project_root: Path | None = None) -> dict[str, Any]:
    blocker_counts: dict[str, int] = {}
    for run in runs:
        for blocker in run.get("promotion_blockers") or []:
            code = str(blocker.get("code") or "unknown")
            blocker_counts[code] = blocker_counts.get(code, 0) + 1

    chronological = list(reversed(runs))
    peak_index = next((idx for idx, run in enumerate(chronological) if peak and run.get("run_id") == peak.get("run_id")), None)
    stale_runs_since_peak = (len(chronological) - peak_index - 1) if peak_index is not None else None
    window = chronological[-5:]
    values = [_metric_value(run) for run in window]
    values = [value for value in values if value is not None]
    plateau_delta = (max(values) - min(values)) if len(values) >= 2 else None
    plateau_window = len(values)
    plateau_epsilon = 0.002
    plateau_detected = plateau_delta is not None and plateau_window >= 5 and plateau_delta <= plateau_epsilon

    target = promotable or robust
    gate = _promotion_gate(runs, loo_summaries or [], target, project_root)
    evidence = _candidate_evidence(target, loo_summaries or [], project_root)

    if promotable:
        next_action = f"Promote or seed-repeat verify {promotable.get('run_id')} before release."
        status = "promotion_eligible"
    elif robust:
        top = robust.get("promotion_blockers") or []
        suffix = f"; top blocker: {top[0].get('code')}" if top else ""
        if any(item["id"] == "seed_repeat_loo" and item["state"] != "done" for item in gate["criteria"]):
            next_action = f"Run seed-repeat leave-one-out for robust candidate {robust.get('run_id')}{suffix}."
        elif any(item["id"] == "full_tile" and item["state"] != "done" for item in gate["criteria"]):
            next_action = f"Run full-tile inference for robust candidate {robust.get('run_id')}{suffix}."
        else:
            next_action = f"Resolve blockers for robust candidate {robust.get('run_id')}{suffix}."
        status = "blocked"
    elif peak:
        next_action = f"Convert peak run {peak.get('run_id')} into a robust held-out candidate."
        status = "needs_robust_candidate"
    else:
        next_action = "Run a held-out validation experiment before promotion review."
        status = "no_runs"

    actions = evidence.get("promotion_actions", []) if isinstance(evidence.get("promotion_actions"), list) else []
    top_action = actions[0] if actions and isinstance(actions[0], dict) else {}
    if top_action and top_action.get("id") != "promotion_review":
        next_action = str(top_action.get("label") or next_action)
        if top_action.get("id") == "weak_fold_full_tile" and "before promotion review" not in next_action:
            next_action = f"{next_action} before promotion review."

    return {
        "status": status,
        "next_action": next_action,
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "promotion_gate": gate,
        "candidate_evidence": evidence,
        "promotion_actions": actions,
        "plateau": {"detected": plateau_detected, "window_runs": plateau_window, "metric_delta": plateau_delta, "epsilon": plateau_epsilon},
        "staleness": {"stale_runs_since_peak": stale_runs_since_peak, "peak_run_id": peak.get("run_id") if peak else None},
    }


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


def _load_loo_summaries(project_root: Path, limit: int = 8) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted((project_root / "logs").glob("*summary.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict) or "promotion_ready" not in data:
            continue
        summaries.append({"path": str(path), "promotion_ready": bool(data.get("promotion_ready")), "warnings": data.get("promotion_warnings", []), "median_over_seeds_median_val_f1": data.get("median_over_seeds_median_val_f1"), "worst_fold_id": data.get("worst_fold_id"), "worst_fold_val_f1": data.get("worst_fold_val_f1"), "mean_average_precision": data.get("mean_average_precision"), "per_fold_val_f1": data.get("per_fold_val_f1"), "per_fold_average_precision": data.get("per_fold_average_precision"), "base_config": data.get("base_config"), "fold_map": data.get("fold_map"), "seeds": data.get("seeds"), "min_seeds_for_promotion": data.get("min_seeds_for_promotion"), "distinct_successful_seeds": data.get("distinct_successful_seeds"), "run_ids": data.get("run_ids")})
        if len(summaries) >= limit:
            break
    return summaries


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
    loo_summaries = _load_loo_summaries(project_root)
    empty = {"count": 0, "best": None, "latest": None, "recent": [], "champions": {"peak_score": None, "robust_candidate": None, "promotion_eligible": None}, "decision": _decision_snapshot([], None, None, None, loo_summaries, project_root), "loo_summaries": loo_summaries, "metric_trends": [], "validation_matrix": [], "config_diffs": {"latest_vs_previous": [], "latest_vs_best": [], "latest_vs_baseline": []}, "hypotheses": []}
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
        run["promotion_blockers"] = _promotion_blockers(run, loo_summaries, project_root)
        run["promotion_status"] = "eligible" if not run["promotion_blockers"] else "blocked"
        run["champion_class"] = None
        run["champion_classes"] = []
        return run

    runs = [row_to_run(row) for row in rows]
    best = min(runs, key=lambda run: _metric_direction(run) * float(run.get("main_metric", 0.0)), default=None)
    robust_candidates = [run for run in runs if _is_robust_candidate(run)]
    robust = min(
        robust_candidates,
        key=lambda run: (_validation_priority(run), _scope_priority(str(_get_nested(run.get("config", {}), ("dataset", "research_scope"), "")), str(_get_nested(run.get("config", {}), ("autoresearch", "scope_policy"), ""))), _metric_direction(run) * float(run.get("main_metric", 0.0))),
        default=None,
    )
    eligible_runs = [run for run in runs if run.get("promotion_status") == "eligible"]
    promotable = max(eligible_runs, key=_run_quality_score, default=None)
    for label, run in (("peak_score", best), ("robust_candidate", robust), ("promotion_eligible", promotable)):
        if isinstance(run, dict):
            run.setdefault("champion_classes", []).append(label)
    for run in runs:
        classes = run.get("champion_classes") or []
        run["champion_class"] = ",".join(classes) if classes else None
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
        "champions": {"peak_score": _compact_run(best), "robust_candidate": _compact_run(robust), "promotion_eligible": _compact_run(promotable)},
        "decision": _decision_snapshot(runs, best, robust, promotable, loo_summaries, project_root),
        "loo_summaries": loo_summaries,
        "metric_trends": [{"run_id": r["run_id"], "timestamp": r["timestamp"], "main_metric": r["main_metric"], "val_f1": r.get("metrics", {}).get("val_f1"), "average_precision": r.get("metrics", {}).get("average_precision")} for r in chronological],
        "validation_matrix": sorted(matrix.values(), key=lambda item: (item["train_segment_id"], item["val_segment_id"])),
        "config_diffs": {"latest_vs_previous": _config_diff(previous.get("config") if previous else None, latest.get("config") if latest else None), "latest_vs_best": _config_diff(best.get("config") if best else None, latest.get("config") if latest else None), "latest_vs_baseline": _config_diff(baseline.get("config") if baseline else None, latest.get("config") if latest else None)},
        "hypotheses": [],
    }

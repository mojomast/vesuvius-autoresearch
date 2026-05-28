from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import yaml

from .artifacts import list_artifact_files
from .quality import metrics_quality_verdict, quality_next_actions


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


def _eligible_full_tile_metrics(metrics: dict[str, Any]) -> bool:
    checks = metrics.get("promotion_checks") if isinstance(metrics.get("promotion_checks"), dict) else {}
    region = metrics.get("evaluation_region") if isinstance(metrics.get("evaluation_region"), dict) else {}
    return checks.get("eligible") is True and region.get("type") == "whole_segment"


def _path_tail(value: Any) -> str:
    return str(value or "").replace("\\", "/").lstrip("./")


def _same_path_tail(left: Any, right: Any) -> bool:
    left_tail = _path_tail(left)
    right_tail = _path_tail(right)
    return bool(left_tail and right_tail and (left_tail.endswith(right_tail) or right_tail.endswith(left_tail)))


def _normalized_segments(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        raw = value
    elif value:
        raw = [value]
    else:
        raw = []
    return sorted({str(segment) for segment in raw if segment is not None and str(segment) not in {"", "?"}})


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
    if metrics.get("full_tile_promotion_ready") or _eligible_full_tile_metrics(metrics):
        return True
    artifact_dir = run.get("artifact_dir")
    if artifact_dir:
        for path in Path(str(artifact_dir)).glob("full_tile*/metrics.json"):
            try:
                full_tile_metrics = json.loads(path.read_text())
            except Exception:
                continue
            if isinstance(full_tile_metrics, dict) and _eligible_full_tile_metrics(full_tile_metrics):
                return True
    return False


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
        quality = metrics_quality_verdict(data)
        actions = quality_next_actions(quality, target="full_tile", segment_id=segment_id or None)
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
            "brier_score": data.get("brier_score"),
            "expected_calibration_error": data.get("expected_calibration_error"),
            "ap_prevalence_lift": data.get("ap_prevalence_lift"),
            "prob_mean": data.get("prob_mean"),
            "prob_p95": data.get("prob_p95"),
            "prob_max": data.get("prob_max"),
            "fixed_threshold_f1": data.get("fixed_threshold_f1"),
            "fixed_threshold_status": data.get("fixed_threshold_status"),
            "fixed_threshold_failure_reason": data.get("fixed_threshold_failure_reason"),
            "threshold_selection": data.get("threshold_selection"),
            "selected_threshold_reason": data.get("selected_threshold_reason"),
            "threshold_risk_summary": data.get("threshold_risk_summary"),
            "quality_verdict": quality,
            "quality_score": quality.get("score"),
            "quality_reasons": quality.get("reasons", []),
            "quality_next_actions": actions,
            "quality_next_action": actions[0].get("label") if actions else None,
        })
    out.sort(key=lambda item: (str(item.get("segment_id") or ""), 0 if item.get("threshold_risk_summary") else 1, str(item.get("relative_path") or item.get("path") or "")))
    return out


def _quality_blocks_promotion(item: dict[str, Any]) -> bool:
    fixed_threshold_status = str(item.get("fixed_threshold_status") or "").lower()
    if fixed_threshold_status and fixed_threshold_status != "ok":
        return True
    quality = item.get("quality_verdict") if isinstance(item.get("quality_verdict"), dict) else {}
    has_quality_metrics = any(item.get(key) is not None for key in ("val_f1", "average_precision", "pred_positive_rate", "val_positive_rate", "fixed_threshold_f1"))
    return quality.get("verdict") == "fail" and has_quality_metrics


def _quality_promotion_actions(full_tiles: list[dict[str, Any]], loo_full_tiles: dict[str, Any]) -> list[dict[str, Any]]:
    items = list(full_tiles)
    if isinstance(loo_full_tiles, dict):
        items.extend(item for item in loo_full_tiles.get("evidence", []) if isinstance(item, dict))
    actions: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()
    for verdict in ("fail", "review"):
        for item in items:
            quality = item.get("quality_verdict") if isinstance(item.get("quality_verdict"), dict) else {}
            if quality.get("verdict") != verdict:
                continue
            segment = item.get("segment_id") or item.get("heldout_segment")
            action_items = quality_next_actions(quality, target="candidate_full_tile", segment_id=segment)
            if not action_items:
                continue
            for action in action_items:
                key = (str(action.get("id")), str(segment) if segment is not None else None)
                if key in seen:
                    continue
                seen.add(key)
                label = str(action.get("label") or "Review quality evidence")
                if segment:
                    label = f"{label} for segment {segment}"
                actions.append({**action, "label": label, "quality_verdict": verdict, "segment_id": segment, "writes_artifacts": False})
    return actions


def _weak_fold_loo_run(summary: dict[str, Any] | None, weak_fold_id: str) -> dict[str, Any] | None:
    if not summary or not weak_fold_id:
        return None
    rows = [row for row in summary.get("rows", []) if isinstance(row, dict) and str(row.get("heldout_segment")) == weak_fold_id and row.get("artifact_dir")]
    if not rows:
        return None
    target = summary.get("worst_fold_val_f1")
    try:
        target_f1 = float(target)
    except (TypeError, ValueError):
        target_f1 = None
    if target_f1 is None:
        return rows[0]
    return min(rows, key=lambda row: abs(float(row.get("val_f1") or 0.0) - target_f1))


def _loo_full_tile_diagnostics(summary: dict[str, Any] | None, project_root: Path | None = None) -> dict[str, Any]:
    if not summary:
        return {"coverage_count": 0, "segments_covered": [], "evidence": []}
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str]] = set()
    for row in summary.get("rows", []):
        if not isinstance(row, dict) or not row.get("artifact_dir") or not row.get("heldout_segment"):
            continue
        heldout = str(row.get("heldout_segment"))
        for item in _full_tile_metrics({"artifact_dir": row.get("artifact_dir")}, project_root):
            if item.get("segment_id") != heldout:
                continue
            key = (item.get("segment_id"), item.get("path"))
            if key in seen:
                continue
            seen.add(key)
            evidence.append({
                **item,
                "heldout_segment": heldout,
                "loo_run_id": row.get("run_id"),
                "seed": row.get("seed"),
                "artifact_dir": row.get("artifact_dir"),
                "loo_val_f1": row.get("val_f1"),
                "loo_average_precision": row.get("average_precision"),
                "loo_best_threshold": row.get("best_threshold"),
                "loo_pred_positive_rate": row.get("pred_positive_rate"),
                "loo_val_positive_rate": row.get("val_positive_rate"),
                "loo_brier_score": row.get("brier_score"),
                "loo_expected_calibration_error": row.get("expected_calibration_error"),
                "loo_ap_prevalence_lift": row.get("ap_prevalence_lift"),
                "loo_fixed_threshold_f1": row.get("fixed_threshold_f1"),
                "loo_fixed_threshold_status": row.get("fixed_threshold_status"),
                "loo_threshold_selection": row.get("threshold_selection"),
            })
    evidence.sort(key=lambda item: str(item.get("segment_id") or ""))
    return {
        "coverage_count": len(evidence),
        "segments_covered": sorted({str(item.get("segment_id")) for item in evidence if item.get("segment_id")}),
        "evidence": evidence,
    }


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    left = _as_float(numerator)
    right = _as_float(denominator)
    if left is None or right is None or abs(right) < 1e-9:
        return None
    ratio = left / right
    return ratio if math.isfinite(ratio) else None


def _safe_metric(value: Any) -> float | None:
    return _as_float(value)


def _positive_rate_risk_summary(run: dict[str, Any], full_tiles: list[dict[str, Any]], loo_full_tiles: dict[str, Any]) -> dict[str, Any]:
    metrics = run.get("metrics", {}) if isinstance(run.get("metrics"), dict) else {}
    candidate_ratio = _safe_ratio(metrics.get("pred_positive_rate"), metrics.get("val_positive_rate"))
    full_tile_ratios: list[dict[str, Any]] = []
    for item in full_tiles:
        ratio = _safe_ratio(item.get("pred_positive_rate"), item.get("val_positive_rate"))
        if ratio is not None:
            full_tile_ratios.append({"segment_id": item.get("segment_id"), "pred_to_val_ratio": ratio, "pred_positive_rate": _safe_metric(item.get("pred_positive_rate")), "val_positive_rate": _safe_metric(item.get("val_positive_rate"))})

    sampled_vs_full_tile: list[dict[str, Any]] = []
    loo_tile_evidence = loo_full_tiles.get("evidence", []) if isinstance(loo_full_tiles, dict) else []
    for item in loo_tile_evidence:
        pred_rate_ratio = _safe_ratio(item.get("loo_pred_positive_rate"), item.get("pred_positive_rate"))
        sampled_ratio = _safe_ratio(item.get("loo_pred_positive_rate"), item.get("loo_val_positive_rate"))
        full_ratio = _safe_ratio(item.get("pred_positive_rate"), item.get("val_positive_rate"))
        full_f1 = _as_float(item.get("val_f1"))
        sampled_f1 = _as_float(item.get("loo_val_f1"))
        full_ap = _as_float(item.get("average_precision"))
        sampled_ap = _as_float(item.get("loo_average_precision"))
        sampled_vs_full_tile.append({
            "segment_id": item.get("segment_id"),
            "seed": item.get("seed"),
            "sampled_pred_positive_rate": _safe_metric(item.get("loo_pred_positive_rate")),
            "full_tile_pred_positive_rate": _safe_metric(item.get("pred_positive_rate")),
            "sampled_to_full_pred_positive_rate_ratio": pred_rate_ratio,
            "sampled_pred_to_val_ratio": sampled_ratio,
            "full_tile_pred_to_val_ratio": full_ratio,
            "sampled_best_threshold": _safe_metric(item.get("loo_best_threshold")),
            "full_tile_best_threshold": _safe_metric(item.get("best_threshold")),
            "val_f1_delta_full_minus_sampled": (full_f1 - sampled_f1) if full_f1 is not None and sampled_f1 is not None else None,
            "average_precision_delta_full_minus_sampled": (full_ap - sampled_ap) if full_ap is not None and sampled_ap is not None else None,
        })

    warnings: list[str] = []
    if candidate_ratio is not None and candidate_ratio >= 3.5:
        warnings.append(f"candidate sampled pred/val positive-rate ratio is high ({candidate_ratio:.2f}x)")
    high_full = [item for item in full_tile_ratios if float(item["pred_to_val_ratio"]) >= 3.5]
    if high_full:
        warnings.append(f"{len(high_full)} candidate full-tile segment(s) have pred/val positive-rate ratio >= 3.5x")
    loo_high = [item for item in sampled_vs_full_tile if item.get("full_tile_pred_to_val_ratio") is not None and float(item["full_tile_pred_to_val_ratio"]) >= 3.5]
    if loo_high:
        warnings.append(f"{len(loo_high)} LOO full-tile segment(s) have pred/val positive-rate ratio >= 3.5x")
    divergent = [item for item in sampled_vs_full_tile if item.get("sampled_to_full_pred_positive_rate_ratio") is not None and (float(item["sampled_to_full_pred_positive_rate_ratio"]) < 0.5 or float(item["sampled_to_full_pred_positive_rate_ratio"]) > 2.0)]
    if divergent:
        warnings.append(f"{len(divergent)} LOO segment(s) have sampled/full-tile predicted positive rates outside 0.5x-2.0x")

    return {
        "risk_level": "warning" if warnings else "ok",
        "warnings": warnings,
        "candidate_pred_to_val_ratio": candidate_ratio,
        "candidate_full_tile_pred_to_val_ratios": full_tile_ratios,
        "loo_sampled_vs_full_tile": sampled_vs_full_tile,
    }


def _candidate_evidence(run: dict[str, Any] | None, loo_summaries: list[dict[str, Any]], project_root: Path | None = None) -> dict[str, Any]:
    if not isinstance(run, dict):
        return {"candidate_run_id": None, "promotion_actions": []}
    config = run.get("config", {}) if isinstance(run.get("config"), dict) else {}
    summary = _linked_loo_summary(run, loo_summaries, project_root)
    full_tiles = _full_tile_metrics(run, project_root)
    loo_full_tiles = _loo_full_tile_diagnostics(summary, project_root)
    full_tile_quality_actions = _quality_promotion_actions(full_tiles, {})
    artifact_dir = Path(str(run.get("artifact_dir") or ""))
    weak_fold_id = str((summary or {}).get("worst_fold_id") or "")
    weak_loo_run = _weak_fold_loo_run(summary, weak_fold_id)
    weak_artifact_dir = Path(str((weak_loo_run or {}).get("artifact_dir") or artifact_dir))
    weak_run = {"artifact_dir": str(weak_artifact_dir)} if weak_artifact_dir else run
    weak_tiles = _full_tile_metrics(weak_run, project_root) if weak_fold_id else []
    weak_tile = next((item for item in weak_tiles if item.get("segment_id") == weak_fold_id), None) if weak_fold_id else None
    if weak_fold_id and not weak_tile:
        weak_tile = next((item for item in full_tiles if item.get("segment_id") == weak_fold_id), None)
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
    if weak_fold_id and weak_artifact_dir:
        artifact_rel = _rel_path(weak_artifact_dir, project_root)
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
    actions.extend(_quality_promotion_actions(full_tiles, loo_full_tiles))
    if run.get("promotion_status") == "eligible":
        actions.append({"id": "promotion_review", "label": f"Review promotion candidate {run.get('run_id')}", "kind": "review", "writes_artifacts": False})

    return {
        "candidate_run_id": run.get("run_id"),
        "candidate_artifact_dir": run.get("artifact_dir"),
        "research_scope": _get_nested(config, ("dataset", "research_scope"), None),
        "scope_policy": _get_nested(config, ("autoresearch", "scope_policy"), None),
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
            "quality_next_actions": full_tile_quality_actions,
            "quality_next_action": full_tile_quality_actions[0].get("label") if full_tile_quality_actions else None,
        },
        "loo_full_tile": loo_full_tiles,
        "risk_summary": _positive_rate_risk_summary(run, full_tiles, loo_full_tiles),
        "weak_fold_full_tile": {
            "weak_fold_id": weak_fold_id or None,
            "status": weak_status,
            "metrics": weak_tile,
            "loo_run": weak_loo_run,
            "artifact_dir": str(weak_artifact_dir) if weak_artifact_dir else None,
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
        if ratio > 3.5 or ratio < 0.1:
            add("pred_positive_rate_ratio_suspicious", "calibration")
    fixed_threshold_status = str(metrics.get("fixed_threshold_status") or "").lower()
    if fixed_threshold_status and fixed_threshold_status != "ok":
        add("fixed_threshold_status_weak", "calibration")
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
    setup = run.get("validation_setup", {}) if isinstance(run.get("validation_setup"), dict) else {}
    train_segment = str(setup.get("train_segment_id") or "")
    val_segment = str(setup.get("val_segment_id") or "")
    train_segments = set(_normalized_segments(setup.get("train_segments")))
    if val_segment and val_segment != "?" and ((train_segment and train_segment == val_segment) or val_segment in train_segments):
        add("validation_segment_leakage", "validation")
    if _is_robust_candidate(run):
        if not _linked_loo_ready(run, loo_summaries or [], project_root):
            add("missing_seed_repeat_loo", "validation")
        if not _full_tile_ready(run):
            add("missing_full_tile_evidence", "inference")
        elif any(_quality_blocks_promotion(item) for item in _full_tile_metrics(run, project_root)):
            add("full_tile_quality_fail", "quality")
        summary = _linked_loo_summary(run, loo_summaries or [], project_root)
        if any(_quality_blocks_promotion(item) for item in _loo_full_tile_diagnostics(summary, project_root).get("evidence", [])):
            add("loo_full_tile_quality_fail", "quality")
    best_threshold = metrics.get("best_threshold")
    if best_threshold is not None:
        threshold = float(best_threshold)
        if threshold <= 0.03 or threshold >= 0.94:
            add("best_threshold_at_sweep_edge", "calibration", "warning")
    fixed_f1 = metrics.get("fixed_threshold_f1")
    if fixed_f1 is None:
        fixed_f1 = metrics.get("val_f1") or 0.0
    if float(metrics.get("val_f1") or 0.0) >= 0.2 and float(fixed_f1) < 0.05:
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
    if top_action:
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
    train_segments = _normalized_segments(setup.get("train_segments") or train_meta.get("train_segments"))
    val_segment = str(setup.get("val_segment_id") or val_meta.get("segment_id") or "?")
    mode = str(setup.get("mode") or _get_nested(cfg, ("dataset", "validation_mode"), "unknown"))
    if mode == "unknown" and train_segment != "?" and val_segment != "?":
        mode = "cross-segment" if train_segment != val_segment else "spatial-same-segment"
    warning = setup.get("warning")
    if not warning and val_segment != "?" and (train_segment == val_segment or val_segment in set(train_segments)):
        warning = "Validation segment overlaps training segments; this is not held-out validation."
    if not warning and mode not in {"cross-segment", "cross-scroll", "leave-one-segment-out"}:
        warning = "Validation is not cross-segment/cross-scroll/leave-one-segment-out."
    return {"mode": mode, "warning": warning, "train_segment_id": train_segment, "train_segments": train_segments, "val_segment_id": val_segment}


def _load_loo_summaries(project_root: Path, limit: int = 8) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted((project_root / "logs").glob("*summary.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict) or "promotion_ready" not in data:
            continue
        rows = []
        jsonl_path = Path(str(path)[:-len(".summary.json")] + ".jsonl") if path.name.endswith(".summary.json") else path.with_suffix(".jsonl")
        if jsonl_path.exists():
            try:
                for line in jsonl_path.read_text().splitlines():
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows.append({key: row.get(key) for key in ("run_id", "artifact_dir", "heldout_segment", "seed", "val_f1", "average_precision", "best_threshold", "pred_positive_rate", "val_positive_rate", "brier_score", "expected_calibration_error", "ap_prevalence_lift", "prob_mean", "prob_p95", "prob_max", "fixed_threshold", "fixed_threshold_f1", "fixed_threshold_status", "threshold_selection", "selected_threshold_reason", "returncode")})
            except Exception:
                rows = []
        summaries.append({"path": str(path), "promotion_ready": bool(data.get("promotion_ready")), "warnings": data.get("promotion_warnings", []), "median_over_seeds_median_val_f1": data.get("median_over_seeds_median_val_f1"), "worst_fold_id": data.get("worst_fold_id"), "worst_fold_val_f1": data.get("worst_fold_val_f1"), "mean_average_precision": data.get("mean_average_precision"), "per_fold_val_f1": data.get("per_fold_val_f1"), "per_fold_average_precision": data.get("per_fold_average_precision"), "base_config": data.get("base_config"), "fold_map": data.get("fold_map"), "seeds": data.get("seeds"), "min_seeds_for_promotion": data.get("min_seeds_for_promotion"), "distinct_successful_seeds": data.get("distinct_successful_seeds"), "run_ids": data.get("run_ids"), "rows": rows})
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


def _leaderboard_rows(runs: list[dict[str, Any]], project_root: Path | None = None, limit: int = 20) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        config = run.get("config", {}) if isinstance(run.get("config"), dict) else {}
        metrics = run.get("metrics", {}) if isinstance(run.get("metrics"), dict) else {}
        setup = run.get("validation_setup", {}) if isinstance(run.get("validation_setup"), dict) else {}
        full_tiles = _full_tile_metrics(run, project_root)
        best_tile = max(full_tiles, key=lambda item: float(item.get("quality_score") or 0.0), default=None)
        metric_quality = metrics_quality_verdict(metrics)
        tile_quality = best_tile.get("quality_verdict") if isinstance(best_tile, dict) else None
        quality = tile_quality if isinstance(tile_quality, dict) else metric_quality
        ratio = _safe_ratio(metrics.get("pred_positive_rate"), metrics.get("val_positive_rate"))
        quality_actions = quality_next_actions(quality, target="leaderboard", run_id=str(run.get("run_id") or ""), segment_id=setup.get("val_segment_id")) if isinstance(quality, dict) else []
        rows.append({
            "run_id": run.get("run_id"),
            "timestamp": run.get("timestamp"),
            "model_name": metrics.get("model_name") or _get_nested(run.get("config", {}), ("model", "name"), "unknown"),
            "research_scope": _get_nested(config, ("dataset", "research_scope"), None),
            "scope_policy": _get_nested(config, ("autoresearch", "scope_policy"), None),
            "extra_train_npz_count": metrics.get("extra_train_npz_count") or len(_get_nested(config, ("resolved_data", "train_extra"), []) or []),
            "extra_train_samples": metrics.get("extra_train_samples"),
            "val_f1": metrics.get("val_f1"),
            "average_precision": metrics.get("average_precision"),
            "promotion_status": run.get("promotion_status"),
            "champion_classes": run.get("champion_classes", []),
            "validation_mode": setup.get("mode"),
            "train_segment_id": setup.get("train_segment_id"),
            "val_segment_id": setup.get("val_segment_id"),
            "pred_positive_rate_ratio": ratio,
            "full_tile_coverage_count": len(full_tiles),
            "best_full_tile_verdict": tile_quality,
            "quality_verdict": quality,
            "quality_score": quality.get("score") if isinstance(quality, dict) else None,
            "quality_reasons": quality.get("reasons", []) if isinstance(quality, dict) else [],
            "quality_next_actions": quality_actions,
            "quality_next_action": (quality_actions or [{}])[0].get("label"),
            "blocker_count": len(run.get("promotion_blockers") or []),
            "top_blockers": [item.get("code") for item in (run.get("promotion_blockers") or [])[:3]],
        })

    def key(row: dict[str, Any]) -> tuple[float, float, float]:
        status_bonus = 0.15 if row.get("promotion_status") == "eligible" else 0.0
        tile_bonus = 0.05 * min(int(row.get("full_tile_coverage_count") or 0), 3)
        return (float(row.get("quality_score") or 0.0) + status_bonus + tile_bonus, float(row.get("val_f1") or 0.0), float(row.get("average_precision") or 0.0))

    ranked = sorted(rows, key=key, reverse=True)[:limit]
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    return ranked


def load_experiments(project_root: Path, limit: int = 500) -> dict[str, Any]:
    db_path = project_root / "experiments" / "experiments.db"
    loo_summaries = _load_loo_summaries(project_root)
    empty = {"count": 0, "best": None, "latest": None, "recent": [], "champions": {"peak_score": None, "robust_candidate": None, "promotion_eligible": None}, "decision": _decision_snapshot([], None, None, None, loo_summaries, project_root), "loo_summaries": loo_summaries, "metric_trends": [], "validation_matrix": [], "leaderboard": [], "config_diffs": {"latest_vs_previous": [], "latest_vs_best": [], "latest_vs_baseline": []}, "hypotheses": []}
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
        blockers_only = [b for b in run["promotion_blockers"] if b.get("severity") != "warning"]
        run["promotion_status"] = "eligible" if not blockers_only else "blocked"
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
        "leaderboard": _leaderboard_rows(runs, project_root),
        "config_diffs": {"latest_vs_previous": _config_diff(previous.get("config") if previous else None, latest.get("config") if latest else None), "latest_vs_best": _config_diff(best.get("config") if best else None, latest.get("config") if latest else None), "latest_vs_baseline": _config_diff(baseline.get("config") if baseline else None, latest.get("config") if latest else None)},
        "hypotheses": [],
    }

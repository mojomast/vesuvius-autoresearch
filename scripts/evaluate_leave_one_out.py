#!/usr/bin/env python3
"""Run and summarize leave-one-segment-out validation for a base config."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.vesuvius_data import validate_prepared_npz
from experiments.runner import ROOT, _check_extra_train_fold_safety, load_config, run_experiment


THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
HARD_FOLD_ID = os.environ.get("AUTORESEARCH_HARD_FOLD_ID", "20230530172803")
HARD_FOLD_MIN_AP = float(os.environ.get("AUTORESEARCH_HARD_FOLD_MIN_AP", "0.02"))
HARD_FOLD_MIN_F1 = float(os.environ.get("AUTORESEARCH_HARD_FOLD_MIN_F1", "0.02"))


def _resolve(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _validate_extra_train_npzs_for_fold(cfg: dict[str, Any], heldout_segment: str) -> None:
    dataset = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
    extra_train_npzs = dataset.get("extra_train_npzs") or []
    if isinstance(extra_train_npzs, (str, Path)):
        extra_train_npzs = [extra_train_npzs]
    patch_size = int(dataset.get("patch_size", 32))
    metas = [validate_prepared_npz(_resolve(path), split="train_extra", patch_size=patch_size) for path in extra_train_npzs]
    _check_extra_train_fold_safety(metas, heldout_segment)


def _row_id(row: dict[str, Any]) -> str:
    fold = str(row.get("heldout_segment", "unknown"))
    if "seed" in row and row.get("seed") is not None:
        return f"{fold}:seed={row['seed']}"
    return fold


def _as_float(row: dict[str, Any], key: str) -> float | None:
    if key not in row or row.get(key) is None:
        return None
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return None


def _summarize(
    rows: list[dict[str, Any]],
    min_seeds_for_promotion: int = 3,
    *,
    expected_folds: list[str] | None = None,
    expected_seeds: list[int | None] | None = None,
    expected_task_count: int | None = None,
    planned_task_count: int | None = None,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    expected_folds = sorted(str(fold) for fold in (expected_folds or []))
    expected_seeds = list(expected_seeds or [])
    expected_task_count = len(rows) if expected_task_count is None else expected_task_count
    planned_task_count = len(rows) if planned_task_count is None else planned_task_count
    task_limit_applied = max_tasks is not None and planned_task_count < expected_task_count
    requested_folds = sorted({str(row["heldout_segment"]) for row in rows if "heldout_segment" in row})
    missing_expected_folds = [fold for fold in expected_folds if fold not in requested_folds]
    expected_seed_values = {seed for seed in expected_seeds if seed is not None}
    planned_seeds_by_fold: dict[str, set[int | None]] = {fold: set() for fold in requested_folds}
    for row in rows:
        if "heldout_segment" in row:
            planned_seeds_by_fold.setdefault(str(row["heldout_segment"]), set()).add(row.get("seed"))
    missing_expected_seed_repeats = [
        f"{fold}:{seed}"
        for fold in (expected_folds or requested_folds)
        for seed in sorted(expected_seed_values - planned_seeds_by_fold.get(fold, set()))
    ]
    if rows and all(row.get("dry_run") for row in rows):
        return {
            "folds_requested": len(rows),
            "folds_successful": len(rows),
            "folds_failed": 0,
            "dry_run": True,
            "expected_folds": expected_folds,
            "planned_folds": requested_folds,
            "missing_expected_folds": missing_expected_folds,
            "missing_expected_seed_repeats": missing_expected_seed_repeats,
            "expected_seeds": expected_seeds,
            "expected_task_count": expected_task_count,
            "planned_task_count": planned_task_count,
            "max_tasks": max_tasks,
            "task_limit_applied": task_limit_applied,
            "folds_with_zero_precision_or_recall": [],
            "folds_with_zero_positive_validation": [],
            "folds_with_positive_rate_alarm": [],
            "folds_with_fixed_threshold_not_ok": [],
            "folds_with_threshold_edge_case": [],
            "folds_with_weak_ap_prevalence_lift": [],
            "min_seeds_for_promotion": min_seeds_for_promotion,
            "distinct_successful_seeds": 0,
            "per_fold_successful_seeds": {},
            "promotion_ready": False,
            "promotion_warnings": [
                "dry_run_no_promotion_metrics",
                *([f"partial_task_limit_applied:{planned_task_count}/{expected_task_count}"] if task_limit_applied else []),
                *[f"missing_expected_fold:{fold}" for fold in missing_expected_folds],
                *[f"missing_expected_seed_repeat:{item}" for item in missing_expected_seed_repeats],
            ],
        }
    promotion_warnings: list[str] = []
    if task_limit_applied:
        promotion_warnings.append(f"partial_task_limit_applied:{planned_task_count}/{expected_task_count}")
    for fold in missing_expected_folds:
        promotion_warnings.append(f"missing_expected_fold:{fold}")
    for item in missing_expected_seed_repeats:
        promotion_warnings.append(f"missing_expected_seed_repeat:{item}")
    failed_rows = [row for row in rows if row.get("returncode", 0) != 0]
    for row in failed_rows:
        detail = row.get("error")
        promotion_warnings.append(f"failed_fold:{_row_id(row)}" + (f":{detail}" if detail else ""))

    successful = [row for row in rows if row.get("returncode", 0) == 0 and _as_float(row, "val_f1") is not None]
    for row in rows:
        if row.get("returncode", 0) == 0 and row not in successful:
            promotion_warnings.append(f"missing_or_invalid_val_f1:{_row_id(row)}")
    summary: dict[str, Any] = {
        "folds_requested": len(rows),
        "folds_successful": len(successful),
        "folds_failed": len(rows) - len(successful),
        "run_ids": [row["run_id"] for row in successful if row.get("run_id")],
        "expected_folds": expected_folds,
        "planned_folds": requested_folds,
        "missing_expected_folds": missing_expected_folds,
        "missing_expected_seed_repeats": missing_expected_seed_repeats,
        "expected_seeds": expected_seeds,
        "expected_task_count": expected_task_count,
        "planned_task_count": planned_task_count,
        "max_tasks": max_tasks,
        "task_limit_applied": task_limit_applied,
    }
    distinct_successful_seeds = {row.get("seed") for row in successful if row.get("seed") is not None}
    successful_seeds_by_fold: dict[str, set[Any]] = {fold: set() for fold in requested_folds}
    for row in successful:
        if row.get("seed") is not None:
            successful_seeds_by_fold.setdefault(str(row["heldout_segment"]), set()).add(row.get("seed"))
    summary["min_seeds_for_promotion"] = min_seeds_for_promotion
    summary["distinct_successful_seeds"] = len(distinct_successful_seeds)
    summary["per_fold_successful_seeds"] = {fold: len(seeds) for fold, seeds in successful_seeds_by_fold.items()}
    if successful:
        by_fold: dict[str, list[float]] = {}
        ap_by_fold: dict[str, list[float]] = {}
        by_seed: dict[str, list[float]] = {}
        ap_by_seed: dict[str, list[float]] = {}
        all_ap: list[float] = []
        zero_precision_or_recall: list[str] = []
        zero_positive_validation: list[str] = []
        positive_rate_alarm: list[str] = []
        fixed_threshold_not_ok: list[str] = []
        threshold_edge_case: list[str] = []
        weak_ap_prevalence_lift: list[str] = []
        for row in successful:
            fold = str(row["heldout_segment"])
            val_f1 = _as_float(row, "val_f1")
            average_precision = _as_float(row, "average_precision")
            if val_f1 is None:
                promotion_warnings.append(f"invalid_val_f1:{_row_id(row)}")
                continue
            if average_precision is None:
                promotion_warnings.append(f"missing_average_precision:{_row_id(row)}")
                average_precision = 0.0
            by_fold.setdefault(fold, []).append(val_f1)
            ap_by_fold.setdefault(fold, []).append(average_precision)
            all_ap.append(average_precision)
            if "seed" in row:
                seed = str(row["seed"])
                by_seed.setdefault(seed, []).append(val_f1)
                ap_by_seed.setdefault(seed, []).append(average_precision)

            precision = _as_float(row, "precision")
            recall = _as_float(row, "recall")
            if precision == 0.0 or recall == 0.0:
                zero_precision_or_recall.append(_row_id(row))

            pred_positive_rate = _as_float(row, "pred_positive_rate")
            val_positive_rate = _as_float(row, "val_positive_rate")
            if val_positive_rate == 0.0:
                zero_positive_validation.append(_row_id(row))
            if pred_positive_rate is not None and val_positive_rate is not None and val_positive_rate > 0:
                ratio = pred_positive_rate / val_positive_rate
                if ratio > 3.5 or ratio < 0.1:
                    positive_rate_alarm.append(_row_id(row))

            fixed_threshold_status = row.get("fixed_threshold_status")
            if fixed_threshold_status is not None and fixed_threshold_status != "ok":
                fixed_threshold_not_ok.append(_row_id(row))

            best_threshold = _as_float(row, "best_threshold")
            if best_threshold is not None and (best_threshold <= 0.01 or best_threshold >= 0.99):
                threshold_edge_case.append(_row_id(row))

            ap_prevalence_lift = _as_float(row, "ap_prevalence_lift")
            if ap_prevalence_lift is None and val_positive_rate is not None and val_positive_rate > 0:
                ap_prevalence_lift = average_precision / val_positive_rate
            if ap_prevalence_lift is not None and ap_prevalence_lift < 1.25:
                weak_ap_prevalence_lift.append(_row_id(row))
        per_fold_f1 = {fold: float(statistics.mean(values)) for fold, values in by_fold.items()}
        per_fold_ap = {fold: float(statistics.mean(values)) for fold, values in ap_by_fold.items()}
        fixed_threshold_status_counts: dict[str, int] = {}
        positive_rate_alarm_details: list[dict[str, Any]] = []
        fixed_threshold_not_ok_details: list[dict[str, Any]] = []
        for row in successful:
            row_id = _row_id(row)
            status = str(row.get("fixed_threshold_status") or "missing")
            fixed_threshold_status_counts[status] = fixed_threshold_status_counts.get(status, 0) + 1
            pred_positive_rate = _as_float(row, "pred_positive_rate")
            val_positive_rate = _as_float(row, "val_positive_rate")
            pred_to_val_ratio = pred_positive_rate / val_positive_rate if pred_positive_rate is not None and val_positive_rate and val_positive_rate > 0 else None
            if row_id in positive_rate_alarm:
                positive_rate_alarm_details.append({"row_id": row_id, "heldout_segment": row.get("heldout_segment"), "seed": row.get("seed"), "pred_positive_rate": pred_positive_rate, "val_positive_rate": val_positive_rate, "pred_to_val_ratio": pred_to_val_ratio})
            if row_id in fixed_threshold_not_ok:
                fixed_threshold_not_ok_details.append({"row_id": row_id, "heldout_segment": row.get("heldout_segment"), "seed": row.get("seed"), "status": row.get("fixed_threshold_status"), "failure_reason": row.get("fixed_threshold_failure_reason"), "fixed_threshold_f1": row.get("fixed_threshold_f1")})
        diagnostic_summary: list[str] = []
        if fixed_threshold_not_ok:
            diagnostic_summary.append(f"{len(fixed_threshold_not_ok)}/{len(successful)} rows fixed_threshold_not_ok")
        if positive_rate_alarm:
            diagnostic_summary.append(f"{len(positive_rate_alarm)}/{len(successful)} rows positive_rate_alarm")
        if weak_ap_prevalence_lift:
            diagnostic_summary.append(f"{len(weak_ap_prevalence_lift)}/{len(successful)} rows weak_ap_prevalence_lift")
        recommended_next_actions: list[dict[str, Any]] = []
        hard_fold_diagnostics: dict[str, Any] = {}
        if HARD_FOLD_ID in per_fold_f1 or HARD_FOLD_ID in per_fold_ap:
            hard_f1 = per_fold_f1.get(HARD_FOLD_ID)
            hard_ap = per_fold_ap.get(HARD_FOLD_ID)
            hard_rows = [row for row in successful if str(row.get("heldout_segment")) == HARD_FOLD_ID]
            hard_lifts = [_as_float(row, "ap_prevalence_lift") for row in hard_rows]
            hard_val_rates = [_as_float(row, "val_positive_rate") for row in hard_rows]
            hard_pred_rates = [_as_float(row, "pred_positive_rate") for row in hard_rows]
            hard_fold_diagnostics = {
                "fold_id": HARD_FOLD_ID,
                "min_val_f1": HARD_FOLD_MIN_F1,
                "min_average_precision": HARD_FOLD_MIN_AP,
                "val_f1": hard_f1,
                "average_precision": hard_ap,
                "average_precision_floor_delta": (hard_ap - HARD_FOLD_MIN_AP) if hard_ap is not None else None,
                "mean_ap_prevalence_lift": float(statistics.mean([value for value in hard_lifts if value is not None])) if any(value is not None for value in hard_lifts) else None,
                "mean_val_positive_rate": float(statistics.mean([value for value in hard_val_rates if value is not None])) if any(value is not None for value in hard_val_rates) else None,
                "mean_pred_positive_rate": float(statistics.mean([value for value in hard_pred_rates if value is not None])) if any(value is not None for value in hard_pred_rates) else None,
                "passed": (hard_f1 is None or hard_f1 >= HARD_FOLD_MIN_F1) and (hard_ap is None or hard_ap >= HARD_FOLD_MIN_AP),
            }
            if hard_f1 is not None and hard_f1 < HARD_FOLD_MIN_F1:
                promotion_warnings.append(f"hard_fold_low_f1:{HARD_FOLD_ID}:{hard_f1:.6f}")
            if hard_ap is not None and hard_ap < HARD_FOLD_MIN_AP:
                promotion_warnings.append(f"hard_fold_low_ap:{HARD_FOLD_ID}:{hard_ap:.6f}")
            if hard_ap is not None and hard_ap <= max(HARD_FOLD_MIN_AP * 1.5, 0.03):
                diagnostic_summary.append(f"hard_fold_low_ap:{HARD_FOLD_ID}:{hard_ap:.6f}")
                recommended_next_actions.append({"id": "audit_hard_fold_labels", "label": f"Audit labels and sampling pressure for hard fold {HARD_FOLD_ID}", "writes_artifacts": False})
        if fixed_threshold_not_ok:
            recommended_next_actions.append({"id": "calibrate_fixed_threshold", "label": "Diagnose fixed-threshold calibration across linked LOO", "writes_artifacts": False})
        if positive_rate_alarm:
            recommended_next_actions.append({"id": "review_positive_rate_calibration", "label": "Review positive-rate alarms before promotion", "writes_artifacts": False})
        worst_fold_id = min(per_fold_f1, key=per_fold_f1.get)
        summary.update({
            "median_val_f1": float(statistics.median(per_fold_f1.values())),
            "mean_val_f1": float(statistics.mean(per_fold_f1.values())),
            "min_val_f1": float(min(per_fold_f1.values())),
            "max_val_f1": float(max(per_fold_f1.values())),
            "mean_average_precision": float(statistics.mean(per_fold_ap.values())),
            "median_average_precision": float(statistics.median(all_ap)),
            "worst_fold_val_f1": float(per_fold_f1[worst_fold_id]),
            "worst_fold_id": worst_fold_id,
            "per_fold_val_f1": per_fold_f1,
            "per_fold_average_precision": per_fold_ap,
            "hard_fold_diagnostics": hard_fold_diagnostics,
            "folds_with_zero_precision_or_recall": zero_precision_or_recall,
            "folds_with_zero_positive_validation": zero_positive_validation,
            "folds_with_positive_rate_alarm": positive_rate_alarm,
            "folds_with_fixed_threshold_not_ok": fixed_threshold_not_ok,
            "folds_with_threshold_edge_case": threshold_edge_case,
            "folds_with_weak_ap_prevalence_lift": weak_ap_prevalence_lift,
            "fixed_threshold_status_counts": fixed_threshold_status_counts,
            "positive_rate_alarm_count": len(positive_rate_alarm),
            "fixed_threshold_not_ok_count": len(fixed_threshold_not_ok),
            "positive_rate_alarm_details": positive_rate_alarm_details,
            "fixed_threshold_not_ok_details": fixed_threshold_not_ok_details,
            "diagnostic_summary": diagnostic_summary,
            "recommended_next_actions": recommended_next_actions,
        })
        if by_seed:
            summary["per_seed_mean_val_f1"] = {seed: float(statistics.mean(values)) for seed, values in by_seed.items()}
            per_seed_median = {seed: float(statistics.median(values)) for seed, values in by_seed.items()}
            summary["per_seed_median_val_f1"] = per_seed_median
            summary["per_seed_min_val_f1"] = {seed: float(min(values)) for seed, values in by_seed.items()}
            summary["per_seed_mean_average_precision"] = {seed: float(statistics.mean(values)) for seed, values in ap_by_seed.items()}
            summary["median_over_seeds_median_val_f1"] = float(statistics.median(per_seed_median.values()))
            summary["worst_seed_median_val_f1"] = float(min(per_seed_median.values()))

        for row_id in zero_precision_or_recall:
            promotion_warnings.append(f"zero_precision_or_recall:{row_id}")
        for row_id in zero_positive_validation:
            promotion_warnings.append(f"zero_positive_validation:{row_id}")
        for row_id in positive_rate_alarm:
            promotion_warnings.append(f"positive_rate_alarm:{row_id}")
        for row_id in fixed_threshold_not_ok:
            promotion_warnings.append(f"fixed_threshold_not_ok:{row_id}")
        for row_id in threshold_edge_case:
            promotion_warnings.append(f"threshold_edge_case:{row_id}")
        for row_id in weak_ap_prevalence_lift:
            promotion_warnings.append(f"weak_ap_prevalence_lift:{row_id}")
        if len(distinct_successful_seeds) < min_seeds_for_promotion:
            promotion_warnings.append(f"insufficient_seed_repeats:{len(distinct_successful_seeds)}/{min_seeds_for_promotion}")
        for fold, seed_count in summary["per_fold_successful_seeds"].items():
            if seed_count < min_seeds_for_promotion:
                promotion_warnings.append(f"insufficient_fold_seed_repeats:{fold}:{seed_count}/{min_seeds_for_promotion}")
        if expected_seed_values:
            for fold in expected_folds or requested_folds:
                seen = successful_seeds_by_fold.get(fold, set())
                for seed in sorted(expected_seed_values - seen):
                    warning = f"missing_expected_seed_repeat:{fold}:{seed}"
                    if warning not in promotion_warnings:
                        promotion_warnings.append(warning)
    else:
        summary["folds_with_zero_precision_or_recall"] = []
        summary["folds_with_zero_positive_validation"] = []
        summary["folds_with_positive_rate_alarm"] = []
        summary["folds_with_fixed_threshold_not_ok"] = []
        summary["folds_with_threshold_edge_case"] = []
        summary["folds_with_weak_ap_prevalence_lift"] = []
        for fold, seed_count in summary["per_fold_successful_seeds"].items():
            if seed_count < min_seeds_for_promotion:
                promotion_warnings.append(f"insufficient_fold_seed_repeats:{fold}:{seed_count}/{min_seeds_for_promotion}")
        if rows:
            promotion_warnings.append("no_successful_folds")

    summary["promotion_warnings"] = promotion_warnings
    summary["promotion_ready"] = bool(successful) and not promotion_warnings
    return summary


def _parse_seeds(raw: str | None, default_seed: int | None) -> list[int | None]:
    if raw is None:
        return [default_seed]
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer seed")
    return seeds


def _limit_worker_threads() -> None:
    for key in THREAD_LIMIT_ENV_VARS:
        os.environ.setdefault(key, "1")


def _run_fold_job(fold_config: str, row: dict[str, Any]) -> dict[str, Any]:
    try:
        result = run_experiment(fold_config)
        metrics = result["metrics"]
        row.update({
            "returncode": 0,
            "run_id": result["run_id"],
            "artifact_dir": result["artifact_dir"],
            "val_f1": float(metrics["val_f1"]),
            "val_f05": float(metrics["val_f05"]),
            "average_precision": float(metrics["average_precision"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "best_threshold": float(metrics["best_threshold"]),
            "val_positive_rate": float(metrics["val_positive_rate"]),
            "pred_positive_rate": float(metrics["pred_positive_rate"]),
        })
        for key in (
            "brier_score",
            "expected_calibration_error",
            "ap_prevalence_lift",
            "prob_mean",
            "prob_p95",
            "prob_max",
            "fixed_threshold",
            "fixed_threshold_f1",
            "fixed_threshold_status",
            "fixed_threshold_failure_reason",
            "threshold_selection",
            "selected_threshold_reason",
            "threshold_risk_summary",
            "max_pred_positive_rate_ratio",
        ):
            if key in metrics:
                row[key] = metrics[key]
    except Exception as exc:
        row.update({"returncode": 1, "error": repr(exc)})
    return row


def _run_fold_jobs_for_heldout_fold(tasks: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [_run_fold_job(config_path, row) for config_path, row in tasks]


def _write_row_jsonl(output_jsonl: Path, row: dict[str, Any]) -> None:
    with output_jsonl.open("a") as fh:
        line = json.dumps(row, sort_keys=True)
        fh.write(line + "\n")
        print(line, flush=True)


def _print_progress(row: dict[str, Any], completed: int, total: int) -> None:
    print(
        "LOO_PROGRESS " + json.dumps({
            "completed": completed,
            "heldout_segment": row.get("heldout_segment"),
            "returncode": row.get("returncode"),
            "seed": row.get("seed"),
            "total": total,
        }, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a config with leave-one-segment-out folds")
    parser.add_argument("--base-config", required=True, help="YAML/JSON config to evaluate")
    parser.add_argument("--fold-map", required=True, help="JSON map of heldout segment to train_npz/val_npz")
    parser.add_argument("--output-jsonl", required=True, help="Per-fold result JSONL path")
    parser.add_argument("--summary-json", default=None, help="Optional summary JSON path")
    parser.add_argument("--label", default=None, help="Label stored in outputs; defaults to base config stem")
    parser.add_argument("--rerun-tag", default=None, help="Optional tag added to fold configs to force fresh non-deduped diagnostic runs")
    parser.add_argument("--seeds", default=None, help="Comma-separated training.seed values to repeat for each held-out segment")
    parser.add_argument("--min-seeds-for-promotion", type=int, default=3, help="Distinct successful seeds required before setting promotion_ready")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel fold jobs for non-dry-run execution; default 1")
    parser.add_argument("--execution-mode", choices=("task", "fold-major"), default="task", help="Schedule one worker job per seed task, or group all seeds for a held-out fold in one worker")
    parser.add_argument("--limit-worker-threads", action=argparse.BooleanOptionalAction, default=True, help="Set common BLAS/OpenMP thread env vars to 1 inside worker processes when unset")
    parser.add_argument("--max-tasks", type=int, default=None, help="Optional cap on fold/seed tasks to run")
    parser.add_argument("--dry-run", action="store_true", help="Write planned fold configs without running experiments")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.max_tasks is not None and args.max_tasks < 1:
        parser.error("--max-tasks must be >= 1")

    base_path = _resolve(args.base_config)
    fold_map_path = _resolve(args.fold_map)
    output_jsonl = _resolve(args.output_jsonl)
    summary_json = _resolve(args.summary_json) if args.summary_json else output_jsonl.with_suffix(".summary.json")
    label = args.label or base_path.stem

    base_cfg = load_config(base_path)
    fold_map = json.loads(fold_map_path.read_text())
    if not fold_map:
        raise SystemExit("fold map contains no folds")
    base_seed = base_cfg.get("training", {}).get("seed")
    seeds = _parse_seeds(args.seeds, int(base_seed) if base_seed is not None else None)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("")

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="vesuvius_loo_") as tmpdir:
        tmp = Path(tmpdir)
        tasks: list[tuple[str, dict[str, Any]]] = []
        tasks_by_fold: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for heldout_segment, fold in sorted(fold_map.items()):
            for seed in seeds:
                cfg = copy.deepcopy(base_cfg)
                cfg.pop("resolved_data", None)
                cfg.pop("validation_setup", None)
                dataset = cfg.setdefault("dataset", {})
                dataset["train_npz"] = fold["train_npz"]
                dataset["val_npz"] = fold["val_npz"]
                dataset["validation_mode"] = "leave-one-segment-out"
                if seed is not None:
                    cfg.setdefault("training", {})["seed"] = seed
                cfg.setdefault("autoresearch", {})["heldout_segment"] = heldout_segment
                cfg["autoresearch"]["run_profile"] = "promotion"
                cfg["autoresearch"]["intent"] = "promotion_validation"
                cfg["autoresearch"]["fold_map"] = str(fold_map_path.relative_to(ROOT) if fold_map_path.is_relative_to(ROOT) else fold_map_path)
                cfg["autoresearch"]["seed_repeat"] = seed
                if args.rerun_tag is not None:
                    cfg["autoresearch"]["rerun_tag"] = args.rerun_tag
                _validate_extra_train_npzs_for_fold(cfg, heldout_segment)

                seed_suffix = f"__seed_{seed}" if seed is not None else ""
                fold_config = tmp / f"{label}__leaveout_{heldout_segment}{seed_suffix}.yaml"
                fold_config.write_text(yaml.safe_dump(_jsonable(cfg), sort_keys=False))
                row: dict[str, Any] = {"label": label, "heldout_segment": heldout_segment, "seed": seed, "config_path": str(fold_config)}
                if args.dry_run:
                    row["returncode"] = 0
                    row["dry_run"] = True
                    rows.append(row)
                else:
                    tasks.append((str(fold_config), row))
                    tasks_by_fold.setdefault(str(heldout_segment), []).append((str(fold_config), row))
        if args.max_tasks is not None:
            if args.dry_run:
                rows = rows[: args.max_tasks]
            else:
                tasks = tasks[: args.max_tasks]
                tasks_by_fold = {}
                for task in tasks:
                    tasks_by_fold.setdefault(str(task[1]["heldout_segment"]), []).append(task)
        total_tasks = len(rows) if args.dry_run else len(tasks)
        if total_tasks == 0:
            raise SystemExit("no fold/seed tasks were planned")
        if args.dry_run:
            for completed, row in enumerate(rows, start=1):
                _write_row_jsonl(output_jsonl, row)
                _print_progress(row, completed, total_tasks)
        elif tasks and args.jobs == 1:
            if args.limit_worker_threads:
                _limit_worker_threads()
            for completed, (config_path, row) in enumerate(tasks, start=1):
                result_row = _run_fold_job(config_path, row)
                rows.append(result_row)
                _write_row_jsonl(output_jsonl, result_row)
                _print_progress(result_row, completed, total_tasks)
        elif tasks:
            executor_kwargs: dict[str, Any] = {"max_workers": args.jobs}
            if args.limit_worker_threads:
                executor_kwargs["initializer"] = _limit_worker_threads
            with ProcessPoolExecutor(**executor_kwargs) as executor:
                if args.execution_mode == "fold-major":
                    fold_task_groups = [tasks_by_fold[fold] for fold in sorted(tasks_by_fold)]
                    future_to_group = {executor.submit(_run_fold_jobs_for_heldout_fold, group): group for group in fold_task_groups}
                    completed = 0
                    for future in as_completed(future_to_group):
                        fold_rows = future.result()
                        rows.extend(fold_rows)
                        for row in fold_rows:
                            completed += 1
                            _write_row_jsonl(output_jsonl, row)
                            _print_progress(row, completed, total_tasks)
                else:
                    future_to_task = {executor.submit(_run_fold_job, config_path, row): (config_path, row) for config_path, row in tasks}
                    for completed, future in enumerate(as_completed(future_to_task), start=1):
                        row = future.result()
                        rows.append(row)
                        _write_row_jsonl(output_jsonl, row)
                        _print_progress(row, completed, total_tasks)

    expected_folds = sorted(str(fold) for fold in fold_map)
    expected_task_count = len(expected_folds) * len(seeds)
    summary = _summarize(
        rows,
        min_seeds_for_promotion=args.min_seeds_for_promotion,
        expected_folds=expected_folds,
        expected_seeds=seeds,
        expected_task_count=expected_task_count,
        planned_task_count=total_tasks,
        max_tasks=args.max_tasks,
    )
    summary.update({"label": label, "base_config": str(base_path), "fold_map": str(fold_map_path), "seeds": seeds})
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("SUMMARY_JSON " + json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary.get("folds_failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run and summarize leave-one-segment-out validation for a base config."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
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


def _summarize(rows: list[dict[str, Any]], min_seeds_for_promotion: int = 3) -> dict[str, Any]:
    if rows and all(row.get("dry_run") for row in rows):
        return {
            "folds_requested": len(rows),
            "folds_successful": len(rows),
            "folds_failed": 0,
            "dry_run": True,
            "folds_with_zero_precision_or_recall": [],
            "folds_with_positive_rate_alarm": [],
            "min_seeds_for_promotion": min_seeds_for_promotion,
            "distinct_successful_seeds": 0,
            "promotion_ready": False,
            "promotion_warnings": ["dry_run_no_promotion_metrics"],
        }
    promotion_warnings: list[str] = []
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
    }
    distinct_successful_seeds = {row.get("seed") for row in successful if row.get("seed") is not None}
    summary["min_seeds_for_promotion"] = min_seeds_for_promotion
    summary["distinct_successful_seeds"] = len(distinct_successful_seeds)
    if successful:
        by_fold: dict[str, list[float]] = {}
        ap_by_fold: dict[str, list[float]] = {}
        by_seed: dict[str, list[float]] = {}
        ap_by_seed: dict[str, list[float]] = {}
        all_ap: list[float] = []
        zero_precision_or_recall: list[str] = []
        positive_rate_alarm: list[str] = []
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
            if pred_positive_rate is not None and val_positive_rate is not None and val_positive_rate > 0:
                ratio = pred_positive_rate / val_positive_rate
                if ratio > 4.0 or ratio < 0.1:
                    positive_rate_alarm.append(_row_id(row))
        per_fold_f1 = {fold: float(statistics.mean(values)) for fold, values in by_fold.items()}
        per_fold_ap = {fold: float(statistics.mean(values)) for fold, values in ap_by_fold.items()}
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
            "folds_with_zero_precision_or_recall": zero_precision_or_recall,
            "folds_with_positive_rate_alarm": positive_rate_alarm,
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
        for row_id in positive_rate_alarm:
            promotion_warnings.append(f"positive_rate_alarm:{row_id}")
        if len(distinct_successful_seeds) < min_seeds_for_promotion:
            promotion_warnings.append(f"insufficient_seed_repeats:{len(distinct_successful_seeds)}/{min_seeds_for_promotion}")
    else:
        summary["folds_with_zero_precision_or_recall"] = []
        summary["folds_with_positive_rate_alarm"] = []
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
            "threshold_selection",
            "selected_threshold_reason",
        ):
            if key in metrics:
                row[key] = metrics[key]
    except Exception as exc:
        row.update({"returncode": 1, "error": repr(exc)})
    return row


def _write_rows_jsonl(output_jsonl: Path, rows: list[dict[str, Any]]) -> None:
    with output_jsonl.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a config with leave-one-segment-out folds")
    parser.add_argument("--base-config", required=True, help="YAML/JSON config to evaluate")
    parser.add_argument("--fold-map", required=True, help="JSON map of heldout segment to train_npz/val_npz")
    parser.add_argument("--output-jsonl", required=True, help="Per-fold result JSONL path")
    parser.add_argument("--summary-json", default=None, help="Optional summary JSON path")
    parser.add_argument("--label", default=None, help="Label stored in outputs; defaults to base config stem")
    parser.add_argument("--seeds", default=None, help="Comma-separated training.seed values to repeat for each held-out segment")
    parser.add_argument("--min-seeds-for-promotion", type=int, default=3, help="Distinct successful seeds required before setting promotion_ready")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel fold jobs for non-dry-run execution; default 1")
    parser.add_argument("--dry-run", action="store_true", help="Write planned fold configs without running experiments")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    base_path = _resolve(args.base_config)
    fold_map_path = _resolve(args.fold_map)
    output_jsonl = _resolve(args.output_jsonl)
    summary_json = _resolve(args.summary_json) if args.summary_json else output_jsonl.with_suffix(".summary.json")
    label = args.label or base_path.stem

    base_cfg = load_config(base_path)
    fold_map = json.loads(fold_map_path.read_text())
    base_seed = base_cfg.get("training", {}).get("seed")
    seeds = _parse_seeds(args.seeds, int(base_seed) if base_seed is not None else None)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="vesuvius_loo_") as tmpdir:
        tmp = Path(tmpdir)
        tasks: list[tuple[str, dict[str, Any]]] = []
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
                cfg["autoresearch"]["fold_map"] = str(fold_map_path.relative_to(ROOT) if fold_map_path.is_relative_to(ROOT) else fold_map_path)
                cfg["autoresearch"]["seed_repeat"] = seed
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
        if tasks and args.jobs == 1:
            rows.extend(_run_fold_job(config_path, row) for config_path, row in tasks)
        elif tasks:
            with ProcessPoolExecutor(max_workers=args.jobs) as executor:
                rows.extend(executor.map(_run_fold_job, [task[0] for task in tasks], [task[1] for task in tasks]))
        _write_rows_jsonl(output_jsonl, rows)

    summary = _summarize(rows, min_seeds_for_promotion=args.min_seeds_for_promotion)
    summary.update({"label": label, "base_config": str(base_path), "fold_map": str(fold_map_path), "seeds": seeds})
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("SUMMARY_JSON " + json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary.get("folds_failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

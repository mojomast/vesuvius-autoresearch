#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_threshold_caps import compare_threshold_caps  # noqa: E402
from scripts.evaluate_leave_one_out import _summarize  # noqa: E402


def _cap_label(cap: float) -> str:
    return str(float(cap)).rstrip("0").rstrip(".").replace(".", "p")


def _parse_caps(value: str) -> list[float]:
    caps = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not caps:
        raise argparse.ArgumentTypeError("at least one cap is required")
    return caps


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _metrics_path_for_row(row: dict[str, Any], repo_root: Path) -> Path | None:
    artifact = row.get("artifact_dir")
    if not artifact:
        return None
    artifact_path = Path(str(artifact))
    if not artifact_path.is_absolute():
        artifact_path = repo_root / artifact_path
    path = artifact_path / "metrics.json"
    return path if path.exists() else None


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _retained_fraction(new_value: Any, original_value: Any) -> float | None:
    original = _as_float(original_value)
    new = _as_float(new_value)
    if original is None or new is None or original <= 0.0:
        return None
    return new / original


def _apply_cap_best(row: dict[str, Any], cap: float, best: dict[str, Any] | None, val_positive_rate: float | None = None) -> dict[str, Any]:
    out = dict(row)
    if not best:
        out["returncode"] = 1
        out["error"] = f"no_threshold_row_under_prratio{_cap_label(cap)}"
        return out
    original_fields = {
        "val_f1": "original_val_f1",
        "val_f05": "original_val_f05",
        "precision": "original_precision",
        "recall": "original_recall",
        "best_threshold": "original_best_threshold",
        "pred_positive_rate": "original_pred_positive_rate",
        "threshold_selection": "original_threshold_selection",
        "selected_threshold_reason": "original_selected_threshold_reason",
    }
    for source, target in original_fields.items():
        if source in out and target not in out:
            out[target] = out[source]
    out.update({
        "val_f1": float(best["f1"]),
        "val_f05": float(best["f05"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "best_threshold": float(best["threshold"]),
        "pred_positive_rate": float(best["pred_positive_rate"]),
        "selected_threshold_reason": f"recomputed_positive_rate_cap_{_cap_label(cap)}",
        "threshold_selection": f"recomputed_positive_rate_cap_{_cap_label(cap)}",
        "recomputed_max_pred_positive_rate_ratio": float(cap),
        "recomputed_pred_to_val_ratio": float(best["pred_to_val_ratio"]),
    })
    if val_positive_rate is not None and out.get("val_positive_rate") is None:
        out["val_positive_rate"] = float(val_positive_rate)
    out["f1_retained_vs_original"] = _retained_fraction(out.get("val_f1"), out.get("original_val_f1"))
    out["f05_retained_vs_original"] = _retained_fraction(out.get("val_f05"), out.get("original_val_f05"))
    return out


def _error_row(row: dict[str, Any], message: str) -> dict[str, Any]:
    out = dict(row)
    out["returncode"] = 1
    out["error"] = message
    return out


def _recompute_row(row: dict[str, Any], cap: float, repo_root: Path) -> dict[str, Any]:
    out = dict(row)
    if int(out.get("returncode") or 0) != 0:
        return out
    metrics_path = _metrics_path_for_row(out, repo_root)
    if metrics_path is None:
        return _error_row(out, "missing_artifact_metrics_json")
    report = compare_threshold_caps(metrics_path, caps=[cap], baseline_cap=None)
    best = (report["caps"][0] or {}).get("best")
    return _apply_cap_best(out, cap, best, _as_float(report.get("val_positive_rate")))


def recompute_loo_threshold_cap(input_jsonl: str | Path, cap: float, *, repo_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = Path(repo_root).resolve() if repo_root is not None else ROOT
    rows: list[dict[str, Any]] = []
    with Path(input_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(_recompute_row(json.loads(line), cap, root))
    return rows


def recompute_loo_threshold_caps(input_jsonl: str | Path, caps: list[float], *, repo_root: str | Path | None = None) -> dict[float, list[dict[str, Any]]]:
    root = Path(repo_root).resolve() if repo_root is not None else ROOT
    caps = sorted(float(cap) for cap in caps)
    rows_by_cap: dict[float, list[dict[str, Any]]] = {cap: [] for cap in caps}
    with Path(input_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("returncode") or 0) != 0:
                for cap in caps:
                    rows_by_cap[cap].append(dict(row))
                continue
            metrics_path = _metrics_path_for_row(row, root)
            if metrics_path is None:
                for cap in caps:
                    rows_by_cap[cap].append(_error_row(row, "missing_artifact_metrics_json"))
                continue
            report = compare_threshold_caps(metrics_path, caps=caps, baseline_cap=None)
            best_by_cap = {float(item["cap"]): item.get("best") for item in report.get("caps") or []}
            for cap in caps:
                rows_by_cap[cap].append(_apply_cap_best(row, cap, best_by_cap.get(cap), _as_float(report.get("val_positive_rate"))))
    return rows_by_cap


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _summary_for_cap(input_jsonl: str | Path, rows: list[dict[str, Any]], cap: float, min_seeds_for_promotion: int) -> dict[str, Any]:
    summary = _summarize(rows, min_seeds_for_promotion=min_seeds_for_promotion)
    successful = [row for row in rows if int(row.get("returncode") or 0) == 0]
    ratios = [float(row["recomputed_pred_to_val_ratio"]) for row in successful if isinstance(row.get("recomputed_pred_to_val_ratio"), (int, float))]
    thresholds = [float(row["best_threshold"]) for row in successful if isinstance(row.get("best_threshold"), (int, float))]
    f1_retained = [float(row["f1_retained_vs_original"]) for row in successful if isinstance(row.get("f1_retained_vs_original"), (int, float))]
    f05_retained = [float(row["f05_retained_vs_original"]) for row in successful if isinstance(row.get("f05_retained_vs_original"), (int, float))]
    selection_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    per_fold_ratios: dict[str, list[float]] = {}
    per_fold_thresholds: dict[str, list[float]] = {}
    for row in successful:
        selection = str(row.get("threshold_selection") or "")
        reason = str(row.get("selected_threshold_reason") or "")
        if selection:
            selection_counts[selection] = selection_counts.get(selection, 0) + 1
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        fold = str(row.get("heldout_segment") or "")
        if fold and isinstance(row.get("recomputed_pred_to_val_ratio"), (int, float)):
            per_fold_ratios.setdefault(fold, []).append(float(row["recomputed_pred_to_val_ratio"]))
        if fold and isinstance(row.get("best_threshold"), (int, float)):
            per_fold_thresholds.setdefault(fold, []).append(float(row["best_threshold"]))
    summary.update({
        "input_jsonl": str(input_jsonl),
        "cap_policy": f"recomputed_positive_rate_cap_{_cap_label(cap)}",
        "recomputed_max_pred_positive_rate_ratio": float(cap),
        "mean_recomputed_pred_to_val_ratio": (sum(ratios) / len(ratios)) if ratios else None,
        "median_recomputed_pred_to_val_ratio": _median(ratios),
        "max_recomputed_pred_to_val_ratio": max(ratios) if ratios else None,
        "mean_selected_threshold": (sum(thresholds) / len(thresholds)) if thresholds else None,
        "mean_f1_retained_vs_original": (sum(f1_retained) / len(f1_retained)) if f1_retained else None,
        "min_f1_retained_vs_original": min(f1_retained) if f1_retained else None,
        "mean_f05_retained_vs_original": (sum(f05_retained) / len(f05_retained)) if f05_retained else None,
        "min_f05_retained_vs_original": min(f05_retained) if f05_retained else None,
        "threshold_selection_counts": selection_counts,
        "selected_threshold_reason_counts": reason_counts,
        "per_fold_recomputed_pred_to_val_ratio": {fold: sum(values) / len(values) for fold, values in sorted(per_fold_ratios.items()) if values},
        "per_fold_selected_threshold": {fold: sum(values) / len(values) for fold, values in sorted(per_fold_thresholds.items()) if values},
    })
    return summary


def _aggregate_summaries(summaries: dict[float, dict[str, Any]]) -> dict[str, Any]:
    sortable = [(cap, summary) for cap, summary in summaries.items() if int(summary.get("folds_failed") or 0) == 0]
    def metric_value(item: tuple[float, dict[str, Any]], key: str) -> float:
        value = item[1].get(key)
        return float(value) if isinstance(value, (int, float)) else float("-inf")

    best_median = max(sortable, key=lambda item: (metric_value(item, "median_over_seeds_median_val_f1"), metric_value(item, "worst_fold_val_f1"), -item[0]), default=None)
    best_worst = max(sortable, key=lambda item: (metric_value(item, "worst_fold_val_f1"), metric_value(item, "median_over_seeds_median_val_f1"), -item[0]), default=None)
    return {
        "caps": [float(cap) for cap in sorted(summaries)],
        "best_cap_by_median_over_seeds_f1": float(best_median[0]) if best_median else None,
        "best_cap_by_worst_fold_f1": float(best_worst[0]) if best_worst else None,
        "cap_summaries": {str(float(cap)): summary for cap, summary in sorted(summaries.items())},
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_aggregate_markdown(aggregate: dict[str, Any]) -> str:
    best_median = aggregate.get("best_cap_by_median_over_seeds_f1")
    best_worst = aggregate.get("best_cap_by_worst_fold_f1")
    lines = [
        "# Recomputed LOO Cap Sweep",
        "",
        f"- best_cap_by_median_over_seeds_f1: {_fmt(best_median)}",
        f"- best_cap_by_worst_fold_f1: {_fmt(best_worst)}",
        "",
        "| cap | mean F1 | median-over-seeds F1 | worst-fold F1 | mean pred/val | max pred/val | min F1 retained | min F0.5 retained | folds failed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summaries = aggregate.get("cap_summaries") or {}
    for cap in aggregate.get("caps") or []:
        summary = summaries.get(str(float(cap))) or {}
        label = f"{_fmt(cap)}"
        if best_median is not None and abs(float(cap) - float(best_median)) < 1e-9:
            label += " best-median"
        if best_worst is not None and abs(float(cap) - float(best_worst)) < 1e-9:
            label += " best-worst"
        lines.append("| " + " | ".join([
            label,
            _fmt(summary.get("mean_val_f1")),
            _fmt(summary.get("median_over_seeds_median_val_f1")),
            _fmt(summary.get("worst_fold_val_f1")),
            _fmt(summary.get("mean_recomputed_pred_to_val_ratio")),
            _fmt(summary.get("max_recomputed_pred_to_val_ratio")),
            _fmt(summary.get("min_f1_retained_vs_original")),
            _fmt(summary.get("min_f05_retained_vs_original")),
            _fmt(summary.get("folds_failed")),
        ]) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recompute an existing LOO JSONL under a positive-rate cap using saved metrics_by_threshold.csv files")
    parser.add_argument("--input-jsonl", required=True, help="Existing LOO JSONL with artifact_dir fields")
    parser.add_argument("--output-jsonl", default=None, help="Output recomputed JSONL for single-cap mode")
    parser.add_argument("--summary-json", default=None, help="Output recomputed summary JSON for single-cap mode")
    parser.add_argument("--cap", type=float, default=None, help="Max pred/val positive-rate ratio to recompute")
    parser.add_argument("--caps", type=_parse_caps, default=None, help="Comma-separated caps for batch mode")
    parser.add_argument("--output-dir", default=None, help="Directory for batch output JSONL/summary files")
    parser.add_argument("--output-prefix", default=None, help="Filename prefix for batch outputs")
    parser.add_argument("--aggregate-json", default=None, help="Optional batch aggregate JSON path")
    parser.add_argument("--aggregate-markdown", default=None, help="Optional batch aggregate Markdown path")
    parser.add_argument("--min-seeds-for-promotion", type=int, default=3)
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args(argv)

    if args.caps is not None:
        if args.output_dir is None:
            parser.error("--output-dir is required with --caps")
        output_prefix = args.output_prefix or Path(args.input_jsonl).stem
        rows_by_cap = recompute_loo_threshold_caps(args.input_jsonl, args.caps, repo_root=args.repo_root)
        summaries: dict[float, dict[str, Any]] = {}
        output_dir = Path(args.output_dir)
        for cap, rows in rows_by_cap.items():
            label = _cap_label(cap)
            jsonl_path = output_dir / f"{output_prefix}_prratio{label}_recomputed.jsonl"
            summary_path = output_dir / f"{output_prefix}_prratio{label}_recomputed.summary.json"
            _write_jsonl(jsonl_path, rows)
            summary = _summary_for_cap(args.input_jsonl, rows, cap, args.min_seeds_for_promotion)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            summaries[cap] = summary
        aggregate = _aggregate_summaries(summaries)
        if args.aggregate_json:
            aggregate_path = Path(args.aggregate_json)
            aggregate_path.parent.mkdir(parents=True, exist_ok=True)
            aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.aggregate_markdown:
            aggregate_markdown_path = Path(args.aggregate_markdown)
            aggregate_markdown_path.parent.mkdir(parents=True, exist_ok=True)
            aggregate_markdown_path.write_text(render_aggregate_markdown(aggregate), encoding="utf-8")
        print("AGGREGATE_JSON " + json.dumps(aggregate, sort_keys=True))
        return 0 if all(int(summary.get("folds_failed") or 0) == 0 for summary in summaries.values()) else 1

    if args.cap is None:
        parser.error("one of --cap or --caps is required")
    if args.output_jsonl is None or args.summary_json is None:
        parser.error("--output-jsonl and --summary-json are required with --cap")

    rows = recompute_loo_threshold_cap(args.input_jsonl, args.cap, repo_root=args.repo_root)
    output = Path(args.output_jsonl)
    _write_jsonl(output, rows)
    summary = _summary_for_cap(args.input_jsonl, rows, args.cap, args.min_seeds_for_promotion)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SUMMARY_JSON " + json.dumps(summary, sort_keys=True))
    return 0 if summary.get("folds_failed") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

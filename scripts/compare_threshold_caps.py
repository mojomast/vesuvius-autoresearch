#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_metrics(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"metrics JSON is not an object: {path}")
    return data


def _load_threshold_rows(path: str | Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            try:
                rows.append({
                    "threshold": float(raw["threshold"]),
                    "precision": float(raw["precision"]),
                    "recall": float(raw["recall"]),
                    "f05": float(raw["f05"]),
                    "f1": float(raw["f1"]),
                    "pred_positive_rate": float(raw["pred_positive_rate"]),
                })
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid threshold row in {path}: {raw}") from exc
    return rows


def _row_summary(row: dict[str, float] | None, val_positive_rate: float) -> dict[str, float] | None:
    if row is None:
        return None
    return {
        "threshold": float(row["threshold"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f05": float(row["f05"]),
        "f1": float(row["f1"]),
        "pred_positive_rate": float(row["pred_positive_rate"]),
        "pred_to_val_ratio": float(row["pred_positive_rate"] / max(val_positive_rate, 1e-12)),
    }


def _best_row(rows: list[dict[str, float]]) -> dict[str, float] | None:
    return max(rows, key=lambda row: (row["f1"], row["precision"], -row["pred_positive_rate"])) if rows else None


def _selected_summary(metrics: dict[str, Any], rows: list[dict[str, float]], val_positive_rate: float) -> dict[str, float] | None:
    risk = metrics.get("threshold_risk_summary") if isinstance(metrics.get("threshold_risk_summary"), dict) else {}
    selected = risk.get("selected") if isinstance(risk.get("selected"), dict) else None
    if selected:
        return {key: float(selected[key]) for key in ("threshold", "precision", "recall", "f05", "f1", "pred_positive_rate", "pred_to_val_ratio") if key in selected}
    best_threshold = metrics.get("best_threshold")
    if best_threshold is not None:
        best_threshold = float(best_threshold)
        closest = min(rows, key=lambda row: abs(float(row["threshold"]) - best_threshold), default=None)
        return _row_summary(closest, val_positive_rate)
    return _row_summary(_best_row(rows), val_positive_rate)


def compare_threshold_caps(
    metrics_path: str | Path,
    threshold_csv: str | Path | None = None,
    *,
    caps: list[float] | None = None,
    min_retained_f1: float = 0.95,
    min_retained_f05: float | None = None,
    baseline_cap: float | None = 2.0,
) -> dict[str, Any]:
    metrics_path = Path(metrics_path)
    threshold_csv = Path(threshold_csv) if threshold_csv is not None else metrics_path.with_name("metrics_by_threshold.csv")
    metrics = _load_metrics(metrics_path)
    rows = _load_threshold_rows(threshold_csv)
    val_positive_rate = float(metrics.get("val_positive_rate") or metrics.get("label_positive_rate") or 0.0)
    if val_positive_rate <= 0.0:
        raise ValueError("metrics must include positive val_positive_rate or label_positive_rate")
    caps = sorted(float(cap) for cap in (caps or [2.0, 2.5, 3.0, 3.5]))
    selected = _selected_summary(metrics, rows, val_positive_rate)
    if not selected or float(selected.get("f1") or 0.0) <= 0.0:
        raise ValueError("could not resolve selected threshold with positive F1")
    selected_f1 = max(float(selected["f1"]), 1e-12)
    min_retained_f05 = min_retained_f1 if min_retained_f05 is None else float(min_retained_f05)
    selected_f05 = max(float(selected["f05"]), 1e-12)
    cap_rows: list[dict[str, Any]] = []
    for cap in caps:
        eligible = [row for row in rows if row["pred_positive_rate"] / max(val_positive_rate, 1e-12) <= cap]
        best = _row_summary(_best_row(eligible), val_positive_rate)
        cap_rows.append({
            "cap": cap,
            "eligible_threshold_count": len(eligible),
            "best": best,
            "f1_retained_vs_selected": (float(best["f1"]) / selected_f1) if best else None,
            "f05_retained_vs_selected": (float(best["f05"]) / selected_f05) if best else None,
        })
    recommended = next((row for row in cap_rows if _passes_retention(row, min_retained_f1, min_retained_f05)), None)
    baseline = next((row for row in cap_rows if baseline_cap is not None and abs(row["cap"] - baseline_cap) < 1e-9), None)
    return {
        "metrics_path": str(metrics_path),
        "threshold_csv": str(threshold_csv),
        "val_positive_rate": val_positive_rate,
        "selected": selected,
        "caps": cap_rows,
        "baseline_cap": baseline_cap,
        "recommended_cap": recommended,
        "improvement_vs_baseline_cap": _improvement(recommended, baseline) if recommended and baseline else None,
        "min_retained_f1": min_retained_f1,
        "min_retained_f05": min_retained_f05,
    }


def _passes_retention(row: dict[str, Any], min_retained_f1: float, min_retained_f05: float) -> bool:
    f1_retained = row.get("f1_retained_vs_selected")
    f05_retained = row.get("f05_retained_vs_selected")
    return (
        f1_retained is not None
        and f05_retained is not None
        and float(f1_retained) >= min_retained_f1
        and float(f05_retained) >= min_retained_f05
    )


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _metrics_paths_from_loo_jsonl(input_jsonl: str | Path, repo_root: str | Path | None = None) -> list[Path]:
    root = Path(repo_root).resolve() if repo_root is not None else ROOT
    paths: list[Path] = []
    seen: set[Path] = set()
    with Path(input_jsonl).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            artifact = row.get("artifact_dir")
            if not artifact:
                continue
            artifact_path = Path(str(artifact))
            if not artifact_path.is_absolute():
                artifact_path = root / artifact_path
            metrics_path = artifact_path / "metrics.json"
            if metrics_path.exists() and metrics_path not in seen:
                seen.add(metrics_path)
                paths.append(metrics_path)
    return paths


def _aggregate_cap_reports(reports: list[dict[str, Any]], min_retained_f1: float, min_retained_f05: float) -> dict[str, Any]:
    caps = sorted({float(row["cap"]) for report in reports for row in report.get("caps") or []})
    aggregate: dict[str, Any] = {}
    for cap in caps:
        cap_rows = []
        for report in reports:
            match = next((row for row in report.get("caps") or [] if abs(float(row["cap"]) - cap) < 1e-9), None)
            if match:
                cap_rows.append(match)
        best_rows = [row.get("best") for row in cap_rows if isinstance(row.get("best"), dict)]
        f1_values = [float(row["f1"]) for row in best_rows]
        f05_values = [float(row["f05"]) for row in best_rows]
        precision_values = [float(row["precision"]) for row in best_rows]
        recall_values = [float(row["recall"]) for row in best_rows]
        ratio_values = [float(row["pred_to_val_ratio"]) for row in best_rows]
        retained_values = [float(row["f1_retained_vs_selected"]) for row in cap_rows if row.get("f1_retained_vs_selected") is not None]
        retained_f05_values = [float(row["f05_retained_vs_selected"]) for row in cap_rows if row.get("f05_retained_vs_selected") is not None]
        aggregate[str(cap)] = {
            "cap": cap,
            "metrics_count": len(cap_rows),
            "eligible_count": len(best_rows),
            "mean_f1": (sum(f1_values) / len(f1_values)) if f1_values else None,
            "median_f1": _median(f1_values),
            "min_f1": min(f1_values) if f1_values else None,
            "mean_f05": (sum(f05_values) / len(f05_values)) if f05_values else None,
            "mean_precision": (sum(precision_values) / len(precision_values)) if precision_values else None,
            "mean_recall": (sum(recall_values) / len(recall_values)) if recall_values else None,
            "mean_pred_to_val_ratio": (sum(ratio_values) / len(ratio_values)) if ratio_values else None,
            "mean_f1_retained_vs_selected": (sum(retained_values) / len(retained_values)) if retained_values else None,
            "min_f1_retained_vs_selected": min(retained_values) if retained_values else None,
            "mean_f05_retained_vs_selected": (sum(retained_f05_values) / len(retained_f05_values)) if retained_f05_values else None,
            "min_f05_retained_vs_selected": min(retained_f05_values) if retained_f05_values else None,
            "passes_retention_gate": (
                len(retained_values) == len(reports)
                and len(retained_f05_values) == len(reports)
                and all(value >= min_retained_f1 for value in retained_values)
                and all(value >= min_retained_f05 for value in retained_f05_values)
            ),
        }
    passing = [row for row in aggregate.values() if row.get("passes_retention_gate")]
    recommended = min(passing, key=lambda row: float(row["cap"]), default=None)
    best_mean_f1 = max(aggregate.values(), key=lambda row: (float(row.get("mean_f1") or float("-inf")), -float(row["cap"])), default=None)
    return {
        "caps": caps,
        "per_cap": aggregate,
        "recommended_cap": recommended,
        "best_cap_by_mean_f1": best_mean_f1,
    }


def compare_threshold_caps_batch(
    metrics_paths: list[str | Path],
    *,
    caps: list[float] | None = None,
    min_retained_f1: float = 0.95,
    min_retained_f05: float | None = None,
    baseline_cap: float | None = 2.0,
) -> dict[str, Any]:
    min_retained_f05 = min_retained_f1 if min_retained_f05 is None else float(min_retained_f05)
    reports = [compare_threshold_caps(path, caps=caps, min_retained_f1=min_retained_f1, min_retained_f05=min_retained_f05, baseline_cap=baseline_cap) for path in metrics_paths]
    return {
        "metrics_count": len(reports),
        "metrics_paths": [str(path) for path in metrics_paths],
        "per_artifact": reports,
        "aggregate": _aggregate_cap_reports(reports, min_retained_f1, min_retained_f05),
        "min_retained_f1": min_retained_f1,
        "min_retained_f05": min_retained_f05,
        "baseline_cap": baseline_cap,
    }


def _improvement(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float] | None:
    candidate_best = candidate.get("best") if isinstance(candidate.get("best"), dict) else None
    baseline_best = baseline.get("best") if isinstance(baseline.get("best"), dict) else None
    if not candidate_best or not baseline_best:
        return None
    return {
        "cap_delta": float(candidate["cap"]) - float(baseline["cap"]),
        "f1_delta": float(candidate_best["f1"]) - float(baseline_best["f1"]),
        "f05_delta": float(candidate_best["f05"]) - float(baseline_best["f05"]),
        "recall_delta": float(candidate_best["recall"]) - float(baseline_best["recall"]),
        "precision_delta": float(candidate_best["precision"]) - float(baseline_best["precision"]),
        "pred_to_val_ratio_delta": float(candidate_best["pred_to_val_ratio"]) - float(baseline_best["pred_to_val_ratio"]),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    selected = report["selected"]
    lines = [
        "# Threshold Cap Comparison",
        "",
        f"- Metrics: `{report['metrics_path']}`",
        f"- Selected cap ratio: {_fmt(selected.get('pred_to_val_ratio'))}x; F1 {_fmt(selected.get('f1'))}; F0.5 {_fmt(selected.get('f05'))}",
    ]
    recommended = report.get("recommended_cap") or {}
    if recommended:
        best = recommended.get("best") or {}
        lines.append(f"- Recommended strict cap: {recommended.get('cap')}x; F1 {_fmt(best.get('f1'))}; ratio {_fmt(best.get('pred_to_val_ratio'))}x")
    improvement = report.get("improvement_vs_baseline_cap")
    if improvement:
        lines.append(f"- Improvement vs {report.get('baseline_cap')}x: F1 {improvement['f1_delta']:+.6f}; F0.5 {improvement['f05_delta']:+.6f}; recall {improvement['recall_delta']:+.6f}; precision {improvement['precision_delta']:+.6f}")
    lines.append(f"- Retention gate: F1 >= {_fmt(report.get('min_retained_f1'))}; F0.5 >= {_fmt(report.get('min_retained_f05'))}")
    lines.extend(["", "## Caps", "", "| cap | eligible | F1 | F0.5 | precision | recall | pred/val | F1 retained | F0.5 retained |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in report["caps"]:
        best = row.get("best") or {}
        lines.append("| " + " | ".join([
            _fmt(row.get("cap")),
            _fmt(row.get("eligible_threshold_count")),
            _fmt(best.get("f1")),
            _fmt(best.get("f05")),
            _fmt(best.get("precision")),
            _fmt(best.get("recall")),
            _fmt(best.get("pred_to_val_ratio")),
            _fmt(row.get("f1_retained_vs_selected")),
            _fmt(row.get("f05_retained_vs_selected")),
        ]) + " |")
    return "\n".join(lines) + "\n"


def render_batch_markdown(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    recommended = aggregate.get("recommended_cap") or {}
    best = aggregate.get("best_cap_by_mean_f1") or {}
    lines = [
        "# Threshold Cap Batch Comparison",
        "",
        f"- Metrics files: {report['metrics_count']}",
        f"- Retention gate: F1 >= {_fmt(report.get('min_retained_f1'))}; F0.5 >= {_fmt(report.get('min_retained_f05'))}",
        f"- Recommended strict cap: {_fmt(recommended.get('cap'))}",
        f"- Best cap by mean F1: {_fmt(best.get('cap'))}",
        "",
        "## Aggregate Caps",
        "",
        "| cap | mean F1 | median F1 | min F1 | mean recall | mean pred/val | min F1 retained | min F0.5 retained | passes retention |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cap in aggregate.get("caps") or []:
        row = aggregate["per_cap"][str(cap)]
        lines.append("| " + " | ".join([
            _fmt(row.get("cap")),
            _fmt(row.get("mean_f1")),
            _fmt(row.get("median_f1")),
            _fmt(row.get("min_f1")),
            _fmt(row.get("mean_recall")),
            _fmt(row.get("mean_pred_to_val_ratio")),
            _fmt(row.get("min_f1_retained_vs_selected")),
            _fmt(row.get("min_f05_retained_vs_selected")),
            _fmt(row.get("passes_retention_gate")),
        ]) + " |")
    return "\n".join(lines) + "\n"


def _parse_caps(value: str) -> list[float]:
    caps = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not caps:
        raise argparse.ArgumentTypeError("at least one cap is required")
    return caps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare threshold positive-rate caps from existing metrics_by_threshold.csv without retraining")
    parser.add_argument("--metrics", action="append", default=[], help="metrics.json path; repeat for batch mode")
    parser.add_argument("--loo-jsonl", default=None, help="Optional LOO JSONL to resolve artifact_dir/metrics.json paths for batch mode")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--threshold-csv", default=None, help="metrics_by_threshold.csv path; defaults beside metrics.json")
    parser.add_argument("--caps", type=_parse_caps, default=[2.0, 2.5, 3.0, 3.5], help="comma-separated cap values")
    parser.add_argument("--min-retained-f1", type=float, default=0.95, help="minimum retained F1 fraction for recommended strict cap")
    parser.add_argument("--min-retained-f05", type=float, default=None, help="minimum retained F0.5 fraction for recommended strict cap; defaults to --min-retained-f1")
    parser.add_argument("--baseline-cap", type=float, default=2.0, help="cap used for improvement deltas")
    parser.add_argument("--markdown", action="store_true", help="emit Markdown instead of JSON")
    args = parser.parse_args(argv)

    metrics_paths = [Path(path) for path in args.metrics]
    if args.loo_jsonl:
        metrics_paths.extend(_metrics_paths_from_loo_jsonl(args.loo_jsonl, args.repo_root))
    if not metrics_paths:
        parser.error("at least one --metrics or --loo-jsonl is required")
    if len(metrics_paths) > 1 or args.loo_jsonl:
        if args.threshold_csv is not None:
            parser.error("--threshold-csv is only supported for single --metrics mode")
        report = compare_threshold_caps_batch(metrics_paths, caps=args.caps, min_retained_f1=args.min_retained_f1, min_retained_f05=args.min_retained_f05, baseline_cap=args.baseline_cap)
        if args.markdown:
            print(render_batch_markdown(report), end="")
        else:
            print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    report = compare_threshold_caps(metrics_paths[0], args.threshold_csv, caps=args.caps, min_retained_f1=args.min_retained_f1, min_retained_f05=args.min_retained_f05, baseline_cap=args.baseline_cap)
    if args.markdown:
        print(render_markdown(report), end="")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

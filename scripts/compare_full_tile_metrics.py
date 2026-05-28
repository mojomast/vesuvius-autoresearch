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


CORE_METRICS = [
    "val_f1",
    "val_f05",
    "average_precision",
    "ap_prevalence_lift",
    "precision",
    "recall",
    "val_positive_rate",
    "pred_positive_rate",
    "pred_to_val_ratio",
    "best_threshold",
    "fixed_threshold_f1",
    "brier_score",
    "expected_calibration_error",
    "prob_mean",
    "prob_p95",
    "prob_max",
]

SELECTED_METRICS = {
    "selected_threshold": "threshold",
    "selected_precision": "precision",
    "selected_recall": "recall",
    "selected_f05": "f05",
    "selected_f1": "f1",
    "selected_pred_positive_rate": "pred_positive_rate",
    "selected_pred_to_val_ratio": "pred_to_val_ratio",
}

LOWER_IS_BETTER = {"brier_score", "expected_calibration_error", "pred_to_val_ratio", "selected_pred_to_val_ratio"}
HIGHER_IS_BETTER = {"val_f1", "val_f05", "average_precision", "ap_prevalence_lift", "fixed_threshold_f1", "selected_f1", "selected_f05"}
CORE_EVIDENCE_METRICS = ["val_f1", "val_f05", "average_precision", "ap_prevalence_lift", "pred_to_val_ratio", "brier_score", "expected_calibration_error"]


def _load_metrics(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"metrics JSON is not an object: {path}")
    return data


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _segment_id(metrics: dict[str, Any]) -> str | None:
    segment = metrics.get("segment")
    if isinstance(segment, dict):
        for key in ("id", "segment_id", "name"):
            value = segment.get(key)
            if value is not None:
                return str(value)
    if segment is not None and not isinstance(segment, (list, dict)):
        return str(segment)
    evaluation_region = metrics.get("evaluation_region")
    if isinstance(evaluation_region, dict):
        for key in ("segment_id", "segment", "id"):
            value = evaluation_region.get(key)
            if value is not None:
                return str(value)
    model_meta = metrics.get("model_meta")
    if isinstance(model_meta, dict):
        value = model_meta.get("heldout_segment_id") or model_meta.get("heldout_segment")
        if value is not None:
            return str(value)
    return None


def _evaluation_type(metrics: dict[str, Any]) -> str | None:
    evaluation_region = metrics.get("evaluation_region")
    if isinstance(evaluation_region, dict):
        value = evaluation_region.get("type") or evaluation_region.get("region_type")
        return str(value) if value is not None else None
    return None


def _selected(metrics: dict[str, Any]) -> dict[str, Any]:
    risk = metrics.get("threshold_risk_summary")
    if not isinstance(risk, dict):
        return {}
    selected = risk.get("selected")
    return selected if isinstance(selected, dict) else {}


def _metric_values(metrics: dict[str, Any]) -> dict[str, Any]:
    values = {key: _number(metrics.get(key)) for key in CORE_METRICS if key != "pred_to_val_ratio"}
    val_positive_rate = _number(metrics.get("val_positive_rate") or metrics.get("label_positive_rate"))
    pred_positive_rate = _number(metrics.get("pred_positive_rate"))
    if val_positive_rate and pred_positive_rate is not None:
        values["pred_to_val_ratio"] = pred_positive_rate / max(val_positive_rate, 1e-12)
    else:
        values["pred_to_val_ratio"] = None

    selected = _selected(metrics)
    for output_key, input_key in SELECTED_METRICS.items():
        values[output_key] = _number(selected.get(input_key))
    values["fixed_threshold_status"] = metrics.get("fixed_threshold_status")
    return values


def _metric_delta(candidate: Any, baseline: Any) -> float | None:
    candidate_value = _number(candidate)
    baseline_value = _number(baseline)
    if candidate_value is None or baseline_value is None:
        return None
    return candidate_value - baseline_value


def _comparison_rows(baseline_values: dict[str, Any], candidate_values: dict[str, Any]) -> list[dict[str, Any]]:
    keys = [*CORE_METRICS, *SELECTED_METRICS.keys()]
    rows = []
    for key in keys:
        baseline_value = baseline_values.get(key)
        candidate_value = candidate_values.get(key)
        if baseline_value is None and candidate_value is None:
            continue
        rows.append({
            "metric": key,
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": _metric_delta(candidate_value, baseline_value),
        })
    return rows


def _warnings(baseline: dict[str, Any], candidate: dict[str, Any], *, require_whole_segment: bool) -> list[str]:
    warnings: list[str] = []
    baseline_segment = _segment_id(baseline)
    candidate_segment = _segment_id(candidate)
    if baseline_segment and candidate_segment and baseline_segment != candidate_segment:
        warnings.append(f"segment mismatch: baseline={baseline_segment}, candidate={candidate_segment}")

    baseline_type = _evaluation_type(baseline)
    candidate_type = _evaluation_type(candidate)
    for label, value in (("baseline", baseline_type), ("candidate", candidate_type)):
        if require_whole_segment and value != "whole_segment":
            warnings.append(f"{label} evaluation_region.type is not whole_segment: {value}")
        elif value and value != "whole_segment":
            warnings.append(f"{label} evaluation_region.type is {value}")
    return warnings


def _verdict(rows: list[dict[str, Any]]) -> str:
    improved = 0
    regressed = 0
    for row in rows:
        metric = row["metric"]
        delta = row.get("delta")
        if delta is None or metric not in HIGHER_IS_BETTER | LOWER_IS_BETTER:
            continue
        if metric in LOWER_IS_BETTER:
            delta = -delta
        if delta > 0.0:
            improved += 1
        elif delta < 0.0:
            regressed += 1
    if improved and not regressed:
        return "candidate_improves_tracked_metrics"
    if regressed and not improved:
        return "candidate_regresses_tracked_metrics"
    return "mixed"


def _delta(report: dict[str, Any], metric: str) -> float | None:
    value = (report.get("deltas") or {}).get(metric)
    return _number(value)


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _core_evidence_status(report: dict[str, Any], *, max_pred_to_val_ratio_delta: float) -> str:
    f1_delta = _delta(report, "val_f1")
    f05_delta = _delta(report, "val_f05")
    ap_delta = _delta(report, "average_precision")
    ratio_delta = _delta(report, "pred_to_val_ratio")
    if f1_delta is None or f05_delta is None or ap_delta is None:
        return "incomplete"
    if f1_delta > 0.0 and f05_delta >= 0.0 and ap_delta > 0.0 and (ratio_delta is None or ratio_delta <= max_pred_to_val_ratio_delta):
        return "core_improved"
    if f1_delta < 0.0 or f05_delta < 0.0 or ap_delta < 0.0 or (ratio_delta is not None and ratio_delta > max_pred_to_val_ratio_delta):
        return "core_regressed"
    return "mixed"


def summarize_comparisons(reports: list[dict[str, Any]], *, max_pred_to_val_ratio_delta: float = 0.01) -> dict[str, Any]:
    statuses = [
        {
            "label": report.get("label"),
            "status": _core_evidence_status(report, max_pred_to_val_ratio_delta=max_pred_to_val_ratio_delta),
            "val_f1_delta": _delta(report, "val_f1"),
            "val_f05_delta": _delta(report, "val_f05"),
            "average_precision_delta": _delta(report, "average_precision"),
            "pred_to_val_ratio_delta": _delta(report, "pred_to_val_ratio"),
        }
        for report in reports
    ]
    status_counts = {status: sum(1 for item in statuses if item["status"] == status) for status in ("core_improved", "core_regressed", "mixed", "incomplete")}
    mean_deltas = {
        metric: _mean([value for report in reports if (value := _delta(report, metric)) is not None])
        for metric in CORE_EVIDENCE_METRICS
    }
    warnings = [warning for report in reports for warning in (report.get("warnings") or [])]
    if reports and status_counts["core_improved"] == len(reports):
        verdict = "candidate_improves_core_metrics_all_pairs"
    elif reports and status_counts["core_regressed"] == len(reports):
        verdict = "candidate_regresses_core_metrics_all_pairs"
    elif status_counts["core_improved"] and status_counts["core_regressed"]:
        verdict = "mixed_core_evidence"
    else:
        verdict = "incomplete_or_mixed_core_evidence"
    return {
        "pair_count": len(reports),
        "labels": [report.get("label") for report in reports],
        "max_pred_to_val_ratio_delta": max_pred_to_val_ratio_delta,
        "status_counts": status_counts,
        "per_pair_status": statuses,
        "mean_deltas": mean_deltas,
        "warning_count": len(warnings),
        "warnings": warnings,
        "verdict": verdict,
    }


def compare_full_tile_metrics(
    baseline_path: str | Path,
    candidate_path: str | Path,
    *,
    label: str | None = None,
    require_whole_segment: bool = False,
) -> dict[str, Any]:
    baseline_path = Path(baseline_path)
    candidate_path = Path(candidate_path)
    baseline = _load_metrics(baseline_path)
    candidate = _load_metrics(candidate_path)
    baseline_values = _metric_values(baseline)
    candidate_values = _metric_values(candidate)
    rows = _comparison_rows(baseline_values, candidate_values)
    return {
        "label": label or _segment_id(candidate) or _segment_id(baseline) or candidate_path.parent.name,
        "baseline_path": str(baseline_path),
        "candidate_path": str(candidate_path),
        "baseline_segment": _segment_id(baseline),
        "candidate_segment": _segment_id(candidate),
        "baseline_evaluation_type": _evaluation_type(baseline),
        "candidate_evaluation_type": _evaluation_type(candidate),
        "baseline_fixed_threshold_status": baseline_values.get("fixed_threshold_status"),
        "candidate_fixed_threshold_status": candidate_values.get("fixed_threshold_status"),
        "metrics": rows,
        "deltas": {row["metric"]: row["delta"] for row in rows if row.get("delta") is not None},
        "warnings": _warnings(baseline, candidate, require_whole_segment=require_whole_segment),
        "verdict": _verdict(rows),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Full-Tile Metrics Comparison",
        "",
        f"- Label: `{report['label']}`",
        f"- Baseline: `{report['baseline_path']}`",
        f"- Candidate: `{report['candidate_path']}`",
        f"- Verdict: `{report['verdict']}`",
    ]
    if report.get("core_evidence_status"):
        lines.append(f"- Core evidence status: `{report['core_evidence_status']}`")
    if report.get("warnings"):
        lines.append(f"- Warnings: {'; '.join(report['warnings'])}")
    lines.extend([
        "",
        "| metric | baseline | candidate | delta |",
        "|---|---:|---:|---:|",
    ])
    for row in report["metrics"]:
        lines.append("| " + " | ".join([
            str(row["metric"]),
            _fmt(row.get("baseline")),
            _fmt(row.get("candidate")),
            _fmt(row.get("delta")),
        ]) + " |")
    return "\n".join(lines) + "\n"


def render_batch_markdown(package: dict[str, Any]) -> str:
    summary = package["summary"]
    lines = [
        "# Full-Tile Evidence Package",
        "",
        f"- Pairs: {summary['pair_count']}",
        f"- Verdict: `{summary['verdict']}`",
        f"- Core-improved pairs: {summary['status_counts'].get('core_improved', 0)}",
        f"- Core-regressed pairs: {summary['status_counts'].get('core_regressed', 0)}",
        f"- Max allowed pred/val delta for core-improved status: {summary['max_pred_to_val_ratio_delta']}",
    ]
    if summary.get("warnings"):
        lines.append(f"- Warnings: {'; '.join(summary['warnings'])}")
    lines.extend([
        "",
        "## Aggregate Deltas",
        "",
        "| metric | mean delta |",
        "|---|---:|",
    ])
    for metric, value in summary["mean_deltas"].items():
        lines.append(f"| {metric} | {_fmt(value)} |")
    lines.extend([
        "",
        "## Pair Status",
        "",
        "| label | status | F1 delta | F0.5 delta | AP delta | pred/val delta |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in summary["per_pair_status"]:
        lines.append("| " + " | ".join([
            str(row.get("label")),
            str(row.get("status")),
            _fmt(row.get("val_f1_delta")),
            _fmt(row.get("val_f05_delta")),
            _fmt(row.get("average_precision_delta")),
            _fmt(row.get("pred_to_val_ratio_delta")),
        ]) + " |")
    for report in package["comparisons"]:
        lines.extend(["", render_markdown(report).rstrip()])
    return "\n".join(lines) + "\n"


def _parse_pair(value: str) -> tuple[str | None, Path, Path]:
    label = None
    pair = value
    if "=" in value:
        label, pair = value.split("=", 1)
        label = label.strip() or None
    if "," not in pair:
        raise argparse.ArgumentTypeError("--pair must be LABEL=baseline.json,candidate.json or baseline.json,candidate.json")
    baseline, candidate = pair.split(",", 1)
    return label, Path(baseline), Path(candidate)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare existing full-tile baseline and candidate metrics.json artifacts")
    parser.add_argument("--baseline", default=None, help="baseline metrics.json path")
    parser.add_argument("--candidate", default=None, help="candidate metrics.json path, such as an ensemble metrics artifact")
    parser.add_argument("--ensemble", default=None, help="alias for --candidate")
    parser.add_argument("--label", default=None, help="label for single-pair output")
    parser.add_argument("--pair", action="append", type=_parse_pair, default=[], help="repeatable LABEL=baseline.json,candidate.json pair")
    parser.add_argument("--require-whole-segment", action="store_true", help="return a non-zero exit code when compared artifacts are not whole-segment metrics")
    parser.add_argument("--max-pred-to-val-ratio-delta", type=float, default=0.01, help="maximum pred/val ratio increase allowed for core-improved aggregate status")
    parser.add_argument("--fail-on-core-regression", action="store_true", help="return non-zero when any pair regresses F1, F0.5, AP, or pred/val ratio beyond the allowed delta")
    parser.add_argument("--markdown", action="store_true", help="emit Markdown instead of JSON")
    args = parser.parse_args(argv)

    candidate = args.candidate or args.ensemble
    pairs = list(args.pair)
    if args.baseline or candidate:
        if not args.baseline or not candidate:
            parser.error("--baseline and --candidate/--ensemble must be provided together")
        pairs.append((args.label, Path(args.baseline), Path(candidate)))
    if not pairs:
        parser.error("provide --baseline/--candidate or at least one --pair")

    reports = [
        compare_full_tile_metrics(baseline, candidate, label=label, require_whole_segment=args.require_whole_segment)
        for label, baseline, candidate in pairs
    ]
    summary = summarize_comparisons(reports, max_pred_to_val_ratio_delta=args.max_pred_to_val_ratio_delta)
    for report, status in zip(reports, summary["per_pair_status"], strict=False):
        report["core_evidence_status"] = status["status"]
    package = {"summary": summary, "comparisons": reports}
    output: dict[str, Any] = reports[0] if len(reports) == 1 else package
    if args.markdown:
        if len(reports) == 1:
            print(render_markdown(reports[0]), end="")
        else:
            print(render_batch_markdown(package), end="")
    else:
        print(json.dumps(output, indent=2, sort_keys=True))
    if args.require_whole_segment and any(report.get("warnings") for report in reports):
        return 1
    if args.fail_on_core_regression and package["summary"]["status_counts"].get("core_regressed", 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

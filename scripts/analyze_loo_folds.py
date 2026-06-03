#!/usr/bin/env python3
"""Summarize leave-one-out fold diagnostics from summary JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


LOW_F1_THRESHOLD = 0.04


def _companion_jsonl(summary_path: Path) -> Path:
    text = str(summary_path)
    if text.endswith(".summary.json"):
        return Path(text[: -len(".summary.json")] + ".jsonl")
    return summary_path.with_suffix(".jsonl")


def _row_id(fold_id: str, seed: Any) -> str:
    return f"{fold_id}:seed={seed}" if seed not in (None, "", "pooled") else fold_id


def _flag_set(summary: dict[str, Any], key: str) -> set[str]:
    return {str(item) for item in summary.get(key, []) if item is not None}


def _summary_cap(summary: dict[str, Any]) -> float | None:
    base_config = summary.get("base_config")
    if not base_config:
        return None
    path = Path(str(base_config)).expanduser()
    if not path.exists():
        return None
    try:
        cfg = json.loads(path.read_text()) if path.suffix == ".json" else yaml.safe_load(path.read_text())
    except (OSError, json.JSONDecodeError, yaml.YAMLError):
        return None
    if not isinstance(cfg, dict):
        return None
    return _float(cfg.get("evaluation", {}).get("max_pred_positive_rate_ratio"))


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 4) -> str:
    number = _float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def _diagnostic_flags(row: dict[str, Any]) -> str:
    flags = []
    if row.get("low_f1"):
        flags.append("low_f1")
    if row.get("over_cap"):
        flags.append("over_cap")
    if row.get("zero_precision_or_recall"):
        flags.append("zero_precision_or_recall")
    if row.get("positive_rate_alarm"):
        flags.append("positive_rate_alarm")
    if row.get("fixed_threshold_not_ok"):
        flags.append("fixed_threshold_not_ok")
    return ", ".join(flags)


def _enrich_row(summary_path: Path, summary: dict[str, Any], raw: dict[str, Any], source_kind: str) -> dict[str, Any]:
    fold_id = str(raw.get("heldout_segment") or raw.get("fold_id") or raw.get("fold") or "")
    seed = raw.get("seed", "pooled")
    val_f1 = _float(raw.get("val_f1"))
    ap = _float(raw.get("average_precision"))
    pred_rate = _float(raw.get("pred_positive_rate"))
    val_rate = _float(raw.get("val_positive_rate"))
    cap = _float(raw.get("max_pred_positive_rate_ratio"))
    if cap is None:
        cap = _summary_cap(summary)
    pred_ratio = pred_rate / val_rate if pred_rate is not None and val_rate and val_rate > 0 else None
    rid = _row_id(fold_id, seed)
    zero_ids = _flag_set(summary, "folds_with_zero_precision_or_recall")
    alarm_ids = _flag_set(summary, "folds_with_positive_rate_alarm")
    fixed_ids = _flag_set(summary, "folds_with_fixed_threshold_not_ok")
    precision = _float(raw.get("precision"))
    recall = _float(raw.get("recall"))
    return {
        "source": summary_path.name,
        "source_kind": source_kind,
        "fold_id": fold_id,
        "seed": seed,
        "val_f1": val_f1,
        "average_precision": ap,
        "pred_positive_rate": pred_rate,
        "val_positive_rate": val_rate,
        "pred_val_ratio": pred_ratio,
        "cap": cap,
        "fixed_threshold_status": raw.get("fixed_threshold_status", ""),
        "fixed_threshold_failure_reason": raw.get("fixed_threshold_failure_reason", ""),
        "selected_threshold_reason": raw.get("selected_threshold_reason", ""),
        "threshold_selection": raw.get("threshold_selection", ""),
        "ap_prevalence_lift": _float(raw.get("ap_prevalence_lift")),
        "prob_mean": _float(raw.get("prob_mean")),
        "prob_p95": _float(raw.get("prob_p95")),
        "prob_max": _float(raw.get("prob_max")),
        "low_f1": val_f1 is not None and val_f1 < LOW_F1_THRESHOLD,
        "over_cap": pred_ratio is not None and cap is not None and pred_ratio > cap,
        "zero_precision_or_recall": rid in zero_ids or precision == 0.0 or recall == 0.0,
        "positive_rate_alarm": rid in alarm_ids,
        "fixed_threshold_not_ok": rid in fixed_ids,
    }


def load_diagnostics(summary_path: Path) -> list[dict[str, Any]]:
    summary = json.loads(summary_path.read_text())
    jsonl = _companion_jsonl(summary_path)
    rows: list[dict[str, Any]] = []
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                rows.append(_enrich_row(summary_path, summary, json.loads(line), "seed"))
        return rows

    per_fold_f1 = summary.get("per_fold_val_f1", {})
    per_fold_ap = summary.get("per_fold_average_precision", {})
    for fold_id in sorted(per_fold_f1):
        raw = {
            "fold_id": fold_id,
            "seed": "pooled",
            "val_f1": per_fold_f1.get(fold_id),
            "average_precision": per_fold_ap.get(fold_id),
        }
        rows.append(_enrich_row(summary_path, summary, raw, "pooled"))
    return rows


def summarize_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    fixed_bad = [row for row in rows if row.get("fixed_threshold_not_ok")]
    alarms = [row for row in rows if row.get("positive_rate_alarm")]
    zero = [row for row in rows if row.get("zero_precision_or_recall")]
    hard = [row for row in rows if str(row.get("fold_id")) == "20230530172803"]
    reason_counts: dict[str, int] = {}
    for row in fixed_bad:
        reason = str(row.get("fixed_threshold_failure_reason") or row.get("fixed_threshold_status") or "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    next_action = "review linked LOO diagnostics"
    if hard:
        mean_ap = sum(float(row.get("average_precision") or 0.0) for row in hard) / max(len(hard), 1)
        mean_lift_values = [float(row["ap_prevalence_lift"]) for row in hard if row.get("ap_prevalence_lift") is not None]
        mean_lift = sum(mean_lift_values) / max(len(mean_lift_values), 1) if mean_lift_values else None
        if mean_ap <= 0.03 or (mean_lift is not None and mean_lift < 1.5):
            next_action = "audit hard-fold labels/sampling before threshold tuning"
    if fixed_bad and len(fixed_bad) == total:
        next_action = "diagnose fixed-threshold calibration across all LOO rows"
    return {
        "total_rows": total,
        "fixed_threshold_not_ok_count": len(fixed_bad),
        "positive_rate_alarm_count": len(alarms),
        "zero_precision_or_recall_count": len(zero),
        "fixed_threshold_failure_reason_counts": reason_counts,
        "hard_fold_rows": hard,
        "next_action": next_action,
    }


def render_markdown(rows: list[dict[str, Any]]) -> str:
    summary = summarize_diagnostics(rows)
    lines = [
        "## LOO Diagnostic Summary",
        "",
        f"- Rows: {summary['total_rows']}",
        f"- Fixed-threshold not OK: {summary['fixed_threshold_not_ok_count']}/{summary['total_rows']}",
        f"- Positive-rate alarms: {summary['positive_rate_alarm_count']}/{summary['total_rows']}",
        f"- Zero precision/recall rows: {summary['zero_precision_or_recall_count']}/{summary['total_rows']}",
        f"- Next action: {summary['next_action']}",
        "",
    ]
    if summary["fixed_threshold_failure_reason_counts"]:
        reasons = ", ".join(f"`{key}`={value}" for key, value in sorted(summary["fixed_threshold_failure_reason_counts"].items()))
        lines.extend([f"- Fixed-threshold reasons: {reasons}", ""])
    if summary["hard_fold_rows"]:
        hard = summary["hard_fold_rows"]
        mean_f1 = sum(float(row.get("val_f1") or 0.0) for row in hard) / max(len(hard), 1)
        mean_ap = sum(float(row.get("average_precision") or 0.0) for row in hard) / max(len(hard), 1)
        lines.extend([f"- Hard fold `20230530172803`: rows={len(hard)}, mean F1={mean_f1:.4f}, mean AP={mean_ap:.4f}", ""])
    header = [
        "source",
        "fold_id",
        "seed",
        "val_f1",
        "AP",
        "pred_rate",
        "val_rate",
        "pred/val",
        "cap",
        "fixed",
        "fixed_reason",
        "AP/prevalence",
        "flags",
    ]
    lines.extend(["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"])
    for row in rows:
        lines.append("| " + " | ".join([
            str(row["source"]),
            str(row["fold_id"]),
            str(row["seed"]),
            _fmt(row.get("val_f1")),
            _fmt(row.get("average_precision")),
            _fmt(row.get("pred_positive_rate")),
            _fmt(row.get("val_positive_rate")),
            _fmt(row.get("pred_val_ratio"), 2),
            _fmt(row.get("cap"), 2),
            str(row.get("fixed_threshold_status") or ""),
            str(row.get("fixed_threshold_failure_reason") or ""),
            _fmt(row.get("ap_prevalence_lift"), 2),
            _diagnostic_flags(row),
        ]) + " |")
    return "\n".join(lines)


def render_text(rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{row['source']} {row['fold_id']} seed={row['seed']} val_f1={_fmt(row.get('val_f1'))} "
        f"AP={_fmt(row.get('average_precision'))} pred/val={_fmt(row.get('pred_val_ratio'), 2)} flags={_diagnostic_flags(row)}"
        for row in rows
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze LOO fold diagnostics from summary JSON files")
    parser.add_argument("--summary-json", action="append", required=True, help="LOO summary JSON path; repeat for multiple summaries")
    parser.add_argument("--markdown", action="store_true", help="Print a Markdown table")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for raw_path in args.summary_json:
        rows.extend(load_diagnostics(Path(raw_path).expanduser()))
    rows.sort(key=lambda row: (str(row["source"]), str(row["fold_id"]), str(row["seed"])))
    print(render_markdown(rows) if args.markdown else render_text(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

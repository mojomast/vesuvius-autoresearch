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


def render_markdown(rows: list[dict[str, Any]]) -> str:
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
        "flags",
    ]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
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

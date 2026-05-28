#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


METRIC_KEYS = [
    "folds_successful",
    "folds_failed",
    "mean_val_f1",
    "median_over_seeds_median_val_f1",
    "mean_average_precision",
    "median_average_precision",
    "worst_fold_id",
    "worst_fold_val_f1",
    "worst_seed_median_val_f1",
    "mean_recomputed_pred_to_val_ratio",
    "max_recomputed_pred_to_val_ratio",
]

WARNING_KEYS = [
    "folds_with_positive_rate_alarm",
    "folds_with_zero_precision_or_recall",
    "folds_with_fixed_threshold_not_ok",
    "folds_with_threshold_edge_case",
    "folds_with_weak_ap_prevalence_lift",
    "promotion_warnings",
]


def _load_summary(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"summary is not a JSON object: {path}")
    return data


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem.replace(".summary", ""), path
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"missing baseline name in {value!r}")
    return name, Path(raw_path)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _selected_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: summary.get(key) for key in METRIC_KEYS}


def _warning_counts(summary: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in WARNING_KEYS:
        value = summary.get(key) or []
        counts[key] = len(value) if isinstance(value, list) else 0
    return counts


def _metric_deltas(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key in METRIC_KEYS:
        candidate_value = _number(candidate.get(key))
        baseline_value = _number(baseline.get(key))
        if candidate_value is not None and baseline_value is not None:
            deltas[key] = candidate_value - baseline_value
    return deltas


def _per_fold_deltas(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    candidate_folds = candidate.get("per_fold_val_f1") or {}
    baseline_folds = baseline.get("per_fold_val_f1") or {}
    if not isinstance(candidate_folds, dict) or not isinstance(baseline_folds, dict):
        return {}
    deltas: dict[str, float] = {}
    for fold_id in sorted(set(candidate_folds) & set(baseline_folds)):
        candidate_value = _number(candidate_folds.get(fold_id))
        baseline_value = _number(baseline_folds.get(fold_id))
        if candidate_value is not None and baseline_value is not None:
            deltas[str(fold_id)] = candidate_value - baseline_value
    return deltas


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _sign_test_two_sided_p(wins: int, losses: int) -> float | None:
    trials = wins + losses
    if trials == 0:
        return None
    low_tail = sum(math.comb(trials, idx) for idx in range(0, min(wins, losses) + 1)) / (2 ** trials)
    return min(1.0, 2.0 * low_tail)


def _paired_delta_summary(candidate: dict[str, Any], baseline: dict[str, Any], key: str, label: str) -> dict[str, Any]:
    candidate_map = candidate.get(key) or {}
    baseline_map = baseline.get(key) or {}
    if not isinstance(candidate_map, dict) or not isinstance(baseline_map, dict):
        return {"metric": label, "paired_count": 0, "common_ids": [], "candidate_only_ids": [], "baseline_only_ids": []}
    common = sorted(set(candidate_map) & set(baseline_map))
    deltas: list[float] = []
    for item_id in common:
        candidate_value = _number(candidate_map.get(item_id))
        baseline_value = _number(baseline_map.get(item_id))
        if candidate_value is not None and baseline_value is not None:
            deltas.append(candidate_value - baseline_value)
    wins = sum(1 for value in deltas if value > 0.0)
    losses = sum(1 for value in deltas if value < 0.0)
    ties = sum(1 for value in deltas if value == 0.0)
    return {
        "metric": label,
        "paired_count": len(deltas),
        "common_ids": common,
        "candidate_only_ids": sorted(set(candidate_map) - set(baseline_map)),
        "baseline_only_ids": sorted(set(baseline_map) - set(candidate_map)),
        "mean_delta": (sum(deltas) / len(deltas)) if deltas else None,
        "median_delta": _median(deltas),
        "min_delta": min(deltas) if deltas else None,
        "max_delta": max(deltas) if deltas else None,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "sign_test_two_sided_p": _sign_test_two_sided_p(wins, losses),
    }


def _paired_comparisons(candidate: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    if baseline is None:
        return {}
    return {
        "per_fold_val_f1": _paired_delta_summary(candidate, baseline, "per_fold_val_f1", "per-fold F1"),
        "per_fold_average_precision": _paired_delta_summary(candidate, baseline, "per_fold_average_precision", "per-fold AP"),
        "per_seed_median_val_f1": _paired_delta_summary(candidate, baseline, "per_seed_median_val_f1", "per-seed median F1"),
    }


def _gate(label: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"label": label, "passed": bool(passed), "detail": detail}


def _build_gates(
    candidate: dict[str, Any],
    *,
    min_median_over_seeds_f1: float | None = None,
    min_mean_ap: float | None = None,
    min_worst_fold_f1: float | None = None,
    max_fixed_threshold_not_ok: int | None = None,
) -> list[dict[str, Any]]:
    gates = [
        _gate("all folds succeeded", int(candidate.get("folds_failed") or 0) == 0, f"folds_failed={candidate.get('folds_failed')}"),
        _gate(
            "no positive-rate alarms",
            not candidate.get("folds_with_positive_rate_alarm"),
            f"count={len(candidate.get('folds_with_positive_rate_alarm') or [])}",
        ),
        _gate(
            "no zero precision/recall alarms",
            not candidate.get("folds_with_zero_precision_or_recall"),
            f"count={len(candidate.get('folds_with_zero_precision_or_recall') or [])}",
        ),
    ]
    if min_median_over_seeds_f1 is not None:
        value = _number(candidate.get("median_over_seeds_median_val_f1"))
        gates.append(
            _gate(
                "median-over-seeds F1 floor",
                value is not None and value >= min_median_over_seeds_f1,
                f"value={value}, floor={min_median_over_seeds_f1}",
            )
        )
    if min_mean_ap is not None:
        value = _number(candidate.get("mean_average_precision"))
        gates.append(_gate("mean AP floor", value is not None and value >= min_mean_ap, f"value={value}, floor={min_mean_ap}"))
    if min_worst_fold_f1 is not None:
        value = _number(candidate.get("worst_fold_val_f1"))
        gates.append(_gate("worst-fold F1 floor", value is not None and value >= min_worst_fold_f1, f"value={value}, floor={min_worst_fold_f1}"))
    if max_fixed_threshold_not_ok is not None:
        count = len(candidate.get("folds_with_fixed_threshold_not_ok") or [])
        gates.append(_gate("fixed-threshold warning cap", count <= max_fixed_threshold_not_ok, f"count={count}, cap={max_fixed_threshold_not_ok}"))
    return gates


def compare_summaries(
    candidate_path: str | Path,
    baselines: dict[str, str | Path],
    *,
    min_median_over_seeds_f1: float | None = None,
    min_mean_ap: float | None = None,
    min_worst_fold_f1: float | None = None,
    max_fixed_threshold_not_ok: int | None = None,
) -> dict[str, Any]:
    candidate = _load_summary(candidate_path)
    loaded_baselines = {name: _load_summary(path) for name, path in baselines.items()}
    primary_name = next(iter(loaded_baselines), None)
    primary = loaded_baselines[primary_name] if primary_name is not None else None
    gates = _build_gates(
        candidate,
        min_median_over_seeds_f1=min_median_over_seeds_f1,
        min_mean_ap=min_mean_ap,
        min_worst_fold_f1=min_worst_fold_f1,
        max_fixed_threshold_not_ok=max_fixed_threshold_not_ok,
    )
    return {
        "candidate_path": str(candidate_path),
        "baselines": {name: {"path": str(path), "metrics": _selected_metrics(loaded_baselines[name]), "warnings": _warning_counts(loaded_baselines[name])} for name, path in baselines.items()},
        "candidate": {"metrics": _selected_metrics(candidate), "warnings": _warning_counts(candidate)},
        "deltas_vs_primary_baseline": _metric_deltas(candidate, primary) if primary is not None else {},
        "per_fold_f1_deltas_vs_primary_baseline": _per_fold_deltas(candidate, primary) if primary is not None else {},
        "paired_comparisons_vs_primary_baseline": _paired_comparisons(candidate, primary),
        "primary_baseline": primary_name,
        "gates": gates,
        "passed": all(gate["passed"] for gate in gates),
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    names = list(report["baselines"].keys()) + ["candidate"]
    sources = {name: report["baselines"][name]["metrics"] for name in report["baselines"]}
    sources["candidate"] = report["candidate"]["metrics"]
    lines = ["# LOO Summary Comparison", "", f"- Primary baseline: `{report.get('primary_baseline')}`", f"- Passed gates: {report['passed']}", "", "## Metrics", "", "| metric | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for key in METRIC_KEYS:
        lines.append("| " + key + " | " + " | ".join(_fmt(sources[name].get(key)) for name in names) + " |")
    if report.get("deltas_vs_primary_baseline"):
        lines.extend(["", "## Candidate Deltas", "", "| metric | delta |", "|---|---:|"])
        for key, value in report["deltas_vs_primary_baseline"].items():
            lines.append(f"| {key} | {_fmt(value)} |")
    if report.get("per_fold_f1_deltas_vs_primary_baseline"):
        lines.extend(["", "## Per-Fold F1 Deltas", "", "| fold | delta |", "|---|---:|"])
        for fold_id, value in report["per_fold_f1_deltas_vs_primary_baseline"].items():
            lines.append(f"| {fold_id} | {_fmt(value)} |")
    paired = report.get("paired_comparisons_vs_primary_baseline") or {}
    if paired:
        lines.extend(["", "## Paired Comparisons", "", "| metric | pairs | mean delta | median delta | wins | losses | ties | sign-test p |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for item in paired.values():
            lines.append("| " + " | ".join([
                _fmt(item.get("metric")),
                _fmt(item.get("paired_count")),
                _fmt(item.get("mean_delta")),
                _fmt(item.get("median_delta")),
                _fmt(item.get("wins")),
                _fmt(item.get("losses")),
                _fmt(item.get("ties")),
                _fmt(item.get("sign_test_two_sided_p")),
            ]) + " |")
    lines.extend(["", "## Gates", "", "| gate | pass | detail |", "|---|---:|---|"])
    for gate in report["gates"]:
        lines.append(f"| {gate['label']} | {gate['passed']} | {gate['detail']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare seed-repeat LOO summary JSON files with explicit promotion gates")
    parser.add_argument("--candidate", required=True, help="Candidate LOO summary JSON")
    parser.add_argument("--baseline", action="append", default=[], help="Baseline as name=path. First baseline is used for deltas")
    parser.add_argument("--min-median-over-seeds-f1", type=float, default=None)
    parser.add_argument("--min-mean-ap", type=float, default=None)
    parser.add_argument("--min-worst-fold-f1", type=float, default=None)
    parser.add_argument("--max-fixed-threshold-not-ok", type=int, default=None)
    parser.add_argument("--markdown", action="store_true", help="Emit Markdown instead of JSON")
    parser.add_argument("--fail-on-gate", action="store_true", help="Exit non-zero when any gate fails")
    args = parser.parse_args(argv)

    baselines = dict(_parse_named_path(value) for value in args.baseline)
    report = compare_summaries(
        args.candidate,
        baselines,
        min_median_over_seeds_f1=args.min_median_over_seeds_f1,
        min_mean_ap=args.min_mean_ap,
        min_worst_fold_f1=args.min_worst_fold_f1,
        max_fixed_threshold_not_ok=args.max_fixed_threshold_not_ok,
    )
    if args.markdown:
        print(render_markdown(report), end="")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.fail_on_gate and not report["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

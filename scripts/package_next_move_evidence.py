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


from research_dashboard.mining import build_hard_negative_plan  # noqa: E402
from research_dashboard.snapshot import build_snapshot  # noqa: E402
from scripts.compare_full_tile_metrics import compare_full_tile_metrics, render_batch_markdown as render_full_tile_batch_markdown, render_markdown as render_full_tile_markdown, summarize_comparisons  # noqa: E402
from scripts.compare_threshold_caps import compare_threshold_caps, render_markdown as render_cap_markdown  # noqa: E402


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON is not an object: {path}")
    return data


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


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


def _snapshot_next_action(snapshot: dict[str, Any]) -> str | None:
    research_summary = snapshot.get("research_summary") if isinstance(snapshot.get("research_summary"), dict) else {}
    decision = research_summary.get("decision") if isinstance(research_summary.get("decision"), dict) else {}
    action = decision.get("next_action")
    return str(action) if action is not None else None


def _candidate_evidence(snapshot: dict[str, Any]) -> dict[str, Any]:
    research_summary = snapshot.get("research_summary") if isinstance(snapshot.get("research_summary"), dict) else {}
    evidence = research_summary.get("candidate_evidence")
    return evidence if isinstance(evidence, dict) else {}


def _promotion_gate(snapshot: dict[str, Any]) -> dict[str, Any]:
    research_summary = snapshot.get("research_summary") if isinstance(snapshot.get("research_summary"), dict) else {}
    decision = research_summary.get("decision") if isinstance(research_summary.get("decision"), dict) else {}
    gate = decision.get("promotion_gate")
    return gate if isinstance(gate, dict) else {}


def _blocker_counts(snapshot: dict[str, Any]) -> dict[str, int]:
    research_summary = snapshot.get("research_summary") if isinstance(snapshot.get("research_summary"), dict) else {}
    decision = research_summary.get("decision") if isinstance(research_summary.get("decision"), dict) else {}
    blockers = decision.get("blocker_counts") if isinstance(decision.get("blocker_counts"), dict) else {}
    counts: dict[str, int] = {}
    for code, count in blockers.items():
        try:
            counts[str(code)] = int(count)
        except (TypeError, ValueError):
            continue
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _cap_reports(root: Path, plan: dict[str, Any], caps: list[float], min_retained_f1: float, min_retained_f05: float) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for command in plan.get("cap_comparison_commands") or []:
        if not isinstance(command, dict) or command.get("writes_artifacts"):
            continue
        metrics_json = command.get("metrics_json")
        if not metrics_json:
            continue
        try:
            report = compare_threshold_caps(
                _resolve(root, str(metrics_json)),
                caps=caps,
                min_retained_f1=min_retained_f1,
                min_retained_f05=min_retained_f05,
                baseline_cap=None,
            )
            report["segment_id"] = command.get("segment_id")
            report["command_id"] = command.get("id")
            reports.append(report)
        except Exception as exc:
            reports.append({"segment_id": command.get("segment_id"), "command_id": command.get("id"), "metrics_path": str(metrics_json), "error": str(exc)})
    return reports


def _cap_recommendation_result(cap_reports: list[dict[str, Any]]) -> dict[str, Any]:
    recommendations = []
    blockers = []
    for report in cap_reports:
        if report.get("error"):
            blockers.append({
                "segment_id": report.get("segment_id"),
                "metrics_path": report.get("metrics_path"),
                "reason": "cap_comparison_error",
                "error": report.get("error"),
            })
            continue
        recommended = report.get("recommended_cap") if isinstance(report.get("recommended_cap"), dict) else None
        if not recommended or recommended.get("cap") is None:
            blockers.append({
                "segment_id": report.get("segment_id"),
                "metrics_path": report.get("metrics_path"),
                "reason": "no_cap_preserves_retention",
            })
            continue
        best = recommended.get("best") if isinstance(recommended.get("best"), dict) else {}
        recommendations.append(
            {
                "segment_id": report.get("segment_id"),
                "metrics_path": report.get("metrics_path"),
                "recommended_cap": recommended.get("cap"),
                "f1": best.get("f1"),
                "f05": best.get("f05"),
                "pred_to_val_ratio": best.get("pred_to_val_ratio"),
            }
        )
    if blockers or not recommendations:
        return {"recommendation": None, "blockers": blockers}
    caps = [float(item["recommended_cap"]) for item in recommendations if item.get("recommended_cap") is not None]
    return {"recommendation": {
        "target_max_pred_positive_rate_ratio": max(caps) if caps else None,
        "segments": recommendations,
        "reason": "strictest per-artifact caps preserve requested F1/F0.5 retention; use the loosest per-artifact recommendation as a shared policy cap",
    }, "blockers": []}


def synthesize_next_move_decision(package: dict[str, Any]) -> dict[str, Any]:
    full_tile_summary = package.get("full_tile_comparison_summary") if isinstance(package.get("full_tile_comparison_summary"), dict) else None
    if full_tile_summary:
        regressed = [item for item in full_tile_summary.get("per_pair_status") or [] if isinstance(item, dict) and item.get("status") == "core_regressed"]
        if regressed:
            return {
                "action": "review_full_tile_regression",
                "readiness": "needs_review_before_claiming_lift",
                "writes_artifacts": False,
                "reason": "one or more full-tile candidate comparisons regress F1, F0.5, AP, or positive-rate safety",
                "segments": [item.get("label") for item in regressed],
                "supporting_evidence": ["full_tile_comparison_summary"],
            }

    cap_recommendation = package.get("cap_recommendation") if isinstance(package.get("cap_recommendation"), dict) else None
    if cap_recommendation and cap_recommendation.get("target_max_pred_positive_rate_ratio") is not None:
        return {
            "action": "tighten_positive_rate_cap",
            "readiness": "ready_from_existing_evidence",
            "writes_artifacts": False,
            "target_max_pred_positive_rate_ratio": cap_recommendation.get("target_max_pred_positive_rate_ratio"),
            "reason": cap_recommendation.get("reason"),
            "segments": [item.get("segment_id") for item in cap_recommendation.get("segments") or []],
            "supporting_evidence": ["cap_comparisons", "hard_negative_plan"],
        }

    plan = package.get("hard_negative_plan") if isinstance(package.get("hard_negative_plan"), dict) else {}
    mine_decision = next((item for item in plan.get("calibration_mining_decisions") or [] if isinstance(item, dict) and item.get("action") == "mine_hard_negatives"), None)
    if mine_decision:
        mine_command = next((item for item in plan.get("mine_commands") or [] if isinstance(item, dict) and str(item.get("segment_id")) == str(mine_decision.get("segment_id"))), None)
        return {
            "action": "review_fold_safe_hard_negative_mining",
            "readiness": "review_before_artifact_writing",
            "writes_artifacts": False,
            "requires_review": True,
            "reason": mine_decision.get("reason") or "planner recommends mining after cap review",
            "segment_id": mine_decision.get("segment_id"),
            "command_text": mine_command.get("command_text") if mine_command else None,
            "command_writes_artifacts": mine_command.get("writes_artifacts") if mine_command else None,
            "safe_to_execute_from_dashboard": mine_command.get("safe_to_execute_from_dashboard") if mine_command else None,
            "artifact_policy": plan.get("artifact_policy"),
            "eligible_extra_train_npzs": plan.get("eligible_extra_train_npzs") or [],
            "rejected_extra_train_npzs": plan.get("rejected_extra_train_npzs") or [],
            "fold_safe_extra_train_npzs_by_heldout": plan.get("fold_safe_extra_train_npzs_by_heldout") or {},
            "supporting_evidence": ["hard_negative_plan"],
        }

    gate = package.get("promotion_gate") if isinstance(package.get("promotion_gate"), dict) else {}
    if gate.get("ready") is True:
        return {
            "action": "promotion_review",
            "readiness": "ready_from_existing_evidence",
            "writes_artifacts": False,
            "reason": "promotion gate is ready and no higher-priority cap, mining, or full-tile regression action was found",
            "supporting_evidence": ["promotion_gate"],
        }

    return {
        "action": "follow_dashboard_next_action",
        "readiness": "needs_more_evidence",
        "writes_artifacts": False,
        "reason": package.get("next_action") or "no synthesized action available",
        "supporting_evidence": ["dashboard_next_action"],
    }


def build_next_move_evidence_package(
    repo_root: str | Path | None = None,
    *,
    snapshot_json: str | Path | None = None,
    pairs: list[tuple[str | None, Path, Path]] | None = None,
    caps: list[float] | None = None,
    min_retained_f1: float = 0.95,
    min_retained_f05: float = 0.95,
    max_commands: int = 6,
    ratio_threshold: float = 2.0,
    target_config: str = "configs/robust_hard_negative_prratio2p0_followup.yaml",
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve() if repo_root else ROOT
    snapshot = _read_json(snapshot_json) if snapshot_json else build_snapshot(root)
    plan = build_hard_negative_plan(root, snapshot, max_commands=max_commands, ratio_threshold=ratio_threshold, target_config=target_config)
    cap_reports = _cap_reports(root, plan, caps or [2.0, 2.5, 3.0, 3.5], min_retained_f1, min_retained_f05)
    full_tile_reports = [
        compare_full_tile_metrics(_resolve(root, baseline), _resolve(root, candidate), label=label)
        for label, baseline, candidate in (pairs or [])
    ]
    full_tile_summary = summarize_comparisons(full_tile_reports) if full_tile_reports else None
    cap_result = _cap_recommendation_result(cap_reports)
    package = {
        "schema": "vesuvius.next_move_evidence_package.v1",
        "repo_root": str(root),
        "writes_artifacts": False,
        "next_action": _snapshot_next_action(snapshot),
        "promotion_gate": _promotion_gate(snapshot),
        "promotion_blockers": _blocker_counts(snapshot),
        "candidate_evidence": _candidate_evidence(snapshot),
        "hard_negative_plan": plan,
        "cap_comparisons": cap_reports,
        "cap_recommendation": cap_result["recommendation"],
        "cap_recommendation_blockers": cap_result["blockers"],
        "full_tile_comparison_summary": full_tile_summary,
        "full_tile_comparisons": full_tile_reports,
    }
    package["recommended_next_move"] = synthesize_next_move_decision(package)
    return package


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_markdown(package: dict[str, Any]) -> str:
    gate = package.get("promotion_gate") if isinstance(package.get("promotion_gate"), dict) else {}
    evidence = package.get("candidate_evidence") if isinstance(package.get("candidate_evidence"), dict) else {}
    plan = package.get("hard_negative_plan") if isinstance(package.get("hard_negative_plan"), dict) else {}
    recommended = package.get("recommended_next_move") if isinstance(package.get("recommended_next_move"), dict) else {}
    lines = [
        "# Next-Move Evidence Package",
        "",
        f"- Repo root: `{package['repo_root']}`",
        f"- Writes artifacts: {package['writes_artifacts']}",
        f"- Next action: {package.get('next_action') or 'unknown'}",
        f"- Promotion gate ready: {gate.get('ready')}",
        f"- Candidate: `{evidence.get('candidate_run_id')}`",
        f"- Planner held-out segment: `{plan.get('heldout_segment')}`",
        f"- Planner mine commands: {len(plan.get('mine_commands') or [])}",
        f"- Planner cap comparison commands: {len(plan.get('cap_comparison_commands') or [])}",
    ]
    if recommended:
        lines.extend([
            "",
            "## Recommended Next Move",
            "",
            f"- Action: `{recommended.get('action')}`",
            f"- Readiness: `{recommended.get('readiness')}`",
            f"- Writes artifacts: {recommended.get('writes_artifacts')}",
        ])
        if recommended.get("target_max_pred_positive_rate_ratio") is not None:
            lines.append(f"- Target cap: {recommended.get('target_max_pred_positive_rate_ratio')}x")
        if recommended.get("segment_id"):
            lines.append(f"- Segment: `{recommended.get('segment_id')}`")
        if recommended.get("segments"):
            lines.append(f"- Segments: {', '.join(str(item) for item in recommended.get('segments') or [])}")
        if recommended.get("reason"):
            lines.append(f"- Reason: {recommended.get('reason')}")
        if recommended.get("command_text"):
            lines.append(f"- Command: `{recommended.get('command_text')}`")
        if recommended.get("command_writes_artifacts") is not None:
            lines.append(f"- Command writes artifacts: {recommended.get('command_writes_artifacts')}")
        if recommended.get("safe_to_execute_from_dashboard") is not None:
            lines.append(f"- Safe to execute from dashboard: {recommended.get('safe_to_execute_from_dashboard')}")
        if recommended.get("artifact_policy"):
            lines.append(f"- Artifact policy: {recommended.get('artifact_policy')}")
    blockers = package.get("promotion_blockers") if isinstance(package.get("promotion_blockers"), dict) else {}
    if blockers:
        lines.extend(["", "## Promotion Blockers", "", "| blocker | count |", "|---|---:|"])
        for code, count in list(blockers.items())[:10]:
            lines.append(f"| `{code}` | {count} |")
    decisions = [item for item in plan.get("calibration_mining_decisions") or [] if isinstance(item, dict)]
    if decisions:
        lines.extend(["", "## Calibration/Mining Decisions", "", "| segment | action | target cap | reason |", "|---|---|---:|---|"])
        for decision in decisions:
            lines.append("| " + " | ".join([
                str(decision.get("segment_id")),
                str(decision.get("action")),
                _fmt(decision.get("target_max_pred_positive_rate_ratio")),
                str(decision.get("reason")),
            ]) + " |")
    cap_blockers = package.get("cap_recommendation_blockers") or []
    if cap_blockers:
        lines.extend(["", "## Cap Recommendation Blockers", "", "| segment | reason | detail |", "|---|---|---|"])
        for blocker in cap_blockers:
            lines.append("| " + " | ".join([
                str(blocker.get("segment_id")),
                str(blocker.get("reason")),
                str(blocker.get("error") or blocker.get("metrics_path") or ""),
            ]) + " |")
    cap_reports = package.get("cap_comparisons") or []
    if cap_reports:
        lines.extend(["", "## Cap Comparisons"])
        for report in cap_reports:
            if report.get("error"):
                lines.extend(["", f"- Segment `{report.get('segment_id')}` cap comparison error: {report.get('error')}"])
            else:
                lines.extend(["", render_cap_markdown(report).rstrip()])
    full_tile_reports = package.get("full_tile_comparisons") or []
    full_tile_summary = package.get("full_tile_comparison_summary")
    if full_tile_reports and full_tile_summary:
        lines.extend(["", "## Full-Tile Comparisons", "", render_full_tile_batch_markdown({"summary": full_tile_summary, "comparisons": full_tile_reports}).rstrip()])
    elif full_tile_reports:
        lines.extend(["", "## Full-Tile Comparisons"])
        for report in full_tile_reports:
            lines.extend(["", render_full_tile_markdown(report).rstrip()])
    return "\n".join(lines) + "\n"


def _parse_caps(value: str) -> list[float]:
    caps = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not caps:
        raise argparse.ArgumentTypeError("at least one cap is required")
    return caps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a read-only next-move evidence package from existing research artifacts")
    parser.add_argument("--repo-root", default=str(ROOT), help="repository root")
    parser.add_argument("--snapshot-json", default=None, help="optional dashboard snapshot JSON to read instead of building one")
    parser.add_argument("--pair", action="append", type=_parse_pair, default=[], help="optional LABEL=baseline.json,candidate.json full-tile comparison pair")
    parser.add_argument("--caps", type=_parse_caps, default=[2.0, 2.5, 3.0, 3.5], help="comma-separated caps for planner-suggested cap comparisons")
    parser.add_argument("--min-retained-f1", type=float, default=0.95)
    parser.add_argument("--min-retained-f05", type=float, default=0.95)
    parser.add_argument("--max-commands", type=int, default=6)
    parser.add_argument("--ratio-threshold", type=float, default=2.0)
    parser.add_argument("--target-config", default="configs/robust_hard_negative_prratio2p0_followup.yaml")
    parser.add_argument("--markdown", action="store_true", help="emit Markdown instead of JSON")
    args = parser.parse_args(argv)

    package = build_next_move_evidence_package(
        args.repo_root,
        snapshot_json=args.snapshot_json,
        pairs=args.pair,
        caps=args.caps,
        min_retained_f1=args.min_retained_f1,
        min_retained_f05=args.min_retained_f05,
        max_commands=args.max_commands,
        ratio_threshold=args.ratio_threshold,
        target_config=args.target_config,
    )
    if args.markdown:
        print(render_markdown(package), end="")
    else:
        print(json.dumps(package, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

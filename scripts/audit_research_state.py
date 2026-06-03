#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _count_generated_configs(root: Path) -> int:
    config_dir = root / "configs"
    if not config_dir.exists():
        return 0
    return sum(1 for pattern in ("auto_*.yaml", "auto_*.yml", "auto_*.json") for _ in config_dir.glob(pattern))


def _experiment_db_summary(root: Path) -> dict[str, Any]:
    db_path = root / "experiments" / "experiments.db"
    summary: dict[str, Any] = {"path": "experiments/experiments.db", "exists": db_path.exists(), "run_count": None}
    if not db_path.exists():
        return summary
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            summary["run_count"] = int(conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0])
        finally:
            conn.close()
    except Exception as exc:
        summary["error"] = str(exc)
    return summary


def _artifact_summary(root: Path) -> dict[str, Any]:
    runs_root = root / "experiments" / "runs"
    summary = {"path": "experiments/runs", "exists": runs_root.exists(), "run_directory_count": 0, "total_bytes": 0}
    if not runs_root.exists():
        return summary
    summary["run_directory_count"] = sum(1 for path in runs_root.iterdir() if path.is_dir())
    total = 0
    for path in runs_root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    summary["total_bytes"] = total
    return summary


def _logs_summary(root: Path) -> dict[str, Any]:
    logs_root = root / "logs"
    count = len(list(logs_root.glob("**/*.summary.json"))) if logs_root.exists() else 0
    return {"path": "logs", "exists": logs_root.exists(), "summary_json_count": count}


def _top_blockers(blocker_counts: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
    items: list[tuple[str, int]] = []
    for code, count in blocker_counts.items():
        try:
            items.append((str(code), int(count)))
        except (TypeError, ValueError):
            continue
    return [{"code": code, "count": count} for code, count in sorted(items, key=lambda item: (-item[1], item[0]))[:limit]]


def _top_calibration_mining_action(mining: dict[str, Any]) -> dict[str, Any] | None:
    decisions = [item for item in mining.get("calibration_mining_decisions") or [] if isinstance(item, dict)]
    if not decisions:
        return None
    priority = {"tighten_positive_rate_cap": 0, "mine_hard_negatives": 1, "review_threshold_risk": 2}
    action = dict(min(decisions, key=lambda item: priority.get(str(item.get("action")), 99)))
    commands = [item for item in mining.get("mine_commands") or [] if isinstance(item, dict)]
    matching = next((item for item in commands if str(item.get("segment_id")) == str(action.get("segment_id"))), None)
    if matching is None and action.get("action") == "mine_hard_negatives" and commands:
        matching = commands[0]
    if matching is not None:
        action["command_text"] = matching.get("command_text")
        action["mine_output"] = matching.get("mine_output")
    cap_commands = [item for item in mining.get("cap_comparison_commands") or [] if isinstance(item, dict)]
    cap_matching = next((item for item in cap_commands if str(item.get("segment_id")) == str(action.get("segment_id"))), None)
    if cap_matching is None and cap_commands:
        cap_matching = cap_commands[0]
    if cap_matching is not None:
        action["cap_comparison_command_text"] = cap_matching.get("command_text")
    return action


def _meaningful_lineage(lineage: dict[str, Any]) -> dict[str, Any]:
    keys = ("proposal_id", "hypothesis_id", "mutation_family", "changed_path", "arm_id")
    return lineage if any(lineage.get(key) for key in keys) else {}


def _dashboard_summary(root: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"snapshot_contract_available": False}
    try:
        from research_dashboard.snapshot import build_snapshot

        snapshot = build_snapshot(root)
    except Exception as exc:
        summary["error"] = str(exc)
        return summary

    research_summary = snapshot.get("research_summary") if isinstance(snapshot, dict) else None
    decision = research_summary.get("decision", {}) if isinstance(research_summary, dict) else {}
    blocker_counts = decision.get("blocker_counts", {}) if isinstance(decision, dict) else {}
    promotion_gate = decision.get("promotion_gate", {}) if isinstance(decision, dict) else {}
    candidate_evidence = decision.get("candidate_evidence", {}) if isinstance(decision, dict) else {}
    mining = snapshot.get("mining", {}) if isinstance(snapshot.get("mining"), dict) else {}
    operations = snapshot.get("operations", {}) if isinstance(snapshot.get("operations"), dict) else {}
    no_progress = operations.get("no_progress", {}) if isinstance(operations.get("no_progress"), dict) else {}
    experiments = snapshot.get("experiments", {}) if isinstance(snapshot.get("experiments"), dict) else {}
    summary.update(
        {
            "snapshot_contract_available": True,
            "schema_version": snapshot.get("schema_version"),
            "champions": research_summary.get("champions", {}) if isinstance(research_summary, dict) else {},
            "research_summary": research_summary or {},
            "next_action": decision.get("next_action") if isinstance(decision, dict) else None,
            "promotion_gate_ready": promotion_gate.get("ready") if isinstance(promotion_gate, dict) else None,
            "promotion_gate_criteria": promotion_gate.get("criteria", []) if isinstance(promotion_gate, dict) else [],
            "candidate_evidence": candidate_evidence,
            "promotion_actions": decision.get("promotion_actions", []) if isinstance(decision, dict) else [],
            "top_promotion_blockers": _top_blockers(blocker_counts if isinstance(blocker_counts, dict) else {}),
            "mining": mining,
            "top_calibration_mining_action": _top_calibration_mining_action(mining),
            "hypotheses": research_summary.get("hypotheses", []) if isinstance(research_summary, dict) else [],
            "proposal_lineage": _meaningful_lineage(research_summary.get("proposal_lineage", {}) if isinstance(research_summary, dict) else {}),
            "staleness": decision.get("staleness", {}) if isinstance(decision, dict) else {},
            "latest_no_progress_cause": no_progress.get("latest"),
            "no_progress_cause_counts": no_progress.get("counts", {}),
            "evidence_packages": experiments.get("evidence_packages", []),
        }
    )
    return summary


def audit_research_state(repo_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve() if repo_root else ROOT
    return {
        "project_root": str(root),
        "generated_auto_config_count": _count_generated_configs(root),
        "experiment_db": _experiment_db_summary(root),
        "artifacts": _artifact_summary(root),
        "logs": _logs_summary(root),
        "dashboard": _dashboard_summary(root),
    }


def render_markdown(report: dict[str, Any]) -> str:
    db = report["experiment_db"]
    artifacts = report["artifacts"]
    logs = report["logs"]
    dashboard = report["dashboard"]
    lines = [
        "# Research State Audit",
        "",
        f"- Project root: `{report['project_root']}`",
        f"- Generated auto configs: {report['generated_auto_config_count']}",
        f"- Experiment DB runs: {db.get('run_count') if db.get('exists') else 'DB missing'}",
        f"- Artifact run directories: {artifacts['run_directory_count']}",
        f"- Artifact total bytes: {artifacts['total_bytes']}",
        f"- Log summary JSON files: {logs['summary_json_count']}",
        f"- Dashboard snapshot contract available: {dashboard['snapshot_contract_available']}",
    ]
    if dashboard.get("schema_version"):
        lines.append(f"- Dashboard schema version: `{dashboard['schema_version']}`")
    if dashboard.get("next_action"):
        lines.append(f"- Next action: {dashboard['next_action']}")
    if dashboard.get("promotion_gate_ready") is not None:
        lines.append(f"- Promotion gate ready: {dashboard['promotion_gate_ready']}")
    criteria = dashboard.get("promotion_gate_criteria") or []
    if criteria:
        lines.extend(["", "## Promotion Gate"])
        lines.extend(f"- {item.get('label', item.get('id', 'criterion'))}: {item.get('state')} - {item.get('detail', '')}" for item in criteria)
    evidence = dashboard.get("candidate_evidence") or {}
    if evidence:
        loo = evidence.get("loo") or {}
        weak = evidence.get("weak_fold_full_tile") or {}
        full = evidence.get("full_tile") or {}
        loo_full = evidence.get("loo_full_tile") or {}
        risk = evidence.get("risk_summary") or {}
        lines.extend(["", "## Candidate Evidence"])
        lines.append(f"- Candidate: `{evidence.get('candidate_run_id')}`")
        if loo.get("worst_fold_id"):
            lines.append(f"- Weak fold: `{loo.get('worst_fold_id')}` F1={loo.get('worst_fold_val_f1')}")
        lines.append(f"- Full-tile segments covered: {', '.join(str(item) for item in full.get('segments_covered') or []) or 'none'}")
        lines.append(f"- LOO full-tile diagnostics covered: {', '.join(str(item) for item in loo_full.get('segments_covered') or []) or 'none'}")
        if risk:
            warnings = risk.get("warnings") or []
            detail = f" - {warnings[0]}" if warnings else ""
            lines.append(f"- Positive-rate risk: {risk.get('risk_level', 'unknown')}{detail}")
        lines.append(f"- Weak-fold full-tile status: {weak.get('status')}")
        actions = evidence.get("promotion_actions") or dashboard.get("promotion_actions") or []
        if actions:
            first = actions[0]
            lines.append(f"- Next evidence action: {first.get('label')}")
            if first.get("command_text"):
                lines.append(f"- Command: `{first.get('command_text')}`")
    blockers = dashboard.get("top_promotion_blockers") or []
    if blockers:
        lines.extend(["", "## Top Promotion Blockers"])
        lines.extend(f"- `{item['code']}`: {item['count']}" for item in blockers)
    lineage = dashboard.get("proposal_lineage") or {}
    hypotheses = dashboard.get("hypotheses") or []
    if lineage or hypotheses:
        lines.extend(["", "## Hypothesis / Proposal Lineage"])
        if lineage:
            if lineage.get("proposal_id"):
                lines.append(f"- Proposal: `{lineage.get('proposal_id')}`")
            if lineage.get("hypothesis_id"):
                lines.append(f"- Hypothesis: `{lineage.get('hypothesis_id')}`")
            if lineage.get("mutation_family") or lineage.get("changed_path"):
                lines.append(f"- Family: `{lineage.get('mutation_family')}` changed `{lineage.get('changed_path')}`")
        if hypotheses:
            lines.append(f"- Recent hypotheses: {len(hypotheses)}")
    latest_cause = dashboard.get("latest_no_progress_cause") or {}
    cause_counts = dashboard.get("no_progress_cause_counts") or {}
    if latest_cause or cause_counts:
        lines.extend(["", "## Stale / No-progress Causes"])
        if latest_cause:
            lines.append(f"- Latest: `{latest_cause.get('code')}` - {latest_cause.get('label')}")
            if latest_cause.get("source"):
                lines.append(f"- Source: `{latest_cause.get('source')}`")
        if cause_counts:
            lines.append("- Counts: " + ", ".join(f"`{code}`={count}" for code, count in sorted(cause_counts.items())))
    packages = dashboard.get("evidence_packages") or []
    if packages:
        lines.extend(["", "## Evidence Packages"])
        for item in packages[:5]:
            path = item.get("relative_path") or item.get("path_json") or item.get("path_markdown") or item.get("path")
            lines.append(f"- `{path}`")
    top_action = dashboard.get("top_calibration_mining_action") or {}
    if top_action:
        lines.extend(["", "## Top Calibration/Mining/Cap Action"])
        lines.append(f"- Action: `{top_action.get('action')}`")
        if top_action.get("segment_id"):
            lines.append(f"- Segment: `{top_action.get('segment_id')}`")
        if top_action.get("target_max_pred_positive_rate_ratio") is not None:
            lines.append(f"- Target cap: {top_action.get('target_max_pred_positive_rate_ratio')}x")
        if top_action.get("reason"):
            lines.append(f"- Reason: {top_action.get('reason')}")
        if top_action.get("selected_pred_to_val_ratio") is not None:
            lines.append(f"- Selected pred/val ratio: {top_action.get('selected_pred_to_val_ratio')}")
        if top_action.get("f1_retained_fraction") is not None:
            lines.append(f"- F1 retained: {top_action.get('f1_retained_fraction')}")
        if top_action.get("f05_retained_fraction") is not None:
            lines.append(f"- F0.5 retained: {top_action.get('f05_retained_fraction')}")
        if top_action.get("command_text"):
            lines.append(f"- Command: `{top_action.get('command_text')}`")
        if top_action.get("cap_comparison_command_text"):
            lines.append(f"- Cap comparison: `{top_action.get('cap_comparison_command_text')}`")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit local Vesuvius AutoResearch state without modifying artifacts")
    parser.add_argument("--repo-root", default=None, help="Repository root to inspect; defaults to this script's project")
    parser.add_argument("--markdown", action="store_true", help="Emit a Markdown report instead of JSON")
    args = parser.parse_args(argv)

    report = audit_research_state(args.repo_root)
    if args.markdown:
        print(render_markdown(report), end="")
    else:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

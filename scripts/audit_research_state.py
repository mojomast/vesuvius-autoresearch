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
        lines.extend(["", "## Candidate Evidence"])
        lines.append(f"- Candidate: `{evidence.get('candidate_run_id')}`")
        if loo.get("worst_fold_id"):
            lines.append(f"- Weak fold: `{loo.get('worst_fold_id')}` F1={loo.get('worst_fold_val_f1')}")
        lines.append(f"- Full-tile segments covered: {', '.join(str(item) for item in full.get('segments_covered') or []) or 'none'}")
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

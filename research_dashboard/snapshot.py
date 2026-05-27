from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION
from .configs import load_configs
from .datasets import dataset_summary, fold_maps, prepared_datasets
from .experiments import load_experiments
from .inventory import build_inventory
from .operations import operations_snapshot
from .progress import build_progress


def resolve_project_root(raw: str | os.PathLike[str] | None = None) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    env = os.getenv("VESUVIUS_DASHBOARD_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def build_snapshot(project_root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    root = resolve_project_root(project_root)
    experiments = load_experiments(root)
    datasets = {"summary": dataset_summary(), "prepared": prepared_datasets(root), "fold_maps": fold_maps(root)}
    operations = operations_snapshot(root)
    progress = build_progress(experiments, datasets, operations)
    decision = experiments.get("decision", {}) if isinstance(experiments.get("decision"), dict) else {}
    research_summary = {
        "status": decision.get("status"),
        "next_action": decision.get("next_action"),
        "blocker_counts": decision.get("blocker_counts", {}),
        "plateau": decision.get("plateau", {}),
        "staleness": decision.get("staleness", {}),
        "candidate_evidence": decision.get("candidate_evidence", {}),
        "promotion_actions": decision.get("promotion_actions", []),
        "decision": decision,
        "champions": experiments.get("champions", {}),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": {"root": str(root), "exists": root.exists(), "python": ".venv/bin/python" if (root / ".venv" / "bin" / "python").exists() else "python3"},
        "inventory": build_inventory(root),
        "datasets": datasets,
        "configs": load_configs(root),
        "experiments": experiments,
        "progress": progress,
        "research_summary": research_summary,
        "operations": operations,
        "capabilities": {"enable_runs": os.getenv("VESUVIUS_DASHBOARD_ENABLE_RUNS") == "1", "artifact_preview": True, "standalone": True},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a Vesuvius dashboard JSON snapshot")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(build_snapshot(args.repo_root), indent=2 if args.pretty else None, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

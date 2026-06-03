from __future__ import annotations

import argparse
import copy
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION
from .configs import load_configs
from .datasets import dataset_summary, fold_maps, prepared_datasets
from .experiments import load_experiments
from .inventory import build_inventory
from .mining import build_hard_negative_plan
from .operations import operations_snapshot
from .progress import build_progress
from .settings import list_settings_snapshots, load_dashboard_settings, redact_dashboard_settings


def _harness_metadata() -> dict[str, Any]:
    """Return dashboard-safe metadata from the active autoresearch harness contract."""
    try:
        from autoresearch import METRIC_CONTRACT, PARAM_BOUNDS
        return {
            "type": "vesuvius",
            "metric_contract_keys": sorted(METRIC_CONTRACT.required_metrics),
            "param_bounds": {".".join(path): {"min": bounds[0], "max": bounds[1]} for path, bounds in PARAM_BOUNDS.items()},
        }
    except Exception as exc:
        return {"type": "vesuvius", "error": str(exc), "metric_contract_keys": [], "param_bounds": {}}


_SNAPSHOT_CACHE_TTL_SEC = 2.0
_snapshot_cache_lock = threading.RLock()
_snapshot_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def resolve_project_root(raw: str | os.PathLike[str] | None = None) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    env = os.getenv("VESUVIUS_DASHBOARD_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def reset_snapshot_cache() -> None:
    with _snapshot_cache_lock:
        _snapshot_cache.clear()


def _snapshot_cache_ttl_sec() -> float:
    raw = os.getenv("VESUVIUS_DASHBOARD_SNAPSHOT_TTL_SEC")
    if raw is None:
        return _SNAPSHOT_CACHE_TTL_SEC
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _SNAPSHOT_CACHE_TTL_SEC


def _build_snapshot_uncached(root: Path) -> dict[str, Any]:
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
        "hypotheses": experiments.get("hypotheses", []),
        "proposal_lineage": (experiments.get("latest") or {}).get("proposal_lineage", {}) if isinstance(experiments.get("latest"), dict) else {},
        "evidence_packages": experiments.get("evidence_packages", []),
        "ledger": experiments.get("ledger", {}),
    }
    contract_status = {
        "source_of_truth": True,
        "producer": "research_dashboard.snapshot.build_snapshot",
        "schema_version": SCHEMA_VERSION,
        "serving_endpoint": "/api/research",
        "candidate_linkage": "candidate-linked LOO/full-tile evidence only clears promotion gate",
        "conflict_fix_status": "dashboard_contract_authoritative",
    }
    research_summary["contract_status"] = contract_status
    partial_snapshot = {"research_summary": research_summary, "experiments": experiments, "datasets": datasets}
    mining = build_hard_negative_plan(root, partial_snapshot)
    harness = _harness_metadata()
    latest = experiments.get("latest") if isinstance(experiments.get("latest"), dict) else {}
    latest_config = latest.get("config", {}) if isinstance(latest.get("config"), dict) else {}
    dashboard_settings = redact_dashboard_settings(load_dashboard_settings(root))
    settings_values = dashboard_settings.get("values", {})
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_status": contract_status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": {"root": str(root), "exists": root.exists(), "python": ".venv/bin/python" if (root / ".venv" / "bin" / "python").exists() else "python3"},
        "inventory": build_inventory(root),
        "datasets": datasets,
        "configs": load_configs(root),
        "experiments": experiments,
        "progress": progress,
        "research_summary": research_summary,
        "harness": harness,
        "param_drift": {"current_config": latest_config, "bounds": harness.get("param_bounds", {})},
        "mining": mining,
        "operations": operations,
        "dashboard_settings": dashboard_settings,
        "settings_snapshots": list_settings_snapshots(root, limit=10),
        "capabilities": {
            "enable_runs": os.getenv("VESUVIUS_DASHBOARD_ENABLE_RUNS") == "1",
            "agent_chat": os.getenv("VESUVIUS_DASHBOARD_AGENT_ENABLED") == "1",
            "agent_settings_write": os.getenv("VESUVIUS_DASHBOARD_AGENT_SETTINGS_WRITE") == "1",
            "agent_provider": os.getenv("VESUVIUS_DASHBOARD_AGENT_PROVIDER") or settings_values.get("agent_provider", "hermes"),
            "agent_model": os.getenv("VESUVIUS_DASHBOARD_AGENT_MODEL") or settings_values.get("agent_model", ""),
            "agent_api_key_configured": bool(os.getenv("VESUVIUS_DASHBOARD_AGENT_API_KEY")),
            "settings_write": True,
            "visual_analysis": os.getenv("VESUVIUS_DASHBOARD_VISUAL_ANALYSIS_ENABLED") == "1" or settings_values.get("visual_analysis_enabled") is True,
            "visual_analysis_provider": os.getenv("VESUVIUS_DASHBOARD_VISUAL_ANALYSIS_PROVIDER") or settings_values.get("visual_analysis_provider", "hermes"),
            "visual_analysis_model": os.getenv("VESUVIUS_DASHBOARD_VISUAL_ANALYSIS_MODEL") or settings_values.get("visual_analysis_model", ""),
            "visual_analysis_api_key_configured": bool(os.getenv("VESUVIUS_DASHBOARD_VISUAL_ANALYSIS_API_KEY") or os.getenv("VESUVIUS_DASHBOARD_AGENT_API_KEY")),
            "artifact_preview": True,
            "standalone": True,
        },
    }


def build_snapshot(project_root: str | os.PathLike[str] | None = None, *, use_cache: bool = True) -> dict[str, Any]:
    root = resolve_project_root(project_root)
    ttl_sec = _snapshot_cache_ttl_sec()
    cache_enabled = use_cache and ttl_sec > 0 and os.getenv("VESUVIUS_DASHBOARD_DISABLE_SNAPSHOT_CACHE") != "1"
    settings_version = load_dashboard_settings(root).get("version")
    cache_key = f"{root}\0enable_runs={os.getenv('VESUVIUS_DASHBOARD_ENABLE_RUNS') == '1'}\0agent={os.getenv('VESUVIUS_DASHBOARD_AGENT_ENABLED') == '1'}\0settings={settings_version}\0visual={os.getenv('VESUVIUS_DASHBOARD_VISUAL_ANALYSIS_ENABLED') == '1'}"
    now = time.monotonic()

    if cache_enabled:
        with _snapshot_cache_lock:
            cached = _snapshot_cache.get(cache_key)
            if cached and now - cached[0] < ttl_sec:
                return copy.deepcopy(cached[1])

    snapshot = _build_snapshot_uncached(root)
    if cache_enabled:
        with _snapshot_cache_lock:
            _snapshot_cache[cache_key] = (time.monotonic(), copy.deepcopy(snapshot))
    return snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a Vesuvius dashboard JSON snapshot")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(build_snapshot(args.repo_root), indent=2 if args.pretty else None, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

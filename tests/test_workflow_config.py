from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    return yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text())


def test_autoresearch_cron_workflow_has_required_triggers_and_artifacts():
    workflow = _load("autoresearch_cron.yml")

    assert workflow["on"]["schedule"][0]["cron"] == "*/30 * * * *"
    assert "workflow_dispatch" in workflow["on"]
    steps = workflow["jobs"]["autoresearch"]["steps"]
    assert any(step.get("uses") == "actions/checkout@v4" for step in steps)
    assert any("python autoresearch.py" in step.get("run", "") for step in steps)
    assert any(step.get("uses") == "actions/upload-artifact@v4" for step in steps)


def test_autoresearch_test_workflow_runs_full_suite_on_push_and_pr():
    workflow = _load("autoresearch_test.yml")

    assert "push" in workflow["on"]
    assert "pull_request" in workflow["on"]
    steps = workflow["jobs"]["tests"]["steps"]
    assert any(step.get("run") == "python -m pytest tests/" for step in steps)

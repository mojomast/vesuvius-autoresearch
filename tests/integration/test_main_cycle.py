from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import autoresearch


def test_plan_json_output_is_valid(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["autoresearch.py", "--plan", "--json"])
    monkeypatch.setattr(autoresearch, "_research_harness", lambda: SimpleNamespace(promotion_ready_payload=lambda: None, best_base_config=lambda runs: {}))
    monkeypatch.setattr(autoresearch._RunHistory, "recent_runs", lambda self: [])

    assert autoresearch.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "needs_baseline"


def test_lock_contention_exits_without_writing(monkeypatch, tmp_path, capsys) -> None:
    cfg_dir = tmp_path / "configs"
    logs_dir = tmp_path / "logs"
    monkeypatch.setattr(sys, "argv", ["autoresearch.py"])
    monkeypatch.setattr(autoresearch, "CONFIGS", cfg_dir)
    monkeypatch.setattr(autoresearch, "LOGS", logs_dir)
    monkeypatch.setattr(autoresearch, "LOCK_PATH", logs_dir / "autoresearch.lock")
    monkeypatch.setattr(autoresearch, "_research_harness", lambda: SimpleNamespace())

    def busy_lock(_lock):
        raise BlockingIOError("busy")

    monkeypatch.setattr(autoresearch, "_acquire_autoresearch_lock", busy_lock)
    assert autoresearch.main() == 0
    assert "another autoresearch run is active" in capsys.readouterr().out
    assert not list(cfg_dir.glob("auto_*.yaml"))

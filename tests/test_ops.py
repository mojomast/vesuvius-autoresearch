from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import autoresearch


def test_prune_stale_configs_deletes_only_old_auto_configs(tmp_path, monkeypatch):
    monkeypatch.setattr(autoresearch, "CONFIGS", tmp_path)
    old_auto = tmp_path / "auto_old.yaml"
    fresh_auto = tmp_path / "auto_fresh.yaml"
    manual = tmp_path / "baseline.yaml"
    old_auto.write_text("x: 1\n")
    fresh_auto.write_text("x: 2\n")
    manual.write_text("x: 3\n")
    old_time = time.time() - 72 * 3600
    fresh_time = time.time()
    for path, mtime in ((old_auto, old_time), (fresh_auto, fresh_time), (manual, old_time)):
        path.touch()
        path.chmod(0o644)
        import os
        os.utime(path, (mtime, mtime))

    assert autoresearch._prune_stale_configs(max_age_hours=48) == 1
    assert not old_auto.exists()
    assert fresh_auto.exists()
    assert manual.exists()


def test_linked_loo_summary_cache_uses_ttl(tmp_path, monkeypatch):
    monkeypatch.setattr(autoresearch, "LOGS", tmp_path)
    monkeypatch.setattr(autoresearch, "_LINKED_LOO_SUMMARY_CACHE", None)
    monkeypatch.setattr(autoresearch, "_LINKED_LOO_SUMMARY_CACHE_AT", 0.0)
    summary = tmp_path / "candidate.summary.json"
    summary.write_text(json.dumps({"promotion_ready": True, "run_ids": ["run-1"]}))

    assert autoresearch._linked_loo_summary_ready({"run_id": "run-1"}) is True
    summary.write_text(json.dumps({"promotion_ready": True, "run_ids": ["run-2"]}))
    assert autoresearch._linked_loo_summary_ready({"run_id": "run-1"}) is True

    monkeypatch.setattr(autoresearch, "_LINKED_LOO_SUMMARY_CACHE_AT", time.time() - 301)
    assert autoresearch._linked_loo_summary_ready({"run_id": "run-1"}) is False


def test_locking_without_backend_raises_runtime_error(tmp_path, monkeypatch):
    lock_path = tmp_path / "lock"
    monkeypatch.setattr(autoresearch, "fcntl", None)

    with lock_path.open("w") as lock, patch.dict("sys.modules", {"msvcrt": None}):
        with pytest.raises(RuntimeError, match="no safe fallback"):
            autoresearch._acquire_autoresearch_lock(lock)

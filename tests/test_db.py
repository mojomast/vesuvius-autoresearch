from __future__ import annotations

import json
import sqlite3

import autoresearch
from experiments.runner import init_db


def test_init_db_creates_promotion_results_table(tmp_path):
    db_path = tmp_path / "experiments.db"

    init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'promotion_results'"
        ).fetchone()
    assert row == ("promotion_results",)


def test_record_promotion_status_inserts_into_initialized_db(tmp_path, monkeypatch):
    db_path = tmp_path / "experiments.db"
    init_db(db_path)
    monkeypatch.setattr(autoresearch, "DB_PATH", db_path)

    autoresearch._record_promotion_status("run-1", "SUCCEEDED", {"ok": True})

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT run_id, status, payload_json FROM promotion_results").fetchone()

    assert row[0] == "run-1"
    assert row[1] == "SUCCEEDED"
    assert json.loads(row[2]) == {"ok": True}

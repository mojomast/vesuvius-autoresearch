from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import autoresearch
from experiments.runner import init_db


def test_automated_promotion_writes_log_and_db(tmp_path, monkeypatch):
    db_path = tmp_path / "experiments.db"
    logs = tmp_path / "logs"
    summary = logs / "candidate.summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"promotion_ready": True, "run_ids": ["candidate"]}))
    monkeypatch.setattr(autoresearch, "DB_PATH", db_path)
    monkeypatch.setattr(autoresearch, "LOGS", logs)
    init_db(db_path)

    with patch("autoresearch.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="ok", stderr="")):
        result = autoresearch._run_automated_promotion(["python", "script.py"], "candidate", summary, timeout=5)

    assert result["automation_status"] == "SUCCEEDED"
    assert Path(result["promotion_log"]).exists()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT status, payload_json FROM promotion_results WHERE run_id = ?", ("candidate",)).fetchone()
    assert row[0] == "SUCCEEDED"
    assert json.loads(row[1])["summary"]["promotion_ready"] is True


def test_automated_promotion_records_failure(tmp_path, monkeypatch):
    db_path = tmp_path / "experiments.db"
    logs = tmp_path / "logs"
    summary = logs / "missing.summary.json"
    monkeypatch.setattr(autoresearch, "DB_PATH", db_path)
    monkeypatch.setattr(autoresearch, "LOGS", logs)
    init_db(db_path)

    with patch("autoresearch.subprocess.run", return_value=SimpleNamespace(returncode=7, stdout="", stderr="bad")):
        result = autoresearch._run_automated_promotion(["python", "script.py"], "candidate", summary, timeout=5)

    assert result["automation_status"] == "FAILED"
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT status FROM promotion_results WHERE run_id = ?", ("candidate",)).fetchone()
    assert row[0] == "FAILED"


def test_automated_promotion_records_missing_summary_after_success(tmp_path, monkeypatch):
    db_path = tmp_path / "experiments.db"
    logs = tmp_path / "logs"
    summary = logs / "missing.summary.json"
    monkeypatch.setattr(autoresearch, "DB_PATH", db_path)
    monkeypatch.setattr(autoresearch, "LOGS", logs)
    init_db(db_path)

    with patch("autoresearch.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="ok", stderr="")):
        result = autoresearch._run_automated_promotion(["python", "script.py"], "candidate", summary, timeout=5)

    assert result["automation_status"] == "SUCCEEDED_NO_SUMMARY"
    assert result["promotion_payload"]["warning"] == "summary_json not found"
    assert result["promotion_payload"]["path"] == str(summary)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT status, payload_json FROM promotion_results WHERE run_id = ?", ("candidate",)).fetchone()
    assert row[0] == "SUCCEEDED_NO_SUMMARY"
    assert json.loads(row[1])["warning"] == "summary_json not found"


def test_automated_promotion_records_artifact_outputs_without_summary(tmp_path, monkeypatch):
    db_path = tmp_path / "experiments.db"
    logs = tmp_path / "logs"
    output_dir = tmp_path / "full_tile"
    output_dir.mkdir()
    metrics = output_dir / "metrics.json"
    probability_map = output_dir / "probability_map.npy"
    threshold_csv = output_dir / "metrics_by_threshold.csv"
    for path in (metrics, probability_map, threshold_csv):
        path.write_text("ok")
    stdout = json.dumps({"outputs": {"metrics_json": str(metrics), "probability_map": str(probability_map), "threshold_csv": str(threshold_csv)}})
    monkeypatch.setattr(autoresearch, "DB_PATH", db_path)
    monkeypatch.setattr(autoresearch, "LOGS", logs)
    init_db(db_path)

    with patch("autoresearch.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=stdout, stderr="")):
        result = autoresearch._run_automated_promotion(["python", "script.py"], "candidate", logs / "missing.summary.json", timeout=5)

    assert result["automation_status"] == "SUCCEEDED_ARTIFACTS"
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT status, payload_json FROM promotion_results WHERE run_id = ?", ("candidate",)).fetchone()
    assert row[0] == "SUCCEEDED_ARTIFACTS"
    assert json.loads(row[1])["outputs"]["metrics_json"] == str(metrics)


def test_automated_promotion_fails_when_reported_outputs_are_missing(tmp_path, monkeypatch):
    db_path = tmp_path / "experiments.db"
    logs = tmp_path / "logs"
    missing = tmp_path / "full_tile" / "metrics.json"
    stdout = json.dumps({"outputs": {"metrics_json": str(missing)}})
    monkeypatch.setattr(autoresearch, "DB_PATH", db_path)
    monkeypatch.setattr(autoresearch, "LOGS", logs)
    init_db(db_path)

    with patch("autoresearch.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=stdout, stderr="")):
        result = autoresearch._run_automated_promotion(["python", "script.py"], "candidate", logs / "missing.summary.json", timeout=5)

    assert result["automation_status"] == "FAILED"
    assert result["promotion_payload"]["missing_outputs"] == [str(missing)]

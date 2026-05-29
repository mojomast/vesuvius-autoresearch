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
    monkeypatch.setattr(autoresearch, "DB_PATH", db_path)
    monkeypatch.setattr(autoresearch, "LOGS", logs)
    init_db(db_path)

    def run_and_write_summary(*_args, **_kwargs):
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps({"promotion_ready": True, "run_ids": ["candidate"]}))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    with patch("autoresearch.subprocess.run", side_effect=run_and_write_summary):
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


def test_automated_promotion_fails_missing_summary_after_success(tmp_path, monkeypatch):
    db_path = tmp_path / "experiments.db"
    logs = tmp_path / "logs"
    summary = logs / "missing.summary.json"
    monkeypatch.setattr(autoresearch, "DB_PATH", db_path)
    monkeypatch.setattr(autoresearch, "LOGS", logs)
    init_db(db_path)

    with patch("autoresearch.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="ok", stderr="")):
        result = autoresearch._run_automated_promotion(["python", "script.py"], "candidate", summary, timeout=5)

    assert result["automation_status"] == "FAILED"
    assert result["promotion_payload"]["error"] == "promotion command succeeded but produced no current summary_json or validated outputs"
    assert result["promotion_payload"]["path"] == str(summary)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT status, payload_json FROM promotion_results WHERE run_id = ?", ("candidate",)).fetchone()
    assert row[0] == "FAILED"
    assert json.loads(row[1])["error"] == "promotion command succeeded but produced no current summary_json or validated outputs"


def test_automated_promotion_rejects_stale_summary_without_outputs(tmp_path, monkeypatch):
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

    assert result["automation_status"] == "FAILED"
    assert result["promotion_payload"]["stale_summary_json"] == str(summary)


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
    assert json.loads(row[1])["outputs_verified"] is True


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


def test_automated_promotion_fails_incomplete_full_tile_outputs(tmp_path, monkeypatch):
    db_path = tmp_path / "experiments.db"
    logs = tmp_path / "logs"
    output_dir = tmp_path / "full_tile"
    output_dir.mkdir()
    metrics = output_dir / "metrics.json"
    metrics.write_text(json.dumps({"promotion_checks": {"eligible": True}}))
    stdout = json.dumps({"outputs": {"metrics_json": str(metrics)}})
    monkeypatch.setattr(autoresearch, "DB_PATH", db_path)
    monkeypatch.setattr(autoresearch, "LOGS", logs)
    init_db(db_path)

    command = [".venv/bin/python", "scripts/infer_full_tile.py", "--output-dir", str(output_dir)]
    with patch("autoresearch.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=stdout, stderr="")):
        result = autoresearch._run_automated_promotion(command, "candidate", logs / "missing.summary.json", timeout=5)

    assert result["automation_status"] == "FAILED"
    assert result["promotion_payload"]["missing_output_keys"] == ["probability_map", "threshold_csv"]


def test_automated_promotion_extracts_noisy_stdout_outputs(tmp_path, monkeypatch):
    db_path = tmp_path / "experiments.db"
    logs = tmp_path / "logs"
    output_dir = tmp_path / "full_tile"
    output_dir.mkdir()
    metrics = output_dir / "metrics.json"
    probability_map = output_dir / "probability_map.npy"
    threshold_csv = output_dir / "metrics_by_threshold.csv"
    metrics.write_text(json.dumps({"promotion_checks": {"eligible": True}}))
    probability_map.write_text("prob")
    threshold_csv.write_text("threshold\n")
    stdout = "progress\n" + json.dumps({"outputs": {"metrics_json": str(metrics), "probability_map": str(probability_map), "threshold_csv": str(threshold_csv)}})
    monkeypatch.setattr(autoresearch, "DB_PATH", db_path)
    monkeypatch.setattr(autoresearch, "LOGS", logs)
    init_db(db_path)

    command = [".venv/bin/python", "scripts/infer_full_tile.py", "--output-dir", str(output_dir)]
    with patch("autoresearch.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=stdout, stderr="")):
        result = autoresearch._run_automated_promotion(command, "candidate", logs / "missing.summary.json", timeout=5)

    assert result["automation_status"] == "SUCCEEDED_ARTIFACTS"
    assert result["promotion_payload"]["outputs_verified"] is True

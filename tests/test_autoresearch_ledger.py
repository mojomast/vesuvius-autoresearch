from __future__ import annotations

import sqlite3
import subprocess

import autoresearch
from src.autoresearch.ledger import DEFAULT_DB_PATH, init_ledger, recent_ledger_items, record_evidence_package, record_proposal, record_proposal_result


def test_ledger_initializes_required_tables(tmp_path):
    db = tmp_path / "experiments.db"

    init_ledger(db)

    with sqlite3.connect(db) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    assert {"hypotheses", "proposals", "proposal_results", "diagnoses", "evidence_packages"}.issubset(tables)


def test_ledger_records_proposal_result_and_evidence_package(tmp_path):
    db = tmp_path / "experiments.db"
    cfg = {
        "autoresearch": {
            "proposal_id": "prop1",
            "hypothesis_id": "hyp1",
            "mutation_family": "augmentation_policy",
            "changed_path": "training.augment_flips",
            "cost_tier": "normal",
            "search_signature": ["training.augment_flips", True],
        }
    }

    assert record_proposal("configs/auto.yaml", cfg, db_path=db) == "prop1"
    record_proposal_result("prop1", run_id="run1", status="completed", metric_deltas={"val_f1": 0.4}, db_path=db)
    record_evidence_package("pkg1", candidate_run_id="run1", path_json="logs/evidence_packages/pkg.json", path_markdown="logs/evidence_packages/pkg.md", reason="test", db_path=db)

    recent = recent_ledger_items(db)

    assert recent["proposals"][0]["proposal_id"] == "prop1"
    assert recent["proposal_results"][0]["run_id"] == "run1"
    assert recent["evidence_packages"][0]["package_id"] == "pkg1"


def test_default_ledger_ignores_absolute_tmp_proposal_paths(tmp_path):
    cfg = {"autoresearch": {"proposal_id": "tmp_prop"}}

    assert record_proposal(tmp_path / "auto.yaml", cfg, db_path=DEFAULT_DB_PATH) is None


def test_failed_subprocess_records_proposal_failure(tmp_path, monkeypatch):
    db = tmp_path / "experiments" / "experiments.db"
    logs = tmp_path / "logs"
    cfg = tmp_path / "configs" / "auto.yaml"
    cfg.parent.mkdir()
    cfg.write_text("autoresearch:\n  proposal_id: prop_fail\n")

    monkeypatch.setattr(autoresearch, "DB_PATH", db)
    monkeypatch.setattr(autoresearch, "LOGS", logs)

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=["run_experiment.py"], returncode=2, stdout="out", stderr="boom")

    monkeypatch.setattr(autoresearch.subprocess, "run", fake_run)

    try:
        autoresearch._run_experiment_checked(cfg)
    except subprocess.CalledProcessError:
        pass
    else:
        raise AssertionError("expected CalledProcessError")

    recent = recent_ledger_items(db)

    assert recent["proposal_results"][0]["proposal_id"] == "prop_fail"
    assert recent["proposal_results"][0]["status"] == "failed"
    assert (logs / "auto_failure.log").exists()

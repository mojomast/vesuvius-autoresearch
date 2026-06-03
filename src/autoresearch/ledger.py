from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = ROOT / "experiments" / "experiments.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_ledger(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hypotheses (
                hypothesis_id TEXT PRIMARY KEY,
                title TEXT,
                expected_effect TEXT,
                diagnosis_code TEXT,
                parent_run_id TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS proposals (
                proposal_id TEXT PRIMARY KEY,
                hypothesis_id TEXT,
                arm_id TEXT,
                mutation_family TEXT,
                config_path TEXT,
                config_signature TEXT,
                search_signature TEXT,
                changed_path TEXT,
                expected_effect TEXT,
                cost_tier TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS proposal_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                proposal_id TEXT,
                run_id TEXT,
                status TEXT NOT NULL,
                metric_deltas_json TEXT,
                blocker_deltas_json TEXT,
                error_class TEXT,
                error_message TEXT,
                returncode INTEGER,
                log_path TEXT,
                duration_seconds REAL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS diagnoses (
                diagnosis_id TEXT PRIMARY KEY,
                candidate_run_id TEXT,
                code TEXT NOT NULL,
                severity TEXT,
                evidence_json TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence_packages (
                package_id TEXT PRIMARY KEY,
                candidate_run_id TEXT,
                path_json TEXT,
                path_markdown TEXT,
                reason TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_proposals_status ON proposals(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_proposal_results_proposal ON proposal_results(proposal_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_packages_candidate ON evidence_packages(candidate_run_id)")


def record_proposal(config_path: str | Path, cfg: dict[str, Any], *, db_path: str | Path = DEFAULT_DB_PATH) -> str | None:
    autoresearch = cfg.get("autoresearch", {}) if isinstance(cfg.get("autoresearch"), dict) else {}
    proposal_id = autoresearch.get("proposal_id")
    if not proposal_id:
        return None
    path = Path(config_path).expanduser()
    resolved_path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    resolved_db = Path(db_path).expanduser().resolve()
    if resolved_db == DEFAULT_DB_PATH.resolve() and not resolved_path.is_relative_to(ROOT):
        return None
    init_ledger(db_path)
    now = _now()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO proposals(
                proposal_id,hypothesis_id,arm_id,mutation_family,config_path,config_signature,
                search_signature,changed_path,expected_effect,cost_tier,status,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,COALESCE((SELECT created_at FROM proposals WHERE proposal_id=?),?))
            """,
            (
                str(proposal_id),
                autoresearch.get("hypothesis_id"),
                autoresearch.get("arm_id") or autoresearch.get("mutation_family"),
                autoresearch.get("mutation_family"),
                str(config_path),
                autoresearch.get("config_signature"),
                json.dumps(autoresearch.get("search_signature"), sort_keys=True),
                autoresearch.get("changed_path"),
                autoresearch.get("parent_reason") or autoresearch.get("expected_effect"),
                autoresearch.get("cost_tier"),
                autoresearch.get("proposal_status") or "generated",
                str(proposal_id),
                now,
            ),
        )
    return str(proposal_id)


def record_proposal_result(
    proposal_id: str | None,
    *,
    run_id: str | None = None,
    status: str,
    metric_deltas: dict[str, Any] | None = None,
    blocker_deltas: dict[str, Any] | None = None,
    error_class: str | None = None,
    error_message: str | None = None,
    returncode: int | None = None,
    log_path: str | Path | None = None,
    duration_seconds: float | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    if not proposal_id:
        return
    init_ledger(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO proposal_results(
                proposal_id,run_id,status,metric_deltas_json,blocker_deltas_json,error_class,
                error_message,returncode,log_path,duration_seconds,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                proposal_id,
                run_id,
                status,
                json.dumps(metric_deltas or {}, sort_keys=True),
                json.dumps(blocker_deltas or {}, sort_keys=True),
                error_class,
                error_message,
                returncode,
                str(log_path) if log_path else None,
                duration_seconds,
                _now(),
            ),
        )
        conn.execute("UPDATE proposals SET status=? WHERE proposal_id=?", (status, proposal_id))


def record_evidence_package(
    package_id: str,
    *,
    candidate_run_id: str | None,
    path_json: str | Path | None,
    path_markdown: str | Path | None,
    reason: str | None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    init_ledger(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO evidence_packages(package_id,candidate_run_id,path_json,path_markdown,reason,created_at)
            VALUES(?,?,?,?,?,?)
            """,
            (package_id, candidate_run_id, str(path_json) if path_json else None, str(path_markdown) if path_markdown else None, reason, _now()),
        )


def record_diagnosis(
    candidate_run_id: str,
    code: str,
    *,
    severity: str | None = None,
    evidence: dict[str, Any] | None = None,
    status: str = "open",
    db_path: str | Path = DEFAULT_DB_PATH,
) -> str:
    diagnosis_id = f"{candidate_run_id}:{code}"
    init_ledger(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO diagnoses(diagnosis_id,candidate_run_id,code,severity,evidence_json,status,created_at)
            VALUES(?,?,?,?,?,?,?)
            """,
            (diagnosis_id, candidate_run_id, code, severity, json.dumps(evidence or {}, sort_keys=True, default=str), status, _now()),
        )
    return diagnosis_id


def recent_ledger_items(db_path: str | Path = DEFAULT_DB_PATH, *, limit: int = 20) -> dict[str, Any]:
    path = Path(db_path)
    if not path.exists():
        return {"hypotheses": [], "proposals": [], "proposal_results": [], "diagnoses": [], "evidence_packages": []}
    init_ledger(path)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        def rows(sql: str) -> list[dict[str, Any]]:
            return [dict(row) for row in conn.execute(sql, (limit,)).fetchall()]
        return {
            "hypotheses": rows("SELECT * FROM hypotheses ORDER BY updated_at DESC LIMIT ?"),
            "proposals": rows("SELECT * FROM proposals ORDER BY created_at DESC LIMIT ?"),
            "proposal_results": rows("SELECT * FROM proposal_results ORDER BY created_at DESC LIMIT ?"),
            "diagnoses": rows("SELECT * FROM diagnoses ORDER BY created_at DESC LIMIT ?"),
            "evidence_packages": rows("SELECT * FROM evidence_packages ORDER BY created_at DESC LIMIT ?"),
        }

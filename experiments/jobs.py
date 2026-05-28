"""SQLite-backed experiment job queue."""
from __future__ import annotations

import json
import os
import sqlite3
import traceback
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .runner import (
    DB_PATH as EXPERIMENTS_DB_PATH,
    SQLITE_BUSY_TIMEOUT_MS,
    SQLITE_TIMEOUT_SECONDS,
    canonical_experiment_config_signature,
    load_config,
    run_experiment,
)

JOB_DB_PATH = Path(__file__).resolve().parent / "jobs.db"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _stamp() -> str:
    return _utcnow().isoformat()


def _connect(db_path: str | os.PathLike[str] = JOB_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=SQLITE_TIMEOUT_SECONDS)
    conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.row_factory = sqlite3.Row
    return conn


def _default_dedupe_key(experiment_config: str | os.PathLike[str]) -> str | None:
    try:
        return f"experiment_config:{canonical_experiment_config_signature(load_config(experiment_config))}"
    except OSError:
        return None


def _row_to_job(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    job = dict(row)
    job["payload"] = json.loads(job.pop("payload_json"))
    return job


def init_db(db_path: str | os.PathLike[str] = JOB_DB_PATH) -> None:
    """Create the job queue schema if needed."""
    with closing(_connect(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                dedupe_key TEXT UNIQUE,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                worker_id TEXT,
                lease_expires_at TEXT,
                result_json TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_experiment_jobs_status ON experiment_jobs(status, lease_expires_at)")
        conn.commit()


def enqueue_experiment_config(
    experiment_config: str | os.PathLike[str],
    db_path: str | os.PathLike[str] = JOB_DB_PATH,
    dedupe_key: str | None = None,
) -> dict[str, Any]:
    """Enqueue an experiment config path, optionally deduplicated by key."""
    init_db(db_path)
    if dedupe_key is None:
        dedupe_key = _default_dedupe_key(experiment_config)
    payload = {"experiment_config": str(experiment_config)}
    payload_json = json.dumps(payload, sort_keys=True)
    now = _stamp()
    with closing(_connect(db_path)) as conn:
        if dedupe_key is None:
            cur = conn.execute(
                """
                INSERT INTO experiment_jobs(job_type, payload_json, status, created_at, updated_at)
                VALUES('experiment_config', ?, 'queued', ?, ?)
                """,
                (payload_json, now, now),
            )
            job_id = int(cur.lastrowid)
        else:
            conn.execute(
                """
                INSERT OR IGNORE INTO experiment_jobs(job_type, payload_json, dedupe_key, status, created_at, updated_at)
                VALUES('experiment_config', ?, ?, 'queued', ?, ?)
                """,
                (payload_json, dedupe_key, now, now),
            )
            job_id = int(conn.execute("SELECT id FROM experiment_jobs WHERE dedupe_key = ?", (dedupe_key,)).fetchone()[0])
        conn.commit()
        return _row_to_job(conn.execute("SELECT * FROM experiment_jobs WHERE id = ?", (job_id,)).fetchone())  # type: ignore[return-value]


def get_job(job_id: int, db_path: str | os.PathLike[str] = JOB_DB_PATH) -> dict[str, Any] | None:
    init_db(db_path)
    with closing(_connect(db_path)) as conn:
        return _row_to_job(conn.execute("SELECT * FROM experiment_jobs WHERE id = ?", (job_id,)).fetchone())


def lease_one_job(
    worker_id: str,
    lease_seconds: int,
    db_path: str | os.PathLike[str] = JOB_DB_PATH,
) -> dict[str, Any] | None:
    """Lease one queued or stale leased job for a worker."""
    init_db(db_path)
    now = _stamp()
    lease_until = (_utcnow() + timedelta(seconds=lease_seconds)).isoformat()
    with closing(_connect(db_path)) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT * FROM experiment_jobs
            WHERE status = 'queued' OR (status = 'leased' AND lease_expires_at <= ?)
            ORDER BY created_at, id
            LIMIT 1
            """,
            (now,),
        ).fetchone()
        if row is None:
            conn.commit()
            return None
        conn.execute(
            """
            UPDATE experiment_jobs
            SET status = 'leased', attempts = attempts + 1, worker_id = ?, lease_expires_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (worker_id, lease_until, now, row["id"]),
        )
        conn.commit()
    return get_job(int(row["id"]), db_path)


def run_leased_job(
    job: dict[str, Any],
    db_path: str | os.PathLike[str] = JOB_DB_PATH,
    experiments_db_path: str | os.PathLike[str] = EXPERIMENTS_DB_PATH,
) -> dict[str, Any]:
    """Run a leased experiment_config job and persist its terminal state."""
    if job.get("status") != "leased":
        raise ValueError(f"job {job.get('id')} is not leased")
    if job.get("job_type") != "experiment_config":
        raise ValueError(f"unsupported job_type: {job.get('job_type')}")
    config_path = job["payload"]["experiment_config"]
    try:
        result = run_experiment(config_path, db_path=Path(experiments_db_path))
    except Exception as exc:
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        with closing(_connect(db_path)) as conn:
            conn.execute(
                """
                UPDATE experiment_jobs
                SET status = 'failed', result_json = NULL, error = ?, lease_expires_at = NULL, updated_at = ?
                WHERE id = ?
                """,
                (error, _stamp(), job["id"]),
            )
            conn.commit()
        raise
    with closing(_connect(db_path)) as conn:
        conn.execute(
            """
            UPDATE experiment_jobs
            SET status = 'succeeded', result_json = ?, error = NULL, lease_expires_at = NULL, updated_at = ?
            WHERE id = ?
            """,
            (json.dumps(result, sort_keys=True), _stamp(), job["id"]),
        )
        conn.commit()
    updated = get_job(int(job["id"]), db_path)
    assert updated is not None
    return updated


def run_worker(
    db_path: str | os.PathLike[str] = JOB_DB_PATH,
    worker_id: str | None = None,
    lease_seconds: int = 3600,
    max_jobs: int | None = None,
    once: bool = False,
    experiments_db_path: str | os.PathLike[str] = EXPERIMENTS_DB_PATH,
) -> int:
    """Lease and run jobs until limits are reached or no work remains."""
    worker = worker_id or f"worker-{os.getpid()}"
    processed = 0
    while max_jobs is None or processed < max_jobs:
        job = lease_one_job(worker, lease_seconds, db_path)
        if job is None:
            break
        try:
            run_leased_job(job, db_path, experiments_db_path)
        except Exception:
            # The job row already records the failure; keep unattended workers alive.
            pass
        processed += 1
        if once:
            break
    return processed

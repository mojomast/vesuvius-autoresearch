#!/usr/bin/env python3
"""Run SQLite-backed experiment config jobs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.jobs import JOB_DB_PATH, run_worker  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run queued experiment config jobs")
    parser.add_argument("--once", action="store_true", help="Run at most one job")
    parser.add_argument("--max-jobs", type=int, default=None, help="Maximum jobs to run before exiting")
    parser.add_argument("--lease-seconds", type=int, default=3600, help="Lease duration for each job")
    parser.add_argument("--worker-id", default=None, help="Worker identifier recorded on leased jobs")
    parser.add_argument("--db-path", default=str(JOB_DB_PATH), help="SQLite job queue path")
    args = parser.parse_args(argv)

    max_jobs = 1 if args.once and args.max_jobs is None else args.max_jobs
    run_worker(
        db_path=args.db_path,
        worker_id=args.worker_id,
        lease_seconds=args.lease_seconds,
        max_jobs=max_jobs,
        once=args.once,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

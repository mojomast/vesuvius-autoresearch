#!/usr/bin/env python3
"""Summarize recent AutoResearch promotion attempts for operators."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.runner import DB_PATH


def _reasons(payload: dict[str, Any]) -> list[str]:
    reasons = payload.get("promotion_failure_reasons")
    if isinstance(reasons, list):
        return [str(item) for item in reasons]
    diagnostic = payload.get("diagnostic_summary")
    if isinstance(diagnostic, dict) and isinstance(diagnostic.get("reasons"), list):
        return [str(item) for item in diagnostic["reasons"]]
    if payload.get("error"):
        return [str(payload["error"])]
    if payload.get("warning"):
        return [str(payload["warning"])]
    return []


def summarize(limit: int = 20, db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT run_id, timestamp, status, payload_json FROM promotion_results ORDER BY timestamp DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    out = []
    for row in rows:
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except Exception:
            payload = {}
        diagnostic = payload.get("diagnostic_summary") if isinstance(payload.get("diagnostic_summary"), dict) else {}
        out.append({
            "run_id": row["run_id"],
            "timestamp": row["timestamp"],
            "status": row["status"],
            "reasons": _reasons(payload),
            "summary_json_verified": bool(diagnostic.get("summary_json_verified") or payload.get("summary")),
            "artifacts_verified": bool(diagnostic.get("artifacts_verified") or payload.get("outputs_verified")),
            "summary_json": payload.get("summary_json"),
            "missing_outputs": payload.get("missing_outputs", []),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize recent promotion_results rows")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    args = parser.parse_args()
    print(json.dumps(summarize(args.limit, args.db_path), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

try:
    import fcntl
except ImportError:
    fcntl = None
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


CAUSE_LABELS = {
    "active_guard": "Active process guard",
    "load_guard": "Load guard",
    "mem_guard": "Memory guard",
    "disk_guard": "Disk guard",
    "proposal_guard": "Proposal guard",
    "promotion_pause": "Promotion pause",
    "promotion_evidence_required": "Promotion evidence required",
    "signature_exhausted": "Signature exhausted",
    "deduped_existing_run": "Deduped existing run",
    "timeout": "Timeout",
    "unknown_no_progress": "Unknown no-progress",
}


def lock_active(project_root: Path) -> bool:
    lock_path = project_root / "logs" / "autoresearch.lock"
    if not lock_path.exists():
        return False
    try:
        if fcntl is not None:
            with lock_path.open("a+") as fh:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(fh, fcntl.LOCK_UN)
                    return False
                except BlockingIOError:
                    return True
        else:
            try:
                import msvcrt
                with lock_path.open("r+") as fh:
                    try:
                        fh.seek(0)
                        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                        # Unlock immediately if we successfully acquired it
                        fh.seek(0)
                        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                        return False
                    except (BlockingIOError, PermissionError, OSError):
                        return True
            except (ImportError, AttributeError, OSError):
                try:
                    # Generic Windows sharing violation check
                    with lock_path.open("r+") as fh:
                        pass
                    return False
                except OSError:
                    return True
    except Exception:
        return False


def tail_log(path: Path, lines: int = 120) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "lines": [], "stat": None}
    try:
        stat = path.stat()
        return {"path": str(path), "lines": path.read_text(errors="replace").splitlines()[-lines:], "stat": {"size_bytes": stat.st_size, "modified_at": stat.st_mtime}}
    except Exception as exc:
        return {"path": str(path), "lines": [f"<failed to read log: {exc}>"], "stat": None}


def _latest_overnight_orchestrator_log(project_root: Path) -> Path | None:
    root = project_root / "logs" / "overnight_safe_loop"
    if not root.exists():
        return None
    candidates = sorted(root.glob("*/orchestrator.log"), key=lambda path: path.stat().st_mtime if path.exists() else 0.0, reverse=True)
    return candidates[0] if candidates else None


def classify_no_progress_line(line: str) -> dict[str, Any] | None:
    raw = line.strip()
    if not raw:
        return None
    explicit = re.search(r"NO_PROGRESS_CAUSE\s+code=([a-z_]+)", raw)
    if explicit:
        code = explicit.group(1)
    elif "SKIP active_guard" in raw:
        code = "active_guard"
    elif "SKIP load_guard" in raw:
        code = "load_guard"
    elif "SKIP mem_guard" in raw:
        code = "mem_guard"
    elif "SKIP disk_guard" in raw:
        code = "disk_guard"
    elif "SKIP proposal_guard" in raw:
        code = "proposal_guard"
    elif "Promotion gate is ready" in raw or "pausing exploration" in raw:
        code = "promotion_pause"
    elif "AutoResearch promotion action required" in raw or "Run linked seed-repeat LOO" in raw:
        code = "promotion_evidence_required"
    elif "No novel one-change proposals remain" in raw:
        code = "signature_exhausted"
    elif "deduped_existing_run" in raw or "deduped" in raw:
        code = "deduped_existing_run"
    elif "timeout" in raw.lower() or "exit_code=124" in raw:
        code = "timeout"
    elif "QUALITY_SKIP no new run produced" in raw or "STALE cycle produced no new run" in raw:
        code = "unknown_no_progress"
    else:
        return None
    timestamp = None
    match = re.match(r"\[?([0-9]{4}-[0-9]{2}-[0-9]{2}T[^\]\s]+)", raw)
    if match:
        timestamp = match.group(1)
    return {"code": code, "label": CAUSE_LABELS.get(code, code), "line": raw, "timestamp": timestamp, "detail": {}}


def collect_no_progress_causes(project_root: Path, *, max_lines: int = 2000) -> dict[str, Any]:
    sources = [project_root / "logs" / "autoresearch_guard.log", project_root / "logs" / "autoresearch.log"]
    overnight = _latest_overnight_orchestrator_log(project_root)
    if overnight:
        sources.append(overnight)
    recent: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for source in sources:
        if not source.exists():
            continue
        try:
            lines = source.read_text(errors="replace").splitlines()[-max_lines:]
        except Exception:
            continue
        for line in lines:
            item = classify_no_progress_line(line)
            if item is None:
                continue
            item["source"] = str(source.relative_to(project_root)) if source.is_relative_to(project_root) else str(source)
            recent.append(item)
            counts[item["code"]] = counts.get(item["code"], 0) + 1
    return {"latest": recent[-1] if recent else None, "counts": counts, "recent": recent[-20:], "sources": [str(path.relative_to(project_root)) if path.exists() and path.is_relative_to(project_root) else str(path) for path in sources if path.exists()]}


def running_processes(project_root: Path, limit: int = 8) -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(["ps", "-eo", "pid=,etimes=,command="], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    rows = []
    root_text = str(project_root)
    for line in proc.stdout.splitlines():
        raw = line.strip()
        if not raw or ("autoresearch.py" not in raw and "run_experiment.py" not in raw):
            continue
        if root_text not in raw and "vesuvius-autoresearch" not in raw:
            continue
        parts = raw.split(None, 2)
        if len(parts) == 3:
            rows.append({"pid": parts[0], "age_seconds": int(parts[1]), "command": parts[2]})
        if len(rows) >= limit:
            break
    return rows


def operations_snapshot(project_root: Path) -> dict[str, Any]:
    overnight = _latest_overnight_orchestrator_log(project_root)
    return {
        "lock_active": lock_active(project_root),
        "processes": running_processes(project_root),
        "logs": tail_log(project_root / "logs" / "autoresearch.log"),
        "guard_log": tail_log(project_root / "logs" / "autoresearch_guard.log", lines=80),
        "overnight_log": tail_log(overnight, lines=120) if overnight else {"path": None, "lines": [], "stat": None},
        "no_progress": collect_no_progress_causes(project_root),
        "generated_at_epoch": time.time(),
        "environment": {"enable_runs": os.getenv("VESUVIUS_DASHBOARD_ENABLE_RUNS") == "1"},
    }

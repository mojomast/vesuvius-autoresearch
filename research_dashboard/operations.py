from __future__ import annotations

try:
    import fcntl
except ImportError:
    fcntl = None
import os
import subprocess
import time
from pathlib import Path
from typing import Any


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
    return {"lock_active": lock_active(project_root), "processes": running_processes(project_root), "logs": tail_log(project_root / "logs" / "autoresearch.log"), "generated_at_epoch": time.time(), "environment": {"enable_runs": os.getenv("VESUVIUS_DASHBOARD_ENABLE_RUNS") == "1"}}

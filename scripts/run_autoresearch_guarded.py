#!/usr/bin/env python3
"""Resource-aware launcher for Vesuvius AutoResearch.

Runs only local NumPy experiments; does not perform web/LLM/API calls.  This wrapper
keeps cron aggressive while protecting the machine with load, memory, disk, and
process guards, then launches autoresearch at low CPU/IO priority.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
AUTORESEARCH = ROOT / "autoresearch.py"
LOGS = ROOT / "logs"
GUARD_LOG = LOGS / "autoresearch_guard.log"

CPU_COUNT = os.cpu_count() or 1
GI_B = 1024 * 1024 * 1024

# Conservative defaults for a 32-core / 121 GiB host.  Override from cron/env if needed.
MAX_LOAD_HARD = float(os.environ.get("SCROLL_RESEARCH_MAX_LOAD_HARD", max(10.0, CPU_COUNT * 0.40)))
MAX_LOAD_SOFT = float(os.environ.get("SCROLL_RESEARCH_MAX_LOAD_SOFT", max(6.0, CPU_COUNT * 0.25)))
MIN_MEM_HARD_GIB = float(os.environ.get("SCROLL_RESEARCH_MIN_MEM_HARD_GIB", "16"))
MIN_MEM_SOFT_GIB = float(os.environ.get("SCROLL_RESEARCH_MIN_MEM_SOFT_GIB", "48"))
MIN_ROOT_FREE_GIB = float(os.environ.get("SCROLL_RESEARCH_MIN_ROOT_FREE_GIB", "100"))
MAX_ACTIVE_AUTORESEARCH = int(os.environ.get("SCROLL_RESEARCH_MAX_ACTIVE", "0"))
TIMEOUT_SECONDS = int(os.environ.get("SCROLL_RESEARCH_TIMEOUT_SECONDS", "480"))


def log(message: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    line = f"{stamp} {message}"
    print(line, flush=True)
    with GUARD_LOG.open("a") as f:
        f.write(line + "\n")


def loadavg() -> tuple[float, float, float]:
    with open("/proc/loadavg", "r") as f:
        a, b, c = f.read().split()[:3]
    return float(a), float(b), float(c)


def mem_available_gib() -> float:
    values: dict[str, int] = {}
    with open("/proc/meminfo", "r") as f:
        for line in f:
            key, rest = line.split(":", 1)
            values[key] = int(rest.strip().split()[0]) * 1024
    return values.get("MemAvailable", 0) / GI_B


def root_free_gib() -> float:
    return shutil.disk_usage(str(ROOT)).free / GI_B


def active_autoresearch_processes() -> list[str]:
    active: list[str] = []
    self_pid = os.getpid()
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == self_pid:
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "ignore")
        except Exception:
            continue
        if str(ROOT) in cmdline and ("autoresearch.py" in cmdline or "run_experiment.py" in cmdline):
            active.append(f"{pid}:{cmdline[:180]}")
    return active


def choose_proposals(load1: float, mem_gib: float) -> int:
    # Idle machine: exploit the slack.  Moderate machine: still useful.  Busy: tiny pulse.
    if load1 <= MAX_LOAD_SOFT and mem_gib >= 64:
        return int(os.environ.get("SCROLL_RESEARCH_PROPOSALS_IDLE", "8"))
    if load1 <= MAX_LOAD_HARD and mem_gib >= MIN_MEM_SOFT_GIB:
        return int(os.environ.get("SCROLL_RESEARCH_PROPOSALS_NORMAL", "6"))
    if load1 <= MAX_LOAD_HARD and mem_gib >= MIN_MEM_HARD_GIB:
        return int(os.environ.get("SCROLL_RESEARCH_PROPOSALS_LOW", "2"))
    return 0


def main() -> int:
    load1, load5, load15 = loadavg()
    mem_gib = mem_available_gib()
    free_gib = root_free_gib()
    active = active_autoresearch_processes()

    if free_gib < MIN_ROOT_FREE_GIB:
        log(f"SKIP disk_guard root_free_gib={free_gib:.1f} min={MIN_ROOT_FREE_GIB:.1f}")
        return 0
    if len(active) > MAX_ACTIVE_AUTORESEARCH:
        log(f"SKIP active_guard count={len(active)} active={active[:3]}")
        return 0
    if load1 > MAX_LOAD_HARD:
        log(f"SKIP load_guard load1={load1:.2f} load5={load5:.2f} load15={load15:.2f} max={MAX_LOAD_HARD:.2f}")
        return 0
    if mem_gib < MIN_MEM_HARD_GIB:
        log(f"SKIP mem_guard mem_available_gib={mem_gib:.1f} min={MIN_MEM_HARD_GIB:.1f}")
        return 0

    proposals = choose_proposals(load1, mem_gib)
    if proposals <= 0:
        log(f"SKIP proposal_guard load1={load1:.2f} mem_available_gib={mem_gib:.1f}")
        return 0

    env = os.environ.copy()
    env.update({
        "AUTORESEARCH_PROPOSALS": str(proposals),
        "AUTORESEARCH_RECENT_LIMIT": env.get("AUTORESEARCH_RECENT_LIMIT", "1000"),
        # Keep BLAS/NumPy from bursting across all cores during MLP matrix ops.
        "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "2"),
        "OPENBLAS_NUM_THREADS": env.get("OPENBLAS_NUM_THREADS", "2"),
        "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS", "2"),
        "NUMEXPR_NUM_THREADS": env.get("NUMEXPR_NUM_THREADS", "2"),
    })

    cmd = ["nice", "-n", "15"]
    if shutil.which("ionice"):
        cmd += ["ionice", "-c2", "-n7"]
    if shutil.which("timeout"):
        cmd += ["timeout", str(TIMEOUT_SECONDS)]
    cmd += [str(PYTHON), str(AUTORESEARCH)]

    log(
        "RUN local_only "
        f"proposals={proposals} load={load1:.2f}/{load5:.2f}/{load15:.2f} "
        f"mem_available_gib={mem_gib:.1f} root_free_gib={free_gib:.1f} "
        "nice=15 ionice=best-effort/7 blas_threads=2 no_web_llm_calls=true"
    )
    completed = subprocess.run(cmd, cwd=ROOT, env=env)
    log(f"DONE exit_code={completed.returncode} proposals={proposals}")
    return 0 if completed.returncode == 0 else completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

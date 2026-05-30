#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
LOCK="$REPO/.overnight_safe_loop.lock"
STOP="$REPO/logs/overnight_safe_loop.stop"
RUN_ROOT="$REPO/logs/overnight_safe_loop/$(date +%Y%m%d_%H%M%S)"
MAX_SECONDS="${OVERNIGHT_MAX_SECONDS:-28800}"
CYCLE_SLEEP_SECONDS="${OVERNIGHT_CYCLE_SLEEP_SECONDS:-900}"
JOB_TIMEOUT_SECONDS="${OVERNIGHT_JOB_TIMEOUT_SECONDS:-5400}"
HEARTBEAT_SECONDS="${OVERNIGHT_HEARTBEAT_SECONDS:-300}"
MAX_STALE_CYCLES="${OVERNIGHT_MAX_STALE_CYCLES:-2}"
MAX_EVIDENCE_ONLY_CYCLES="${OVERNIGHT_MAX_EVIDENCE_ONLY_CYCLES:-3}"

mkdir -p "$RUN_ROOT" "$REPO/logs"

exec 200>"$LOCK"
if ! flock -n 200; then
  echo "$(date -Iseconds) another overnight loop is already running"
  exit 0
fi

cd "$REPO"

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$RUN_ROOT/orchestrator.log"
}

latest_run_id() {
  "$PY" - <<'PY'
import sqlite3
from pathlib import Path

db = Path("experiments/experiments.db")
if not db.exists():
    raise SystemExit(0)
con = sqlite3.connect(db)
row = con.execute("SELECT run_id FROM experiments ORDER BY timestamp DESC LIMIT 1").fetchone()
con.close()
if row:
    print(row[0])
PY
}

latest_promotion_log() {
  local latest
  latest=$(ls -t logs/promotion_*.log 2>/dev/null | sed -n '1p' || true)
  if [ -n "$latest" ]; then
    printf '%s\n' "$latest"
  fi
}

run_with_heartbeat() {
  local label="$1"
  shift
  local logfile="$RUN_ROOT/${label}.log"
  log "START $label"
  "$@" >"$logfile" 2>&1 &
  local pid=$!
  local started
  started=$(date +%s)
  local next_heartbeat=$HEARTBEAT_SECONDS
  while kill -0 "$pid" 2>/dev/null; do
    sleep 5
    local now elapsed
    now=$(date +%s)
    elapsed=$((now - started))
    if [ "$elapsed" -ge "$next_heartbeat" ]; then
      log "HEARTBEAT $label pid=$pid elapsed_sec=$elapsed logfile=$logfile"
      next_heartbeat=$((next_heartbeat + HEARTBEAT_SECONDS))
    fi
  done
  wait "$pid"
  local rc=$?
  log "DONE $label exit_code=$rc logfile=$logfile"
  return "$rc"
}

gate_latest_artifact() {
  "$PY" - <<'PY'
import json
import math
import os
import sqlite3
import sys
from pathlib import Path

root = Path.cwd()
db = root / "experiments" / "experiments.db"
stop_file = root / "logs" / "overnight_safe_loop.stop"
min_val_f1 = float(os.environ.get("OVERNIGHT_MIN_VAL_F1", "0.02"))
min_ap_lift = float(os.environ.get("OVERNIGHT_MIN_AP_PREVALENCE_LIFT", "1.25"))
min_ratio = float(os.environ.get("OVERNIGHT_MIN_PRED_VAL_RATIO", "0.1"))
max_ratio = float(os.environ.get("OVERNIGHT_MAX_PRED_VAL_RATIO", "3.5"))

def fail(reason: str) -> int:
    stop_file.write_text(reason + "\n")
    print("QUALITY_STOP " + reason)
    return 42

if not db.exists():
    print("QUALITY_SKIP no experiment DB yet")
    raise SystemExit(0)

con = sqlite3.connect(db)
row = con.execute(
    "SELECT run_id,timestamp,secondary_metrics_json,artifact_dir "
    "FROM experiments ORDER BY timestamp DESC LIMIT 1"
).fetchone()
con.close()
if not row:
    print("QUALITY_SKIP no experiment rows yet")
    raise SystemExit(0)

run_id, _timestamp, metrics_json, artifact_dir = row
try:
    metrics = json.loads(metrics_json)
except Exception as exc:
    raise SystemExit(fail(f"{run_id}: invalid metrics JSON: {exc!r}"))

artifact = Path(artifact_dir)
missing_files = [name for name in ("config.json", "metrics.json", "metrics_by_threshold.csv") if not (artifact / name).exists()]
if missing_files:
    raise SystemExit(fail(f"{run_id}: missing artifact files: {missing_files}"))

required = ["val_f1", "average_precision", "precision", "recall", "val_positive_rate", "pred_positive_rate", "fixed_threshold_status"]
missing_metrics = [key for key in required if key not in metrics]
if missing_metrics:
    raise SystemExit(fail(f"{run_id}: missing metrics: {missing_metrics}"))

for key in ("val_f1", "average_precision", "precision", "recall", "val_positive_rate", "pred_positive_rate"):
    try:
        value = float(metrics[key])
    except Exception:
        raise SystemExit(fail(f"{run_id}: nonnumeric metric {key}={metrics.get(key)!r}"))
    if not math.isfinite(value):
        raise SystemExit(fail(f"{run_id}: nonfinite metric {key}={value!r}"))

val_f1 = float(metrics["val_f1"])
ap = float(metrics["average_precision"])
precision = float(metrics["precision"])
recall = float(metrics["recall"])
val_rate = float(metrics["val_positive_rate"])
pred_rate = float(metrics["pred_positive_rate"])
fixed_status = metrics.get("fixed_threshold_status")
if val_f1 < min_val_f1:
    raise SystemExit(fail(f"{run_id}: val_f1 {val_f1:.6f} < {min_val_f1:.6f}"))
if precision <= 0.0 or recall <= 0.0:
    raise SystemExit(fail(f"{run_id}: zero precision/recall precision={precision:.6f} recall={recall:.6f}"))
if fixed_status != "ok":
    raise SystemExit(fail(f"{run_id}: fixed_threshold_status={fixed_status!r}"))
if val_rate <= 0.0:
    raise SystemExit(fail(f"{run_id}: val_positive_rate={val_rate:.6f}"))
ratio = pred_rate / val_rate
if ratio < min_ratio or ratio > max_ratio:
    raise SystemExit(fail(f"{run_id}: pred/val ratio {ratio:.6f} outside {min_ratio}..{max_ratio}"))
ap_lift = float(metrics.get("ap_prevalence_lift", ap / val_rate))
if ap_lift < min_ap_lift:
    raise SystemExit(fail(f"{run_id}: ap_prevalence_lift {ap_lift:.6f} < {min_ap_lift:.6f}"))
print(
    "QUALITY_OK "
    f"run_id={run_id} val_f1={val_f1:.6f} ap={ap:.6f} "
    f"fixed_threshold_status={fixed_status} pred_val_ratio={ratio:.6f} artifact={artifact_dir}"
)
PY
}

export AUTORESEARCH_PROPOSALS="${AUTORESEARCH_PROPOSALS:-1}"
export SCROLL_RESEARCH_MAX_ACTIVE="${SCROLL_RESEARCH_MAX_ACTIVE:-0}"
export SCROLL_RESEARCH_TIMEOUT_SECONDS="$JOB_TIMEOUT_SECONDS"
export SCROLL_RESEARCH_PROPOSALS_IDLE="${SCROLL_RESEARCH_PROPOSALS_IDLE:-1}"
export SCROLL_RESEARCH_PROPOSALS_NORMAL="${SCROLL_RESEARCH_PROPOSALS_NORMAL:-1}"
export SCROLL_RESEARCH_PROPOSALS_LOW="${SCROLL_RESEARCH_PROPOSALS_LOW:-1}"
export SCROLL_RESEARCH_MAX_LOAD_HARD="${SCROLL_RESEARCH_MAX_LOAD_HARD:-8}"
export SCROLL_RESEARCH_MAX_LOAD_SOFT="${SCROLL_RESEARCH_MAX_LOAD_SOFT:-4}"
export SCROLL_RESEARCH_MIN_MEM_HARD_GIB="${SCROLL_RESEARCH_MIN_MEM_HARD_GIB:-24}"
export SCROLL_RESEARCH_MIN_ROOT_FREE_GIB="${SCROLL_RESEARCH_MIN_ROOT_FREE_GIB:-100}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export AUTORESEARCH_NUM_WORKERS="${AUTORESEARCH_NUM_WORKERS:-2}"
export AUTORESEARCH_LOO_JOBS="${AUTORESEARCH_LOO_JOBS:-1}"
export AUTORESEARCH_PAUSE_WHEN_PROMOTION_READY="${AUTORESEARCH_PAUSE_WHEN_PROMOTION_READY:-1}"
export AUTORESEARCH_CONTINUE_AFTER_PROMOTION_ACTION="${AUTORESEARCH_CONTINUE_AFTER_PROMOTION_ACTION:-0}"

rm -f "$STOP"
log "run_dir=$RUN_ROOT"
git rev-parse HEAD >"$RUN_ROOT/git_head.txt"
git status --short --branch >"$RUN_ROOT/git_status.txt" || true

log "auditing villa fold map"
if "$PY" scripts/audit_villa_fold_map.py \
  --fold-map data/fold_map_villa_labels.json \
  --summary-out "$RUN_ROOT/fold_map_villa.audit.json" \
  --filtered-fold-map-out "$RUN_ROOT/fold_map_villa.filtered.json" \
  >"$RUN_ROOT/fold_map_villa.audit.log" 2>&1; then
  log "villa fold map has no dropped folds"
else
  log "villa fold map has dropped folds; filtered map written to $RUN_ROOT/fold_map_villa.filtered.json"
fi

started=$(date +%s)
cycle=0
stale_cycles=0
evidence_only_cycles=0
while true; do
  now=$(date +%s)
  if [ $((now - started)) -ge "$MAX_SECONDS" ]; then
    log "STOP max runtime reached"
    exit 0
  fi
  if [ -f "$STOP" ]; then
    log "STOP marker exists: $STOP"
    exit 0
  fi
  cycle=$((cycle + 1))
  before_run_id="$(latest_run_id || true)"
  before_promotion_log="$(latest_promotion_log || true)"
  log "CYCLE $cycle begin"
  if ! run_with_heartbeat "autoresearch_cycle_${cycle}" \
    timeout --foreground "$JOB_TIMEOUT_SECONDS" nice -n 15 ionice -c2 -n7 \
    "$PY" scripts/run_autoresearch_guarded.py; then
    log "STOP autoresearch cycle failed"
    exit 1
  fi
  if ! gate_latest_artifact | tee -a "$RUN_ROOT/orchestrator.log"; then
    log "STOP artifact quality gate failed"
    exit 1
  fi
  after_run_id="$(latest_run_id || true)"
  after_promotion_log="$(latest_promotion_log || true)"
  if [ -n "$before_run_id" ] && [ "$after_run_id" = "$before_run_id" ]; then
    if [ -n "$after_promotion_log" ] && [ "$after_promotion_log" != "$before_promotion_log" ]; then
      stale_cycles=0
      evidence_only_cycles=$((evidence_only_cycles + 1))
      log "EVIDENCE_PROGRESS latest_run_id=$after_run_id promotion_log=$after_promotion_log evidence_only_cycles=$evidence_only_cycles/$MAX_EVIDENCE_ONLY_CYCLES"
    else
      stale_cycles=$((stale_cycles + 1))
      log "STALE cycle produced no new run or evidence latest_run_id=$after_run_id stale_cycles=$stale_cycles/$MAX_STALE_CYCLES"
    fi
  else
    stale_cycles=0
    evidence_only_cycles=0
    log "NEW_RUN latest_run_id=$after_run_id previous_run_id=$before_run_id"
  fi
  if [ "$evidence_only_cycles" -ge "$MAX_EVIDENCE_ONLY_CYCLES" ]; then
    log "EVIDENCE_LIMIT reached; next cycle will allow exploration past promotion action"
    export SCROLL_RESEARCH_ALLOW_PROMOTION_OVERRIDE=1
    export AUTORESEARCH_PAUSE_WHEN_PROMOTION_READY=0
    export AUTORESEARCH_CONTINUE_AFTER_PROMOTION_ACTION=1
    evidence_only_cycles=0
  else
    export SCROLL_RESEARCH_ALLOW_PROMOTION_OVERRIDE="${SCROLL_RESEARCH_ALLOW_PROMOTION_OVERRIDE:-0}"
    if [ "$SCROLL_RESEARCH_ALLOW_PROMOTION_OVERRIDE" = "1" ]; then
      export AUTORESEARCH_PAUSE_WHEN_PROMOTION_READY=0
      export AUTORESEARCH_CONTINUE_AFTER_PROMOTION_ACTION=1
    else
      export AUTORESEARCH_PAUSE_WHEN_PROMOTION_READY=1
      export AUTORESEARCH_CONTINUE_AFTER_PROMOTION_ACTION=0
    fi
  fi
  if [ "$stale_cycles" -ge "$MAX_STALE_CYCLES" ]; then
    log "STALE_LIMIT reached; enabling one exploration override cycle instead of stopping"
    export SCROLL_RESEARCH_ALLOW_PROMOTION_OVERRIDE=1
    export AUTORESEARCH_PAUSE_WHEN_PROMOTION_READY=0
    export AUTORESEARCH_CONTINUE_AFTER_PROMOTION_ACTION=1
    stale_cycles=0
  fi
  log "CYCLE $cycle complete; sleeping ${CYCLE_SLEEP_SECONDS}s"
  sleep "$CYCLE_SLEEP_SECONDS"
done

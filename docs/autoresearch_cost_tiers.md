# AutoResearch Cost Tiers

AutoResearch assigns every generated proposal an `autoresearch.cost_tier` so cron can prefer high-throughput exploration before expensive validation.

## Tiers

- `cheap`: small NumPy/logreg-style runs with bounded pixel counts.
- `normal`: bounded single-model torch or small MLP runs suitable for routine cron exploration.
- `expensive`: seed ensembles, TTA, large sample/pixel budgets, or long training schedules.

Normal cron exploration defaults to `AUTORESEARCH_MAX_COST_TIER=normal`. Expensive proposals are reserved for explicit promotion actions, candidate-linked follow-ups, or manual runs.

## Method-Family Arms

AutoResearch can propose bounded method-family arms in addition to scalar hyperparameter changes:

- `augmentation_policy`: toggles CPU-safe train-time flips and rotations.
- `z_context`: proposes prepared-data-compatible 2.5D z-offset contexts such as `[-4, 0, 4]` or `[-6, -3, 0, 3, 6]`.
- `sampling_strategy`: proposes hard-mining or warmup-then-hard sampling metadata under the existing exploration profile.

These arms are enabled by default through `AUTORESEARCH_ENABLE_METHOD_FAMILY_ARMS=1` semantics and still pass through `autoresearch.cost_tier` and runner profile checks. `full_tile_inference` and `label_audit` are fail-closed/manual arms; they are not emitted as automatic training configs unless explicitly wired later.

## Training Profiles

Generated exploration configs include:

```yaml
autoresearch:
  run_profile: exploration
  cost_tier: normal
```

The runner enforces exploration safety before training: bounded torch sample counts, no seed ensembles by default, bounded epochs, and lightweight defaults for augmentation/TTA when those fields are absent.

Promotion evidence configs should use:

```yaml
autoresearch:
  run_profile: promotion
  heldout_segment: "..."
```

Promotion profiles allow heavier evidence runs, but they must explicitly identify the held-out segment.

## Profiling

Set `AUTORESEARCH_PROFILE=1` to print per-cycle timings to stderr:

```bash
AUTORESEARCH_PROFILE=1 AUTORESEARCH_PROPOSALS=1 .venv/bin/python autoresearch.py --plan --json
```

Timing labels include `recent_runs`, `strategy_phase`, `proposal_generation`, `config_dump`, and `experiment_subprocess`. Use this mode when changing planner logic or investigating cron latency.

## Overnight Guarded Loop

Use `scripts/overnight_safe_loop.sh` for autonomous overnight research. The loop runs one guarded AutoResearch cycle at a time, records logs under `logs/overnight_safe_loop/<timestamp>/`, honors `logs/overnight_safe_loop.stop`, and uses `.overnight_safe_loop.lock` to avoid concurrent orchestrators.

Promotion-evidence cycles may run LOO or full-tile commands without creating a new experiment row. The loop treats those as `EVIDENCE_PROGRESS` and skips artifact quality gating for stale latest DB rows. If a cycle creates a new run, the loop quality-gates that specific new run before continuing.

The loop is designed to run continuously by default. `OVERNIGHT_MAX_SECONDS=0` means unlimited runtime; set a positive value to restore a wall-clock cap. Failed artifact quality gates are logged as `QUALITY_FAIL` and reject that run-local candidate, but they do not stop the orchestrator. Explicit stop remains available by creating `logs/overnight_safe_loop.stop`, and hard process failures, lock contention, resource guards, and promotion gates remain in force.

Low-load background mode keeps the machine responsive:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 AUTORESEARCH_NUM_WORKERS=2 scripts/overnight_safe_loop.sh
```

High-load overnight mode is appropriate when the machine can be saturated. On a 32-core host, 24 threads leaves some headroom while improving throughput:

```bash
SCROLL_RESEARCH_MAX_LOAD_HARD=64 SCROLL_RESEARCH_MAX_LOAD_SOFT=48 \
OVERNIGHT_MAX_SECONDS=0 OVERNIGHT_CYCLE_SLEEP_SECONDS=60 \
OVERNIGHT_JOB_TIMEOUT_SECONDS=5400 OVERNIGHT_HEARTBEAT_SECONDS=300 \
OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 NUMEXPR_NUM_THREADS=24 \
AUTORESEARCH_NUM_WORKERS=24 scripts/overnight_safe_loop.sh
```

Keep strict promotion gates enabled. Do not use high-load mode to bypass linked LOO, full-tile evidence, fixed-threshold checks, or quality verdicts. The loop also checks configured prepared-data paths before launching generated experiments, so stale pivot configs with missing NPZs are skipped instead of crashing the orchestrator.

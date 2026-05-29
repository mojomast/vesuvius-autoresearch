# AutoResearch Cost Tiers

AutoResearch assigns every generated proposal an `autoresearch.cost_tier` so cron can prefer high-throughput exploration before expensive validation.

## Tiers

- `cheap`: small NumPy/logreg-style runs with bounded pixel counts.
- `normal`: bounded single-model torch or small MLP runs suitable for routine cron exploration.
- `expensive`: seed ensembles, TTA, large sample/pixel budgets, or long training schedules.

Normal cron exploration defaults to `AUTORESEARCH_MAX_COST_TIER=normal`. Expensive proposals are reserved for explicit promotion actions, candidate-linked follow-ups, or manual runs.

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

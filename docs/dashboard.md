# Standalone Dashboard

The standalone dashboard is a Vesuvius-native view of configs, experiment runs, prepared-data metadata, fold maps, logs, and promotion commands. It does not depend on Hermes and does not include Hermes chat, sessions, memory, skills, cron management, or self-improvement panels.

## Launch

From the repository root:

```bash
python3 run_dashboard.py --host 127.0.0.1 --port 8765
```

Then open `http://127.0.0.1:8765`.

To export the same dashboard contract as JSON without starting a server:

```bash
python3 scripts/export_dashboard_snapshot.py --pretty
```

Use another repo root if needed:

```bash
python3 run_dashboard.py --repo-root /path/to/vesuvius-autoresearch
```

## What It Reads

- `experiments/experiments.db`
- `experiments/runs/*` artifact metadata and previewable text/JSON/CSV files
- `configs/*.yaml`, `*.yml`, and `*.json`
- prepared-data metadata under `data/prepared`, `data/real*`, and fold roots
- `data/**/fold_map.json`
- `logs/autoresearch.log` and `logs/autoresearch.lock`
- `logs/*summary.json` leave-one-out summaries with promotion readiness fields

The dashboard does not load model weights, NPZ arrays, or full tile probability maps by default.

## Safety Model

The default dashboard is read-only. Feature cards show copyable commands for terminal use, including seed-repeat LOO dry-runs, prepared-segment verification, fold-map dry-runs, full-tile self-tests, residual 2.5D smoke runs, and safe data expansion.

Run controls are intentionally not wired into the first standalone version. If future execution controls are added, they should be gated behind `VESUVIUS_DASHBOARD_ENABLE_RUNS=1`, use fixed allowlisted commands, and reject arbitrary shell input.

Artifact preview is path-guarded: files must resolve under `experiments/runs`.

## Shared Contract

Both the standalone dashboard and any external UI should consume the same neutral snapshot contract from `research_dashboard.snapshot.build_snapshot()` or `/api/research`:

```text
schema_version: vesuvius-dashboard/v1
project
inventory
datasets
configs
experiments
progress
research_summary
operations
capabilities
```

`research_summary.decision.promotion_gate.criteria` is the compact promotion checklist the UI renders as the Promotion Gate. It is intentionally stricter than peak validation score and expects robust held-out validation, seed-repeat leave-one-out evidence, full-tile evidence, and no promotion blockers.

Keep Vesuvius-specific data parsing and command inventory in `research_dashboard/*`. Hermes or any other host dashboard should render or proxy this contract instead of copying the domain logic.

## Parallel Development Workflow

- Add Vesuvius feature data to `research_dashboard/*` first.
- Update standalone UI rendering in `research_dashboard/app.py` if the contract changes.
- Update Hermes or other dashboards as consumers of the same `vesuvius-dashboard/v1` fields.
- Keep Hermes-specific UI, session state, memory, skills, and self-improvement features outside this repository.
- Run `python3 -m unittest discover -s tests` before committing.

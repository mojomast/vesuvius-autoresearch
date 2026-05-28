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

Dashboard snapshots use a short in-process cache by default (`2` seconds) to avoid repeatedly scanning configs, run artifacts, and fold metadata during active experiments. Set `VESUVIUS_DASHBOARD_SNAPSHOT_TTL_SEC=<seconds>` to tune the TTL, or `VESUVIUS_DASHBOARD_DISABLE_SNAPSHOT_CACHE=1` while debugging source changes. Restart the dashboard process after Python source edits; the cache only affects data snapshots.

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
mining
capabilities
```

`research_summary.decision.promotion_gate.criteria` is the compact promotion checklist the UI renders as the Promotion Gate. It is intentionally stricter than peak validation score and expects robust held-out validation, seed-repeat leave-one-out evidence, full-tile evidence, and no promotion blockers.

Seed-repeat LOO and full-tile evidence are candidate-linked: unrelated global summaries or full-tile artifacts may still be listed for context, but they do not clear the gate for a different robust candidate. LOO summaries should include at least three distinct successful seeds before `promotion_ready` is true.

`research_summary.candidate_evidence` and `research_summary.promotion_actions` expose the candidate-linked evidence used to accelerate review: linked LOO summary, weakest fold, full-tile segments covered, weak-fold full-tile status, quality verdict next actions, and copyable next commands. Weak-fold full-tile status is `done` only when an eligible whole-segment full-tile metrics file covers the worst LOO fold; when direct weak-fold artifact evidence is absent, linked candidate full-tile evidence for the same segment may satisfy the status. Weak-fold full-tile commands use the matching LOO held-out artifact when the seed-repeat JSONL is available, not a candidate artifact that held out a different segment. Public-directory full-tile commands include conservative whole-fetch and chunk-level `429` retry/backoff flags. These commands are read-only from the dashboard perspective; they are not executed by the UI and artifact-writing commands remain marked `safe_to_execute_from_dashboard: false`.

When candidate evidence has an unfinished diagnostic or quality action, `research_summary.decision.next_action` should prefer that action over generic promotion text. A ready gate starts review; it does not hide missing weak-fold full-tile diagnostics or quality-review findings. Candidate and linked LOO full-tile `quality_verdict: fail` blocks promotion when quality metrics are present; `review` stays promotable but inserts a quality review before `promotion_review`.

Full-tile evidence and leaderboard rows include `quality_next_actions` and `quality_next_action`. `pass` has no action, `review` asks for full-tile quality review, and `fail` asks for quality remediation before promotion.

`mining` exposes a dry-run hard-negative retrain plan. It inventories `data/mined/**/*.npz`, rejects mined files whose provenance matches the held-out segment, and emits copyable `scripts/infer_full_tile.py --mine-output ...` commands for overpredicting full-tile outputs. Those commands keep `--mine-output` under `data/mined/`, use a fresh `mining_refresh_<segment_id>` output directory, and omit `--overwrite` so prior full-tile evidence is not replaced. `fold_safe_extra_train_npzs_by_heldout` is the fold-scoped safety map; top-level `eligible_extra_train_npzs` is only the active weak-fold shortcut. The UI renders this as the Mining & Calibration Plan panel with calibration decision, fold-safe, inventory, and config-preview safety summaries. The dashboard never executes these artifact-writing commands.

`mining.calibration_mining_decisions` connects threshold-risk evidence to the next action: tighten the positive-rate cap when lower ratio caps preserve selected F1, mine hard negatives when lower caps collapse F1, or review threshold risk when evidence is incomplete.

When `config_preview` is present, `valid: false` is blocking. The dashboard may show warnings, but unsafe held-out overrides omit the YAML preview so the JSON output does not look runnable.

`research_summary.candidate_evidence.loo_full_tile` summarizes candidate-linked full-tile diagnostics across the seed-repeat LOO panel. It is separate from `full_tile`, which only describes full-tile outputs under the selected candidate artifact directory.

Calibration fields in `metrics.json` are advisory but important for review. `average_precision` should be compared against positive-label prevalence; AP near prevalence is random-like, while AP several times prevalence indicates useful ranking but not a calibrated threshold. `pred_positive_rate / val_positive_rate` above `3.5x` is promotion-blocking overprediction risk. `fixed_threshold_f1_low` is warning-level, but any present `fixed_threshold_status` other than `ok` blocks promotion until probability scale is calibrated for a fixed `0.5` threshold.

Keep Vesuvius-specific data parsing and command inventory in `research_dashboard/*`. Hermes or any other host dashboard should render or proxy this contract instead of copying the domain logic.

## Parallel Development Workflow

- Add Vesuvius feature data to `research_dashboard/*` first.
- Update standalone UI rendering in `research_dashboard/app.py` if the contract changes.
- Update Hermes or other dashboards as consumers of the same `vesuvius-dashboard/v1` fields.
- Keep Hermes-specific UI, session state, memory, skills, and self-improvement features outside this repository.
- Run `python3 -m unittest discover -s tests` before committing.

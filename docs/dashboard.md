# Interactive Dashboard

The standalone dashboard is a Vesuvius-native view of configs, experiment runs, prepared-data metadata, fold maps, logs, promotion commands, safe controls, and optional agent chat.

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

Interactive run controls are available only when `VESUVIUS_DASHBOARD_ENABLE_RUNS=1` and the server is launched with token auth. The backend accepts command IDs, not shell strings; it resolves IDs from the snapshot, requires `safe_to_execute_from_dashboard: true`, rejects artifact-writing commands, runs with `shell=False`, and captures stdout/stderr for display. Unsafe commands remain copy-only.

Example launch with controls:

```bash
VESUVIUS_DASHBOARD_TOKEN=change-me VESUVIUS_DASHBOARD_ENABLE_RUNS=1 \
  .venv/bin/python run_dashboard.py --host 127.0.0.1 --port 8765
```

Then open `http://127.0.0.1:8765?token=change-me`.

## Agent Chat

The dashboard includes an optional Agent Chat panel for asking about promotion blockers, next experiments, evidence triage, or dashboard settings. It is disabled by default and does not execute commands. The server sends a compact dashboard context to the configured agent endpoint and returns the reply.

Hermes is the default provider name for local users of this workspace:

```bash
VESUVIUS_DASHBOARD_TOKEN=change-me \
VESUVIUS_DASHBOARD_AGENT_ENABLED=1 \
VESUVIUS_DASHBOARD_AGENT_PROVIDER=hermes \
VESUVIUS_DASHBOARD_AGENT_BASE_URL=http://127.0.0.1:8766/api/agent/chat \
  .venv/bin/python run_dashboard.py
```

Users outside Hermes can point the same proxy at their own agent endpoint and supply a key either as an environment variable or in the UI's session-only API-key field:

```bash
VESUVIUS_DASHBOARD_TOKEN=change-me \
VESUVIUS_DASHBOARD_AGENT_ENABLED=1 \
VESUVIUS_DASHBOARD_AGENT_PROVIDER=openai-compatible \
VESUVIUS_DASHBOARD_AGENT_BASE_URL=https://example.com/v1/chat/completions \
VESUVIUS_DASHBOARD_AGENT_MODEL=my-agent-model \
VESUVIUS_DASHBOARD_AGENT_API_KEY=... \
  .venv/bin/python run_dashboard.py
```

Secrets are not included in `/api/research`, snapshots, command inventories, URLs, settings files, or logs. Browser-entered API keys are sent only with the chat or visual-analysis request and are not persisted by the dashboard beyond the current page/session.

### Agent Settings Fixes

Agent settings writes are a separate action mode. Enable them only when you want the agent to propose fixes to dashboard preferences:

```bash
VESUVIUS_DASHBOARD_TOKEN=change-me \
VESUVIUS_DASHBOARD_AGENT_ENABLED=1 \
VESUVIUS_DASHBOARD_AGENT_SETTINGS_WRITE=1 \
  .venv/bin/python run_dashboard.py
```

The agent still cannot edit files, run shell commands, or mutate experiment configs. In action mode it may return a structured `settings_patch` over a server-side allowlist of non-secret dashboard settings. The dashboard validates the patch, shows the diff-like proposal, and requires the user to click `Apply after review`. Unknown keys, secret-like values, embedded URL credentials, stale base versions, and invalid values are rejected by the backend.

Mutable settings currently cover only dashboard behavior: polling interval, decoded-gallery limit, default agent endpoint/model labels, and visual-analysis defaults. API keys, tokens, and passwords are never accepted as settings.

### Versioning And Recovery

Dashboard settings are stored under `.dashboard/settings.json` and `.dashboard/settings_snapshots/`. `.dashboard/` is ignored by git because it is local runtime state.

Every settings apply creates an immutable pre-change snapshot before writing the new version. Manual `Snapshot Now` creates an explicit recovery point. Rollback restores a selected snapshot and first snapshots the current settings, so rollback is itself reversible. The settings file includes a version id, update timestamp, and actor; the audit stream is appended to `.dashboard/settings_audit.jsonl`.

Relevant endpoints:

- `GET /api/settings`
- `GET /api/settings/snapshots`
- `POST /api/settings/apply`
- `POST /api/settings/snapshot`
- `POST /api/settings/rollback`

All settings mutation endpoints require dashboard token auth.

### Visual Analysis

Decoded-output visual analysis is opt-in and token-gated. It lets the configured agent inspect the guarded preview images already generated by the dashboard, such as `probability_map.npy` heatmaps and threshold masks, and return advisory feedback about coherent ink structure, blockiness, flooding, speckles, or noise.

Enable it with either the environment flag or the versioned dashboard setting:

```bash
VESUVIUS_DASHBOARD_TOKEN=change-me \
VESUVIUS_DASHBOARD_VISUAL_ANALYSIS_ENABLED=1 \
VESUVIUS_DASHBOARD_VISUAL_ANALYSIS_PROVIDER=hermes \
VESUVIUS_DASHBOARD_VISUAL_ANALYSIS_BASE_URL=http://127.0.0.1:8766/api/agent/chat \
  .venv/bin/python run_dashboard.py
```

The visual request uses the same secret policy as chat. Set `VESUVIUS_DASHBOARD_VISUAL_ANALYSIS_API_KEY` for a server-side key, reuse `VESUVIUS_DASHBOARD_AGENT_API_KEY`, or enter a session-only key in the UI. The backend reuses the artifact path guard, strips the request down to preview data URLs plus metrics/context, caps image payload size, and treats image contents as untrusted. Visual replies are advisory and cannot apply settings or run commands.

### Design Basis

The May 2026 safety posture follows current LLM application guidance: keep tools least-privileged, treat model output as untrusted input, use strict server-side schemas, require human approval before writes, avoid sending secrets to models, and keep rollback/audit paths outside the agent path. Useful references include the OWASP Top 10 for LLM Applications, the OWASP Prompt Injection Prevention Cheat Sheet, OpenAI structured-output/function-calling guidance, Anthropic Claude Code security guidance, NIST AI RMF, and Azure App Configuration snapshot guidance for immutable configuration rollback.

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
dashboard_settings
settings_snapshots
capabilities
```

`research_summary.decision.promotion_gate.criteria` is the compact promotion checklist the UI renders as the Promotion Gate. It is intentionally stricter than peak validation score and expects robust held-out validation, seed-repeat leave-one-out evidence, full-tile evidence, and no promotion blockers.

Seed-repeat LOO and full-tile evidence are candidate-linked: unrelated global summaries or full-tile artifacts may still be listed for context, but they do not clear the gate for a different robust candidate. LOO summaries should include at least three distinct successful seeds before `promotion_ready` is true.

`research_summary.candidate_evidence` and `research_summary.promotion_actions` expose the candidate-linked evidence used to accelerate review: linked LOO summary, weakest fold, full-tile segments covered, weak-fold full-tile status, quality verdict next actions, and copyable next commands. Weak-fold full-tile status is `done` only when an eligible whole-segment full-tile metrics file covers the worst LOO fold; when direct weak-fold artifact evidence is absent, linked candidate full-tile evidence for the same segment may satisfy the status. Weak-fold full-tile commands use the matching LOO held-out artifact when the seed-repeat JSONL is available, not a candidate artifact that held out a different segment. Public-directory full-tile commands include conservative whole-fetch and chunk-level `429` retry/backoff flags. These commands are read-only from the dashboard perspective; they are not executed by the UI and artifact-writing commands remain marked `safe_to_execute_from_dashboard: false`.

When candidate evidence has an unfinished diagnostic or quality action, `research_summary.decision.next_action` should prefer that action over generic promotion text. A ready gate starts review; it does not hide missing weak-fold full-tile diagnostics or quality-review findings. Candidate and linked LOO full-tile `quality_verdict: fail` blocks promotion when quality metrics are present; `review` stays promotable but inserts a quality review before `promotion_review`.

Full-tile evidence and leaderboard rows include `quality_next_actions` and `quality_next_action`. `pass` has no action, `review` asks for full-tile quality review, and `fail` asks for quality remediation before promotion.

`mining` exposes a dry-run hard-negative retrain and calibration plan. It inventories `data/mined/**/*.npz`, rejects mined files whose provenance matches the held-out segment, and emits both read-only cap comparison commands and artifact-writing mining commands. `cap_comparison_commands` run `scripts/compare_threshold_caps.py` against existing `metrics.json`/`metrics_by_threshold.csv` artifacts, are marked `writes_artifacts: false` and `safe_to_execute_from_dashboard: true`, and include 2.0x, 2.5x, 3.0x, and 3.5x retention checks by default. Mining commands run `scripts/infer_full_tile.py --mine-output ...`, keep `--mine-output` under `data/mined/`, use a fresh `mining_refresh_<segment_id>` output directory, and omit `--overwrite` so prior full-tile evidence is not replaced; they are marked `writes_artifacts: true` and `safe_to_execute_from_dashboard: false`. `fold_safe_extra_train_npzs_by_heldout` is the fold-scoped safety map; top-level `eligible_extra_train_npzs` is only the active weak-fold shortcut. The UI renders this as the Mining & Calibration Plan panel with read-only cap commands, artifact-writing mine commands, fold-safe inventory, and config-preview safety summaries. The dashboard never executes artifact-writing commands.

`mining.calibration_mining_decisions` connects threshold-risk evidence to the next action: tighten the positive-rate cap when lower ratio caps preserve about 95% of selected F1 and F0.5, mine hard negatives when lower caps collapse F1 or F0.5, or review threshold risk when evidence is incomplete. Each decision may include `cap_comparison_command_text` so reviewers can rerun the read-only cap check before mining or retraining.

When `config_preview` is present, `valid: false` is blocking. The dashboard may show warnings, but unsafe held-out overrides omit the YAML preview so the JSON output does not look runnable.

`research_summary.candidate_evidence.loo_full_tile` summarizes candidate-linked full-tile diagnostics across the seed-repeat LOO panel. It is separate from `full_tile`, which only describes full-tile outputs under the selected candidate artifact directory.

Calibration fields in `metrics.json` are advisory but important for review. `average_precision` should be compared against positive-label prevalence; AP near prevalence is random-like, while AP several times prevalence indicates useful ranking but not a calibrated threshold. `pred_positive_rate / val_positive_rate` above `3.5x` is promotion-blocking overprediction risk. `fixed_threshold_f1_low` is warning-level, but any present `fixed_threshold_status` other than `ok` blocks promotion until probability scale is calibrated for a fixed `0.5` threshold.

Keep Vesuvius-specific data parsing and command inventory in `research_dashboard/*`. Hermes or any other host dashboard should render or proxy this contract instead of copying the domain logic.

## Parallel Development Workflow

- Add Vesuvius feature data to `research_dashboard/*` first.
- Update standalone UI rendering in `research_dashboard/app.py` if the contract changes.
- Update Hermes or other dashboards as consumers of the same `vesuvius-dashboard/v1` fields.
- Keep Hermes-specific UI, session state, memory, skills, and self-improvement features outside this repository.
- Run `python3 -m pytest tests/` before committing.

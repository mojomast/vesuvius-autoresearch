# Vesuvius AutoResearch — Progress Prize Submission

## What This Is

Vesuvius AutoResearch is an evidence-gated ML research automation system for Vesuvius ink-detection experiments: it proposes bounded experiment changes, runs them against Vesuvius-style prepared segment data, records metrics and provenance, and blocks promotion unless candidates pass validation gates that are visible in a reviewer-safe dashboard.

## Why It Matters For The Challenge

Reading the scrolls requires repeated, careful experiments across fragile data splits, thresholds, seeds, and full-tile evaluations. Manual hyperparameter exploration makes it easy to overfit a patch, leak validation data, or promote a visually tempting false positive. This repository turns that workflow into a reproducible loop with explicit parameter bounds, metric contracts, provenance checks, leave-one-out evidence, full-tile checks, and documented failure analysis.

The Progress Prize criteria favor open-source tools that solve a concrete Vesuvius data problem, demonstrate use, document the path for others, and integrate modularly with community workflows. This submission targets that tooling need: safer, auditable experiment governance for Vesuvius segment research.

## What It Does (Technical Summary)

- Autonomous experiment proposal with `PARAM_BOUNDS` clamping and duplicate-config avoidance.
- `MetricContract` enforcement across 1,600+ recorded experiments.
- Evidence-gated promotion: leave-one-out seed-repeat checks, full-tile inference, positive-rate alarms, fixed-threshold diagnostics, and eligibility checks.
- An interactive standalone dashboard that turns the experiment database, configs, fold maps, full-tile artifacts, LOO summaries, promotion gates, quality verdicts, safe diagnostic controls, and recommended next actions into one reviewer-facing control room with collapsed long panels, fold-matrix actions, folded decoded outputs, and a filterable scroll-contained run ledger.
- `VesuviusHarness` abstraction so the same propose/evaluate/promote loop can pivot to other scroll research workflows.
- GitHub Actions CI with 222+ tests, a 30-minute cron workflow, baseline guards, and artifact upload.
- Synthetic demo data generator so reviewers can exercise the loop without downloading protected or large Vesuvius data first.

## Dashboard Value

The dashboard is one of the most useful parts of the submission because it makes the research loop inspectable instead of opaque. A reviewer can see which run is being considered, why it is or is not promotable, which fold or segment is blocking progress, whether a full-tile artifact is eligible, and what command should be run next. It surfaces the evidence that normally gets buried across SQLite rows, JSON metrics, threshold CSVs, LOO summaries, and local artifacts.

This matters for Vesuvius work because false positives can look convincing. The dashboard helps prevent accidental overclaiming by exposing positive-rate ratios, AP/prevalence lift, fixed-threshold status, full-tile quality checks, provenance eligibility, and promotion warnings in one place. It is not just a UI; it is a practical safety layer for deciding whether an ink-detection result is ready for more compute, more review, or rejection.

The dashboard now includes gated interactive controls and an optional Agent Chat panel. Controls require token auth plus `VESUVIUS_DASHBOARD_ENABLE_RUNS=1`, execute only allowlisted non-artifact-writing commands, and reject arbitrary shell input. The reviewer UI keeps long research surfaces manageable: the usefulness leaderboard is collapsed by default with search/status controls, the validation matrix includes filter and best-run actions plus safe validation command cards, the decoded-output gallery is folded until needed, and the experiment ledger is a sticky-header scroll window with text, status, segment, and F1 filters. Candidate evidence profiles the hard `20230530172803` fold so low AP/prevalence failures lead to fold audit and sampling-pressure review instead of threshold relaxation. Agent Chat defaults to a Hermes-style local provider for this workspace, while reviewers outside Hermes can configure their own agent endpoint and API key through `VESUVIUS_DASHBOARD_AGENT_*` settings or a session-only UI key. With `VESUVIUS_DASHBOARD_AGENT_SETTINGS_WRITE=1`, the agent can propose allowlisted dashboard settings fixes that the user must review and apply. Every settings apply creates a local recovery snapshot, and rollback snapshots the current state first so recovery is reversible. Optional visual analysis lets the agent inspect decoded-output previews when explicitly enabled; those replies are advisory and cannot execute commands or modify settings.

## Quick Start (Anyone Can Run This)

```bash
git clone https://github.com/mojomast/vesuvius-autoresearch.git
cd vesuvius-autoresearch
python3 -m pip install -e .
python3 scripts/generate_synthetic_data.py
python3 autoresearch.py --plan --json
```

For a clean local setup that also regenerates configs:

```bash
python3 scripts/generate_synthetic_data.py --n-train 256 --n-val 64
python3 scripts/setup_data.py --data-dir ./data
python3 autoresearch.py --plan --json
```

The synthetic data follows the runner schema used by this repo: `images` are `[N,C,H,W]` with three z-offset channels and `labels` are `[N,1,H,W]`. The generated metadata marks provenance as `synthetic` so demo artifacts cannot be confused with real scroll evidence.

## Research Results

This repository has run 1,600+ recorded experiments with full metric/provenance tracking. The infrastructure is hardened: metric contracts, parameter bounds, Vesuvius-specific harnessing, promotion evidence packaging, positive-rate alarms, full-tile inference checks, CI, synthetic demo generation, and dashboard-driven review are all in place.

The ink detector itself is not a champion model. The best eligible tiled result so far is `val_f1=0.2363` on a fragment/full-tile-style validation region, and promotion remains blocked by cross-seed generalization failures. Seeds `11001` and `11018` behaved acceptably in recent checks, while `15050` showed threshold-cliff flooding and replacement seed `15073` produced zero-precision folds. The current bottleneck is model architecture and calibration robustness, not the experiment infrastructure.

The submission claim is therefore intentionally narrow: this is a reusable open-source research automation and evidence-gating tool for the Vesuvius community, not a state-of-the-art ink model.

## Work Completed Since The Initial Prize Package

- Added a reproducible synthetic data generator and quickstart path so reviewers can run the loop without downloading real Vesuvius data first.
- Prepared and documented the ScrollPrize Community Projects PR: <https://github.com/ScrollPrize/villa/pull/991>.
- Expanded the health checks to `226 passed` tests and kept the final branch clean after each research cycle.
- Continued the real-data research loop through positive-rate alarm diagnosis, seed-analysis follow-up, and residual 2.5D architecture exploration.
- Added CPU-safe residual 2.5D configs and evidence logs. The best sampled residual candidate, `20260529T034850Z_d0ee3406`, reached sampled `val_f1=0.4664` and eligible full-tile `val_f1=0.2132`, but failed 2-seed LOO on zero folds for `20230530172803`, so it remains diagnostic-only.
- Updated research status docs and best-practice guidance to explain what worked, what failed, and where the next useful work should focus.

## How To Extend This

The harness layer is documented in [`harness/README.md`](harness/README.md). To adapt the loop, implement `ResearchHarness` for a different model, data format, surface-detection task, annotation-quality loop, or scroll-processing workflow while keeping the same bounded proposal, experiment recording, and promotion-gating lifecycle.

The most useful next integrations would be:

- OME-Zarr/Zarr adapters for direct community-format ingestion.
- Stronger 2.5D/3D model families behind the same promotion gates.
- Additional full-scroll or cross-scroll evaluation harnesses.
- Community-contributed proposal families for augmentations, calibration, and data-quality loops.

## Links

- Repo: <https://github.com/mojomast/vesuvius-autoresearch>
- Harness docs: [`harness/README.md`](harness/README.md)
- Research best practices: [`docs/research_best_practices_may2026.md`](docs/research_best_practices_may2026.md)
- Evidence and next moves: [`docs/next_best_moves_may2026.md`](docs/next_best_moves_may2026.md)
- Prize requirements research: [`docs/prize_submission_research.md`](docs/prize_submission_research.md)
- Competitive landscape: [`docs/competitive_landscape.md`](docs/competitive_landscape.md)

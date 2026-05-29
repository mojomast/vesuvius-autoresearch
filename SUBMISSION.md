# Vesuvius AutoResearch — Progress Prize Submission

## What This Is

Vesuvius AutoResearch is an autonomous ML research loop for Vesuvius ink-detection experiments: it proposes bounded experiment changes, runs them against Vesuvius-style prepared segment data, records metrics and provenance, and blocks promotion unless candidates pass evidence-gated validation.

## Why It Matters For The Challenge

Reading the scrolls requires repeated, careful experiments across fragile data splits, thresholds, seeds, and full-tile evaluations. Manual hyperparameter exploration makes it easy to overfit a patch, leak validation data, or promote a visually tempting false positive. This repository turns that workflow into a reproducible loop with explicit parameter bounds, metric contracts, provenance checks, leave-one-out evidence, full-tile checks, and documented failure analysis.

The Progress Prize criteria favor open-source tools that solve a concrete Vesuvius data problem, demonstrate use, document the path for others, and integrate modularly with community workflows. This submission targets that tooling need: safer, auditable experiment governance for Vesuvius segment research.

## What It Does (Technical Summary)

- Autonomous experiment proposal with `PARAM_BOUNDS` clamping and duplicate-config avoidance.
- `MetricContract` enforcement across 1,419+ recorded experiments.
- Evidence-gated promotion: leave-one-out seed-repeat checks, full-tile inference, positive-rate alarms, fixed-threshold diagnostics, and eligibility checks.
- `VesuviusHarness` abstraction so the same propose/evaluate/promote loop can pivot to other scroll research workflows.
- GitHub Actions CI with 222+ tests, a 30-minute cron workflow, baseline guards, and artifact upload.
- Synthetic demo data generator so reviewers can exercise the loop without downloading protected or large Vesuvius data first.

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

This repository has run 1,419+ recorded experiments with full metric/provenance tracking. The infrastructure is hardened: metric contracts, parameter bounds, Vesuvius-specific harnessing, promotion evidence packaging, positive-rate alarms, full-tile inference checks, CI, and synthetic demo generation are all in place.

The ink detector itself is not a champion model. The best eligible tiled result so far is `val_f1=0.2363` on a fragment/full-tile-style validation region, and promotion remains blocked by cross-seed generalization failures. Seeds `11001` and `11018` behaved acceptably in recent checks, while `15050` showed threshold-cliff flooding and replacement seed `15073` produced zero-precision folds. The current bottleneck is model architecture and calibration robustness, not the experiment infrastructure.

The submission claim is therefore intentionally narrow: this is a reusable open-source research automation and evidence-gating tool for the Vesuvius community, not a state-of-the-art ink model.

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

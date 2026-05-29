# Contributing

This project is research software for Vesuvius Scroll Prize experiments. Contributions should improve reproducibility, validation safety, and reviewer visibility.

## Local Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[ingestion]'
```

## Tests

Run the full unit suite before proposing source changes:

```bash
.venv/bin/python -m pytest tests/
```

Use targeted tests while iterating, then run the full suite before committing. Build steps that change behavior should add or update tests in the same commit.

## Development Workflow

Create configs with:

```bash
.venv/bin/python scripts/setup_data.py --data-dir ./data
```

Inspect the next autoresearch tick without writing artifacts:

```bash
.venv/bin/python autoresearch.py --plan --json
```

Promotion automation is opt-in with `AUTORESEARCH_AUTO_PROMOTE=1`; leave it disabled unless the candidate and fold map have been reviewed.

## Artifact Policy

Do not commit generated artifacts:

- prepared data under `data/`
- `*.npy`, `*.npz`, model weights, checkpoints, or ONNX files
- `experiments/experiments.db`
- `experiments/runs/`
- `logs/`
- full-tile predictions or decoded-output images
- generated `configs/auto_*`

If a result depends on large files, document where they are stored and include checksums in `artifacts/README.md` or `weights/README.md`.

## Research Claims

Research claims should include:

- exact config path and commit hash
- data source, segment IDs, and held-out region
- validation mode and fold map
- AP/prevalence lift, positive-rate ratio, F1, precision, recall, and fixed-threshold status
- full-tile quality verdict when available
- a statement that training labels/crops/mined negatives do not overlap the prediction region

## Configs

Keep hand-authored configs in `configs/`. Generated `configs/auto_*` files are local search artifacts and should not be added unless intentionally curated into documentation.

CI cron runs require `configs/baseline.yaml` to be present in the repo; regenerate it with `scripts/setup_data.py` before enabling live scheduled runs.

## Harness Changes

New research loops should implement `harness.ResearchHarness` and include tests that prove proposal, evaluation, promotion decision, and promotion hooks compose without dashboard assumptions.

## Pre-commit Checklist

Before every commit:

- [ ] `python -m pytest tests/` passes with zero failures
- [ ] `python autoresearch.py --plan --json` exits 0
- [ ] `from harness.vesuvius_harness import VesuviusHarness` succeeds
- [ ] No legacy unittest test-runner references remain in docs

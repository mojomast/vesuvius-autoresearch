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
.venv/bin/python -m unittest discover -s tests
```

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

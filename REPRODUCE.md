# Reproducibility

These commands are intended for a clean checkout. They avoid committing generated data or artifacts.

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

For public Vesuvius data ingestion, install optional dependencies and accept official tooling terms if required:

```bash
.venv/bin/python -m pip install -e '.[ingestion]'
.venv/bin/python -m vesuvius.accept_terms --yes
```

## Sanity Tests

```bash
.venv/bin/python -m unittest discover -s tests
```

## Prepare Public Segment Data

```bash
.venv/bin/python scripts/prepare_vesuvius_segment_npz.py \
  --segment-id 20230827161847 \
  --catalog-source public-directory \
  --output-dir data/real/segment_20230827161847 \
  --level 1 \
  --patch-size 64
```

## Run One Experiment

```bash
.venv/bin/python run_experiment.py --config configs/baseline.yaml
```

Outputs are written to `experiments/runs/<run_id>/` and should not be committed.

## Dry-Run Leave-One-Out Configs

```bash
.venv/bin/python scripts/evaluate_leave_one_out.py \
  --base-config configs/robust_hard_negative_prratio2p0_followup.yaml \
  --fold-map data/real_cross_folds_expanded_combined/fold_map.json \
  --output-jsonl logs/prratio2_followup_loo.jsonl \
  --summary-json logs/prratio2_followup_loo.summary.json \
  --seeds 11001,11018,15050 \
  --dry-run
```

## Full-Tile Inference And Fold-Safe Mining

```bash
.venv/bin/python scripts/infer_full_tile.py \
  --artifact experiments/runs/<run_id> \
  --segment-id <segment_id> \
  --output-dir experiments/runs/<run_id>/full_tile_<segment_id> \
  --catalog-source public-directory \
  --level 1 \
  --z-offsets=-4,0,4 \
  --patch-size 64 \
  --stride 32 \
  --batch-size 8 \
  --device cpu \
  --mine-output data/mined/<name>.npz \
  --mine-max-patches 512 \
  --overwrite
```

Only add `data/mined/<name>.npz` to `dataset.extra_train_npzs` for folds whose held-out segment is different from the mined segment.

## Dashboard And Audit

```bash
.venv/bin/python run_dashboard.py --host 127.0.0.1 --port 8765
.venv/bin/python scripts/export_dashboard_snapshot.py --pretty
.venv/bin/python scripts/audit_research_state.py --markdown
```

## Expected Generated Paths

- `data/**`: prepared NPZs and metadata.
- `experiments/experiments.db`: local experiment index.
- `experiments/runs/**`: run configs, metrics, weights, full-tile outputs.
- `logs/**`: LOO summaries and cron logs.
- `data/mined/**`: hard-negative NPZs.

All of the above are local artifacts and should remain outside git.

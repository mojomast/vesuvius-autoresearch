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
Identical resolved configs are deduped by `config_signature`; inspect the returned `deduped` field before assuming a new run directory was created.

## Dry-Run Leave-One-Out Configs

```bash
.venv/bin/python scripts/evaluate_leave_one_out.py \
  --base-config configs/robust_hard_negative_prratio2p0_followup.yaml \
  --fold-map data/real_cross_folds_expanded_combined/fold_map.json \
  --output-jsonl logs/prratio2_followup_loo.jsonl \
  --summary-json logs/prratio2_followup_loo.summary.json \
  --seeds 11001,11018,15050 \
  --execution-mode fold-major \
  --dry-run
```

For non-dry-run seed repeats, prefer `--execution-mode fold-major --limit-worker-threads` so all seeds for one held-out fold run in the same worker and common BLAS/OpenMP thread pools do not oversubscribe CPU cores.

## Full-Tile Inference And Fold-Safe Mining

Before writing any mining artifacts, generate a dry-run plan from the current dashboard snapshot:

```bash
.venv/bin/python scripts/plan_hard_negative_retrain.py --pretty
```

The planner emits copyable `scripts/infer_full_tile.py --mine-output ...` commands, currently eligible mined NPZs, rejected held-out leaks, and preview-only config patches for `dataset.extra_train_npzs`. Planner mining commands write under `data/mined/` and a fresh `experiments/runs/<run_id>/mining_refresh_<segment_id>` output directory; they do not include `--overwrite` and must not reuse an existing full-tile metrics directory.

Preview a fold-specific config without writing files:

```bash
.venv/bin/python scripts/plan_hard_negative_retrain.py \
  --heldout-segment 20230522181603 \
  --base-config configs/robust_hard_negative_prratio2p0_followup.yaml \
  --pretty
```

Read `fold_safe_extra_train_npzs_by_heldout` before running LOO: mined files from a segment are rejected for that same held-out segment but may be eligible for other folds.

`--base-config` is intentionally path-guarded: it must resolve to a `.yaml` or `.yml` file inside the repository root, and the preview is printed in JSON only. If `--heldout-segment` changes `autoresearch.heldout_segment` but the config train/val NPZ paths still point to another fold, `config_preview.valid` is false and no YAML preview is emitted.

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
  --mine-max-patches 512
```

Only add `data/mined/<name>.npz` to `dataset.extra_train_npzs` for folds whose held-out segment is different from the mined segment.

## Dashboard And Audit

```bash
.venv/bin/python run_dashboard.py --host 127.0.0.1 --port 8765
.venv/bin/python scripts/export_dashboard_snapshot.py --pretty
.venv/bin/python scripts/audit_research_state.py --markdown
```

## Queued Worker Smoke Test

The optional SQLite job queue dispatches existing config paths through the same `experiments.runner.run_experiment` code path. To run at most one queued job:

```bash
.venv/bin/python scripts/run_job_worker.py --once
```

Queue databases are local generated artifacts and must not be committed.
Queued experiment configs default to config-signature dedupe keys, so duplicate enqueues reuse an existing queued/completed job row when the config can be loaded.

The guarded AutoResearch launcher honors explicit `AUTORESEARCH_PROPOSALS`, but keeps promotion-action pauses enabled for unattended safety. To intentionally bypass the promotion pause through the guard for diagnostics, set both `AUTORESEARCH_CONTINUE_AFTER_PROMOTION_ACTION=1` and `SCROLL_RESEARCH_ALLOW_PROMOTION_OVERRIDE=1`.

## Expected Generated Paths

- `data/**`: prepared NPZs and metadata.
- `experiments/experiments.db`: local experiment index.
- `experiments/jobs.db`: local queued-worker state.
- `experiments/runs/**`: run configs, metrics, weights, full-tile outputs.
- `logs/**`: LOO summaries and cron logs.
- `data/mined/**`: hard-negative NPZs.

All of the above are local artifacts and should remain outside git.

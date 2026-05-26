# Vesuvius AutoResearch

Minimal continuous experiment pipeline for Vesuvius ScrollPrize ink-detection research.

The active workflow uses real Vesuvius data only. If real prepared NPZs or official data access are unavailable, runs fail loudly instead of falling back to fake data.

## Research Workflow Principles

- Optimize for rare-positive ink detection, not generic accuracy or loss alone.
- Use cross-segment/cross-scroll validation when labeled real data exists. The installed `vesuvius` package catalog is stale, but the public Scroll 1 segment directory exposes 33 exact Zarr-plus-inklabel pairs, so `configs/baseline.yaml` now uses cross-segment validation.
- Promote runs by threshold-swept `val_f1`; a lower loss that predicts no positive ink is not useful.
- Log threshold diagnostics for every run in `metrics_by_threshold.csv` and inspect `best_threshold`, `average_precision`, `pred_positive_rate`, and probability quantiles before trusting a result.
- Keep AutoResearch proposals interpretable: one change per generated config.
- Keep each research cycle focused on one data scope. The active baseline uses one train segment and one held-out validation segment; AutoResearch preserves those exact NPZ paths and only changes model/training/evaluation knobs. Add additional segments only by creating an explicit fold config, then compare folds separately.
- Treat `source: vesuvius_segment_zarr`, `vesuvius_public_segment_zarr`, and validated `prepared_npz` as valid active research sources.

## Run Artifacts

Each experiment writes:

- `config.json`: resolved config and data metadata.
- `metrics.json`: headline metrics, threshold-swept best F1/F0.5, positive rates, and probability diagnostics.
- `metrics_by_threshold.csv`: precision/recall/F0.5/F1 over candidate thresholds.
- `weights.npy`: tiny NumPy logistic-regression weights.
- `run_summary.md`: human-readable steering summary.

## Real Vesuvius Data Ingestion

The runner consumes prepared NPZ files with this schema:

```text
images: float32-like [N, C, H, W], normalized image patches; C>=1 allows multi-slice Zarr context
labels: float32-like [N, 1, H, W], binary ink labels
```

Use explicit NPZ paths to train on real prepared data without synthetic fallback:

```yaml
dataset:
  research_scope: focused_pair
  train_npz: /path/to/real_train_patches.npz
  val_npz: /path/to/real_val_patches.npz
  patch_size: 64
```

If either `train_npz` or `val_npz` is provided, both are required and are validated before training. Invalid real-data paths fail loudly instead of falling back to synthetic data.

Optional official tooling for preparing those NPZ files:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install vesuvius tifffile pillow zarr s3fs
.venv/bin/vesuvius.accept_terms --yes
```

Practical ingestion targets:

- Official `vesuvius.Volume("<segment_id>")` segment surface volumes with `segment.inklabel` when labels are available.
- Local Kaggle-style fragment folders containing `surface_volume/*.tif`, `inklabels.png`, and optionally `mask.png`.
- OME-Zarr CT volumes from public open-data mirrors for unlabeled inference/evaluation, paired with labels only where available.

For local TIFF/PNG fragment or segment folders, prepare NPZs with:

```bash
python scripts/prepare_local_vesuvius_npz.py \
  --root /path/to/fragment_or_segment \
  --output data/real/train_frag.npz \
  --split train \
  --patch-size 64 \
  --samples 512 \
  --positive-fraction 0.5
```

Then point a config at the generated train/validation NPZ files.

For a same-segment comparison dataset:

```bash
.venv/bin/python scripts/prepare_vesuvius_segment_npz.py \
  --segment-id 20230827161847 \
  --output-dir data/real/segment_20230827161847 \
  --level 1 \
  --patch-size 64
```

To inspect all official catalog candidates before preparing data:

```bash
.venv/bin/python scripts/prepare_vesuvius_segment_npz.py --list-labeled --catalog-source public-directory
```

To prepare every labeled segment exposed by the public Scroll 1 segment directory, without synthetic fallback:

```bash
.venv/bin/python scripts/prepare_vesuvius_segment_npz.py --all-labeled --catalog-source public-directory --output-root data/real
```

The installed `vesuvius==0.2.4` catalog exposes only one segment, but the public directory lists 33 exact labeled segment pairs. The active cross-segment baseline uses:

- Train segment: `20230827161847`
- Validation segment: `20230520175435`
- Zarr/label base: `https://dl.ash2txt.org/other/dev/scrolls/1/segments/54keV_7.91um/`
- Current prepared cross-segment paths: `data/real_cross/segment_20230827161847/train.npz` and `data/real_cross/segment_20230520175435/val.npz`

Do not load all 33 labeled segments into one headline experiment by default. Use them to create focused fold configs, for example train on one segment and validate on one held-out segment, then rotate the held-out segment when the current fold has been understood.

The stale installed-catalog blocker can also be partly resolved by installing optional package dependencies (`lxml`, `nest_asyncio`) and refreshing the package catalog, but public-directory label filtering is still required because many public Zarrs do not have labels.

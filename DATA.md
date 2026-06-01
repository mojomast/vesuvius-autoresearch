# Data Policy

This repository does not redistribute raw Vesuvius Challenge or Scroll Prize data, prepared NPZ datasets, labels, model outputs, mined hard negatives, or trained weights.

## Sources

Active research uses public Vesuvius/Scroll Prize data, primarily Scroll 1 labeled segment Zarr volumes and ink labels from the public directory:

- Scroll Prize data portal: https://scrollprize.org/data
- Public labeled segment mirror: `https://dl.ash2txt.org/other/dev/scrolls/1/segments/54keV_7.91um/`

The code also supports local fragment-style folders with `surface_volume/*.tif`, `inklabels.png`, and optional `mask.png`.

## Licenses And Attribution

Vesuvius Challenge data is not covered by this repository's MIT code license. Follow the data portal terms and dataset-specific licenses before downloading or publishing outputs.

For Scrolls 1-4 and Fragments 1-6, cite the EduceLab-Scrolls dataset and Parsons et al. 2023 as requested by the Scroll Prize data documentation. For newer scans, cite the Vesuvius Challenge CT scans of Herculaneum papyri dataset.

## Expected Local Paths

Generated data lives under `data/` and is intentionally ignored by git. Common generated paths include:

- `data/real/`
- `data/real_cross/`
- `data/real_cross_folds_v2/`
- `data/real_cross_folds_expanded_combined/`
- `data/mined/`

Prepared NPZ files must use this schema:

```text
images: float32-like [N, C, H, W]
labels: float32-like [N, 1, H, W]
```

Each NPZ should have a sidecar `*.metadata.json` with source, segment ID, patch size, split, and validation-safety metadata.

Curated configs that point at prepared NPZ files are expected to reference real existing local paths. AutoResearch skips pivot or generated configs whose `dataset.train_npz` or `dataset.val_npz` is missing instead of substituting synthetic data.

## Safety Rules

- Do not train on labels, crops, mined negatives, or manually reviewed regions from a held-out validation or submitted prediction region.
- Keep full-tile mined negatives fold-safe: a mined NPZ from segment `X` must not be listed in `dataset.extra_train_npzs` for a fold where `X` is held out.
- Treat `64 x 64` patch windows at public 8 micron scale as the default hallucination-control window size unless a submission protocol explicitly justifies otherwise.
- Do not commit raw scans, labels, NPZs, NPYs, model checkpoints, experiment databases, full-tile predictions, or decoded text outputs.
- Keep dated research notes under `docs/archive/`; keep generated data and evidence artifacts under ignored runtime paths such as `data/`, `experiments/runs/`, and `logs/`.

## Preparing Data

List labeled public segments:

```bash
.venv/bin/python scripts/prepare_vesuvius_segment_npz.py --list-labeled --catalog-source public-directory
```

Prepare one labeled segment with 2.5D context:

```bash
.venv/bin/python scripts/prepare_vesuvius_segment_npz.py \
  --segment-id 20230827161847 \
  --catalog-source public-directory \
  --output-dir data/real_25d/segment_20230827161847 \
  --patch-size 64 \
  --z-offsets=-8,-4,0,4,8 \
  --val-tiled \
  --val-stride 64
```

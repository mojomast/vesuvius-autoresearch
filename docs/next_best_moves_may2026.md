# Next-Best Moves, May 2026

These moves are the next promotion path for this repo. They are ordered to reduce false confidence from lucky seeds, positive-biased patches, or accidental segment leakage.

## 1. Seed-Repeat Leave-One-Out

Run the same committed config through `scripts/evaluate_leave_one_out.py` for at least three seeds before promotion. Keep each seed as a separate config so `config.json`, `metrics.json`, and the LOO JSONL files remain reproducible.

Promotion summary should report:

- Per-seed median LOO `val_f1`.
- Median-over-seeds of median LOO `val_f1`.
- Worst fold across all seeds.
- Mean average precision across successful folds.
- Any fold with high `pred_positive_rate` or near-zero recall.

Dry-run the fold wiring before launching remote-data or GPU work:

```bash
python scripts/evaluate_leave_one_out.py \
  --base-config configs/robust_multisegment_dice035_expanded.yaml \
  --fold-map data/real_cross_folds_expanded_combined/fold_map.json \
  --output-jsonl logs/robust_multisegment_dice035_expanded_seed11001_loo.jsonl \
  --summary-json logs/robust_multisegment_dice035_expanded_seed11001_loo.summary.json \
  --dry-run
```

## 2. Safe Data Expansion

Expand labeled data by preparing additional public segments into separate segment directories, then regenerate a fold map where each held-out segment has a training NPZ that excludes it. Do not append all segments into one headline training set unless the validation segment is outside that combined file.

Safe expansion checklist:

- Store each segment under `data/real_cross_folds_v*/segment_<id>/` or an equivalent explicit versioned root.
- Preserve metadata JSON next to every NPZ.
- Keep `patch_size: 64` unless the run is clearly marked diagnostic-only.
- Use `--negative-max-positive-rate` for training negatives to avoid ambiguous ink-contaminated negatives.
- Use tiled validation when measuring promotion quality.

Example preparation command:

```bash
python scripts/prepare_vesuvius_segment_npz.py \
  --segment-id 20230827161847 \
  --catalog-source public-directory \
  --output-dir data/real_cross_folds_v3/segment_20230827161847 \
  --patch-size 64 \
  --train-samples 2048 \
  --val-tiled \
  --val-stride 64 \
  --z-offsets -4,0,4
```

## 3. Full-Tile Inference

Patch-sampled F1 can overstate segment utility when validation patches are positive-biased. Promotion runs should use validation NPZs prepared with `--val-tiled --val-stride 64` so every candidate sees a uniform tile grid over the held-out validation region.

Record these fields from `metrics.json` for full-tile runs:

- `evaluation_region.type: whole_segment`
- `best_threshold`
- `average_precision`
- `val_f1`, `val_f05`, and `val_loss`
- `val_positive_rate`
- `pred_positive_rate`
- `prob_p95`, `prob_mean`, and `prob_max`
- `promotion_checks.validation_setup` and `promotion_checks.resolved_data`
- `promotion_checks.eligible` and `promotion_checks.warnings`

Promotion criteria for full-tile results:

- Use only runs with whole-segment evaluation provenance, matching validation setup metadata, and `promotion_checks.eligible: true`.
- Preserve or improve tiled `val_f1` and `val_f05` without a large jump in `pred_positive_rate`.
- Keep `val_positive_rate` consistent with the held-out segment metadata.
- Inspect probability quantiles before promotion when a threshold sweep is the main source of lift.

Public-directory full-tile reads may be rate limited. Prefer dashboard-generated weak-fold commands with `--public-retry-count` and `--public-retry-delay-sec` so zarr fetches cool down between `429 Too Many Requests` responses instead of immediately failing and being rerun manually.

If a model only wins on positive-biased validation but fails on tiled validation, keep it diagnostic-only.

## 4. TTA And Seed Ensembling

Add ensembling only after the single-seed LOO gate is stable. The initial ensemble should be simple probability averaging:

- Horizontal and vertical flip test-time augmentation.
- Three independently trained seeds.
- Per-fold threshold selection from the validation sweep, not a threshold copied from another fold.

Report ensemble lift against the best individual seed on the same tiled folds. If lift comes mainly from increased prediction rate, inspect precision and probability quantiles before promotion.

## 5. 2.5D Residual U-Net

The current `tiny_torch_unet` already consumes multi-channel NPZs, so 2.5D data can be prepared now with `--z-offsets`. The next model-family change should be a residual U-Net variant that keeps 64 x 64 patches and treats adjacent z slices as channels.

Minimum config intent:

- `model.name: residual_25d_torch_unet` for residual 2.5D experiments.
- `model.input_mode: z_offsets_as_channels`.
- `dataset.z_offsets` matching the preparation command.
- LOO seed-repeat promotion before comparing against champions.

Use `configs/next_best_moves_robust_template.yaml` as the planning template, and keep runnable experiments tied to model names supported by the current training and inference entry points.

# Next-Best Moves, May 2026

These moves are the next promotion path for this repo. They are ordered to reduce false confidence from lucky seeds, positive-biased patches, or accidental segment leakage.

## 1. Seed-Repeat Leave-One-Out

Status on 2026-05-29: COMPLETE for post-fix candidate `20260529T005819Z_28af43a2` generated from `auto_20260529T005748Z_3_evaluation_threshold_0p35.yaml`. The sampled validation run recorded `val_f1=0.3740`, `average_precision=0.2598`, `pred_positive_rate=0.3839`, and `pred_positive_rate / val_positive_rate=2.99` under `max_pred_positive_rate_ratio=3.0`, `positive_rate_loss_weight=0.03`, `positive_rate_loss_tolerance=0.02`, and `max_train_samples=4096`.

Seed-repeat LOO results from `logs/20260529T005819Z_28af43a2_seedrepeat_loo.summary.json`: `promotion_ready=true`, median-over-seeds median `val_f1=0.1747`, mean AP `0.1196`, and worst fold `20230522181603` with `val_f1=0.0426`. This is measured evidence only; the weak fold remains the main promotion risk and should be inspected before any broader robustness claim.

Promotion decision on 2026-05-29: DO NOT PROMOTE. Full-tile inference completed for `20230520175435` and `20230522181603`, but both candidate full-tile results have `promotion_checks.eligible=false` because the artifact lineage records `spatial-same-segment` validation provenance. See `logs/promotion_evidence_28af43a2.md`.

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

Status on 2026-05-29: COMPLETE for candidate `20260529T005819Z_28af43a2` on validation segment `20230520175435` and worst-fold segment `20230522181603`.

| Segment | val_f1 | val_f05 | AP | pred/val ratio | eligible |
| --- | ---: | ---: | ---: | ---: | --- |
| 20230520175435 | 0.2187 | 0.1755 | 0.1463 | 2.9745 | false |
| 20230522181603 | 0.1267 | 0.1081 | 0.0868 | 2.9641 | false |

Hard-negative mining was not applied: the worst-fold full-tile pred/val ratio was `2.9641`, below the `3.5` flooding threshold, and `prob_p95=0.3535` was slightly above `best_threshold=0.3500`. The current blocker is promotion-ineligible provenance, not confirmed flooding.

Next active milestone: keep TTA/seed-ensemble and positive-rate calibration sweeps diagnostic until a promotion-eligible held-out full-tile lineage is generated.

Patch-sampled F1 can overstate segment utility when validation patches are positive-biased. Promotion runs should use validation NPZs prepared with `--val-tiled --val-stride 64` so every candidate sees a uniform tile grid over the held-out validation region.

Record these fields from `metrics.json` for full-tile runs:

- `evaluation_region.type: whole_segment`
- `best_threshold`
- `average_precision`
- `val_f1`, `val_f05`, and `val_loss`
- `val_positive_rate`
- `pred_positive_rate`
- `prob_p95`, `prob_mean`, and `prob_max`
- `brier_score`, `expected_calibration_error`, and `ap_prevalence_lift`
- `fixed_threshold_f1`, `fixed_threshold_status`, and `threshold_selection`
- `promotion_checks.validation_setup` and `promotion_checks.resolved_data`
- `promotion_checks.eligible` and `promotion_checks.warnings`
- dashboard-derived `quality_verdict`, `quality_next_action`, and `quality_next_actions` from the snapshot

Promotion criteria for full-tile results:

- Use only runs with whole-segment evaluation provenance, matching validation setup metadata, and `promotion_checks.eligible: true`.
- Preserve or improve tiled `val_f1` and `val_f05` without a large jump in `pred_positive_rate`.
- Keep `val_positive_rate` consistent with the held-out segment metadata.
- Inspect probability quantiles before promotion when a threshold sweep is the main source of lift.
- Resolve dashboard quality next actions before promotion review. A `review` verdict requires human inspection of full-tile quality; a `fail` verdict blocks promotion until remediated.

Public-directory full-tile reads may be rate limited. Prefer dashboard-generated weak-fold commands with `--public-retry-count`, `--public-retry-delay-sec`, `--public-chunk-delay-sec`, `--public-chunk-retry-count`, and `--public-chunk-retry-delay-sec` so Zarr fetches cool down between `429 Too Many Requests` responses instead of immediately failing and being rerun manually. The chunk flags are opt-in; without them, layer reads keep the faster direct path.

When a decoded full-tile output fails because of flooding, speckles, or suspicious positive-rate ratio, mine false-positive hard negatives from the same full-tile pass instead of launching another blind sweep. `scripts/infer_full_tile.py` can write a runner-schema mined NPZ with `--mine-output path/to/hard_negatives.npz --mine-max-patches 512`; by default mining uses the selected best threshold and keeps patches with near-zero label positives. Only combine mined negatives into training folds where the mined segment is training-eligible, never into the held-out segment for that fold.

If a model only wins on positive-biased validation but fails on tiled validation, keep it diagnostic-only.

Metric interpretation:

- `average_precision` measures ranking quality across thresholds. Its random baseline is approximately the positive-label prevalence. AP around `0.09` on a segment with `0.05` prevalence is useful signal; AP around `0.10` on `0.087` prevalence is only a small lift.
- `fixed_threshold_f1` at `0.5` is currently diagnostic only. Recent calibrated full-tile panels often have `fixed_threshold_f1=0.0` because probability maxima are below `0.5`; best operating thresholds are selected by sweep and positive-rate constraints near `0.28-0.34`.
- `prratio2p5` is the intermediate safety setting when `2.0x` loses too much recall but still avoids the floodier `3.0x`/`3.5x` range. `prratio3` is stricter than `3.5x` while recovering more recall than `2.5x`. The original `4x` cap remains a recall/F1 reference, not a default promotion target when full-tile runs ride the cap.
- AutoResearch now treats `evaluation.max_pred_positive_rate_ratio` and `training.positive_rate_loss_tolerance` as first-class search dimensions so future proposals can test the `2.5-3.0` cap band and tighter residual positive-rate tolerances without one-off config edits.
- Promote-phase sweeps now include loss-calibration proposals before Dice-weight mutations so `positive_rate_loss_tolerance: 0.01` can be exercised even when plateau logic requires mutation-family diversity. The first tolerance-inclusive batch produced `20260529T011645Z_20f27abf` (`val_f1=0.3783`, AP `0.2621`, pred/val ratio `2.99`, tolerance `0.01`, cap `3.0`) and rejected unconstrained tolerance/TTA variants with pred/val ratios above `4.0` as unsafe.

## 4. TTA And Seed Ensembling

Add ensembling only after the single-seed LOO gate is stable. The initial ensemble should be simple probability averaging:

- Horizontal and vertical flip test-time augmentation.
- Three independently trained seeds.
- Per-fold threshold selection from the validation sweep, not a threshold copied from another fold.

Report ensemble lift against the best individual seed on the same tiled folds. If lift comes mainly from increased prediction rate, inspect precision and probability quantiles before promotion.

Before packaging ensemble evidence, compare the linked single-seed and ensemble full-tile artifacts without writing new outputs:

```bash
python3 scripts/compare_full_tile_metrics.py \
  --pair 20230522181603=experiments/runs/<single_seed_run>/full_tile_20230522181603/metrics.json,experiments/runs/<ensemble_run>/full_tile_20230522181603/metrics.json \
  --pair 20230530212931=experiments/runs/<single_seed_run>/full_tile_20230530212931/metrics.json,experiments/runs/<ensemble_run>/full_tile_20230530212931/metrics.json \
  --fail-on-core-regression \
  --markdown
```

Treat a mixed package as diagnostic evidence: promote the specific improved segment only if candidate lineage is linked, and do not generalize ensemble lift across folds that regress F1, F0.5, or AP.

Seed ensembling and TTA can change probability scale. Re-run threshold calibration and full-tile checks after any ensemble/TTA change; do not assume an ensemble makes threshold `0.5` valid.

## 5. 2.5D Residual U-Net

Post-fix 2026-05-29 evidence supports keeping 4096-sample, positive-rate-controlled candidates in the search space. The new tiny torch post-fix batch produced `20260529T005748Z_7d9b2b4d` (`prratio2.5`, `val_f1=0.3447`, AP `0.2420`, pred/val ratio `2.45`) and `20260529T005819Z_28af43a2` (`prratio3.0`, `val_f1=0.3740`, AP `0.2598`, pred/val ratio `2.99`). The unconstrained seed-repeat candidate `20260529T005804Z_12a24594` had higher sampled `val_f1=0.3887` but pred/val ratio `3.87`, outside the target safe band.

The current `tiny_torch_unet` already consumes multi-channel NPZs, so 2.5D data can be prepared now with `--z-offsets`. The next model-family change should be a residual U-Net variant that keeps 64 x 64 patches and treats adjacent z slices as channels.

Minimum config intent:

- `model.name: residual_25d_torch_unet` for residual 2.5D experiments.
- `model.input_mode: z_offsets_as_channels`.
- `dataset.z_offsets` matching the preparation command.
- LOO seed-repeat promotion before comparing against champions.
- For residual CPU searches, `max_train_samples: 4096` has stronger local LOO evidence than the stricter `2048/2.0x` path; enable it only when the sample budget is explicit and promotion checks remain fold-safe.

Use `configs/next_best_moves_robust_template.yaml` as the planning template, and keep runnable experiments tied to model names supported by the current training and inference entry points.

# Villa E Hard-Fold Root Cause Research

Read-only sources: `experiments/experiments.db` queried with `.venv/bin/python` and Python `sqlite3`; existing full-tile metrics at `experiments/runs/20260530T004150Z_0b659aec/full_tile_weak_fold_20230530172803/metrics.json`.

## Summary

`20230530172803` is a consistently hard validation/heldout segment. Across 54 DB runs where this segment is the validation segment, best sampled `val_f1` is only `0.048638`, median `val_f1` is `0.028024`, best AP is only `0.035827`, and no run reaches `AP > 0.1`. This is not an isolated seed failure: both random tiny U-Net and residual 2.5D hard-mining families fail, and LOO promotion summaries identify this fold as the minimum fold even when overall LOO evidence is promotion-ready.

The strongest current root-cause hypothesis is label/prevalence mismatch plus local sampling scarcity, not model capacity or z-layer choice. The segment has low validation prevalence (`0.018045` in the current full validation cohort, `0.021615` full tile), but training mixtures feeding it are much richer (`~0.25-0.34` train positive rate), so models overpredict weak ink-like texture while AP remains near prevalence.

## Target Fold Evidence

Rows selected from `experiments` where `validation_setup.val_segment_id = 20230530172803` or `dataset.val_npz` contains `segment_20230530172803`.

| Metric | Value |
| --- | ---: |
| Target validation rows | 54 |
| Best sampled `val_f1` | 0.048638 |
| Median sampled `val_f1` | 0.028024 |
| Mean sampled `val_f1` | 0.025343 |
| Best sampled AP | 0.035827 |
| Median sampled AP | 0.024699 |
| Runs with `AP > 0.1` | 0 |
| Runs with `val_f1 > 0.1` | 0 |
| Fixed-threshold status | 51 `ok`, 3 `weak` |

Validation cohort split:

| Val samples / prevalence | Runs | Best F1 | Median F1 | Best AP | Median AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| 120 / 0.000000 | 6 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 128 / 0.008770 | 14 | 0.034888 | 0.010713 | 0.019933 | 0.012700 |
| 1633 / 0.018045 | 34 | 0.048638 | 0.035369 | 0.035827 | 0.028805 |

Top target-validation runs:

| Run | F1 | AP | Fixed | Fixed F1 | Prev | Model | Seed | Samples | Sampling | Pos patch | Hard neg | Focal | Scope |
| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `20260529T130542Z_c1df5ccd` | 0.048638 | 0.035152 | ok | 0.072535 | 0.018045 | `residual_25d_torch_unet` | 11045 | 1024 | hard_mining | 0.45 | 0.75 | | `multi_segment_residual25d_prcontrol_worstfold` |
| `20260529T130314Z_8ef112e4` | 0.047179 | 0.035152 | ok | 0.072535 | 0.018045 | `residual_25d_torch_unet` | 11045 | 1024 | hard_mining | 0.45 | 0.75 | | `multi_segment_residual25d_prcontrol_worstfold` |
| `20260530T004156Z_f649496e` | 0.046497 | 0.035827 | ok | 0.073009 | 0.018045 | `residual_25d_torch_unet` | 11045 | 1024 | hard_mining | 0.45 | 0.75 | 0.25 | `multi_segment_targeted_promo_focal_bce_sparse_ink` |
| `20260529T130040Z_db8e64f6` | 0.045172 | 0.035152 | ok | 0.072535 | 0.018045 | `residual_25d_torch_unet` | 11045 | 1024 | hard_mining | 0.45 | 0.75 | | `multi_segment_residual25d_prcontrol_worstfold` |
| `20260529T050843Z_55b63039` | 0.044317 | 0.030303 | ok | 0.046183 | 0.018045 | `tiny_torch_unet` | 11018 | 4096 | random | | | | `multi_segment_balanced_prratio2p75_tolerance0p008` |
| `20260530T004150Z_0b659aec` | 0.043435 | 0.030085 | ok | 0.058120 | 0.018045 | `residual_25d_torch_unet` | 11018 | 1024 | hard_mining | 0.45 | 0.75 | 0.25 | `multi_segment_targeted_promo_focal_bce_sparse_ink` |

## Prevalence Compared With Other Folds

`20230530172803` is among the lowest-prevalence nonzero folds and is the weakest by both F1 and AP.

| Validation segment | Runs | Best F1 | Median F1 | Best AP | Median AP | Median prevalence | AP > 0.1 runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20230520175435` | 1188 | 0.781806 | 0.366736 | 0.849842 | 0.234134 | 0.125562 | 1165 |
| `20230522181603` | 152 | 0.315339 | 0.056011 | 0.188170 | 0.029822 | 0.015821 | 1 |
| `20230522215721` | 101 | 0.201437 | 0.146169 | 0.134228 | 0.096278 | 0.035723 | 35 |
| `20230530172803` | 54 | 0.048638 | 0.028024 | 0.035827 | 0.024699 | 0.018045 | 0 |
| `20230530212931` | 101 | 0.299175 | 0.114843 | 0.248149 | 0.072058 | 0.048857 | 10 |
| `20230531121653` | 101 | 0.361247 | 0.278525 | 0.255013 | 0.176131 | 0.156399 | 101 |
| `20230601193301` | 42 | 0.125063 | 0.082499 | 0.079257 | 0.057026 | 0.026884 | 0 |
| `20230611014200` | 42 | 0.270491 | 0.182722 | 0.190848 | 0.117577 | 0.082914 | 36 |

Low prevalence alone does not fully explain the collapse: `20230522181603` has slightly lower median prevalence (`0.015821`) but still reaches `0.315339` best F1 and one AP over `0.1`. The target fold appears uniquely difficult beyond prevalence.

## LOO And Full-Tile Evidence

Promotion summaries containing the target segment show the same pattern.

| Base run | Promotion ready | Target fold F1 | Target fold AP | Overall mean F1 | Overall median F1 | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `20260529T035202Z_8f01550f` | false | 0.006588 | 0.012951 | 0.189003 | 0.164020 | Target fold generated zero-precision/recall warnings for seeds `11001` and `11018`. |
| `20260530T003819Z_6b3111c8` | true | 0.039621 | 0.029879 | 0.187498 | 0.163957 | Target fold is still the minimum F1 fold, though all three seeds are nonzero. |

The weak-fold full-tile check for `20260530T004150Z_0b659aec` on `20230530172803` reinforces that sampled validation is not hiding a good full-tile result:

| Full-tile metric | Value |
| --- | ---: |
| Label positive rate | 0.021615 |
| AP | 0.027372 |
| Selected F1 at cap 2.75 | 0.014940 |
| Selected precision | 0.010278 |
| Selected recall | 0.027337 |
| Selected pred positive rate | 0.057491 |
| Selected pred/val ratio | 2.659699 |
| Fixed-threshold F1 | 0.041316 |
| Fixed-threshold pred/val ratio | 7.767470 |

The full-tile result says the model can emit positives, but calibrated positives are mostly false positives; letting the threshold overpredict improves recall but not practical quality.

## Root-Cause Hypotheses

Label noise or label sparsity is the leading hypothesis. The target fold has very low AP lift and full-tile precision near `0.01`, indicating that model ranking barely separates labeled ink from background. The early 120-sample cohort has zero label prevalence, while later cohorts have nonzero prevalence, suggesting sampling/label-window sensitivity for this segment.

Sampling mismatch is also likely. Target-fold training positive rates are typically `0.25-0.34`, while validation/full-tile prevalence is only `0.018-0.022`. Hard mining with `positive_patch_fraction=0.45` may still overrepresent ink-positive contexts and train a detector that is too liberal for this villa segment. Runs with train-positive-rate bucket around `0.329` had the best median target F1, but even there median F1 was only `0.043998` and AP stayed below `0.036`, so sampling changes alone are unlikely to solve the fold without label-aware checks.

Capacity is not the primary blocker in current evidence. Both `tiny_torch_unet` base 8 and `residual_25d_torch_unet` base 8 fail similarly: residual 2.5D best F1 `0.048638`, tiny U-Net best F1 `0.044317`. There is no target-fold evidence that increasing channels or depth has been tried, but the near-prevalence AP suggests more capacity may overfit false positives unless labels/sampling are fixed first.

Z-layer choice is not distinguishable from current DB evidence. All 54 target-validation runs use resolved z-offsets `[-4, 0, 4]`. Because no target-fold runs test a different z stack, z-layer failure remains possible but unproven. A z ablation should come after a label/sampling sanity check, or be bundled as a single controlled contrast.

## Minimal Villa-Label Experiment

Goal: determine whether `20230530172803` is hard because the labels/sampled patches are misleading, before spending compute on larger models.

1. Freeze the current strongest family: `residual_25d_torch_unet`, base channels `8`, `[-4,0,4]`, `max_train_samples=1024`, hard mining, `hard_negative_fraction=0.75`, flips/TTA on, positive-rate cap `2.75`, and seeds `11018` and `11045`.
2. Build two tiny validation manifests for `20230530172803`: a label-positive patch set from existing mask-positive pixels and a matched high-texture label-negative patch set from nearby same-tile regions. Keep this as an evaluation-only artifact, not a training change.
3. Run the existing best artifact(s) on that villa-label audit set and report AP, precision/recall at the current selected threshold, and a small grid of thresholds. If AP is still near `0.03`, labels/features are not separable under current inputs; if AP jumps, the current validation/full-tile sampling is misrepresentative.
4. Train exactly one controlled variant with lower positive patch pressure, `positive_patch_fraction=0.25` or `0.30`, keeping hard negatives at `0.75`. Evaluate only on `20230530172803` and the villa-label audit set. This tests sampling mismatch without changing model capacity, z layers, loss, and thresholding all at once.
5. Only if the audit AP is meaningfully above current AP should run a single z ablation `[-8,-4,0,4,8]`; otherwise prioritize manual label review or villa-specific hard-negative sampling.

Success criteria should be AP-first: require target sampled AP above `0.1` or villa-label audit AP at least `3x` current full-tile AP before treating F1 threshold gains as meaningful.

## Conclusion

`20230530172803` is consistently the hard fold across sampled validation, LOO, and full-tile evidence. No run has AP above `0.1`, and the best available full-tile AP is only `0.027372`. The next smallest useful experiment is a villa-label audit plus one lower-positive-pressure sampling run, not a broad capacity or z-stack sweep.

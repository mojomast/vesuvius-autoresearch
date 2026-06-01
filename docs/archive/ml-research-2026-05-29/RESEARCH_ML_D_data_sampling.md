# ML Data Sampling Research

Read-only sources: `DATA.md`, `METHOD.md`, `IMPROVEMENTS.md`, `autoresearch.py`, `src/autoresearch/{schemas,proposals,strategy}.py`, `experiments/runner.py`, `configs/*.yaml`, and `experiments/experiments.db`.

## Summary

The strongest evidence is not from broad sample-count scaling alone. The best multi-segment residual 2.5D results combine fold-safe 64x64 patches, resolved `[-4, 0, 4]` z-channel data, bounded positive-rate controls, hard-negative-oriented patch sampling, flip augmentation/TTA, and validation prevalence-aware thresholding. Validation prevalence varies by nearly an order of magnitude across folds, so sampling should be fold-conditioned rather than globally fixed.

## Validation Positive Rate By Fold

Experiment DB unique-config grouping shows these validation prevalence medians:

| Validation segment | Runs | Median val_positive_rate | Range |
| --- | ---: | ---: | ---: |
| `20230520175435` | 1108 | 0.125562 | 0.067303-0.139774 |
| `20230522181603` | 147 | 0.015821 | 0.014345-0.016081 |
| `20230522215721` | 98 | 0.035723 | 0.035723-0.035723 |
| `20230530164535` | 5 | 0.000000 | 0.000000-0.000000 |
| `20230530172803` | 51 | 0.018045 | 0.000000-0.018045 |
| `20230530212931` | 98 | 0.048857 | 0.048857-0.048857 |
| `20230531121653` | 98 | 0.156399 | 0.156399-0.156399 |
| `20230601193301` | 39 | 0.026884 | 0.000000-0.026884 |
| `20230611014200` | 39 | 0.082914 | 0.000000-0.082914 |
| `20230827161847` | 119 | 0.073009 | 0.073009-0.333206 |
| `20230901184804` | 109 | 0.028230 | 0.028230-0.028230 |

Implication: a single global positive sampling target is unsafe. Folds near `0.016-0.036` are very different from `0.125-0.156`, and runs with selected prediction rates around `0.38-0.52` are still 3x to 20x prevalence depending on fold.

## Z-Offset Evidence

The DB stores `dataset.z_offsets` as absent for all runs, but resolved NPZ metadata shows `[-4, 0, 4]` for 1905 runs. Those runs include the current best sampled result: `20260529T182028Z_8718e6cb`, residual 2.5D, focused pair, `val_f1=0.781806`, `AP=0.849842`, `AP/prevalence=12.627`, `val_positive_rate=0.067303`, `pred_positive_rate=0.057491`.

Only 14 legacy same-segment/focused real runs had no resolved z-offset metadata; their best `val_f1=0.591138` on `val_positive_rate=0.333206` is less relevant to held-out multi-segment promotion.

Config inventory contains one unrun/planning template with `dataset.z_offsets: [-8, -4, 0, 4, 8]` in `next_best_moves_robust_template.yaml`. No DB evidence shows that 5-channel z context has been trained/evaluated. The tried production evidence is therefore effectively `[-4, 0, 4]`; `[-8, -4, 0, 4, 8]` remains a clean next ablation.

## Sample Budgets Vs F1

Torch `max_train_samples` tried in DB:

| max_train_samples | Runs | Best val_f1 | Median val_f1 | Notes |
| ---: | ---: | ---: | ---: | --- |
| 0 | 420 | 0.390414 | 0.183095 | Full/all available patches; not best under current residual scope. |
| 256 | 9 | 0.370931 | 0.123059 | Too small except quick smoke tests. |
| 384 | 3 | 0.364839 | 0.364284 | Limited evidence. |
| 512 | 268 | 0.781806 | 0.365285 | Contains focused-pair residual best; strong but not multi-segment proof. |
| 768 | 3 | 0.366846 | 0.365616 | Limited evidence. |
| 1024 | 304 | 0.489411 | 0.387133 | Best multi-segment residual hard-mining result. |
| 1536 | 85 | 0.376487 | 0.202076 | Mostly earlier LOO sweeps. |
| 3072 | 37 | 0.389958 | 0.159700 | Earlier hard-mining LOO/residual sweeps. |
| 4096 | 356 | 0.430552 | 0.146807 | Useful for robust expanded runs, but not consistently better than 1024. |

NumPy MLP `max_train_pixels` tried:

| max_train_pixels | Runs | Best val_f1 | Median val_f1 |
| ---: | ---: | ---: | ---: |
| 300000 | 1 | 0.364826 | 0.364826 |
| 600000 | 2 | 0.366149 | 0.365753 |
| 700000 | 1 | 0.364378 | 0.364378 |
| 1200000 | 1 | 0.366864 | 0.366864 |

Implication: pixel count scaling in the NumPy MLP barely moved F1. Torch patch budget has a stronger interaction with model family and scope; `1024` is the best CPU-safe multi-segment budget observed, while `4096` should be reserved for confirmatory/fold-specific runs after sampling mix is fixed.

## Hard Mining In Multi-Segment Scope

Runner behavior: `patch_sampling` or `sampling_strategy` of `hard_mining` selects positive patches plus high-texture zero-ink patches. It records `selected_patch_positive_rate`, `positive_patches_selected`, and related metrics. Fold safety for external `extra_train_npzs` is enforced separately.

Best multi-segment hard-mining results:

| Scope | max_train_samples | hard_negative_fraction | positive_patch_fraction | Best val_f1 | Best run | Notes |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `multi_segment_residual25d_prcontrol_worstfold` | 1024 | 0.75 | 0.45 | 0.489411 | `20260529T035202Z_8f01550f` | `precision=0.409`, `recall=0.609`, `pred/val=1.49x`. |
| `multi_segment_residual25d_prcontrol_worstfold_stride32_allval` | 1024 | 0.75 | 0.45 | 0.482921 | `20260529T043409Z_2ed634ce` | Higher recall, higher pred rate. |
| `multi_segment_residual25d_prcontrol_worstfold` | 1024 | 0.85 | 0.45 | 0.443343 | `20260529T035036Z_b7f61463` | Worse than 0.75; more negative pressure reduced quality. |
| `multi_segment_residual25d_prcontrol_worstfold` | 4096 | 0.75 | 0.45 | 0.428599 | `20260528T155452Z_8846f887` | Larger sample count did not beat 1024. |
| `multi_segment_robust_expanded` | 1024 | 0.5 | implicit/default | 0.429987 | `20260529T124014Z_560e8dbe` | Competitive but below residual 2.5D hard-mining setup. |

Against non-hard multi-segment comparators, the best non-hard robust expanded `1024` result reached `0.432147`, and best non-hard balanced `4096` result reached `0.430552`. Hard mining is therefore beneficial mainly in the residual 2.5D worst-fold scope and should not be treated as universally beneficial without holding scope/model fixed.

## Positive Sampling And Validation Prevalence

Explicit `training.sample_positive_fraction` was only observed at `0.25` for the NumPy MLP. It produced `train_positive_rate` median `0.231857` against `val_positive_rate` median `0.125562`, with best `val_f1=0.366654`. That was not materially better than un-enriched MLP/logreg runs in the same focused-pair family.

Prepared-path positive fraction evidence is stronger: `data/real_cross/segment_20230827161847_pf0p15_n2048/train.npz` yielded median `train_positive_rate=0.105223`, close to the focused validation prevalence median `0.125562`, and includes the current best focused residual run. In contrast, older `data/real_cross/segment_20230827161847/train.npz` had median `train_positive_rate=0.362533` versus `val_positive_rate=0.125562`, a clear prevalence mismatch.

Patch-level hard mining used `positive_patch_fraction` values `0.25`, `0.35`, `0.40`, `0.45`, `0.50`, `0.65`, and `0.80` in different scopes/configs. The best residual 2.5D multi-segment result used `0.45` with `hard_negative_fraction=0.75`; a planned weak-fold fallback uses `0.35` and `hard_negative_fraction=0.85`, but DB evidence says `0.85` underperformed `0.75` in the 1024-sample residual scope.

## Augmentation Evidence And Untried Options

Current runner support:

- Train-time `augment_flips` concatenates horizontal and vertical flips.
- Eval `tta_flips` averages identity, horizontal flip, vertical flip, and both-flip predictions.
- No random intensity, rotation, z-channel, elastic, crop-jitter, or cutout augmentation exists in `experiments/runner.py` today.

Observed augmentation results are confounded by scope, but useful:

- Best `(augment_flips=True, tta_flips=True)` multi-segment residual hard-mining run: `val_f1=0.489411`.
- Best `(augment_flips=False, tta_flips=False)` multi-segment robust run: `val_f1=0.436621`.
- Best `(augment_flips=False, tta_flips=True)` focused residual run: `val_f1=0.781806`, not a multi-segment promotion result.

Small-change augmentations the runner could support next:

- 90-degree rotations and transpose flips using `np.rot90`/axis swaps for train and matching TTA. This is the lowest-risk extension because labels transform identically.
- Mild brightness/contrast jitter per patch/channel after loading `Xtr_img`, bounded to preserve normalized CT scale assumptions.
- Additive Gaussian noise or blur on inputs only, applied after patch selection and before tensor conversion.
- Z-channel dropout or z-channel reversal for `z_offsets_as_channels`; useful for testing whether the model overfits one layer.
- Small integer translation/crop-pad jitter within 64x64 patches, with identical label shift.
- Random cutout/coarse dropout on inputs only, not labels, to reduce crackle/texture overfit.

Do not add all at once. Rotations and z-channel dropout are the cleanest first ablations because they test geometric and 2.5D robustness without changing fold composition.

## Recommended Sampling Curriculum

1. Lock validation reporting by fold prevalence. Always report `val_positive_rate`, selected `pred_positive_rate / val_positive_rate`, and fixed-threshold status by fold before comparing F1.
2. Use `[-4, 0, 4]` as the baseline z context because it is the only materially tried resolved z-offset set. Run one controlled `[-8, -4, 0, 4, 8]` ablation only after reproducing the baseline on the same fold/scope.
3. For focused/fold-diagnostic residual CPU runs, start with `max_train_samples=512` only for quick checks. Do not promote from focused-pair evidence alone.
4. For multi-segment residual 2.5D, use `max_train_samples=1024`, `patch_sampling=hard_mining`, `positive_patch_fraction=0.45`, `hard_negative_fraction=0.75`, `augment_flips=true`, and `tta_flips=true` as the current best sampling baseline.
5. For weak low-prevalence folds such as `20230522181603` and `20230530172803`, lower positive patch pressure cautiously before raising hard-negative pressure: test `positive_patch_fraction=0.35` with `hard_negative_fraction=0.75` before `0.85`.
6. Only scale to `4096` after the 1024-sample configuration gives acceptable pred/val ratios on the target fold. Existing evidence does not show that 4096 alone improves F1.
7. Avoid pixel-level `sample_positive_fraction=0.25` as a primary path. It over-enriches train prevalence relative to most validation folds and has weak F1 evidence.
8. Add one augmentation ablation at a time: first 90-degree rotations, then z-channel dropout/reversal, then intensity jitter. Evaluate each under the same fold and sampling setup.
9. Promote only with leave-one-segment-out and full-tile checks, per `METHOD.md`; sampled F1 is insufficient when validation prevalence differs this much.

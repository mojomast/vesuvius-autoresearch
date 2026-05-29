# ML Model Gap Research

Read-only research performed against `/home/mojo/projects/vesuvius-autoresearch` on 2026-05-29.

## Sources Read

- Code: `autoresearch.py`, `src/autoresearch/schemas.py`, `src/autoresearch/proposals.py`, `src/autoresearch/strategy.py`, `experiments/runner.py`, `harness/base.py`, `harness/vesuvius_harness.py`, `harness/README.md`.
- Docs: `IMPROVEMENTS.md`, `METHOD.md`, `DATA.md`.
- Configs: all visible `configs/*.yaml`, with emphasis on pivot configs in `PIVOT_CONFIGS`: `robust_multisegment_dice035_expanded.yaml`, `robust_calibrated_prloss_w0p03_lr0012_prratio3_seed11018.yaml`, `robust_tta_seed_ensemble.yaml`, `residual_25d_torch_unet_cpu.yaml`.
- Database: queried `experiments/experiments.db` read-only. Root `experiments.db` exists but is empty and has no `experiments` table.

## Runner And Harness Model Support

`experiments/runner.py` supports four effective model families:

| Model name | Implementation | Notes |
| --- | --- | --- |
| `tiny_numpy_ink_logreg` | NumPy pixel logistic regression over local/channel/texture features | Fast sanity-check baseline. |
| `tiny_numpy_mlp` | One-hidden-layer NumPy MLP over the same feature extractor | Lightweight nonlinear baseline. |
| `tiny_torch_unet` | Small Torch U-Net | One encoder block, mid block, one decoder block. Uses `base_channels`; runner ignores `model.depth`. |
| `residual_25d_torch_unet` / `torch_residual_25d_unet` | Residual 2.5D Torch U-Net | Two-level residual U-Net with group norm; intended for z-offset channels. Uses `base_channels`; runner ignores `model.depth`. |

The harness does not add architectures. `VesuviusHarness` is an adapter over the existing autoresearch planner/runner, and custom harnesses can propose arbitrary configs, but no additional concrete model implementation is present in `harness/`.

## Architectures Tried In `experiments/experiments.db`

Database scope: 1,919 experiment rows from `2026-05-25T20:56:39Z` through `2026-05-29T19:10:21Z`.

| Architecture | Runs | Best `val_f1` | AP on best-F1 run | Best-F1 run | Best `average_precision` | F1 on best-AP run | Best-AP run |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `residual_25d_torch_unet` | 232 | 0.781806 | 0.849842 | `20260529T182028Z_8718e6cb` | 0.849842 | 0.781806 | `20260529T182028Z_8718e6cb` |
| `tiny_torch_unet` | 1,253 | 0.437516 | 0.326367 | `20260529T180703Z_0e0005c8` | 0.370841 | 0.425949 | `20260529T161015Z_55fb5421` |
| `tiny_numpy_ink_logreg` | 374 | 0.591138 | 0.424039 | `20260525T210418Z_2983030f` | 0.424062 | 0.573754 | `20260525T211312Z_c1be72f0` |
| `tiny_numpy_mlp` | 60 | 0.367187 | 0.232878 | `20260526T052001Z_18be3e47` | 0.239023 | 0.366736 | `20260526T045005Z_da5fd5c1` |

Best-F1 config sketches:

| Architecture | Key settings on best-F1 run |
| --- | --- |
| `residual_25d_torch_unet` | `base_channels=8`, `epochs=5`, `batch_size=4`, `max_train_samples=512`, `learning_rate=0.0012`, `augment_flips=false`, `tta_flips=true`, no positive-rate cap. |
| `tiny_torch_unet` | `base_channels=8`, `epochs=5`, `batch_size=8`, `max_train_samples=1024`, `learning_rate=0.0012`, `augment_flips=true`, `tta_flips=true`, no positive-rate cap. |
| `tiny_numpy_ink_logreg` | `depth=2`, `epochs=4`, `learning_rate=0.16`, fixed threshold `0.5`. |
| `tiny_numpy_mlp` | `depth=3`, `hidden_units=12`, `epochs=2`, `learning_rate=0.04147`, threshold `0.031`. |

## Supported Architectures Not Tried Or Undertried

- Not tried as separate DB architecture: `torch_residual_25d_unet` alias. This is only an alias for the same residual 2.5D implementation, so it is not a true architecture gap.
- Undertried: `tiny_numpy_mlp`, with only 60 rows and poor best AP/F1 relative to logreg and Torch. It is supported, but current evidence suggests it was not competitive.
- Undertried: `residual_25d_torch_unet`, with 232 rows versus 1,253 `tiny_torch_unet` rows despite much better best F1/AP. This is the clearest model-family search gap.
- Not implemented: deeper U-Nets, attention U-Nets, pretrained encoders, nnU-Net-style variants, transformers, and 3D CNNs. They are outside current runner/harness support.

## Torch U-Net Coverage Vs `PARAM_BOUNDS`

`PARAM_BOUNDS` allows `base_channels=4..16`, `model.depth=1..3`, `batch_size=2..32`, `epochs=2..20`, and `max_train_samples=1..4096`. Torch runner coverage is highly concentrated near CPU-safe defaults.

| Parameter | Bound | DB values observed for Torch rows | Coverage gap |
| --- | --- | --- | --- |
| `model.base_channels` | 4..16 | 4, 6, 8, 10, 12, 16 | Mostly `8` (1,393/1,485 Torch rows). `16` only 21 rows, `4` only 15, `12` only 2, `10` only 1. |
| `model.depth` | 1..3 | none on Torch rows | Runner does not use `model.depth` for Torch models. The bound/proposal surface suggests a tunable parameter that has no Torch effect. |
| `training.batch_size` | 2..32 | 2, 4, 8, 12, 16, 24 | Mostly `8` (1,433/1,485). `32` never observed; `2`, `12`, `24` only one run each; `16` three runs. |
| `training.epochs` | 2..20 | 1, 2, 3, 4, 5, 6, 7, 8 | Mostly `5` (1,177/1,485). `9..20` absent; six `1`-epoch rows are outside current bounds. |
| `training.max_train_samples` | 1..4096 | 0, 256, 384, 512, 768, 1024, 1536, 3072, 4096 | `0` means all samples and appears in 420 rows despite being outside the numeric bound. Best residual run used 512; `4096` is common but not best on sampled F1. |

Current `configs/*.yaml` coverage mirrors this concentration: 222 configs total, with 179 `tiny_torch_unet`, 36 `residual_25d_torch_unet`, 5 logreg, and 2 MLP configs. Torch YAMLs are mostly `base_channels=8`, `epochs=5`, `batch_size=8`; only a few YAMLs explore `base_channels=4/16`, `epochs=6/7/8`, or `batch_size=4/16`.

## Pivot Config Interpretation

- `robust_multisegment_dice035_expanded.yaml`: `tiny_torch_unet`, `base_channels=8`, full training set (`max_train_samples=0`), Dice weight `0.35`, leave-one-segment-out scope.
- `robust_calibrated_prloss_w0p03_lr0012_prratio3_seed11018.yaml`: `tiny_torch_unet`, positive-rate loss `0.03`, tolerance `0.02`, cap `max_pred_positive_rate_ratio=3.0`, seed `11018`.
- `robust_tta_seed_ensemble.yaml`: `tiny_torch_unet`, three-seed ensemble with `tta_flips=true`, full training set.
- `residual_25d_torch_unet_cpu.yaml`: `residual_25d_torch_unet`, focused pair, `base_channels=8`, `epochs=5`, `batch_size=4`, `max_train_samples=512`, `augment_flips=true`, `tta_flips=true`.

The pivot set recognizes the residual model but most robust multi-segment pivots remain `tiny_torch_unet`. Later non-pivot configs such as `residual_25d_cpu_safe.yaml`, `residual25d_hneg085.yaml`, and cap/tolerance sweeps move residual 2.5D into expanded leave-one-out scopes, but the DB still shows residual coverage far below tiny U-Net coverage.

## Augmentation And Mining Evidence

Aggregate Torch rows:

| Strategy group | Rows | Mean `val_f1` | Mean AP | Best run | Best `val_f1` | Best AP |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `augment_flips=true` | 258 | 0.220681 | 0.160299 | `20260529T035158Z_8f01550f` | 0.489411 | 0.469463 |
| `augment_flips=false` | 1,227 | 0.253709 | 0.164274 | `20260529T182028Z_8718e6cb` | 0.781806 | 0.849842 |
| `tta_flips=true` | 303 | 0.251868 | 0.180161 | `20260529T182028Z_8718e6cb` | 0.781806 | 0.849842 |
| `tta_flips=false` | 1,182 | 0.246972 | 0.159333 | `20260529T180009Z_237fb3bb` | 0.436621 | 0.319348 |
| hard mining via `sampling_strategy` or `patch_sampling` | 295 | 0.238351 | 0.167125 | `20260529T035158Z_8f01550f` | 0.489411 | 0.469463 |
| no hard mining | 1,190 | 0.250356 | 0.162705 | `20260529T182028Z_8718e6cb` | 0.781806 | 0.849842 |

Matched-pair deltas are more informative than raw aggregates:

| Toggle | Matched pairs | Mean delta `val_f1` | Median delta `val_f1` | Mean delta AP | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `augment_flips` on vs off | 13 | -0.048328 | +0.000590 | -0.057193 | Usually small, but one residual outlier strongly favored no train flips. Excluding outlier, flips look near-neutral to mildly positive on tiny U-Net. |
| `tta_flips` on vs off | 32 | +0.020701 | +0.000369 | +0.024421 | TTA is near-neutral in most pairs but has positive average due to the best residual outlier. It is still the more promising augmentation switch. |
| hard mining on vs off | 11 | -0.054067 | not computed in DB output | -0.066672 | Hard mining is not yet proven. One residual pair strongly favored the non-hard-mined best run; several small tiny-U-Net comparisons are mixed. |

Notable deltas:

- Best residual run `20260529T182028Z_8718e6cb` (`augment_flips=false`, `tta_flips=true`, no hard mining) dominates the augmentation/mining comparisons: `+0.655689` F1 and `+0.785797` AP versus a matched no-TTA/off-axis residual counterpart, and the inverse negative delta for train flips/hard mining comparisons.
- Tiny U-Net matched TTA/flip effects are usually small, often within about `0.001..0.011` F1.
- Hard mining has enough rows (295) to be considered tried, but not enough clean paired evidence to claim a lift; it appears under-controlled rather than absent.

## Underfitting Vs Overfitting Evidence

Evidence for underfitting or capacity/data limits:

- `residual_25d_torch_unet` greatly outperforms `tiny_torch_unet` at best F1/AP despite far fewer runs. This suggests the tiny U-Net family is capacity/architecture limited for the richer 2.5D setting.
- `base_channels=16`, batch sizes away from `8`, and epochs beyond `8` are barely explored. The current search cannot rule out wider/deeper or longer-training Torch variants because the effective search surface is narrow.
- `model.depth` is proposed/bounded but ignored by Torch implementations, so any perceived Torch depth search is illusory.

Evidence for overfitting or calibration failure:

- Fixed-threshold behavior is weak for many rows: `tiny_numpy_ink_logreg` 374/374 weak, `tiny_numpy_mlp` 60/60 weak, `tiny_torch_unet` 993/1,253 weak, and `residual_25d_torch_unet` 51/232 weak.
- Positive-rate ratios are often too high: median ratios are about `4.21x` for logreg, `4.16x` for MLP, `3.96x` for tiny U-Net, and `2.49x` for residual 2.5D. Ratios above the promotion blocker of `3.5x` occur in 360 logreg rows, 60 MLP rows, 734 tiny-U-Net rows, and 27 residual rows.
- Epoch matched pairs are mixed and lean negative: among 16 comparable Torch epoch groups, 6 improved with more epochs and 10 worsened. Examples show 5 to 8 epochs dropping F1 from `0.201222` to `0.140252`, and from `0.147079` to `0.118480`, while train loss changed little or decreased. This is more consistent with overfitting/calibration drift than simple undertraining.
- The best residual sampled-F1 run has outstanding AP/F1 but `fixed_threshold_status=weak`, so promotion risk remains calibration/full-tile robustness rather than sampled ranking alone.

Net: architecture capacity is a real gap for the tiny U-Net branch, but the dominant promotion blocker across families is probability calibration and positive-rate control. Residual 2.5D is the best-supported path, but it needs broader controlled sweeps and promotion-grade validation rather than more unconstrained F1 sweeps.

## Recommended Model-Gap Actions

1. Prioritize `residual_25d_torch_unet` over more `tiny_torch_unet` runs. It has the strongest best F1/AP and is still underrepresented.
2. Run controlled residual sweeps over `base_channels=4,8,16`, `batch_size=2,4,8,16`, `epochs=3,4,5,6,8`, and `max_train_samples=512,1024,2048,4096`, holding fold scope and calibration settings fixed.
3. Treat `model.depth` as non-actionable for Torch until the runner implements depth-variable Torch networks or the planner stops proposing it for Torch.
4. Re-evaluate TTA on residual 2.5D with fixed seeds and caps; current evidence is promising but dominated by one outlier.
5. Use hard-negative mining only in paired, fold-safe residual experiments. Existing evidence is mixed and should not be assumed beneficial.
6. Keep promotion criteria centered on fixed-threshold status, positive-rate ratio, leave-one-out/full-tile evidence, and fold safety, because best sampled F1 alone is misleading in this DB.

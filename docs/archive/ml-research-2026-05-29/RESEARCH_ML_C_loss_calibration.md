# RESEARCH_ML_C: Loss Calibration

## Scope

Read-only ML research over the current Vesuvius AutoResearch code, configs, docs, and `experiments/experiments.db`. The database review used unique `config_signature` values where available: 1,911 unique experiment signatures total, including 1,477 torch/loss-related signatures.

Relevant implementation facts:

- `TrainingConfig` exposes `pos_weight`, `dice_loss_weight`, `positive_rate_loss_weight`, `positive_rate_loss_tolerance`, `positive_rate_loss_target`, `tversky_loss_weight`, `tversky_alpha`, `tversky_beta`, and `focal_tversky_gamma`.
- Torch training optimizes `BCEWithLogitsLoss(pos_weight=...)` plus optional Dice loss, optional positive-rate squared excess penalty, and optional Tversky/Focal-Tversky loss.
- `pos_weight: auto` resolves to `min(sqrt(negative_pixels / positive_pixels), 10.0)`, not raw inverse prevalence.
- Positive-rate loss is active only when `positive_rate_loss_target` is set; the residual configs that worked used `positive_rate_loss_target: 0.04`.
- Promotion quality blocks include zero precision/recall, suspicious `pred_positive_rate / val_positive_rate`, and weak/missing `fixed_threshold_status`.

## Tried Loss Combinations

Observed values across unique loss-related signatures:

| Field | Values tried |
| --- | --- |
| `dice_loss_weight` | `0.0`, `0.1`, `0.15`, `0.2`, `0.25`, `0.35`, `0.5`, `0.65`, `0.75`, `1.0` |
| `tversky_loss_weight` | `0.0`, `0.15`, `0.25`, `0.5` |
| `positive_rate_loss_weight` | `0.0`, `0.01`, `0.02`, `0.03`, `0.05`, `0.06`, `0.08`, `0.10` |
| `positive_rate_loss_tolerance` | `0.0`, `0.0025`, `0.004`, `0.005`, `0.008`, `0.01`, `0.02` |
| `max_pred_positive_rate_ratio` | unset, `2.0`, `2.5`, `2.75`, `3.0`, `3.5`, `4.0` |

Top combinations by best observed F1, with fixed-threshold status counts:

| Dice | Tversky | PR loss | PR tol | PR cap | n | best F1 | median F1 | fixed ok/weak/missing | median pred/val ratio | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.00 | 0.00 | 0.00 | 0.0000 | unset | 263 | 0.7818 | 0.3638 | 3/17/243 | 4.336 | Best historical sampled F1, but mostly missing fixed-threshold evidence and overpredicting. |
| 0.35 | 0.00 | 0.08 | 0.0100 | 2.5 | 69 | 0.4894 | 0.1595 | 63/6/0 | 2.435 | Best calibrated residual 2.5D family; strongest fixed-threshold behavior. |
| 0.35 | 0.00 | 0.06 | 0.0100 | 3.0 | 19 | 0.4798 | 0.1793 | 16/3/0 | 2.864 | Slightly looser cap and lower PR loss; good F1, still mostly fixed-ok. |
| 0.35 | 0.00 | 0.08 | 0.0100 | 2.75 | 1 | 0.4664 | 0.4664 | 1/0/0 | 1.399 | Single strong intermediate-cap run; worth repeating. |
| 0.35 | 0.00 | 0.08 | 0.0050 | 3.0 | 1 | 0.4592 | 0.4592 | 1/0/0 | 2.704 | Tighter tolerance increased recall but needs replication. |
| 0.35 | 0.00 | 0.08 | 0.0100 | 3.0 | 44 | 0.4433 | 0.1660 | 42/2/0 | 2.931 | Reliable fixed-ok, lower best F1 than cap 2.5/2.75 variants. |
| 0.35 | 0.00 | 0.10 | 0.0100 | 3.0 | 4 | 0.4427 | 0.0381 | 4/0/0 | 2.838 | Strong rate control, but median F1 suggests over-regularization/collapse risk. |
| 0.00 | 0.00 | 0.03 | 0.0025 | 2.5 | 1 | 0.4366 | 0.4366 | 1/0/0 | 2.477 | Tiny U-Net calibration probe; good enough to keep as architecture-control comparison. |
| 0.00 | 0.00 | 0.03 | 0.0200 | 3.0 | 113 | 0.4326 | 0.1496 | 80/33/0 | 2.971 | Common tiny U-Net PR-loss baseline; fixed-ok often, but lower ceiling than residual+dice. |
| 0.35 | 0.00 | 0.00 | 0.0000 | unset | 123 | 0.4321 | 0.3878 | 3/18/102 | 4.069 | Dice alone improves segmentation objective but does not solve probability/rate calibration. |

## Best Fixed-Threshold-OK Loss Configs

The best `fixed_threshold_status == ok` runs were dominated by residual 2.5D U-Net with Dice 0.35 and explicit positive-rate loss.

| Run | Model | Dice | PR weight | PR target/tol | PR cap | pos_weight resolved | F1 | F0.5 | AP | Precision | Recall | Pred/val ratio | Best threshold |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20260529T035158Z_8f01550f` | `residual_25d_torch_unet` | 0.35 | 0.08 | `0.04` / 0.01 | 2.5 | 1.785 | 0.4894 | 0.4593 | 0.4695 | 0.4090 | 0.6091 | 1.489 | 0.484 |
| `20260529T043018Z_4b5c76cd` | `residual_25d_torch_unet` | 0.35 | 0.08 | `0.04` / 0.01 | 2.5 | 1.808 | 0.4829 | 0.4135 | 0.4108 | 0.3579 | 0.7420 | 2.073 | 0.506 |
| `20260529T043711Z_c956bf4a` | `residual_25d_torch_unet` | 0.35 | 0.08 | `0.04` / 0.01 | 2.5 | 1.808 | 0.4817 | 0.4396 | 0.4538 | 0.3744 | 0.6753 | 1.803 | 0.366 |
| `20260529T034424Z_cb93d539` | `residual_25d_torch_unet` | 0.35 | 0.06 | `0.04` / 0.01 | 3.0 | 1.785 | 0.4798 | 0.4487 | 0.4459 | 0.3927 | 0.6165 | 1.570 | 0.495 |
| `20260529T034841Z_d0ee3406` | `residual_25d_torch_unet` | 0.35 | 0.08 | `0.04` / 0.01 | 2.5 | 1.785 | 0.4664 | 0.4472 | 0.4084 | 0.3999 | 0.5595 | 1.399 | 0.387 |
| `20260529T034955Z_e1c3ea6f` | `residual_25d_torch_unet` | 0.35 | 0.08 | `0.04` / 0.005 | 3.0 | 1.785 | 0.4592 | 0.4411 | 0.4190 | 0.3145 | 0.8504 | 2.704 | 0.297 |

Interpretation: the consistent calibrated region is not Dice alone and not PR loss alone. It is `BCE(auto pos_weight) + 0.35 Dice + 0.06-0.08 positive-rate penalty` with target `0.04`, tolerance around `0.005-0.01`, and cap `2.5-3.0`.

## Pos Weight vs Predicted Positive-Rate Ratio

Across unique loss-related signatures with resolved `pos_weight` and ratio metrics, the Pearson correlation between `pos_weight_resolved` and `pred_positive_rate / val_positive_rate` was `+0.29`. This is a moderate positive relationship: higher positive weighting generally increases overprediction risk, but architecture, cap selection, sampling, and PR loss have large effects.

Grouped by resolved `pos_weight`:

| Resolved pos_weight bin | n | Median pred/val ratio | Mean pred/val ratio | Zero precision/recall pass rate | Fixed-ok rate | Best F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `<1.8` | 282 | 2.490 | 2.715 | 0.979 | 0.830 | 0.4894 |
| `1.8-2.2` | 146 | 2.926 | 2.917 | 1.000 | 0.411 | 0.4829 |
| `2.2-3.0` | 272 | 4.257 | 5.036 | 1.000 | 0.051 | 0.7818 |
| `>=3.0` | 754 | 3.971 | 6.579 | 1.000 | 0.143 | 0.4286 |

Conclusions:

- The best fixed-ok loss configs used resolved `pos_weight` around `1.78-1.81` from `auto`, not larger explicit weights.
- Very high sampled F1 can occur with `pos_weight` near `2.98`, but the associated median ratio was `4.257` in the `2.2-3.0` bin and fixed-ok rate was only `5.1%`.
- `zero_precision_or_recall` is too weak as a loss-calibration criterion here: almost all higher-pos-weight bins pass it, while fixed-threshold calibration still fails often.
- The practical target should be fixed-ok plus ratio control, not merely nonzero precision and recall.

## Tversky Findings

Tversky was implemented and proposed but not yet competitive in the observed database:

- Tried weights were `0.15`, `0.25`, and `0.5`, with default `alpha=0.3`, `beta=0.7`, and one proposal using `alpha=0.2`, `beta=0.8`.
- Best observed `tversky_loss_weight=0.15` combinations were around F1 `0.4316` and fixed-threshold weak, below the residual Dice+PR-loss region.
- Focal-Tversky is technically implemented through `focal_tversky_gamma`, but the observed successful region used `gamma=1.0`; non-1 focal values do not appear materially explored.

Tversky should not replace the Dice+PR-loss baseline yet. If tested, add it lightly to the best calibrated residual base instead of testing it as a standalone replacement.

## Missing Loss Functions From Sparse Binary Segmentation Literature

Applicable losses not currently exposed as first-class config options:

- Focal BCE / sigmoid focal loss: useful when easy background dominates gradients; directly addresses sparse positive labels and hard false positives. Candidate parameters: `focal_loss_weight`, `focal_alpha`, `focal_gamma`.
- Combo loss: weighted BCE plus Dice with explicit false-positive/false-negative tradeoff. This overlaps the current BCE+Dice setup but would make the BCE/Dice balance and beta-style asymmetry explicit.
- Boundary-aware losses: boundary Dice, Hausdorff-distance-inspired loss, level-set/contour loss, or distance-transform-weighted BCE. These are relevant because ink traces are thin and topology/boundaries matter, but they require stable masks or distance transforms per patch.
- Focal Dice / Focal Tversky as a deliberate search family: code has focal Tversky gamma but the search space does not meaningfully exercise `gamma > 1`.
- Lovasz hinge / Lovasz sigmoid: optimizes IoU-like surrogates and can help binary segmentation with class imbalance, though it may be noisy for very small positive regions.

Most applicable next addition: Focal BCE, because it is simple, local, CPU-safe, and directly targets sparse binary segmentation without requiring distance transforms.

## Recommended Next 5 Loss Configs

These are specific loss configurations to run on the residual 2.5D base unless otherwise stated. Keep model/data scope, fold safety, threshold reporting, and positive-rate cap diagnostics unchanged.

| Priority | Loss config | Rationale |
| ---: | --- | --- |
| 1 | `pos_weight: auto`, `dice_loss_weight: 0.35`, `positive_rate_loss_weight: 0.08`, `positive_rate_loss_target: 0.04`, `positive_rate_loss_tolerance: 0.01`, `max_pred_positive_rate_ratio: 2.75` | Interpolates the best robust cap-2.5 family and the looser cap-3.0 family; one existing 2.75 run was strong but under-replicated. |
| 2 | `pos_weight: auto`, `dice_loss_weight: 0.35`, `positive_rate_loss_weight: 0.06`, `positive_rate_loss_target: 0.04`, `positive_rate_loss_tolerance: 0.01`, `max_pred_positive_rate_ratio: 2.5` | Tests whether the high-F1 PR-weight-0.06 result keeps calibration under the stricter cap. Could recover F1 while avoiding PR-weight-0.10 collapse. |
| 3 | `pos_weight: auto`, `dice_loss_weight: 0.35`, `positive_rate_loss_weight: 0.08`, `positive_rate_loss_target: 0.04`, `positive_rate_loss_tolerance: 0.005`, `max_pred_positive_rate_ratio: 2.5` | Extends the high-recall tolerance-0.005 result to the safer cap. Useful if the current blocker is residual overprediction near the selected threshold. |
| 4 | `pos_weight: auto`, `dice_loss_weight: 0.25`, `positive_rate_loss_weight: 0.08`, `positive_rate_loss_target: 0.04`, `positive_rate_loss_tolerance: 0.01`, `max_pred_positive_rate_ratio: 2.5` | Slightly reduces Dice pressure to test whether precision improves without losing the calibrated behavior of PR-weight-0.08. |
| 5 | `pos_weight: auto`, `dice_loss_weight: 0.35`, `tversky_loss_weight: 0.05`, `tversky_alpha: 0.4`, `tversky_beta: 0.6`, `positive_rate_loss_weight: 0.08`, `positive_rate_loss_target: 0.04`, `positive_rate_loss_tolerance: 0.01`, `max_pred_positive_rate_ratio: 2.5` | Tversky has not won standalone; a very light, less recall-biased Tversky term may reduce false positives while preserving the proven Dice+PR-loss backbone. |

If new code changes are allowed later, replace priority 5 with a Focal BCE addition: `BCEWithLogits + Dice 0.35 + PR loss 0.08 + focal_bce_weight 0.25, focal_gamma 2.0, focal_alpha 0.25`, then compare against priority 1 under identical fold/sampling settings.

## Bottom Line

The best calibrated region is residual 2.5D U-Net with `pos_weight: auto` resolving near `1.8`, `dice_loss_weight: 0.35`, positive-rate target `0.04`, PR loss weight `0.06-0.08`, tolerance `0.005-0.01`, and cap `2.5-2.75`. Higher resolved `pos_weight` correlates with higher predicted-positive-rate ratio and much lower fixed-threshold-ok rates, even when precision and recall are nonzero.

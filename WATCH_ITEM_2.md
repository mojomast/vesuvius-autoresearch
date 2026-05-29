# Watch Item 2

## Findings

All `configs/targeted_promo_*.yaml` files parse as valid YAML, include the required non-null fields, set `autoresearch.promotable: true`, and use model names handled by `experiments/runner.py`.

Runner model support confirmed for targeted configs:

- `residual_25d_torch_unet` is handled in `_train_torch_unet()`.
- `tiny_torch_unet` and `torch_residual_25d_unet` are also supported by the same torch branch.

New key handling confirmed in `experiments/runner.py`:

- `training.focal_loss_weight` is read and applied in the torch loss loop.
- `training.focal_gamma` is read and passed to focal BCE loss.
- `training.augment_rotation` is read and applied only to training arrays.
- `training.sampling_curriculum` is read and used to build epoch-specific sampled training batches.

| Config | Status | Issues Found | Fixed? |
|--------|--------|--------------|--------|
| `targeted_promo_combo_balanced_loss.yaml` | READY | None | N/A |
| `targeted_promo_fixed_threshold_cap275.yaml` | READY | None | N/A |
| `targeted_promo_fixed_threshold_prw006_cap250.yaml` | READY | None | N/A |
| `targeted_promo_focal_bce_sparse_ink.yaml` | READY | Uses handled keys `focal_gamma`, `focal_loss_weight` | N/A |
| `targeted_promo_hardfold_20230530172803_tol005.yaml` | READY | None | N/A |
| `targeted_promo_hardfold_dualheldout_22181603_30172803.yaml` | READY | None | N/A |
| `targeted_promo_light_tversky_precision.yaml` | READY | None | N/A |
| `targeted_promo_prratio_strict_cap250.yaml` | READY | None | N/A |
| `targeted_promo_rotation_aug_residual.yaml` | READY | Uses handled key `augment_rotation` | N/A |
| `targeted_promo_sampling_curriculum_residual.yaml` | READY | Uses handled key `sampling_curriculum` | N/A |

## Validation

- `.venv/bin/python -m pytest tests/ -x -q` passed with `275 passed` after the code fixes.

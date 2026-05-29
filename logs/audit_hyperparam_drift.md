# Hyperparameter Drift Audit

Historical `auto_*.yaml` configs show geometric drift. `training.pos_weight` reached `84.375`; `training.weight_decay` reached `0.4555557`; `evaluation.threshold` drifted down to `0.031`; non-torch `training.learning_rate` reached `0.24` while current code allowed up to `1.2`.

Suggested bounds:

| Parameter | Min | Max |
| --- | ---: | ---: |
| `training.pos_weight` | 0.25 | 25.0 |
| `training.learning_rate` | 0.001 | 0.25 |
| `training.weight_decay` | 0.0 | 0.05 |
| `evaluation.threshold` | 0.05 | 0.95 |
| `training.epochs` | 2 | 20 |
| `training.dice_loss_weight` | 0.0 | 0.8 |
| `training.positive_rate_loss_weight` | 0.0 | 0.1 |
| `training.tversky_loss_weight` | 0.0 | 0.5 |
| `training.tversky_beta` | 0.1 | 0.9 |

The highest-risk current gap is the non-torch `weight_decay * 3.0` proposal, which lacks an upper clamp.

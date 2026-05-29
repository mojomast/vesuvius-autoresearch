# Watch Item 1

## Scope

Audited `autoresearch.py` for these paths:

- `training.focal_loss_weight`
- `training.focal_gamma`
- `training.sampling_curriculum`
- `training.augment_rotation`

## Findings

| Path | SEARCH_PATHS | SIGNATURE_DEFAULTS | PARAM_BOUNDS | Mutation Family | Proposal Candidate |
|------|--------------|--------------------|--------------|-----------------|--------------------|
| `training.focal_loss_weight` | Present | Present as `None` | Present as `(0.0, 0.5)` | Fixed to `loss_calibration` | Added `0.1`, `0.2` |
| `training.focal_gamma` | Present | Present as `None` | Fixed from `(0.5, 4.0)` to `(0.5, 5.0)` | Fixed to `loss_calibration` | Added `2.0`, `3.0` when focal loss is enabled |
| `training.sampling_curriculum` | Present | Present as `None` | Not needed | Fixed to `data_sampling` | Added `warmup_then_hard` when unset |
| `training.augment_rotation` | Present | Present as `None` | Not needed | Fixed to `data_sampling` | Added boolean toggle |

## Validation

- `.venv/bin/python -m pytest tests/ -x -q` passed with `275 passed`.
- `.venv/bin/python autoresearch.py --plan --json` produced proposals with `changed_path: training.focal_loss_weight`.

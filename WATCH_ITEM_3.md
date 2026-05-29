# Watch Item 3

## augment_rotation

Status: `WAS_WIRED`

Findings:

- `training.augment_rotation` was read by `_train_torch_unet()`.
- Rotation augmentation was applied only to training arrays, not validation arrays.
- The original implementation expanded the dataset with all 90/180/270-degree rotations. It was adjusted to the requested behavior: each training sample receives a random 0/90/180/270-degree rotation with equal probability.

Validation:

- Added coverage that rotation changes a dummy batch when enabled.
- Added coverage that disabled rotation returns the original batch.

## sampling_curriculum

Status: `WAS_STUB`

Findings:

- `training.sampling_curriculum` was read by `_train_torch_unet()` and used in the epoch loop, but the scheduler did not implement the requested default half-epoch transition and did not recognize the new `warmup_then_hard` proposal value.
- The scheduler now defaults the transition epoch to half of total epochs when no explicit switch is configured.
- `warmup_then_hard` and `uniform_to_hard_mining` use uniform/random sampling before the switch, then `hard_mining` after the switch.
- The epoch loop now logs `Sampling curriculum: switching to hard_mining at epoch N` when the transition happens.

Validation:

- `.venv/bin/python -m pytest tests/unit/test_runner_sampling_augmentation.py -v` passed with `4 passed`.
- `.venv/bin/python -m pytest tests/ -x -q` passed with `275 passed`.

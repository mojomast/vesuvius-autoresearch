# AutoResearch Profile Observations

Observed after running short bounded cycles on 2026-05-29.

## Recent Run Distribution

Last 50 rows in `experiments/experiments.db`:

- Model family: 50/50 torch, 0/50 NumPy logreg, 0/50 NumPy MLP.
- Run profile metadata: 13/50 `exploration`, 37/50 missing profile metadata from pre-profile runs.
- Cost tier metadata: 13/50 `normal`, 37/50 missing tier metadata from pre-tier runs.
- New metadata-bearing rows: 13/13 torch `exploration` runs at `normal` cost.

## Artifact Checks

Recent `normal` exploration artifacts match the intended routine-cron profile:

- `20260529T135547Z_af9c4d5f`: `tiny_torch_unet`, 5 epochs, seed `11018`, no seed ensemble, `max_train_samples=1024`, no TTA, no train-time flips.
- `20260529T135540Z_5ba7af85`: `tiny_torch_unet`, 5 epochs, seed `11018`, no seed ensemble, `max_train_samples=1024`, no TTA, no train-time flips.
- `20260529T134415Z_fefb27a9`: `tiny_torch_unet`, `exploration`, `normal`, fixed-threshold status `ok`, no TTA or ensemble.
- `20260529T140147Z_6a5ecead`: `tiny_torch_unet`, `exploration`, `normal`, 5 epochs, seed `11018`, `max_train_samples=1024`, no TTA or ensemble.

Older recent artifacts without tier/profile metadata still generally have bounded torch settings, but their missing `autoresearch.cost_tier` and `autoresearch.run_profile` fields predate the current planner labels.

## Expensive Runs

No metadata-bearing run in the inspected recent window was `expensive`.

Planning and bounded real cycles both skipped the generated TTA proposal under the normal max-cost policy before the planner update. The planner now keeps that normal-cron preference by ordering cheap/normal proposals first and only allowing expensive backfill in explicit plateau, promotion-action, or promotion-profile contexts.

## Mismatches

The only mismatch with `docs/autoresearch_cost_tiers.md` is historical: runs created before cost tiers and profiles were added have missing metadata. New generated configs and artifacts carry the expected metadata and match the documented exploration profile.

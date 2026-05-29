# Changelog

## Watch Item Fixes

- Watch Item 1: audited new focal/augmentation/curriculum search paths, widened `training.focal_gamma` bounds to `0.5..5.0`, classified new mutation families, and added torch proposals for focal loss, focal gamma, rotation augmentation, and sampling curriculum.
- Watch Item 2: validated all targeted promotion configs as executable YAML with required fields, supported model names, `autoresearch.promotable: true`, and runner-handled new keys.
- Watch Item 3: confirmed torch training reads the new features, changed rotation augmentation to random per-sample 0/90/180/270-degree training-only rotations, completed the sampling curriculum half-epoch transition behavior, and added transition logging/tests.

## Bug Fixes

- Fixed `_RunHistory` derived caches to use content-based run-id keys instead of `id(runs)`.
- Guarded linked LOO summary cache access with a lock and return cache copies.
- Added promotion snapshot error logging and `AUTORESEARCH_FAIL_ON_SNAPSHOT_ERROR` fail-closed behavior.
- Added boot-time validation that numeric `SIGNATURE_DEFAULTS` respect `PARAM_BOUNDS`.
- Made generated config writes atomic.
- Moved stale generated-config pruning behind the autoresearch lock and kept `--plan` read-only.
- Clean up generated proposal config files when experiment subprocess launch fails.

## Configuration & Cleanup

- Centralized key AutoResearch seeds, paths, thresholds, cost-tier limits, quality-score weights, and promotion command defaults behind named constants and environment variables.
- Made `pyproject.toml` the dependency source of truth and removed the duplicate `requirements.txt` mirror.
- Moved root audit reports into `logs/` and added generated audit/config patterns to `.gitignore`.

## Type Safety

- Added a pydantic v2 `ExperimentConfig` schema and `load_typed_config()` helper under `src/autoresearch/schemas.py`.
- Re-exported typed schema helpers from `src/autoresearch/__init__.py` for incremental adoption.

## Refactor

- Added a transitional `src/autoresearch/` package layout with `config_io`, `cache`, `proposals`, `promotion`, `strategy`, `cli`, and `legacy_api` modules exposing the legacy implementation through stable package boundaries.

## Testing

- Added unit tests for the pydantic config schema and transitional package modules.
- Added integration tests for JSON plan output and lock-contention behavior.

## ML Search Strategy

- Added `src/autoresearch/search_strategy.py` with heuristic and optional Optuna-backed Bayesian candidate ordering.
- Added `AUTORESEARCH_SEARCH_STRATEGY=bayesian` support and an optional `search` dependency extra.
- Added `scripts/backtest_quality_score.py` and `docs/search_strategy.md`.

## Targeted Promotion Configs

- Added `configs/targeted_promo_fixed_threshold_cap275.yaml` to replicate the strongest residual 2.5D calibrated family at the under-tested 2.75x positive-rate cap.
- Added `configs/targeted_promo_fixed_threshold_prw006_cap250.yaml` to test whether lower PR loss recovers F1 while keeping fixed-threshold behavior under the strict 2.5x cap.
- Added `configs/targeted_promo_hardfold_20230530172803_tol005.yaml` to target the low-ink LOO fold that was only 0.0064 F1 below the downstream weak-fold gate.
- Added `configs/targeted_promo_hardfold_dualheldout_22181603_30172803.yaml` for fold-safe weak-fold/full-tile evidence on the repeated LOO blockers.
- Added `configs/targeted_promo_prratio_strict_cap250.yaml` to directly suppress suspicious predicted-positive-rate ratios with stricter sampling and tolerance.
- Added `configs/targeted_promo_light_tversky_precision.yaml` to test a light precision-biased Tversky term on top of the proven Dice+PR-loss backbone.

## Loss Functions

- Added configurable Torch Focal BCE loss with `training.focal_loss_weight`, `training.focal_alpha`, and `training.focal_gamma`.
- Added configurable Torch Combo BCE+Dice loss with `training.combo_loss_weight`, `training.combo_bce_weight`, and `training.combo_dice_weight`.
- Added `configs/targeted_promo_focal_bce_sparse_ink.yaml` and `configs/targeted_promo_combo_balanced_loss.yaml` on the calibrated residual 2.5D base.

## Bayesian Search Full Integration

- Added persistent Optuna study storage in `logs/optuna.db` with historical experiment backfill.
- Added bounded Bayesian proposal asking through `AUTORESEARCH_SEARCH_STRATEGY=bayesian` and `bayesian_multi`.
- Generated Bayesian proposals now include Optuna study/trial metadata and expose `search_strategy` in plan JSON.
- Added `scripts/inspect_optuna_study.py` to inspect best trials and Pareto fronts.

## Augmentation & Sampling

- Added Torch train-time `training.augment_rotation` for 90-degree rotation augmentation.
- Added `training.sampling_curriculum` to transition from uniform sampling to hard mining across epochs.
- Added bounded `dataset.z_offsets` search metadata and two targeted configs for rotation and sampling-curriculum ablations.

## Quality Score Calibration

- Expanded `scripts/backtest_quality_score.py` to grid AP, F0.5, and calibration-penalty weights.
- Updated `_run_quality_score` defaults to AP `0.20`, F0.5 `0.10`, and an explicit capped predicted-rate calibration penalty.
- Added `docs/quality_score_analysis.md` with backtesting rationale.

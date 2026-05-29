# Changelog

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

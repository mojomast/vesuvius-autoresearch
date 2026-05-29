# Improvements

## Bugs Fixed

Stable run-history cache keys:

```python
# Before
key = id(runs)

# After
key = tuple(sorted(str(run.get("run_id") or "") for run in runs))
```

Thread-safe LOO summary cache:

```python
# Before: unsynchronized global reads/writes
return _LINKED_LOO_SUMMARY_CACHE

# After: lock-protected copy
with _LINKED_LOO_SUMMARY_CACHE_LOCK:
    return list(_LINKED_LOO_SUMMARY_CACHE)
```

Promotion snapshot errors now use `logging.exception()` and can fail closed with `AUTORESEARCH_FAIL_ON_SNAPSHOT_ERROR=1` instead of silently continuing exploration.

Generated config writes are now atomic through a temporary file plus `replace()`, and stale config pruning only runs after the autoresearch lock is acquired. Plan mode remains read-only.

Subprocess experiment launches now clean up generated proposal configs on `CalledProcessError`, preventing failed configs from reserving signatures indefinitely.

## Structural Improvements

- Added `RESEARCH_A` through `RESEARCH_F` reports documenting architecture, bugs, type safety, config hygiene, tests, and search strategy.
- Added `src/autoresearch/` package boundaries for config I/O, cache, proposals, promotion, strategy, schemas, search strategy, CLI, and legacy API exports.
- Added pydantic v2 `ExperimentConfig` and `load_typed_config()`.
- Centralized key thresholds, quality weights, cost limits, seeds, paths, and promotion command defaults behind constants/env vars.
- Made `pyproject.toml` canonical and removed `requirements.txt`.
- Moved root audit reports under `logs/` and expanded ignore rules.

## Test Coverage

Current coverage from `TEST_COVERAGE.md`: `273 passed`, `89.38%` total coverage for `src/autoresearch`.

## Bayesian Search Mode

Install the optional search extra and set the strategy variable:

```bash
.venv/bin/python -m pip install -e '.[search]'
AUTORESEARCH_SEARCH_STRATEGY=bayesian .venv/bin/python autoresearch.py --plan --json
```

Bayesian mode now persists Optuna studies in `logs/optuna.db`, backfills historical experiment evidence, and proposes bounded candidates through the same signature dedupe, cost-tier limits, and promotion metadata used by heuristic planning.

## ML Improvements

- Added six read-only ML research reports covering promotion blockers, model gaps, loss/calibration, data sampling, LOO strategy, and Bayesian readiness.
- Added 10 targeted promotion configs focused on fixed-threshold robustness, predicted-positive-rate control, hard-fold evidence, sparse-ink focal/Combo losses, rotation augmentation, and sampling curriculum ablations.
- Added optional Torch Focal BCE and Combo BCE+Dice losses while preserving existing Dice, Tversky, and positive-rate loss behavior.
- Added train-time 90-degree rotation augmentation, epoch-based sampling curriculum support, and bounded z-offset search metadata.
- Rebalanced quality scoring toward fixed-threshold and calibration reliability with AP/F0.5 weights plus an explicit capped predicted-positive-rate penalty.
- Kept promotion gates intact: targeted configs still require seed-repeat LOO evidence, full-tile validation, and promotion-check eligibility before release consideration.

## Remaining Limitations

- The package split is transitional: modules expose legacy implementation boundaries without fully moving all logic out of `autoresearch.py` yet.
- `schemas.py` is permissive to preserve historical config compatibility.
- Bayesian mode persists proposal trials and historical backfill, but still uses existing bounded proposal execution instead of a standalone ask/tell worker.
- Resource warnings from existing SQLite test fixtures remain visible in coverage runs but do not fail tests.

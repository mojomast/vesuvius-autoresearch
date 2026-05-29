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

Current coverage from `TEST_COVERAGE.md`: `268 passed`, `89%` total coverage for `src/autoresearch`.

## Bayesian Search Mode

Install the optional search extra and set the strategy variable:

```bash
.venv/bin/python -m pip install -e '.[search]'
AUTORESEARCH_SEARCH_STRATEGY=bayesian .venv/bin/python autoresearch.py --plan --json
```

Bayesian mode conservatively reorders the existing bounded candidate set. Normal signature dedupe, cost-tier limits, and promotion metadata still apply.

## Remaining Limitations

- The package split is transitional: modules expose legacy implementation boundaries without fully moving all logic out of `autoresearch.py` yet.
- `schemas.py` is permissive to preserve historical config compatibility.
- Bayesian mode is candidate-ordering only; it does not yet persist an Optuna trial ledger or perform full ask/tell experiment management.
- Resource warnings from existing SQLite test fixtures remain visible in coverage runs but do not fail tests.

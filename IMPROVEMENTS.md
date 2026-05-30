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

Current coverage from `TEST_COVERAGE.md`: `291 passed`, `92.20%` total coverage for `src/autoresearch`.

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

## High-Compute Pass Results

- Completed a DB audit in `DB_AUDIT.md`; the top overall audit-score run was diagnostic-only, while the strongest calibrated sampled base remained `20260529T035158Z_8f01550f` / `20260529T035202Z_8f01550f`.
- Reused complete seed-repeat LOO evidence for focal candidate `20260530T003819Z_6b3111c8`: `24/24` folds, 3 seeds, `promotion_ready=true`, median F1 `0.163957`, mean F1 `0.187498`, and worst fold `20230530172803=0.039621`.
- Completed full-tile validation for `20260530T004030Z_1d9489f1`: promotion checks eligible, AP `0.162021`, selected tile F1 `0.242266`, fixed-threshold status `ok`, and selected pred/val ratio `2.740630` under the configured `2.75` cap.
- Completed weak-fold full-tile validation for `20260530T004150Z_0b659aec` on `20230530172803`: promotion checks eligible and fixed-threshold status `ok`, but selected tile F1 was only `0.014940`, confirming this fold as the main blocker.
- Ran 10 additional high-compute Bayesian sampled proposals; none displaced the focal candidate because all reported `fixed_threshold_status=weak`.
- Fixed recursive search-signature normalization so dict/list-valued config fields do not crash dedupe or Bayesian proposal paths.

## Villa Integration Results

- Integrated `StatefulShuffledSampler` and `GroupStratifiedBatchSampler` adapted from ScrollPrize/villa with explicit MIT attribution.
- Integrated `StreamingBinarySegmentationMetrics` as a fixed-threshold F1/Dice cross-check; local threshold semantics match villa for sigmoid probabilities, while AP and threshold-swept `val_f1` remain local metrics.
- Rebuilt local label NPZs from villa cleaned labels without overwriting originals; `20230530172803` validation IoU is `0.974698`, so label cleanup alone does not explain the catastrophic full-tile blocker.
- Hard-fold before: weak-fold full-tile `20260530T004150Z_0b659aec` on `20230530172803` had AP `0.027372` and selected F1 `0.014940`.
- Best villa-label sampled rescue: focal run `20260530T013457Z_1df6dddd` reached AP `0.062981`, `val_f1=0.136424`, fixed-threshold status `ok`, and pred/val ratio `0.882`.
- Group-stratified fallback `20260530T013847Z_eb682e02` reached AP `0.044858` and `val_f1=0.096390`; it did not beat the focal villa-label run.
- LOO was not launched for villa-label candidates because none met the LOO entry rule (`val_f1 >= 0.42`, fixed-threshold `ok`, ratio `0.5..3.5`); this avoids treating a weak sampled hard-fold run as promotion evidence.

## Seed Sensitivity Results

- Hard-fold sampled before local seed search: seed `11001` run `20260530T020142Z_4e5cb820` reached AP `0.064540`, `val_f1=0.142771`, and fixed-threshold status `ok`.
- New best hard-fold sampled run: seed `11071`, 20 epochs, run `20260530T032140Z_d4d69b0f`, reached AP `0.121940`, `val_f1=0.199885`, fixed-threshold status `ok`, and pred/val ratio `1.841176`.
- Relative sampled improvement on `20230530172803`: AP improved by `0.057400` absolute (`1.89x`) and F1 improved by `0.057113` absolute (`1.40x`) versus seed `11001`.
- The lower-LR seed `11071` run `20260530T032140Z_386d0c85` also crossed AP `0.1`, reaching AP `0.106733` and `val_f1=0.198718`.
- Built-in probability ensembling underperformed: ensemble `11001,11045,11073` reached AP `0.059997`, so the current best path is seed/epoch refinement rather than probability averaging.

## Remaining Limitations

- The package split is transitional: modules expose legacy implementation boundaries without fully moving all logic out of `autoresearch.py` yet.
- `schemas.py` is permissive to preserve historical config compatibility.
- Bayesian mode persists proposal trials and historical backfill, but still uses existing bounded proposal execution instead of a standalone ask/tell worker.
- Resource warnings from existing SQLite test fixtures remain visible in coverage runs but do not fail tests.
- The current LOO-backed focal family is still hard-fold-limited; `20230530172803` should remain the first target for additional robustness work.
- Villa labels improved the hard fold but did not reach AP `0.1`; next work should focus on feature separability and hard-negative/threshold behavior, not gate relaxation.

# Research F: ML Search Strategy Improvements

The minute-slot rotation in `_propose_configs` only changes candidate order across six 10-minute buckets. Real deduplication comes from tested/reserved search signatures and runner config signatures. Better alternatives are hash-based ordering per cycle, a DB-backed proposal ledger, or an Optuna trial registry.

Optional Bayesian search design:

- Add `AUTORESEARCH_SEARCH_STRATEGY=bayesian`.
- Keep heuristic proposal flow as the default.
- Lazily import Optuna and fail with a clear message if the optional dependency is missing.
- Use TPE for single-objective optimization over the existing bounded search space.
- Convert historical runs into completed trials, ask for candidates, convert trial params back into configs, then still apply search-signature dedupe and cost-tier guards.

The current quality score `val_f1 + 0.25*average_precision + 0.10*val_f05` is plausible but not backtested. A backtesting script should read `experiments/experiments.db`, rank historical candidates under alternative weights, and evaluate rolling-origin ranking quality against future promotion evidence or later robust metrics.

Multi-objective design can use Optuna NSGA-II with objectives for F1, AP, F0.5/precision, positive-rate calibration error, and cost. Pareto-front selection should still diversify mutation families and respect cost limits.

Backwards-compatible API target: `src/autoresearch/search_strategy.py` with `SearchContext`, `SearchStrategy`, `HeuristicSearchStrategy`, `OptunaBayesianSearchStrategy`, and `strategy_from_env()`.

# Search Strategy

AutoResearch defaults to the existing heuristic one-change proposal flow.

Set `AUTORESEARCH_SEARCH_STRATEGY=bayesian` to enable optional Optuna-backed candidate ordering:

```bash
pip install -e '.[search]'
AUTORESEARCH_SEARCH_STRATEGY=bayesian .venv/bin/python autoresearch.py --plan --json
```

Bayesian mode asks Optuna for a bounded parameter value, injects that as the first candidate, then keeps the normal signature dedupe, cost-tier gates, and proposal metadata rules. Studies are persisted in `logs/optuna.db`, and historical experiment rows are backfilled as completed trials on each planning cycle.

Set `AUTORESEARCH_SEARCH_STRATEGY=bayesian_multi` to use an NSGA-II multi-objective study over `val_f1`, `average_precision`, and calibration closeness. Multi-objective mode uses the same safe proposal path and falls back to heuristic ordering if no bounded Optuna candidate is available.

Inspect study state with:

```bash
.venv/bin/python scripts/inspect_optuna_study.py
```

If Optuna is not installed, Bayesian mode raises a clear runtime error telling the operator to install the `search` extra.

Quality-score weights can be explored with:

```bash
.venv/bin/python scripts/backtest_quality_score.py --db experiments/experiments.db
```

The backtest script ranks historical runs under AP/F0.5 weight combinations and reports top-k validation summaries. It is a lightweight first pass; promotion-result and full-tile labels should be added before treating new weights as authoritative.

# Search Strategy

AutoResearch defaults to the existing heuristic one-change proposal flow.

Set `AUTORESEARCH_SEARCH_STRATEGY=bayesian` to enable optional Optuna-backed candidate ordering:

```bash
pip install -e '.[search]'
AUTORESEARCH_SEARCH_STRATEGY=bayesian .venv/bin/python autoresearch.py --plan --json
```

Bayesian mode is intentionally conservative. It reorders the same bounded candidate set produced by the current proposal generator, then the normal signature dedupe, cost-tier gates, and proposal metadata rules still apply.

If Optuna is not installed, Bayesian mode raises a clear runtime error telling the operator to install the `search` extra.

Quality-score weights can be explored with:

```bash
.venv/bin/python scripts/backtest_quality_score.py --db experiments/experiments.db
```

The backtest script ranks historical runs under AP/F0.5 weight combinations and reports top-k validation summaries. It is a lightweight first pass; promotion-result and full-tile labels should be added before treating new weights as authoritative.

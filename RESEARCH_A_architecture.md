# Research A: Architecture & Decomposition

`autoresearch.py` currently owns path constants, metric contracts, DB reads, run-history caching, nested config mutation, search signatures, candidate proposal generation, cost gating, promotion gates, quality scoring, linked LOO summary discovery, promotion artifact validation, dashboard promotion readiness integration, process locking, config writes, harness wiring, and CLI execution.

Proposed package split under `src/autoresearch/`:

- `paths.py`: `ROOT`, `CONFIGS`, `LOGS`, `LOCK_PATH`, `BASELINE`, `_repo_arg`.
- `contracts.py`: `MetricContract`, `METRIC_CONTRACT`, `validate_metric_contract`.
- `config_io.py`: `_set_nested`, `_get_nested`, `_candidate_is_noop`, `_apply_candidate`, `_dump_config_with_comment`, constants for `SEARCH_PATHS`, `SIGNATURE_DEFAULTS`, `PARAM_BOUNDS`.
- `cache.py`: `_CycleProfiler`, `_RunHistory`, linked LOO summary cache helpers.
- `proposals.py`: `_clamp_param`, `_bounded_candidates`, `_proposal_value_slug`, `_proposal_candidates`, `_mutation_family`, `_propose_configs`, `_proposal_plan`, `_generate_promotion_action_proposals`.
- `strategy.py`: `_strategy_phase`, `_ranked_recent_torch_bases`, `_propose_from_recent_winners`, `_pivot_bases`, `_propose_best_path`, `_propose_with_pivots`, `_promotion_or_fallback_proposals`.
- `promotion.py`: `_promotion_gate`, `_run_quality_score`, `_promotion_next_action`, `_record_promotion_status`, `_reported_promotion_outputs`, `_validate_reported_promotion_outputs`, `_run_automated_promotion`, manual promotion helpers, dashboard readiness helpers.
- `schemas.py`: typed pydantic config and payload models.
- `search_strategy.py`: heuristic/Optuna strategy abstraction.
- `cli.py`: `main`, locking, planning, experiment launch orchestration.

Circular risks:

- `cache.py` must not become a low-level module if `_RunHistory` calls strategy/promotion functions. Either keep it orchestration-level or inject callbacks.
- `strategy.py` can import `proposals.py`, but `proposals.py` should not import `strategy.py`.
- Dashboard imports must remain lazy because `research_dashboard.snapshot` imports `autoresearch` symbols.
- The top-level shim must preserve `import autoresearch` compatibility for tests and harness monkeypatches.

Thin shim target:

```python
#!/usr/bin/env python3
from __future__ import annotations

from src.autoresearch.legacy_api import *  # noqa: F401,F403
from src.autoresearch.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

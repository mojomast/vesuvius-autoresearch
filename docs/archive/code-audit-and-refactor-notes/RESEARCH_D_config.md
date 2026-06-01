# Research D: Hardcoding & Configuration Hygiene

Hardcoded values worth centralizing include path defaults (`configs`, `logs`, baseline, DB path, run dir), interpreter/script paths, default seed `1337`, seed-repeat deltas `17,31,53`, promotion seeds, cost-tier thresholds, quality-score weights, positive-rate thresholds, promotion edge thresholds, and full-tile command defaults.

Existing environment variables include `AUTORESEARCH_RECENT_LIMIT`, `AUTORESEARCH_PROFILE`, `AUTORESEARCH_MAX_COST_TIER`, `AUTORESEARCH_TORCH_MAX_TRAIN_SAMPLES`, `AUTORESEARCH_ALLOW_SEED_ENSEMBLE`, `AUTORESEARCH_PENDING_CONFIG_TTL_HOURS`, `AUTORESEARCH_PLATEAU_WINDOW`, `AUTORESEARCH_PLATEAU_MIN_DELTA`, `AUTORESEARCH_PROMOTION_SEEDS`, `AUTORESEARCH_LOO_JOBS`, `AUTORESEARCH_AUTO_PROMOTE`, `AUTORESEARCH_PROMOTION_TIMEOUT_SECONDS`, `AUTORESEARCH_PROPOSALS`, `AUTORESEARCH_DEADLINE_SECONDS`, `AUTORESEARCH_EXPLORATION_MAX_EPOCHS`, `SCROLL_RESEARCH_*`, and `VESUVIUS_DASHBOARD_*`.

Recommended additions: `VESUVIUS_PROJECT_ROOT`, `VESUVIUS_CONFIG_DIR`, `VESUVIUS_LOG_DIR`, `VESUVIUS_EXPERIMENT_DB`, `VESUVIUS_RUNS_DIR`, `VESUVIUS_BASELINE_CONFIG`, `VESUVIUS_PYTHON`, `AUTORESEARCH_DEFAULT_SEED`, `AUTORESEARCH_SEED_REPEAT_DELTAS`, `AUTORESEARCH_QUALITY_AP_WEIGHT`, `AUTORESEARCH_QUALITY_F05_WEIGHT`, and cost/threshold env overrides.

`pyproject.toml` should be canonical. `requirements.txt` duplicates base dependencies and can drift; either remove it or replace it with `-e .`.

Root `audit_*.md` files are tracked operational reports and should move under `logs/` or curated docs. Runtime patterns such as `logs/audit_*.md`, `auto_*.yaml`, and lock files should be ignored.

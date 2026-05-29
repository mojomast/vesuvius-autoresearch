# Research E: Testing Strategy

Existing tests cover promotion artifact validation, proposal cost tiers, pivot/proposal behavior, metric contracts, guard defaults, and runner validation. Gaps remain around direct unit coverage for small pure helpers and full-cycle integration.

Functions needing unit tests:

- `_clamp_param`: low/high bounds, ints preserve int, booleans and unknown paths unchanged.
- `_config_cost_tier`: defaults, malformed sections, epoch/pixel/sample boundaries, seed ensembles, TTA, unknown models.
- `_promotion_gate`: missing metrics, zero precision/recall, positive-rate ratio boundaries, promotion checks, cron safety, scope variants.
- `_run_quality_score`: formula, AP/F0.5 weights, ratio penalties, fixed-threshold penalties, gate bonus/warning penalties.
- `_search_signature`: defaults, list normalization, float rounding, depth saturation, metadata ignored, every search path changes signature.
- `_proposal_candidates`: torch and numpy path sets, bounds, sample-cap env, toggles, MLP branch.
- `_RunHistory`: cache-per-limit, invalidate, content-keyed derived caches, `None` candidate caching.
- `_validate_reported_promotion_outputs`: missing/empty outputs, full-tile required keys, output-dir checks, invalid metrics JSON, relative paths.
- `validate_metric_contract`: row-field errors, config/metrics object checks, bool rejection, optional metric validation, custom contract.

Mock filesystem with `tmp_path` and monkeypatch module paths. Mock subprocess by patching `autoresearch.subprocess.run`. Mock sqlite with a temp DB and `experiments.runner.init_db` or by patching `_recent_runs` for high-level `main()` tests.

Integration tests should cover plan JSON, no-runs baseline, lock contention, promotion-ready pause, safe auto-promotion, targeted promotion proposals, normal proposal cycle, deadline stopping, and invalid CLI combinations.

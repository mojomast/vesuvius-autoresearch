# Runner Contract Audit

`experiments/runner.py` writes SQLite columns `run_id`, `timestamp`, `config_json`, `main_metric`, `secondary_metrics_json`, `artifact_dir`, and `config_signature`. `autoresearch.py` reads all except `config_signature`; `_strategy_phase` returns `run_id`, scoring reads `main_metric` and selected secondary metrics, and promotion logic reads config scope plus selected secondary metrics.

Required secondary metrics written by every runner path and read by autoresearch are `val_loss`, `val_f1`, `val_f05`, `best_threshold`, `precision`, `recall`, `average_precision`, `ap_prevalence_lift`, `val_positive_rate`, `pred_positive_rate`, and `fixed_threshold_status`.

Optional promotion-evidence keys read by autoresearch but not written by the runner are `promotion_checks`, `loo_promotion_ready`, and `full_tile_promotion_ready`. These are external validation evidence and should default to `{}`, `False`, and `False` during contract validation.

Mismatches flagged: `_score_run` is named `_run_quality_score`; `config_signature` is written but not selected by `_recent_runs`; loss fallback in scoring can reward high loss if `val_f1` is absent; old rows with missing metrics silently become low-score/non-promotable.

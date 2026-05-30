# Experiment Report

## DB Audit Summary

- Database audited: `experiments/experiments.db`.
- Best run by audit score: `20260529T182028Z_8718e6cb`, score `1.080931`, but it is not promotion-usable because it is a focused diagnostic run with `fixed_threshold_status=weak` and not multisegment scope.
- Best calibrated sampled base from the audit: `20260529T035158Z_8f01550f` / `20260529T035202Z_8f01550f`, with `val_f1=0.489411`, AP `0.469463`, and pred/val ratio `1.489`.
- Fresh focal candidate with complete LOO evidence: `20260530T003819Z_6b3111c8`, sampled `val_f1=0.4586`, AP `0.3784`, fixed threshold `ok`, pred/val ratio `2.66`.
- Existing LOO summaries included promotion-ready evidence, so manual Batches A-C were skipped per the high-compute pass rule.

## Batch Results Table

| Batch | Status | Result |
|---|---|---|
| Phase 1 DB audit | Completed | Wrote `DB_AUDIT.md`; identified calibrated and focal candidates. |
| Batch A seed ensemble | Skipped | Existing `promotion_ready=true` LOO evidence found before launch. |
| Batch B focal sweep | Skipped | Existing focal LOO candidate available. |
| Batch C hard-fold configs | Skipped | Existing LOO evidence available; hard folds remain the main risk. |
| Batch D seed-repeat LOO | Reused existing run | `logs/20260530T003819Z_6b3111c8_seedrepeat_loo.summary.json` is complete and promotion-ready. |
| Batch E full-tile validation | Completed | `20260530T004030Z_1d9489f1` candidate full-tile metrics are eligible; weak-fold full-tile evidence remains a blocker. |
| Phase 3 Bayesian cycles | Completed | Added 10 runs; all had weak fixed-threshold status and did not replace the focal candidate. |

## LOO Evidence

- Summary: `logs/20260530T003819Z_6b3111c8_seedrepeat_loo.summary.json`.
- Base config: `experiments/runs/20260530T003819Z_6b3111c8/config.json`.
- Promotion ready: `true`.
- Promotion warnings: none.
- Folds requested/successful: `24/24`.
- Seeds: `11001`, `11018`, `11045`.
- Median `val_f1`: `0.163957`.
- Mean `val_f1`: `0.187498`.
- Mean AP: `0.135966`.
- Worst fold: `20230530172803`, `val_f1=0.039621`.

Per-fold median F1:

| Fold | F1 |
|---|---:|
| `20230520175435` | `0.451547` |
| `20230522181603` | `0.094101` |
| `20230522215721` | `0.139636` |
| `20230530172803` | `0.039621` |
| `20230530212931` | `0.188278` |
| `20230531121653` | `0.291312` |
| `20230601193301` | `0.094525` |
| `20230611014200` | `0.200966` |

## Full Tile Result

- Artifact: `experiments/runs/20260530T004030Z_1d9489f1`.
- Metrics: `logs/full_tile_20260530T004030Z_1d9489f1_20230520175435/metrics.json`.
- Segment: `20230520175435`.
- Promotion checks eligible: `true`.
- Full-tile AP: `0.162021`.
- Selected tile F1: `0.242266`.
- Selected tile F0.5: `0.190517`.
- Selected threshold: `0.463937`.
- Fixed threshold status: `ok`.
- Fixed threshold F1: `0.261701`.
- Selected pred/val ratio: `2.740630` under configured cap `2.75`.
- Inference: CPU, batch size `16`, stride `32`, TTA flips enabled, ensemble size `1`.

The output set was also copied under `experiments/runs/20260530T004030Z_1d9489f1/full_tile_20230520175435/` so local dashboard/planner evidence discovery can link it to the candidate artifact.

## Weak-Fold Full Tile Result

- Artifact: `experiments/runs/20260530T004150Z_0b659aec`.
- Metrics: `experiments/runs/20260530T004150Z_0b659aec/full_tile_weak_fold_20230530172803/metrics.json`.
- Segment: `20230530172803`.
- Promotion checks eligible: `true`.
- Full-tile AP: `0.027372`.
- Selected tile F1: `0.014940`.
- Selected tile F0.5: `0.039336`.
- Selected threshold: `0.423410`.
- Fixed threshold status: `ok`.
- Fixed threshold F1: `0.041316`.
- Selected pred/val ratio: `2.659699` under configured cap `2.75`.
- Interpretation: this validates the hard-fold concern. The candidate family has eligible output and fixed-threshold positives, but full-tile quality on `20230530172803` is too weak to treat as final promotion evidence.

## Autonomous Loop Results

High-compute Bayesian cycles used `AUTORESEARCH_MAX_COST_TIER=expensive`, `AUTORESEARCH_TORCH_MAX_TRAIN_SAMPLES=4096`, `AUTORESEARCH_ALLOW_SEED_ENSEMBLE=1`, `AUTORESEARCH_PROPOSALS=5`, `AUTORESEARCH_LOO_JOBS=8`, `AUTORESEARCH_RECENT_LIMIT=500`, and `AUTORESEARCH_SEARCH_STRATEGY=bayesian`.

Cycle 1 paused on the existing promotion action. Cycles 2 and 3 continued with `AUTORESEARCH_CONTINUE_AFTER_PROMOTION_ACTION=1` and added 10 runs, increasing the DB count from `2004` to `2014`.

| Run | Changed path | `val_f1` | AP | Fixed threshold | Ratio |
|---|---|---:|---:|---|---:|
| `20260530T005107Z_fa4b1283` | `training.positive_rate_loss_tolerance` | `0.420430` | `0.315742` | `weak` | `2.073` |
| `20260530T005123Z_83523754` | `evaluation.max_pred_positive_rate_ratio` | `0.420430` | `0.315742` | `weak` | `2.073` |
| `20260530T005139Z_d340abfd` | `balanced_calibration` | `0.420430` | `0.315742` | `weak` | `2.073` |
| `20260530T005156Z_40279cbb` | `training.positive_rate_loss_tolerance` | `0.424722` | `0.321726` | `weak` | `2.971` |
| `20260530T005212Z_3b3057b2` | `evaluation.max_pred_positive_rate_ratio` | `0.419893` | `0.321243` | `weak` | `2.746` |
| `20260530T005252Z_b3e09e0b` | `training.positive_rate_loss_tolerance` | `0.420430` | `0.315742` | `weak` | `2.073` |
| `20260530T005308Z_73d35c4d` | `balanced_calibration` | `0.420430` | `0.315742` | `weak` | `2.073` |
| `20260530T005324Z_af7028c6` | `evaluation.tta_flips` | `0.422814` | `0.323582` | `weak` | `2.297` |
| `20260530T005340Z_54d9526f` | `training.dice_loss_weight` | `0.420192` | `0.321243` | `weak` | `2.971` |
| `20260530T005357Z_37fb346e` | `balanced_calibration` | `0.422887` | `0.321726` | `weak` | `2.746` |

Cycle 2 initially exposed a planner bug when dict-valued fields entered the search signature. `_normalize_signature_value()` now recursively normalizes dicts and lists to hashable tuples.

## Next Actions

- Treat `20260530T003819Z_6b3111c8` / `20260530T004030Z_1d9489f1` as the current evidence-backed focal family, not as a final promotion answer.
- Keep promotion gates unchanged; sampled metrics alone remain insufficient.
- Prioritize hard-fold robustness on `20230530172803` because its LOO F1 is borderline at `0.039621` and weak-fold full-tile F1 is only `0.014940`.
- Re-run targeted hard-fold configs for `20230530172803` and `20230522181603` if more compute is available.
- Preserve the focal calibration family around `focal_loss_weight=0.25`, `focal_gamma=2.0`, positive-rate loss weight `0.08`, and cap `2.75` unless a hard-fold run provides stronger full evidence.

# Promotion Evidence: 20260529T023543Z_021a01b0

Decision: DO NOT PROMOTE.

Reason: targeted positive-rate tightening reduced but did not eliminate LOO positive-rate alarms, and the worst fold regressed below the promotion target.

## Diagnosis

The `20260529T021416Z_570f6775` positive-rate alarms were model probability-scale floods on discrete threshold plateaus, not fixed-threshold failures.

Original alarming fold/seed pairs:

| Fold | Seed | val_f1 | AP | pred/val ratio | threshold_selection | fixed_threshold_status |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `20230522181603` | 11035 | 0.0821 | 0.0415 | 11.88 | `best_f1` | ok |
| `20230522215721` | 11035 | 0.1772 | 0.0925 | 9.69 | `best_f1` | ok |
| `20230530212931` | 11035 | 0.1398 | 0.0763 | 9.66 | `best_f1` | ok |

The LOO summary alarm itself is hardcoded at pred/val `>3.5` or `<0.1`, which is looser than the base config's `max_pred_positive_rate_ratio=3.0`; this was not a too-strict alarm-threshold bug. Threshold CSV inspection showed no useful under-cap operating point for several folds: under-cap thresholds collapsed to near-zero recall, while the next lower threshold flooded.

## Fix Attempt

Config: `configs/tightened_positive_rate_570f6775.yaml`

- Parent: `20260529T021416Z_570f6775`
- `training.positive_rate_loss_tolerance: 0.005`
- `evaluation.max_pred_positive_rate_ratio: 2.5`
- Run: `20260529T023543Z_021a01b0`
- Sampled validation: `val_f1=0.3912`, AP `0.2959`, pred/val `2.4770`, fixed threshold `ok`

## LOO Evidence

Two-seed check: `logs/20260529T023543Z_021a01b0_tightened_loo.summary.json`

- positive-rate alarms: none
- zero precision/recall folds: none
- median-over-seeds median `val_f1=0.0954`
- mean AP `0.1062`
- worst fold: `20230530172803`, `val_f1=0.0239`

Three-seed check: `logs/20260529T023543Z_021a01b0_3seed_loo.summary.json`

- `promotion_ready=false`
- positive-rate alarms: `20230530212931:seed=15050`, `20230531121653:seed=15050`
- zero precision/recall folds: none
- median-over-seeds median `val_f1=0.1045`
- mean AP `0.1065`
- worst fold: `20230530172803`, `val_f1=0.0179`

Single-seed diagnostic on the originally failing seed: `logs/20260529T023543Z_021a01b0_11035_loo.summary.json`

- positive-rate alarms: `20230522181603:seed=11035`, `20230530212931:seed=11035`, `20230611014200:seed=11035`
- median `val_f1=0.1267`
- worst fold: `20230530172803`, `val_f1=0.0083`

## Interpretation

The tighter positive-rate loss and `2.5x` cap help, but do not solve fold-level probability-scale cliffs. The next fix should improve probability smoothness or segment-specific calibration, not merely adjust the LOO promotion alarm. Post-hoc cap fallback would select near-zero-recall thresholds on several folds and would not be a valid promotion fix.

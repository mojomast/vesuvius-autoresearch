# Promotion Evidence: 20260529T024258Z_8b44032d

Decision: DO NOT PROMOTE.

Reason: full-tile metrics improved on `20230520175435`, but seed-repeat LOO failed with zero precision/recall folds and weak fixed-threshold behavior.

## Seed Analysis

The prior `seed=15050` alarms were not marginal. On `20260529T023543Z_021a01b0`, the two remaining three-seed alarms had pred/val ratios around `5.66x`:

| Fold | Seed | val_f1 | pred/val | prob_mean | prob_p95 | threshold_selection |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `20230530212931` | 11001 | 0.1039 | 2.46 | 0.3659 | 0.4823 | `positive_rate_constrained` |
| `20230530212931` | 11018 | 0.1096 | 2.08 | 0.3488 | 0.4745 | `positive_rate_constrained` |
| `20230530212931` | 15050 | 0.1442 | 5.67 | 0.3586 | 0.4772 | `best_f1` |
| `20230531121653` | 11001 | 0.2435 | 2.49 | 0.4080 | 0.4807 | `positive_rate_constrained` |
| `20230531121653` | 11018 | 0.2552 | 2.49 | 0.3993 | 0.4719 | `positive_rate_constrained` |
| `20230531121653` | 15050 | 0.2989 | 5.66 | 0.4085 | 0.4791 | `best_f1` |

`seed=15050` did not simply produce uniformly higher raw probabilities; it selected floodier best-F1 thresholds on small validation strips with threshold cliffs. Replacing the seed was tested as a diagnostic, not accepted as a promotion shortcut.

## Full-Tile Decision Table

| run | sampled val_f1 | tiled val_f1 | AP | pred/val | eligible | proceed to LOO? |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `20260529T024258Z_8b44032d` prratio3.5 | 0.4110 | 0.2363 | 0.1508 | 3.44 | true | yes |
| `20260529T024350Z_caa81174` prratio3.0 | 0.3971 | 0.2266 | 0.1511 | 2.97 | true | no |

`8b44032d` was selected for LOO because it improved full-tile `val_f1` while staying below the `3.5x` full-tile pred/val limit.

## LOO Evidence

LOO summary: `logs/20260529T024258Z_8b44032d_seeds11001_11018_15073_loo.summary.json`

- Seeds: `11001,11018,15073`
- `promotion_ready=false`
- median-over-seeds median `val_f1=0.0681`
- mean AP `0.1056`
- worst fold: `20230530172803`, `val_f1=0.0133`
- zero precision/recall folds: `20230522215721:seed=15073`, `20230530212931:seed=15073`
- positive-rate alarms: `20230522215721:seed=15073`, `20230530212931:seed=15073`

## Interpretation

The best current full-tile F1 still does not generalize through LOO. The next search should target the gap between cap `2.5` and cap `3.0`, not loosen to `3.5` for promotion. AutoResearch now includes balanced calibration proposals for `max_pred_positive_rate_ratio=2.75`, `positive_rate_loss_tolerance=0.008`, and their combined candidate.

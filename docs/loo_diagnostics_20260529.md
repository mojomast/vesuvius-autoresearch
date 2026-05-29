# LOO Diagnostics, 2026-05-29

These tables were generated with `scripts/analyze_loo_folds.py` from existing LOO summaries and their companion JSONL files. They focus on the weak and alarm folds rather than promotion decisions.

## Pre-Fix Reference: 570f6775

Command:

```bash
.venv/bin/python scripts/analyze_loo_folds.py \
  --summary-json logs/20260529T021416Z_570f6775_promotion_loo.summary.json \
  --markdown
```

Selected analyzer rows:

| source | fold_id | seed | val_f1 | AP | pred_rate | val_rate | pred/val | cap | fixed | flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260529T021416Z_570f6775_promotion_loo.summary.json | 20230522181603 | 11001 | 0.0523 | 0.0400 | 0.0391 | 0.0143 | 2.72 | 3.00 | ok |  |
| 20260529T021416Z_570f6775_promotion_loo.summary.json | 20230522181603 | 11018 | 0.0428 | 0.0375 | 0.0387 | 0.0143 | 2.70 | 3.00 | ok |  |
| 20260529T021416Z_570f6775_promotion_loo.summary.json | 20230522181603 | 11035 | 0.0821 | 0.0415 | 0.1705 | 0.0143 | 11.88 | 3.00 | ok | over_cap, positive_rate_alarm |
| 20260529T021416Z_570f6775_promotion_loo.summary.json | 20230522215721 | 11018 | 0.0345 | 0.0913 | 0.0073 | 0.0357 | 0.20 | 3.00 | ok | low_f1 |
| 20260529T021416Z_570f6775_promotion_loo.summary.json | 20230522215721 | 11035 | 0.1772 | 0.0925 | 0.3462 | 0.0357 | 9.69 | 3.00 | ok | over_cap, positive_rate_alarm |
| 20260529T021416Z_570f6775_promotion_loo.summary.json | 20230530172803 | 11001 | 0.0217 | 0.0104 | 0.0198 | 0.0088 | 2.26 | 3.00 | ok | low_f1 |
| 20260529T021416Z_570f6775_promotion_loo.summary.json | 20230530172803 | 11018 | 0.0349 | 0.0157 | 0.0261 | 0.0088 | 2.98 | 3.00 | ok | low_f1 |
| 20260529T021416Z_570f6775_promotion_loo.summary.json | 20230530172803 | 11035 | 0.0229 | 0.0144 | 0.0261 | 0.0088 | 2.98 | 3.00 | ok | low_f1 |
| 20260529T021416Z_570f6775_promotion_loo.summary.json | 20230530212931 | 11035 | 0.1398 | 0.0763 | 0.4718 | 0.0489 | 9.66 | 3.00 | ok | over_cap, positive_rate_alarm |
| 20260529T021416Z_570f6775_promotion_loo.summary.json | 20230611014200 | 11035 | 0.0375 | 0.1125 | 0.0173 | 0.0829 | 0.21 | 3.00 | ok | low_f1 |

## Post-Fix Reference: 478e28aa

Command:

```bash
.venv/bin/python scripts/analyze_loo_folds.py \
  --summary-json logs/20260529T043342Z_478e28aa_3seed_loo.summary.json \
  --markdown
```

Selected analyzer rows:

| source | fold_id | seed | val_f1 | AP | pred_rate | val_rate | pred/val | cap | fixed | flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260529T043342Z_478e28aa_3seed_loo.summary.json | 20230522181603 | 11001 | 0.1043 | 0.0466 | 0.0136 | 0.0143 | 0.94 | 2.50 | weak | fixed_threshold_not_ok |
| 20260529T043342Z_478e28aa_3seed_loo.summary.json | 20230522181603 | 11018 | 0.0273 | 0.0339 | 0.0324 | 0.0143 | 2.26 | 2.50 | ok | low_f1 |
| 20260529T043342Z_478e28aa_3seed_loo.summary.json | 20230522181603 | 11045 | 0.1595 | 0.0730 | 0.0324 | 0.0143 | 2.26 | 2.50 | weak | fixed_threshold_not_ok |
| 20260529T043342Z_478e28aa_3seed_loo.summary.json | 20230530172803 | 11001 | 0.0369 | 0.0213 | 0.0449 | 0.0180 | 2.49 | 2.50 | ok | low_f1 |
| 20260529T043342Z_478e28aa_3seed_loo.summary.json | 20230530172803 | 11018 | 0.0253 | 0.0274 | 0.0449 | 0.0180 | 2.49 | 2.50 | ok | low_f1 |
| 20260529T043342Z_478e28aa_3seed_loo.summary.json | 20230530172803 | 11045 | 0.0386 | 0.0351 | 0.0449 | 0.0180 | 2.49 | 2.50 | ok | low_f1 |
| 20260529T043342Z_478e28aa_3seed_loo.summary.json | 20230530212931 | 11045 | 0.2992 | 0.2481 | 0.0261 | 0.0489 | 0.53 | 2.50 | weak | fixed_threshold_not_ok |

## Summary

- Dense stride-32 validation fixed the most severe artifact: `20230530172803` no longer produces zero precision/recall folds, and its validation prevalence now appears as `0.0180` instead of `0.0088`.
- The remaining `20230530172803` issue is low ranking strength, not flooding: all three post-fix seeds sit near the cap (`pred/val=2.49`) but remain below the strict `0.04` worst-fold gate.
- The earlier positive-rate alarms on `20230522181603`, `20230522215721`, and `20230530212931` were high-ratio threshold cliffs under the `3.0` cap; the `2.5` cap removed those alarms but left some fixed-threshold weakness.
- Future tiny-torch configs should target smoother cap behavior in the `2.5-3.0` range without relaxing the worst-fold gate or claiming promotion from sampled F1 alone.

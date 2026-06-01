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

## Later Balanced-Calibration Diagnostics

Generated analyzer logs:

- `logs/diagnostics/loo_c3d8317a.md`
- `logs/diagnostics/loo_112aebf1.md`

`20260529T050630Z_c3d8317a` tested the `2.75x` positive-rate cap with tolerance `0.008`. It completed three-seed LOO with `promotion_ready=false`, median-over-seeds median `val_f1=0.1243`, mean AP `0.1163`, worst fold `20230522181603=0.0216`, and one positive-rate alarm on `20230522215721:seed=11045` where pred/val jumped to `6.00`. Eligible full-tile evidence remained weak: `20230520175435` reached `val_f1=0.2237`, AP `0.1504`, pred/val `2.74`; `20230522181603` reached `val_f1=0.1183`, AP `0.0870`, pred/val `2.66`.

`20260529T051459Z_112aebf1` tightened the same family to a `2.5x` cap and tolerance `0.004`. It was stopped after the two-seed LOO gate because `promotion_ready=false`, median-over-seeds median `val_f1=0.1393`, mean AP `0.1144`, worst fold `20230522181603=0.0158`, and `20230530172803=0.0326`; it also triggered a positive-rate alarm on `20230530212931:seed=11018` with pred/val `4.52`. Its eligible full-tile checks were lower than the `2.75x` run: `20230520175435` reached `val_f1=0.2113`, AP `0.1492`, pred/val `2.43`; `20230522181603` reached `val_f1=0.1123`, AP `0.0855`, pred/val `2.43`.

## Summary

- Dense stride-32 validation fixed the most severe artifact: `20230530172803` no longer produces zero precision/recall folds, and its validation prevalence now appears as `0.0180` instead of `0.0088`.
- The remaining `20230530172803` issue is low ranking strength, not flooding: all three post-fix seeds sit near the cap (`pred/val=2.49`) but remain below the strict `0.04` worst-fold gate.
- The earlier positive-rate alarms on `20230522181603`, `20230522215721`, and `20230530212931` were high-ratio threshold cliffs under the `3.0` cap; the `2.5` cap removed those alarms but left some fixed-threshold weakness.
- Balanced `2.75x` and strict `2.5x` follow-ups preserved full-tile eligibility but worsened `20230522181603` LOO robustness relative to the post-fix `478e28aa` reference, so sampled F1 remains insufficient evidence for promotion.
- The cap band is still brittle: under-cap thresholds can suppress recall on `20230522181603`, while nearby threshold plateaus still overrun the cap on `20230522215721` or `20230530212931`.
- Future tiny-torch configs should target smoother cap behavior in the `2.5-3.0` range without relaxing the worst-fold gate or claiming promotion from sampled F1 alone.

# Promotion Evidence: 20260529T005819Z_28af43a2

## LOO Summary

| Metric | Value |
| --- | ---: |
| promotion_ready | true |
| Median-over-seeds median val_f1 | 0.1747 |
| Mean AP | 0.1196 |
| Worst fold | 20230522181603 |
| Worst fold val_f1 | 0.0426 |

Source: `logs/20260529T005819Z_28af43a2_seedrepeat_loo.summary.json`.

## Full-Tile Results

| Segment | evaluation_region.type | val_f1 | val_f05 | AP | pred/val ratio | promotion_checks.eligible |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 20230520175435 | whole_segment | 0.2187 | 0.1755 | 0.1463 | 2.9745 | false |
| 20230522181603 | whole_segment | 0.1267 | 0.1081 | 0.0868 | 2.9641 | false |

Validation segment details (`20230520175435`): best_threshold `0.3468`, val_positive_rate `0.0805`, pred_positive_rate `0.2395`, prob_p95 `0.3529`, prob_mean `0.1949`, prob_max `0.3898`, brier_score `0.0915`, expected_calibration_error `0.1143`, ap_prevalence_lift `1.8166`, fixed_threshold_f1 `0.1743`, fixed_threshold_status `ok`.

Worst-fold segment details (`20230522181603`): best_threshold `0.3500`, val_positive_rate `0.0546`, pred_positive_rate `0.1618`, prob_p95 `0.3535`, prob_mean `0.2160`, prob_max `0.3839`, brier_score `0.0909`, expected_calibration_error `0.1614`, ap_prevalence_lift `1.5902`, fixed_threshold_f1 `0.1267`, fixed_threshold_status `ok`.

Both full-tile runs are positive-rate constrained and remain below the 3.5 pred/val ratio ceiling. Both are ineligible for promotion because `promotion_checks.validation_setup.mode` is `spatial-same-segment`; the artifact config records validation provenance from the sampled validation segment rather than held-out full-tile promotion provenance.

## Comparison vs Previous Best (20260528T205650Z_4b8dff1c)

No previous full-tile metrics file was found for `20260528T205650Z_4b8dff1c`; this comparison uses the previous best contract-compliant DB row and the candidate validation-segment full-tile result.

| Metric | Previous | Candidate | Delta |
| --- | ---: | ---: | ---: |
| val_f1 | 0.1462 | 0.2187 | +0.0725 |
| AP | 0.1151 | 0.1463 | +0.0311 |
| ap_prevalence_lift | 4.0787 | 1.8166 | -2.2621 |
| pred/val ratio | 1.1056 | 2.9745 | +1.8689 |

The candidate improves validation-segment full-tile val_f1 and AP relative to the previous sampled DB row, but AP/prevalence lift is lower and promotion eligibility is false. This is not sufficient for promotion.

## Hard Negative Mining

Not applied. Worst-fold full-tile pred/val ratio is `2.9641`, which is below the flooding threshold of `3.5`, and `prob_p95=0.3535` is slightly above `best_threshold=0.3500`. The failure mode is not uncontrolled flooding; it is weak generalization/low precision on segment `20230522181603` under a constrained operating point.

## Decision

DO NOT PROMOTE.

Reasons:

- `promotion_checks.eligible=false` on both full-tile segments.
- Full-tile validation is whole-segment, but the artifact provenance still records `spatial-same-segment`, which fails the promotion gate.
- The worst fold remains borderline: sampled LOO worst-fold val_f1 is `0.0426`, while whole-segment full-tile on `20230522181603` is better at `0.1267` but still low precision and not promotion-eligible.

## Promotion Gate

| Gate | Status | Evidence |
| --- | --- | --- |
| promotion_checks.eligible=true on validation segment full-tile | FAIL | `20230520175435` eligible=false |
| val_f1 and AP >= previous best on validation segment | PASS | val_f1 `0.2187 > 0.1462`, AP `0.1463 > 0.1151` |
| pred/val ratio <= 3.5 on all full-tile segments | PASS | ratios `2.9745` and `2.9641` |
| LOO promotion_ready=true | PASS | seed-repeat LOO summary |
| Worst fold val_f1 >= 0.04 | PASS with caveat | sampled LOO `0.0426`; full-tile `0.1267` but eligible=false |

Next action: continue evidence-backed proposal sweeps in the 2.5-3.0 positive-rate cap band and fix or generate promotion-eligible held-out full-tile lineage before another promotion attempt.

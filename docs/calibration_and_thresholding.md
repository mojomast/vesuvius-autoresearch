# Calibration And Thresholding

Current robust models are useful rankers but are not calibrated for a fixed `0.5` decision threshold. Full-tile runs often report `fixed_threshold_f1=0.0` while selected thresholds are near `0.28-0.34`, so `threshold: 0.50` should be treated as a diagnostic baseline until calibration improves.

## Required Metrics

Every sampled-validation and full-tile metrics file should expose:

- `average_precision`: ranking quality across thresholds.
- `val_positive_rate`: expected random AP baseline.
- `ap_prevalence_lift`: `average_precision / val_positive_rate`.
- `brier_score`: mean squared probability error.
- `expected_calibration_error`: bin-weighted confidence vs accuracy gap.
- `fixed_threshold_f1` and `fixed_threshold_status`: fixed `0.5` threshold health.
- `threshold_selection` / `selected_threshold_reason`: whether the operating threshold was unconstrained or positive-rate-constrained.

## Interpreting AP

For rare-positive ink detection, AP must be read against prevalence. If a segment has `val_positive_rate=0.05`, random ranking is expected to score around AP `0.05`. AP `0.09` is signal, but it is not a deployable threshold by itself.

## Positive-Rate Caps

- `prratio3`: stricter overprediction control; prefer when precision/rate safety matters most.
- `prratio3p5`: balanced current operating candidate; recovers much of the `4x` F1 while reducing flooding risk.
- `4x`: recall/F1 reference and stress test; do not treat as the default promotion target when predictions ride the cap.

## Promotion Guidance

Promotion notes should report AP/prevalence lift, Brier score, ECE, fixed-threshold status, selected threshold, and pred/val positive-rate ratio. If fixed threshold `0.5` fails but threshold-swept full-tile evidence improves, the run can remain a diagnostic or calibrated-threshold candidate, but should not be described as fixed-threshold deployable.

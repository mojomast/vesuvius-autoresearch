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
- `threshold_risk_summary`: selected, fixed, unconstrained, and cap-constrained threshold rows for `2.0x`, `3.0x`, and `3.5x` positive-rate ratios.

## Interpreting AP

For rare-positive ink detection, AP must be read against prevalence. If a segment has `val_positive_rate=0.05`, random ranking is expected to score around AP `0.05`. AP `0.09` is signal, but it is not a deployable threshold by itself.

## Positive-Rate Caps

- `prratio3`: stricter overprediction control; prefer when precision/rate safety matters most.
- `prratio3p5`: balanced current operating candidate; recovers much of the `4x` F1 while reducing flooding risk.
- `4x`: recall/F1 reference and stress test; do not treat as the default promotion target when predictions ride the cap.

Use `threshold_risk_summary.best_under_prratio2p0`, `best_under_prratio3p0`, and `best_under_prratio3p5` to decide the next cap before retraining. If `2.0x` keeps at least about 90% of selected F1/F0.5, tighten the cap to `max_pred_positive_rate_ratio: 2.0` before mining. If only `3.5x` keeps recall and `cap_binding` is true, prefer calibration or fold-safe hard-negative mining before another threshold sweep.

Tune `training.positive_rate_loss_weight` only after cap evidence shows whether overprediction is threshold-only or requires loss-level pressure. A bounded follow-up should keep the fold, seed, model, patch size, and data fixed while changing the positive-rate loss weight/tolerance.

The hard-negative planner consumes this summary. It emits `calibration_mining_decisions` so a full-tile output can recommend `tighten_positive_rate_cap`, `mine_hard_negatives`, or `review_threshold_risk` before any mined data is generated. Explicit quality actions to mine hard negatives can still surface a secondary mining command when cap tightening is the primary threshold-risk decision.

## Promotion Guidance

Promotion notes should report AP/prevalence lift, Brier score, ECE, fixed-threshold status, selected threshold, and pred/val positive-rate ratio. If fixed threshold `0.5` fails but threshold-swept full-tile evidence improves, the run can remain a diagnostic or calibrated-threshold candidate, but should not be described as fixed-threshold deployable.

## Hard-Negative Retrain Readiness

Use the planner before mining or retraining:

```bash
.venv/bin/python scripts/plan_hard_negative_retrain.py --pretty
```

Important fields:

- `mine_commands`: copyable full-tile mining commands; they write local artifacts, use fresh refresh output directories without `--overwrite`, and are never executed by the dashboard.
- `mined_inventory`: mined NPZ metadata, sample counts, positive contamination warnings, and provenance segments.
- `fold_safe_extra_train_npzs_by_heldout`: per-fold eligibility map for `dataset.extra_train_npzs`.
- `config_preview`: optional preview-only YAML when `--base-config` is provided; unsafe held-out/train-val mismatches set `valid: false` and omit YAML.

Do not use a mined NPZ in a fold whose held-out segment appears in that NPZ's provenance or forbidden segment list.

Do not mine just because a `2.0x` cap is available. Prefer cap tightening when lower caps retain signal; mine hard negatives when lower caps collapse F1 or dashboard quality actions explicitly identify flooding, speckles, or false-positive structure.

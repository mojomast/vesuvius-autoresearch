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
- `threshold_risk_summary`: selected, fixed, unconstrained, and cap-constrained threshold rows for `2.0x`, `2.5x`, `3.0x`, and `3.5x` positive-rate ratios.

## Interpreting AP

For rare-positive ink detection, AP must be read against prevalence. If a segment has `val_positive_rate=0.05`, random ranking is expected to score around AP `0.05`. AP `0.09` is signal, but it is not a deployable threshold by itself.

## Positive-Rate Caps

- `prratio2p5`: intermediate safety cap; prefer when `2.0x` loses too much recall but still needs less flooding than `3.0x`.
- `prratio3`: stricter overprediction control; prefer when precision/rate safety matters most.
- `prratio3p5`: balanced current operating candidate; recovers much of the `4x` F1 while reducing flooding risk.
- `4x`: recall/F1 reference and stress test; do not treat as the default promotion target when predictions ride the cap.

Use `threshold_risk_summary.best_under_prratio2p0`, `best_under_prratio2p5`, `best_under_prratio3p0`, and `best_under_prratio3p5` to decide the next cap before retraining. Pick the strictest cap that keeps about 95% of selected F1 and F0.5; treat 90-95% retention as review-only until seed-repeat LOO confirms it. If `2.0x` loses too much recall but `2.5x` preserves signal, prefer `max_pred_positive_rate_ratio: 2.5` before mining. If only `3.5x` keeps recall and `cap_binding` is true, prefer calibration or fold-safe hard-negative mining before another threshold sweep.

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

## No-Retrain Cap Sweeps

Use saved `metrics_by_threshold.csv` artifacts to evaluate cap policies before launching new training. Single-artifact cap comparison and next-move packaging are read-only; batch LOO recomputation reuses each run artifact once but writes recomputed JSONL/summary outputs under the paths you provide.

For the safest current-state review, package the planner and read-only cap comparisons first:

```bash
.venv/bin/python scripts/package_next_move_evidence.py \
  --caps 1.75,2.0,2.5,3.0,3.5 \
  --markdown
```

This report reuses dashboard full-tile evidence and planner-emitted cap-comparison commands. It should be run before mining or retraining when the planner action is `tighten_positive_rate_cap`.
Its `Recommended Next Move` is intentionally conservative: full-tile core regressions block cap or ensemble claims, cap tightening wins over mining when retention passes, and mining remains review-only until an artifact-writing command is deliberately executed outside the report.
Cap recommendations are all-artifact gated: if any planner-suggested cap comparison errors or lacks a retained-cap row, the report surfaces a cap blocker instead of synthesizing a cap-tightening action.

```bash
.venv/bin/python scripts/recompute_loo_threshold_cap.py \
  --input-jsonl logs/current_candidate_loo.jsonl \
  --caps 2.0,2.25,2.5,2.75,3.0 \
  --output-dir logs/cap_sweeps \
  --output-prefix current_candidate \
  --aggregate-json logs/cap_sweeps/current_candidate.aggregate.json \
  --aggregate-markdown logs/cap_sweeps/current_candidate.aggregate.md
```

Prefer this before creating a new cap-only config. It can distinguish a threshold-policy improvement from a training improvement without writing model artifacts, but the recomputed summaries are generated log artifacts and should not be committed.

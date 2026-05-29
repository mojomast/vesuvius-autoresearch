# Promotion Evidence: 20260529T014355Z_5fc7c1ca

## Provenance Fix

Root cause was confirmed as training/validation provenance contamination: prepared NPZ sidecars did not record `source_segments` or `spatial_overlap_checked`, combined train NPZs did not preserve source segment lists, and the candidate training path included held-out segment `20230520175435`. Full-tile eligibility also rejected leave-one-out diagnostic segments that were not the single recorded `val_segment_id`.

Fixes applied:

- Prepared segment metadata now records `source_segments`, `provenance`, `spatial_overlap_checked`, `val_stride`, `z_offsets`, and spatial bounding boxes.
- Combined fold metadata now records `source_segments` and `train_segments`.
- Generated robust configs now use `leaveout_20230520175435/train.npz` instead of contaminated `all_segments/train.npz`.
- Full-tile promotion checks now allow leave-one-out inference on any non-training held-out segment.

## Clean Retrains

| run_id | train provenance | sampled val_f1 | AP | pred/val ratio | fixed_threshold_status |
| --- | --- | ---: | ---: | ---: | --- |
| `20260529T013717Z_27879bc4` | excludes `20230520175435` | 0.4291 | 0.3506 | 2.9768 | ok |
| `20260529T014355Z_5fc7c1ca` | excludes `20230520175435` and `20230522181603` | 0.4258 | 0.2638 | 2.9710 | weak |

`20260529T013717Z_27879bc4` is clean for the validation segment but not for full-tile segment `20230522181603`, because that segment remains in its training source list. `20260529T014355Z_5fc7c1ca` is the strict dual-heldout artifact used for the final full-tile gate below.

## LOO Summary

| Metric | Value |
| --- | ---: |
| promotion_ready | false |
| Median-over-seeds median val_f1 | 0.0766 |
| Mean AP | 0.0873 |
| Worst fold | 20230530172803 |
| Worst fold val_f1 | 0.0000 |

Source: `logs/20260529T014355Z_5fc7c1ca_seedrepeat_loo.summary.json`.

LOO is blocked by zero precision/recall on folds `20230530172803`, `20230601193301`, and `20230611014200`. Those folds have zero validation positive rate in the sampled/tiled NPZs and need data curation before they can support a robustness promotion claim.

## Full-Tile Results

| Segment | evaluation_region.type | val_f1 | val_f05 | AP | pred/val ratio | promotion_checks.eligible |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 20230520175435 | whole_segment | 0.2168 | 0.1759 | 0.1434 | 2.9745 | true |
| 20230522181603 | whole_segment | 0.1431 | 0.1110 | 0.0933 | 2.8938 | true |

Validation segment details (`20230520175435`): best_threshold `0.2383`, val_positive_rate `0.0805`, pred_positive_rate `0.2395`, prob_p95 `0.2610`, prob_mean `0.1226`, prob_max `0.2992`, brier_score `0.0730`, expected_calibration_error `0.0444`, ap_prevalence_lift `1.7813`, fixed_threshold_f1 `0.0000`, fixed_threshold_status `weak`.

Worst-fold segment details (`20230522181603`): best_threshold `0.2517`, val_positive_rate `0.0546`, pred_positive_rate `0.1579`, prob_p95 `0.2612`, prob_mean `0.1389`, prob_max `0.3092`, brier_score `0.0621`, expected_calibration_error `0.0850`, ap_prevalence_lift `1.7101`, fixed_threshold_f1 `0.0000`, fixed_threshold_status `weak`.

## Comparison vs Previous Best (20260528T205650Z_4b8dff1c)

| Metric | Previous | Candidate | Delta |
| --- | ---: | ---: | ---: |
| val_f1 | 0.1462 | 0.2168 | +0.0706 |
| AP | 0.1151 | 0.1434 | +0.0283 |
| ap_prevalence_lift | 4.0787 | 1.7813 | -2.2974 |
| pred/val ratio | 1.1056 | 2.9745 | +1.8689 |

The dual-heldout candidate keeps the validation full-tile F1/AP improvement over the previous contract-compliant DB row, but probability scale is weak at the fixed threshold and LOO robustness fails.

## Decision

DO NOT PROMOTE.

Reasons:

- LOO `promotion_ready=false` with zero precision/recall on three folds.
- Fixed-threshold diagnostics are weak on the dual-heldout sampled run and both full-tile runs.
- Full-tile provenance is now eligible on both key segments, so the original blocker is fixed, but robustness gates still fail.

## Promotion Gate

| Gate | Status | Evidence |
| --- | --- | --- |
| promotion_checks.eligible=true on both full-tile segments | PASS | both key segments eligible=true |
| val_f1 and AP >= previous best on validation segment | PASS | val_f1 `0.2168 > 0.1462`, AP `0.1434 > 0.1151` |
| pred/val ratio <= 3.5 on all full-tile segments | PASS | ratios `2.9745` and `2.8938` |
| LOO promotion_ready=true | FAIL | seed-repeat LOO summary promotion_ready=false |
| fixed_threshold_status ok | FAIL | sampled and full-tile fixed-threshold statuses are weak |

Next action: curate/remove zero-positive validation folds from the promotion fold map, then rerun LOO and full-tile with a probability-calibrated clean candidate.

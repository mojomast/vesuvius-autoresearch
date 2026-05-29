| run_id | config | val_f1 | AP | pred/val | contract | notes |
|--------|--------|--------|-----|----------|----------|-------|
| 20260529T034229Z_497ee935 | configs/residual25d_prw006.yaml | 0.4505 | 0.4459 | 2.971 | YES | Full-tile check passed on 20230520175435: eligible=true, val_f1=0.2785, AP=0.1770, pred/val=2.974. Proceeding to 2-seed LOO per gate. |
| 20260529T034313Z_f2db2bbd | configs/residual25d_prw010.yaml | 0.4427 | 0.4149 | 2.791 | YES | Sampled gate passes; lower than prw006 and not selected for LOO. |
| 20260529T034850Z_d0ee3406 | configs/residual25d_cap250.yaml | 0.4664 | 0.4084 | 1.399 | YES | Full-tile check passed on 20230520175435: eligible=true, val_f1=0.2132, AP=0.1483, pred/val=2.429. 2-seed LOO failed on zero folds. |
| 20260529T034921Z_e5e7c928 | configs/residual25d_cap275.yaml | 0.4664 | 0.4084 | 1.399 | YES | Same sampled metrics as cap250; not selected for LOO after cap250 zero-fold failure. |
| 20260529T035001Z_e1c3ea6f | configs/residual25d_tol005.yaml | 0.4592 | 0.4190 | 2.704 | YES | Sampled gate passes; not selected for LOO because prw006 and cap250 already exposed the same zero-fold bottleneck. |
| 20260529T035036Z_b7f61463 | configs/residual25d_hneg085.yaml | 0.4433 | 0.3693 | 2.746 | YES | Sampled gate passes; hard-negative increase did not improve sampled AP/F1 over top candidates. |

## LOO Gate Result

- Base run: 20260529T034229Z_497ee935
- 2-seed LOO outputs: logs/20260529T034229Z_497ee935_2seed_loo.jsonl and logs/20260529T034229Z_497ee935_2seed_loo.summary.json
- 2-seed median-over-folds median val_f1: 0.1680
- 2-seed mean AP: 0.1372
- Worst fold: 20230530172803, median val_f1 0.0000
- Zero folds: 20230530172803 seed 11001 val_f1=0.0000, seed 11018 val_f1=0.0000
- Promotion ready: false

## Zero-Fold Diagnosis

Failure mode: segment mismatch / weak generalization on 20230530172803, not flooding. Both zero-fold runs had pred_positive_rate=0.007278 versus val_positive_rate=0.008770, so prediction rate was close to prevalence and below the flooding profile. However precision=0 and recall=0 with weak AP lift (0.726 and 0.793) despite non-collapsed probability distributions (prob_mean about 0.25, prob_p95 about 0.58, prob_max about 0.70). The model ranks positives below negatives for this held-out segment under the selected positive-rate-constrained threshold.

Decision: do not run 3-seed LOO and do not promote. A fix is required before re-running LOO, likely fold-specific data expansion or a segment-robust architecture/training change targeting 20230530172803 rather than another seed repeat.

## Cap250 Fix Attempt

Candidate: 20260529T034850Z_d0ee3406 from configs/residual25d_cap250.yaml

Full-tile check on 20230520175435:
- eligible=true
- val_f1=0.2132
- AP=0.1483
- pred/val=2.429

2-seed LOO outputs:
- logs/20260529T034850Z_d0ee3406_2seed_loo.jsonl
- logs/20260529T034850Z_d0ee3406_2seed_loo.summary.json

2-seed LOO result:
- promotion_ready=false
- median-over-folds median val_f1=0.1730
- mean AP=0.1442
- worst fold: 20230530172803, val_f1=0.0000
- zero/near-zero folds: 20230522181603 seed 11018 val_f1=0.0075, 20230530172803 seed 11001 val_f1=0.0000, 20230530172803 seed 11018 val_f1=0.0000

Fix outcome: stricter sampled positive-rate cap did not fix the zero-fold bottleneck. The repeated zeroes on 20230530172803 indicate a segment-generalization/ranking failure, not simply threshold flooding. Do not run 3-seed LOO or promotion pipeline.

Follow-up diagnosis on 2026-05-29: `scripts/evaluate_leave_one_out.py` already performs per-fold threshold sweeps, so this was not a stale global-threshold problem. For `20260529T034850Z_d0ee3406`, the 20230530172803 zero rows had non-collapsed probabilities (`prob_max=0.729` and `0.672`) and constrained `pred_positive_rate=0.007278` versus `val_positive_rate=0.008770`, but precision and recall stayed zero at the selected under-cap thresholds. Lower thresholds produced overlap only by flooding the sampled strip: the unconstrained best row reached only `val_f1=0.0231` at pred/val about `84.7` for seed 11001.

Diagnostic full-tile inference on 20230530172803 using the base `d0ee3406` artifact and config-inferred z offsets produced `val_f1=0.0854`, AP `0.0544`, pred/val `2.369`, and `prob_max=0.6065`. This is diagnostic-only because the base artifact trained on 20230530172803, but it shows the segment is not probability-collapsed. Before another LOO retrain, run candidate-linked full-tile diagnostics from the actual LOO artifacts that held out 20230530172803, or regenerate denser tiled validation NPZs/fold maps so the weak fold is measured more like the full tile.

## REASSESS

The sampled residual 2.5D axis is strong on 20230520175435, and full-tile checks for prw006 and cap250 remained eligible with full-tile val_f1 above 0.18. Promotion is blocked by leave-one-out robustness: both 2-seed LOO attempts produced zero val_f1 on held-out segment 20230530172803 for seeds 11001 and 11018.

Likely bottleneck: segment-specific generalization to low-prevalence 20230530172803. Positive-rate caps and stronger sampled metrics do not solve ranking/overlap failure on that segment.

Recommended next step: build a residual 2.5D follow-up that targets 20230530172803 explicitly, such as adding segment-aware augmentation/data expansion, reviewing fold-map/data provenance for that segment, preparing denser tiled validation folds, or training a proposal that optimizes low-prevalence folds before repeating LOO.

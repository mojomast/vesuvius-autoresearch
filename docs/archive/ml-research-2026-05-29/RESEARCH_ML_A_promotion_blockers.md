# ML A Promotion Blockers Research

Read-only inputs inspected: `autoresearch.py`, `src/autoresearch/schemas.py`, `src/autoresearch/proposals.py`, `src/autoresearch/strategy.py`, `IMPROVEMENTS.md`, `METHOD.md`, `DATA.md`, `configs/*.yaml`, `experiments/experiments.db`, and `logs/` LOO/full-tile-relevant JSON artifacts. No repository files were modified while collecting evidence.

## 1. Warning Counts

Raw DB query:
```sql
WITH e AS (
  SELECT run_id,
         json_extract(config_json,'$.dataset.research_scope') AS research_scope,
         json_extract(config_json,'$.autoresearch.scope_policy') AS scope_policy,
         CAST(json_extract(secondary_metrics_json,'$.precision') AS REAL) AS precision,
         CAST(json_extract(secondary_metrics_json,'$.recall') AS REAL) AS recall,
         CAST(json_extract(secondary_metrics_json,'$.ap_prevalence_lift') AS REAL) AS ap_lift,
         json_extract(secondary_metrics_json,'$.fixed_threshold_status') AS fixed_status,
         CAST(json_extract(secondary_metrics_json,'$.pred_positive_rate') AS REAL) AS pred_rate,
         CAST(json_extract(secondary_metrics_json,'$.val_positive_rate') AS REAL) AS val_rate
  FROM experiments
)
SELECT 'not_multisegment_scope' AS warning, SUM(CASE WHEN COALESCE(research_scope, scope_policy, '') NOT LIKE '%multi_segment%' AND COALESCE(research_scope, scope_policy, '') NOT LIKE '%leave_one_out%' THEN 1 ELSE 0 END) AS runs FROM e
UNION ALL SELECT 'fixed_threshold_status_weak', SUM(CASE WHEN fixed_status IS NOT NULL AND lower(fixed_status) <> 'ok' THEN 1 ELSE 0 END) FROM e
UNION ALL SELECT 'zero_precision_or_recall', SUM(CASE WHEN COALESCE(precision,0.0) <= 0.0 OR COALESCE(recall,0.0) <= 0.0 THEN 1 ELSE 0 END) FROM e
UNION ALL SELECT 'weak_ap_lift', SUM(CASE WHEN ap_lift IS NOT NULL AND ap_lift < 1.25 THEN 1 ELSE 0 END) FROM e
UNION ALL SELECT 'pred_positive_rate_ratio_suspicious', SUM(CASE WHEN pred_rate IS NOT NULL AND val_rate IS NOT NULL AND (pred_rate / max(val_rate, 0.000001) > 3.5 OR pred_rate / max(val_rate, 0.000001) < 0.1) THEN 1 ELSE 0 END) FROM e;
```
Raw DB query result:
| warning | runs |
|---|---:|
| not_multisegment_scope | 780 |
| fixed_threshold_status_weak | 227 |
| zero_precision_or_recall | 29 |
| weak_ap_lift | 95 |
| pred_positive_rate_ratio_suspicious | 1206 |

Analysis: most promotion blocking in the experiment table is dominated by scope and fixed-threshold weakness. Positive-rate ratio alarms are also common; zero precision/recall and weak AP lift appear in smaller but still material subsets.

## 2. Top 10 Runs By `_run_quality_score` And Gate Warnings

Raw DB basis: all rows from `experiments`; `_run_quality_score` and `_promotion_gate` reproduced from `autoresearch.py` constants and warning order.

| rank | run_id | score | val_f1 | AP | fixed_status | pred/val ratio | gate warnings |
|---:|---|---:|---:|---:|---|---:|---|
| 1 | `20260529T182028Z_8718e6cb` | 1.010931 | 0.781806 | 0.849842 | weak | 0.854 | fixed_threshold_status_weak, not_multisegment_scope |
| 2 | `20260529T035158Z_8f01550f` | 0.702706 | 0.489411 | 0.469463 | ok | 1.489 | <none> |
| 3 | `20260529T035202Z_8f01550f` | 0.702706 | 0.489411 | 0.469463 | ok | 1.489 | <none> |
| 4 | `20260529T043711Z_c956bf4a` | 0.689146 | 0.481746 | 0.453764 | ok | 1.803 | <none> |
| 5 | `20260529T034424Z_cb93d539` | 0.686145 | 0.479808 | 0.445875 | ok | 1.570 | <none> |
| 6 | `20260525T210418Z_2983030f` | 0.685273 | 0.591138 | 0.424039 | None | 2.229 | validation_not_held_out, missing_fixed_threshold_status, not_multisegment_scope |
| 7 | `20260525T211312Z_4447e41e` | 0.685273 | 0.591138 | 0.424039 | None | 2.229 | validation_not_held_out, missing_fixed_threshold_status, not_multisegment_scope |
| 8 | `20260525T210001Z_095b6013` | 0.677493 | 0.583026 | 0.424035 | None | 2.024 | validation_not_held_out, missing_fixed_threshold_status, not_multisegment_scope |
| 9 | `20260525T210418Z_8798905d` | 0.677493 | 0.583026 | 0.424035 | None | 2.024 | validation_not_held_out, missing_fixed_threshold_status, not_multisegment_scope |
| 10 | `20260529T043018Z_4b5c76cd` | 0.676976 | 0.482921 | 0.410828 | ok | 2.073 | <none> |

Analysis: the highest-scoring row is blocked by weak fixed-threshold status and non-multisegment scope. Several top rows have no sampled-run gate warnings and are promotion candidates only after linked LOO/full-tile evidence; older high-F1 rows are blocked by non-held-out validation, missing fixed-threshold status, and non-multisegment scope. High score therefore ranks promising diagnostics, not complete promotion readiness.

## 3. `best_threshold` Distribution

Raw DB query:
```sql
WITH e AS (SELECT CAST(json_extract(secondary_metrics_json,'$.best_threshold') AS REAL) AS best_threshold FROM experiments WHERE json_type(secondary_metrics_json,'$.best_threshold') IN ('integer','real'))
SELECT CASE WHEN best_threshold < 0.1 THEN '<0.1' WHEN best_threshold < 0.2 THEN '0.1-0.2' WHEN best_threshold < 0.3 THEN '0.2-0.3' WHEN best_threshold < 0.4 THEN '0.3-0.4' WHEN best_threshold < 0.5 THEN '0.4-0.5' WHEN best_threshold < 0.6 THEN '0.5-0.6' WHEN best_threshold < 0.7 THEN '0.6-0.7' WHEN best_threshold < 0.8 THEN '0.7-0.8' WHEN best_threshold < 0.9 THEN '0.8-0.9' ELSE '>=0.9' END AS bucket, COUNT(*) AS runs, ROUND(MIN(best_threshold),6) AS min_threshold, ROUND(MAX(best_threshold),6) AS max_threshold
FROM e GROUP BY bucket ORDER BY MIN(best_threshold);
```
Raw DB query result:
| bucket | runs | min_threshold | max_threshold |
|---|---:|---:|---:|
| <0.1 | 39 | 0.000634 | 0.099149 |
| 0.1-0.2 | 68 | 0.101459 | 0.198185 |
| 0.2-0.3 | 351 | 0.202062 | 0.299362 |
| 0.3-0.4 | 614 | 0.300011 | 0.39982 |
| 0.4-0.5 | 316 | 0.401017 | 0.499642 |
| 0.5-0.6 | 287 | 0.5 | 0.599792 |
| 0.6-0.7 | 133 | 0.602242 | 0.692766 |
| 0.7-0.8 | 60 | 0.707236 | 0.797612 |
| 0.8-0.9 | 47 | 0.80355 | 0.890638 |
| >=0.9 | 4 | 0.910426 | 0.95 |

Extreme query result:
| n | lt_0p1 | gt_0p9 | pct_extreme |
|---:|---:|---:|---:|
| 1919 | 39 | 4 | 2.24 |

Analysis: extremes do not dominate: 2.24% are below 0.1 or above 0.9. The mass is in mid-range calibrated thresholds, so promotion failure is not mainly caused by sweep-edge threshold selection in the DB rows.

## 4. `pred_positive_rate / val_positive_rate` Distribution

Raw DB query:
```sql
WITH e AS (SELECT CAST(json_extract(secondary_metrics_json,'$.pred_positive_rate') AS REAL)/max(CAST(json_extract(secondary_metrics_json,'$.val_positive_rate') AS REAL),0.000001) AS ratio FROM experiments WHERE json_type(secondary_metrics_json,'$.pred_positive_rate') IN ('integer','real') AND json_type(secondary_metrics_json,'$.val_positive_rate') IN ('integer','real'))
SELECT CASE WHEN ratio < 0.1 THEN '<0.1x' WHEN ratio < 0.5 THEN '0.1-0.5x' WHEN ratio < 1.0 THEN '0.5-1x' WHEN ratio < 2.0 THEN '1-2x' WHEN ratio < 3.5 THEN '2-3.5x' WHEN ratio < 10.0 THEN '3.5-10x' ELSE '>=10x' END AS bucket, COUNT(*) AS runs, ROUND(MIN(ratio),6) AS min_ratio, ROUND(MAX(ratio),6) AS max_ratio FROM e GROUP BY bucket ORDER BY MIN(ratio);
```
Raw DB query result:
| bucket | runs | min_ratio | max_ratio |
|---|---:|---:|---:|
| <0.1x | 25 | 0.0 | 0.0 |
| 0.1-0.5x | 11 | 0.11418 | 0.390651 |
| 0.5-1x | 26 | 0.507512 | 0.944954 |
| 1-2x | 88 | 1.040753 | 1.991067 |
| 2-3.5x | 588 | 2.007542 | 3.498005 |
| 3.5-10x | 1038 | 3.507188 | 9.903417 |
| >=10x | 143 | 10.04078 | 63.208784 |

Fraction query result:
| n | gt_3p5 | pct_gt_3p5 | avg_ratio |
|---:|---:|---:|---:|
| 1919 | 1181 | 61.54 | 4.866184 |

Analysis: 61.54% of rows with both rates exceed the 3.5x promotion cap. Ratio control is a primary promotion blocker, and some rows are pathological because val prevalence is zero or near zero.

## 5. LOO Summary Fold Failures

Raw log scan result: 48 LOO-style `*.summary.json` files and 884 relevant JSONL rows were parsed.

Promotion-ready summary count:
| status | summaries |
|---|---:|
| not_ready | 23 |
| ready | 25 |

Top repeated fold warnings from summaries:
| fold | worst_fold_count | zero_precision_or_recall | fixed_threshold_not_ok | weak_ap_lift | positive_rate_alarm |
|---|---:|---:|---:|---:|---:|
| 20230530172803 | 15 | 14 | 3 | 21 | 0 |
| 20230531121653 | 0 | 0 | 6 | 41 | 4 |
| 20230522181603 | 28 | 0 | 14 | 0 | 5 |
| 20230611014200 | 0 | 6 | 3 | 8 | 1 |
| 20230520175435 | 0 | 0 | 15 | 0 | 3 |
| 20230601193301 | 0 | 6 | 3 | 6 | 0 |
| 20230530212931 | 0 | 1 | 3 | 1 | 8 |
| 20230522215721 | 0 | 1 | 5 | 0 | 6 |
| 20230827161847 | 0 | 0 | 3 | 0 | 3 |
| 20230901184804 | 0 | 0 | 3 | 0 | 2 |

Top repeated fold warnings from relevant JSONL rows:
| fold | rows | zero_precision_or_recall | fixed_threshold_not_ok | weak_ap_lift | positive_rate_alarm |
|---|---:|---:|---:|---:|---:|
| 20230520175435 | 142 | 9 | 17 | 0 | 60 |
| 20230522181603 | 135 | 9 | 20 | 0 | 55 |
| 20230531121653 | 95 | 9 | 7 | 47 | 14 |
| 20230901184804 | 83 | 9 | 4 | 0 | 53 |
| 20230827161847 | 83 | 9 | 4 | 0 | 53 |
| 20230530172803 | 52 | 20 | 3 | 27 | 6 |
| 20230530212931 | 95 | 10 | 3 | 1 | 18 |
| 20230522215721 | 95 | 10 | 5 | 0 | 16 |
| 20230611014200 | 52 | 6 | 3 | 8 | 7 |
| 20230601193301 | 52 | 6 | 3 | 6 | 6 |

Analysis: one hard fold can block promotion when it contributes zero precision/recall or fixed-threshold failure to `promotion_warnings`; the summaries mark `promotion_ready: false` for those warnings even if all folds ran. The repeated hard folds are dominated by zero-label or zero-recall behavior on `20230530172803`, `20230601193301`, and `20230611014200`, while `20230520175435` appears repeatedly for fixed-threshold weakness and worst-fold status in other runs.

## 6. Config Parameters Correlating With Fixed-Threshold Pass/Fail

Raw DB query for fixed-threshold status distribution:
```sql
SELECT COALESCE(lower(json_extract(secondary_metrics_json,'$.fixed_threshold_status')),'<missing>') status, COUNT(*) runs FROM experiments GROUP BY status ORDER BY runs DESC;
```
Raw DB query result:
| status | runs |
|---|---:|
| <missing> | 1251 |
| ok | 441 |
| weak | 227 |

Strongest numeric point-biserial correlations with `fixed_threshold_status = ok` among DB rows where status is present:
| param | r | n | min | max | pass_rate |
|---|---:|---:|---:|---:|---:|
| evaluation.threshold | -0.780 | 668 | 0.31 | 0.5 | 0.660 |
| training.augment_flips | 0.618 | 350 | 0 | 1 | 0.591 |
| evaluation.tta_flips | 0.487 | 287 | 0 | 1 | 0.652 |
| training.positive_rate_loss_tolerance | -0.475 | 569 | 0.0025 | 0.02 | 0.759 |
| evaluation.max_pred_positive_rate_ratio | -0.432 | 600 | 2 | 4 | 0.725 |
| training.max_train_samples | 0.362 | 668 | 0 | 4096 | 0.660 |
| training.epochs | -0.332 | 668 | 1 | 8 | 0.660 |
| training.learning_rate | -0.311 | 668 | 0.00072 | 0.002 | 0.660 |
| training.positive_rate_loss_weight | 0.259 | 567 | 0 | 0.1 | 0.762 |
| training.batch_size | 0.246 | 668 | 4 | 8 | 0.660 |

Strongest categorical associations with `fixed_threshold_status = ok`:
| param | cramers_v | n | levels | level pass rates shown |
|---|---:|---:|---:|---|
| evaluation.threshold | 0.810 | 668 | 6 | 0.31: 218 rows/0.91, 0.35: 265 rows/0.88, 0.5: 178 rows/0.04 |
| dataset.research_scope | 0.720 | 668 | 20 | focused_pair_residual_25d_cpu: 21 rows/0.05, multi_segment_balanced_prratio2p5_tolerance0p004: 46 rows/1.00, multi_segment_balanced_prratio2p75_tolerance0p008: 70 rows/1.00, multi_segment_residual25d_prcontrol_worstfold: 169 rows/0.93, multi_segment_residual25d_prcontrol_worstfold_stride32_allval: 25 rows/0.88, multi_segment_robust_expanded: 114 rows/0.22, multi_segment_robust_expanded_prratio2p5_tolerance0p005: 35 rows/1.00, multi_segment_robust_expanded_prratio3_seed11018: 149 rows/0.54 |
| training.augment_flips | 0.618 | 350 | 2 | False: 131 rows/0.20, True: 219 rows/0.83 |
| evaluation.max_pred_positive_rate_ratio | 0.494 | 600 | 6 | 2.0: 44 rows/0.91, 2.5: 166 rows/0.91, 2.75: 79 rows/0.90, 3.0: 265 rows/0.63, 3.5: 43 rows/0.12 |
| evaluation.tta_flips | 0.487 | 287 | 2 | False: 40 rows/0.07, True: 247 rows/0.74 |
| training.positive_rate_loss_tolerance | 0.478 | 569 | 6 | 0.004: 46 rows/1.00, 0.005: 58 rows/0.98, 0.008: 78 rows/0.90, 0.01: 181 rows/0.87, 0.02: 205 rows/0.49 |
| training.max_train_samples | 0.425 | 668 | 6 | 0: 51 rows/0.35, 1024: 211 rows/0.52, 256: 6 rows/1.00, 3072: 23 rows/1.00, 4096: 356 rows/0.80, 512: 21 rows/0.05 |
| training.learning_rate | 0.391 | 668 | 6 | 0.001: 196 rows/0.92, 0.0012: 458 rows/0.57, 0.002: 6 rows/0.00 |
| training.positive_rate_loss_weight | 0.377 | 567 | 8 | 0.0: 5 rows/0.00, 0.01: 5 rows/0.00, 0.02: 5 rows/0.00, 0.03: 350 rows/0.72, 0.06: 22 rows/0.82, 0.08: 144 rows/0.92, 0.1: 34 rows/0.85 |
| training.epochs | 0.371 | 668 | 5 | 1: 6 rows/1.00, 4: 196 rows/0.92, 5: 463 rows/0.55 |

Analysis: fixed-threshold pass/fail correlates most with validation fold/scope lineage and calibration controls, not a single universal scalar. Positive-rate constrained residual/robust configs with explicit caps and positive-rate loss/tolerance are more often `ok`; focused or early sweep configs and some threshold/seed-only variants frequently remain weak. Because many parameters co-vary by campaign, these are associations, not causal estimates.

## 7. Config YAML Scan

Parsed config summary:
```json
{
  "changed_paths": [
    [
      "<none>",
      58
    ],
    [
      "training.seed",
      22
    ],
    [
      "evaluation.max_pred_positive_rate_ratio",
      21
    ],
    [
      "evaluation.threshold",
      18
    ],
    [
      "training.positive_rate_loss_tolerance",
      17
    ],
    [
      "training.positive_rate_loss_weight",
      15
    ],
    [
      "evaluation.tta_flips",
      12
    ],
    [
      "training.dice_loss_weight",
      10
    ],
    [
      "model.base_channels",
      9
    ],
    [
      "balanced_calibration",
      8
    ],
    [
      "training.learning_rate",
      6
    ],
    [
      "training.sampling_strategy",
      5
    ]
  ],
  "model_names": [
    [
      "tiny_torch_unet",
      179
    ],
    [
      "residual_25d_torch_unet",
      36
    ],
    [
      "tiny_numpy_ink_logreg",
      5
    ],
    [
      "tiny_numpy_mlp",
      2
    ]
  ],
  "parsed": 222,
  "scopes": [
    [
      "multi_segment_robust_expanded",
      79
    ],
    [
      "multi_segment_robust_expanded_prratio3_seed11018",
      48
    ],
    [
      "multi_segment_robust_expanded_tta_ensemble",
      25
    ],
    [
      "focused_pair_residual_25d_cpu",
      22
    ],
    [
      "focused_pair",
      22
    ],
    [
      "multi_segment_residual25d_prcontrol_worstfold",
      13
    ],
    [
      "multi_segment_robust_expanded_seed_ensemble_notta",
      3
    ],
    [
      "multi_segment_robust_expanded_prratio2p5_tolerance0p005",
      3
    ],
    [
      "multi_segment_robust_expanded_seed_ensemble_prratio3_notta",
      2
    ],
    [
      "next_best_moves_robust_protocol",
      1
    ]
  ],
  "yaml_files": 222
}
```

Analysis: current config inventory is heavily skewed toward auto-generated one-change follow-ups plus robust/residual 2.5D promotion-action configs. This matches the DB evidence: the search is mostly changing calibration, threshold/cap, seed, and small training knobs around a few robust bases rather than discovering a completely new model family.

## Conclusion

Promotion is blocked less by raw sampled F1 than by evidence gates: held-out/multi-segment scope, fixed-threshold behavior, positive-rate ratio control, fold-level zero precision/recall, and linked LOO/full-tile promotion checks. The most actionable blocker is calibration at fixed threshold and positive-rate cap/tolerance on hard folds; a single failing LOO fold is sufficient to keep `promotion_ready` false.

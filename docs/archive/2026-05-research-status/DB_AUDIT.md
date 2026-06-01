# DB Audit

Phase 1 completed before launching new experiments. Database source: `experiments/experiments.db`, queried with Python `sqlite3` against the local SQLite DB. Scoring formula for this audit: `val_f1 + 0.25 * average_precision + 0.10 * val_f05`, with a positive-rate penalty for `pred_positive_rate / val_positive_rate > 3.0`.

## Best Run By Audit Score

- `BEST_RUN`: `20260529T182028Z_8718e6cb`
- Audit score: `1.080931`
- Artifact dir: `experiments/runs/20260529T182028Z_8718e6cb`
- Gate warnings: `fixed_threshold_status_weak`, `not_multisegment_scope`
- Key issue: excellent ranking signal on a focused-pair diagnostic run, but not promotion-ready because it is not multi-segment LOO scope and fixed-threshold output predicts no positives.

Metrics:

| Metric | Value |
|---|---:|
| `val_f1` | `0.781806` |
| `val_f05` | `0.866645` |
| `average_precision` | `0.849842` |
| `precision` | `0.848517` |
| `recall` | `0.724820` |
| `pred_positive_rate` | `0.057491` |
| `val_positive_rate` | `0.067303` |
| pred/val ratio | `0.854220` |
| `fixed_threshold_status` | `weak` |
| `fixed_threshold_f1` | `0.0` |
| `best_threshold` | `0.332448` |

Config sections:

```json
{
  "model": {"name": "residual_25d_torch_unet", "base_channels": 8},
  "dataset": {
    "research_scope": "focused_pair_residual_25d_cpu",
    "train_npz": "data/real_cross/segment_20230827161847_pf0p15_n2048/train.npz",
    "val_npz": "data/real_cross/segment_20230520175435/val.npz",
    "patch_size": 64,
    "validation_mode": "cross-segment"
  },
  "training": {
    "allow_cuda": false,
    "augment_flips": false,
    "batch_size": 4,
    "deterministic": true,
    "epochs": 5,
    "learning_rate": 0.0012,
    "max_train_samples": 512,
    "pos_weight": "auto",
    "weight_decay": 0.0001
  },
  "evaluation": {"main_metric": "val_f1", "threshold": 0.5, "tta_flips": true}
}
```

## Top 5 Calibration-Passing Runs

Criteria: `fixed_threshold_status == ok` and `0.5 <= pred_positive_rate / val_positive_rate <= 3.0`.

| Rank | Run | Score | `val_f1` | AP | `val_f05` | Ratio | Config note |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `20260529T035158Z_8f01550f` | `0.652706` | `0.489411` | `0.469463` | `0.459297` | `1.489` | residual 2.5D, PR weight `0.08`, cap `2.5`, seed `11001` |
| 2 | `20260529T035202Z_8f01550f` | `0.652706` | `0.489411` | `0.469463` | `0.459297` | `1.489` | same signature duplicate artifact |
| 3 | `20260529T205010Z_b3a735dc` | `0.642116` | `0.492384` | `0.430042` | `0.422212` | `2.477` | targeted fixed-threshold cap `2.75`, Dice `0.20`, seed `11018` |
| 4 | `20260529T043711Z_c956bf4a` | `0.639146` | `0.481746` | `0.453764` | `0.439584` | `1.803` | stride32 all-val residual 2.5D, seed `11045` |
| 5 | `20260529T034424Z_cb93d539` | `0.636145` | `0.479808` | `0.445875` | `0.448688` | `1.570` | residual 2.5D, PR weight `0.06`, cap `3.0`, seed `11001` |

## Top 3 Ranking-Signal Runs By Average Precision

| Rank | Run | AP | `val_f1` | Ratio | Gate warnings |
|---:|---|---:|---:|---:|---|
| 1 | `20260529T182028Z_8718e6cb` | `0.849842` | `0.781806` | `0.854` | `fixed_threshold_status_weak`, `not_multisegment_scope` |
| 2 | `20260529T182008Z_52c69462` | `0.510735` | `0.482899` | `1.227` | `fixed_threshold_status_weak`, `not_multisegment_scope` |
| 3 | `20260529T035158Z_8f01550f` | `0.469463` | `0.489411` | `1.489` | none |

## LOO Summary Scan

- Summary files scanned: `66`
- Files with `promotion_ready=true`: `31`
- Per instruction, because at least one `promotion_ready=true` summary exists, manual Batches A-C are skipped and execution proceeds to Phase 3.

Most relevant fresh ready summary:

- `logs/20260530T003819Z_6b3111c8_seedrepeat_loo.summary.json`
- Base config: `experiments/runs/20260530T003819Z_6b3111c8/config.json`
- `promotion_ready=true`, `promotion_warnings=[]`
- Seeds: `11001`, `11018`, `11045`
- Requested/successful folds: `24/24`
- Median `val_f1`: `0.163957`
- Mean `val_f1`: `0.187498`
- Worst fold: `20230530172803`, `val_f1=0.039621`
- Note: this is script-level promotion-ready, but the downstream weak-fold `0.04` gate is effectively at the edge because worst fold is `0.000379` below `0.04`.

Per-fold F1 for the fresh focal LOO summary:

| Fold | F1 |
|---|---:|
| `20230520175435` | `0.451547` |
| `20230522181603` | `0.094101` |
| `20230522215721` | `0.139636` |
| `20230530172803` | `0.039621` |
| `20230530212931` | `0.188278` |
| `20230531121653` | `0.291312` |
| `20230601193301` | `0.094525` |
| `20230611014200` | `0.200966` |

## Audit Decision

- Best score overall is a focused diagnostic run, not usable for promotion.
- Best calibrated promotion-style sampled base is `20260529T035158Z_8f01550f` / duplicate `20260529T035202Z_8f01550f`.
- Best fresh candidate with completed LOO evidence is `20260530T003819Z_6b3111c8`, but it remains borderline on downstream worst-fold F1 for `20230530172803`.
- Proceeding directly to high-compute Bayesian autonomous cycles per the explicit Phase 1 skip rule.

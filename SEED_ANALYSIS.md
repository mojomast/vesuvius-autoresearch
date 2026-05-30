# Seed Analysis

## Summary

The hard fold `20230530172803` is extremely seed-sensitive under the villa-label focal hard-mining recipe. The original useful seed `11001` reached AP `0.064540` and F1 `0.142771`; local seed search found a stronger seed, `11071`, and a 20-epoch refinement reached AP `0.121940` and F1 `0.199885`.

This is real sampled progress, but not promotion evidence. The result still needs LOO and full-tile validation before it can support promotion.

## Best Runs

| Run | Seed(s) | Variant | AP | F1 | Fixed status | Pred/val ratio |
|---|---:|---|---:|---:|---|---:|
| `20260530T032140Z_d4d69b0f` | `11071` | 20 epochs | `0.121940` | `0.199885` | ok | `1.841176` |
| `20260530T032140Z_386d0c85` | `11071` | LR `0.0006` | `0.106733` | `0.198718` | ok | `1.902428` |
| `20260530T030713Z_af36e8c5` | `11071` | 15 epochs | `0.098751` | `0.185276` | ok | `1.526962` |
| `20260530T023312Z_26f70635` | `11073` | 15 epochs | `0.096917` | `0.171957` | ok | `2.554911` |
| `20260530T020142Z_4e5cb820` | `11001` | 15 epochs | `0.064540` | `0.142771` | ok | `2.554911` |

## Sweep Results

| Run | Seed(s) | AP | F1 | Fixed status |
|---|---:|---:|---:|---|
| `20260530T023539Z_41fa95c0` | `11201` | `0.058139` | `0.116586` | ok |
| `20260530T024043Z_884d07d1` | `11317` | `0.060176` | `0.130553` | weak |
| `20260530T030713Z_fb5449a9` | `11072` | `0.056636` | `0.088360` | weak |
| `20260530T031402Z_0bb8c438` | `11074` | `0.095185` | `0.165224` | ok |
| `20260530T031402Z_80cf715b` | `11075` | `0.032566` | `0.064382` | weak |
| `20260530T023539Z_1a1b29aa` | `11099` | `0.033960` | `0.061527` | weak |
| `20260530T020403Z_5883d17b` | `11018` | `0.026957` | `0.032896` | weak |

## Ensemble Result

The built-in probability ensemble of `11001,11045,11073` (`20260530T024329Z_530c2561`) reached AP `0.059997` and F1 `0.130830`, worse than the strongest single seeds. Averaging probabilities appears to dilute the high-confidence ranking signal from the best seed rather than improve AP on this fold.

## Interpretation

The fold is learnable: multiple seeds now exceed AP `0.09`, and the best sampled run exceeds AP `0.1`. However, convergence is unstable. Seeds `11071`, `11073`, and `11074` form a local high-performing cluster, while adjacent seeds such as `11072` and `11075` fail fixed-threshold evidence.

The most plausible cause is optimizer/initialization sensitivity on a sparse, low-prevalence validation fold. Lower LR helped seed `11071`, and longer training helped more. Architecture changes, group stratification, and positive patch pressure changes were not the lever.

## Recommendation

Use `configs/seedsweep_hardfold_seed11071_epoch20.yaml` as the current best sampled hard-fold rescue config. Treat `11071` as the new golden seed for `20230530172803`, with `11073` and `11074` as backup seeds. Do not use probability ensembling as the default until a different ensemble method proves it can preserve AP ranking.

Promotion gates should remain unchanged. The next validation step is LOO only if the team accepts spending compute despite sampled F1 still being below the historical LOO entry threshold.

## Follow-Up Fold Calibration

After the hard-fold breakthrough, the next meaningful weak villa-label fold was `20230522215721`; folds `20230522181603` and `20230601193301` currently have zero validation positives in `data/fold_map_villa_labels.json`, so AP/F1 are not meaningful there.

For `20230522215721`, the hard-fold seed `11071` recipe underperformed (`20260530T054336Z_b8e15468`: AP `0.060314`, F1 `0.075191`, fixed threshold `weak`). Restoring that fold's historically stronger seed/loss family and calibrating the fixed threshold produced a clear improvement:

| Run | Fold | Variant | AP | F1 | Fixed F1 | Fixed status | Pred/val ratio |
|---|---|---|---:|---:|---:|---|---:|
| `20260530T055508Z_78c9b225` | `20230522215721` | seed `11018`, PR weight `0.08`, threshold `0.35` | `0.121645` | `0.209780` | `0.096339` | weak | `1.624341` |
| `20260530T060027Z_98d3bb41` | `20230522215721` | seed `11018`, PR weight `0.08`, threshold `0.20` | `0.203345` | `0.317771` | `0.271323` | ok | `0.768921` |

This shows the second weak fold is primarily threshold/calibration-sensitive rather than architecture-limited. The low-load calibration run used 8 compute threads and 4 DataLoader workers to avoid CPU saturation.

# Quality Score Analysis

`scripts/backtest_quality_score.py` was expanded to grid AP, F0.5, and calibration-penalty weights over historical `experiments/experiments.db` rows.

The tested grid was:

- `val_f1` weight fixed at `1.0`
- AP weight: `0.1, 0.2, 0.3, 0.4`
- F0.5 weight: `0.0, 0.05, 0.1, 0.2`
- calibration penalty weight: `0.0, 0.03, 0.05, 0.1`

The sampled-F1-dominated top 10 remains stable because historical high-F1 focused rows are much higher than promotion-safe rows. The useful change is therefore not a wholesale ranking replacement, but adding an explicit capped calibration penalty while keeping existing promotion-gate warning penalties.

Selected defaults:

- AP weight: `0.20`
- F0.5 weight: `0.10`
- calibration penalty: `0.05 * min(abs(pred_positive_rate / val_positive_rate - 1.0), 0.5)`

This keeps AP/F0.5 signal in the score while directly penalizing prediction-rate drift before the existing hard ratio and promotion-warning penalties apply.

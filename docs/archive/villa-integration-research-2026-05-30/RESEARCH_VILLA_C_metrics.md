# Phase 1 Research Agent C: Villa Metrics Comparison

## Scope

Compared local metric computation in `experiments/runner.py` against Villa's `ink-detection/metrics/binary_segmentation.py`, with stitched metric semantics checked where relevant.

## Villa Reference Behavior

`StreamingBinarySegmentationMetrics` accepts `logits`, `targets`, and optional `mask`.

- Validates `logits.shape == targets.shape` and, when present, `mask.shape == targets.shape`.
- Applies `mask.bool()` before metric computation, so excluded pixels do not affect counts.
- Converts targets to foreground with `targets.float() >= 0.5`.
- Converts logits to probabilities with `torch.sigmoid(logits).float()`.
- Applies a probability threshold with `preds = probs >= threshold`.
- Uses `confusion_counts(preds, targets_bool)` with boolean tensors and float64 reductions for `tp`, `fp`, and `fn`.
- Computes Dice/F1 as `2TP / (2TP + FP + FN + eps)`.
- Does not compute AP or threshold sweeps in `binary_segmentation.py`.

Villa stitched metrics use the same probability-threshold convention for postprocessing: `pred_prob >= threshold`, combined with an evaluation mask. Stitched metrics also validate threshold is finite and in `[0, 1]`.

## Local Runner Behavior

Local metrics are centralized in `_pixel_metrics_from_probs` and `_binary_metrics`.

- Inputs to `_pixel_metrics_from_probs` are probabilities, not logits.
- NumPy and torch paths both apply sigmoid before calling `_pixel_metrics_from_probs`.
- `_binary_metrics` thresholds with `pred = probs >= threshold`, matching Villa's inclusive probability-threshold rule.
- Labels are flattened and treated as positive by exact comparison in confusion counts, `labels == 1`; other local metrics use `labels > 0.5`.
- Counts are produced by NumPy boolean sums, then converted to Python `float`; there is no explicit float64 accumulator object.
- The fixed configured threshold is computed first, then a sweep over linspace thresholds, score quantiles, and the configured threshold selects `best_f1_row`.
- `average_precision` is computed locally by sorting probabilities descending and averaging precision at positive labels.
- There is no explicit metric mask parameter; evaluation inclusion is determined by which pixels are present in the `probs` and `labels` arrays passed to `_pixel_metrics_from_probs`.

## Key Comparisons

### Sigmoid Threshold Semantics

Villa thresholds sigmoid probabilities derived from logits. This runner thresholds probabilities that have already had sigmoid applied in each model path. Therefore threshold semantics are equivalent when local callers pass calibrated probabilities in `[0, 1]`.

Important distinction: Villa's API name makes logits the contract; local `_pixel_metrics_from_probs` makes probabilities the contract. Passing logits directly to local metrics would be wrong and would corrupt loss, F1, AP, calibration, and threshold selection.

### Confusion Counts And F1/Dice

Villa and local runner both use inclusive foreground prediction, `>= threshold`, and compute the same F1/Dice formula conceptually.

Differences:

- Villa targets are positive at `>= 0.5`; local `_binary_metrics` uses `labels == 1`.
- Villa accumulates `tp/fp/fn` as float64 tensors; local metrics convert integer NumPy sums to Python floats after counting.
- Villa computes only fixed-threshold Dice in the streaming binary metric; local `val_f1` is selected from a threshold sweep and `fixed_threshold_f1` is reported separately.
- Villa uses denominator `+ 1e-12`; local precision/recall/F1 use `max(..., 1.0)` or `max(..., 1e-9)`. Empty-positive/empty-prediction cases should usually report zero in both, but exact edge behavior and tiny-denominator behavior are not identical.

### Float64 Accumulators

Villa explicitly reduces counts with `dtype=torch.float64`, suitable for large streaming validation sets. Local counts are exact while held in NumPy integer sums, then converted to Python float. Python floats are double precision, but the local implementation is not a streaming float64 accumulator and does not document the stability contract.

For current flattened in-memory arrays this is likely numerically safe. For future streaming or stitched full-tile parity, explicit int64 or float64 count accumulation would better match Villa.

### Mask Handling

Villa binary metrics support an optional mask and remove excluded pixels before thresholding/counting. Villa stitched metrics form `eval_mask = pred_has & valid`, crop to it, and zero predictions/GT outside the mask before computing metrics.

Local runner has no explicit mask argument in `_pixel_metrics_from_probs`. It assumes `probs` and `labels` already represent the evaluation population. This is acceptable for pre-cropped NPZ validation but is not equivalent to Villa's metric contract for partial coverage, invalid regions, or stitched full-tile evaluation unless masking is handled upstream.

### AP Availability

Villa `StreamingBinarySegmentationMetrics` does not expose AP. It returns only `dice` from fixed-threshold confusion counts.

Local runner requires and reports `average_precision`, plus `ap_prevalence_lift`. AP is computed from all flattened probabilities and labels, independent of the selected threshold. This is additional local behavior, not directly comparable to Villa's streaming binary metric output.

### Threshold Selection And Fixed Threshold Status

Villa streaming binary metrics are fixed-threshold only. The configured threshold is the metric threshold; there is no selected best threshold and no status classification.

Local runner separates:

- `fixed_threshold`: configured evaluation threshold, default `0.5`.
- `fixed_threshold_f1`: F1 at that configured threshold.
- `best_threshold`: selected threshold from the sweep, optionally constrained by positive-rate ratio.
- `val_f1`: F1 at `best_threshold`, not at the fixed threshold.
- `fixed_threshold_status`: `ok` if fixed-threshold F1 is at least 50% of selected-threshold F1, else `weak`.
- `fixed_threshold_failure_reason`: local diagnostic categories such as `no_fixed_positive_predictions`, `fixed_zero_precision`, `fixed_zero_recall`, or `weak_relative_to_selected_f1`.

Therefore `fixed_threshold_status` is a local robustness diagnostic, not a Villa metric. It should not be interpreted as a pass/fail equivalent from Villa.

## Compatibility Assessment

For fixed-threshold F1/Dice on already-masked probability arrays, local `fixed_threshold_f1` should be conceptually comparable to Villa's `dice` from `StreamingBinarySegmentationMetrics`, subject to target binarization and tiny-denominator differences.

Local `val_f1` is not directly comparable to Villa's streaming `dice` because it is threshold-selected. The comparable local field is `fixed_threshold_f1` at the same probability threshold, with the same evaluated pixel mask.

Local AP and AP prevalence lift have no direct counterpart in Villa's streaming binary metric.

## Risks To Track

- A caller could pass logits into `_pixel_metrics_from_probs`; Villa would handle logits correctly, local metrics would not.
- Local metrics do not carry an explicit mask, so full-tile or partial-coverage evaluation can diverge from Villa unless upstream arrays are pre-filtered exactly.
- Local target binarization is inconsistent between `_binary_metrics` (`labels == 1`) and AP/calibration/rates (`labels > 0.5`). This is harmless for strict binary labels but differs from Villa's `>= 0.5` contract.
- `val_f1` is threshold-optimized and can overstate fixed-threshold deployment behavior relative to Villa's fixed-threshold Dice.
- Local float handling is probably safe for current in-memory arrays but does not match Villa's explicit float64 streaming accumulator design.

## Bottom Line

The runner's `fixed_threshold_f1` is the closest local analog to Villa's streaming `dice`, provided inputs are probabilities and masking has already been applied. The runner's headline `val_f1`, AP, threshold sweep, and `fixed_threshold_status` are local additions and should be reported as such when comparing against Villa metrics.

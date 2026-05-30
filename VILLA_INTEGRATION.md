# Villa Integration

## Source And Credits

Components adapted from ScrollPrize/villa, MIT License, by Youssef Nader, Luke Farritor, and Julian Schilliger:

- `src/autoresearch/villa_samplers.py`: `StatefulShuffledSampler`, `GroupStratifiedBatchSampler`.
- `src/autoresearch/villa_metrics.py`: `StreamingBinarySegmentationMetrics`, `confusion_counts`, Dice helpers.
- `scripts/rebuild_npz_with_villa_labels.py`: downloads and applies cleaned labels from `ink-detection/all_labels/`.

See `CREDITS.md` for attribution details.

## Label Delta Findings

The exact hard fold `20230530172803` is present in villa as `20230530172803_inklabels.png`. Its reliable tiled validation delta is small:

| Segment | Split | Current rate | Villa rate | IoU | Villa-not-current | Current-not-villa |
|---|---:|---:|---:|---:|---:|
| `20230530172803` | val | `0.018045` | `0.017589` | `0.974698` | `0` | `3054` |

Other folds show larger deltas, especially some tiled validation regions. Full details are in `RESEARCH_VILLA_A_label_delta.md` and `data/npz_villa/rebuild_summary.json`.

## Sampler Integration

`StatefulShuffledSampler` is available behind `training.stateful_sampler: true` or `training.sampling_strategy: stateful_shuffled`. It preserves a cursor across iterator creation, which matters most for partial-epoch or batch-limited training.

`GroupStratifiedBatchSampler` is available behind `training.group_stratified_sampling: true`. For combined NPZs, segment groups are reconstructed from `metadata.inputs`; batch size is adjusted down when needed to be divisible by the group count.

## Metrics Cross-Validation

Villa's streaming binary metric computes fixed-threshold Dice/F1 from logits using sigmoid probabilities and float64 confusion-count accumulators. Local `fixed_threshold_f1` is the equivalent local metric; local AP and threshold-swept `val_f1` are additional project metrics.

The runner now computes villa fixed-threshold F1 when `evaluation.use_villa_metrics: true` and logs `METRIC DISCREPANCY` if it differs from the local fixed-threshold F1 by more than `0.005`.

## Hard-Fold Results

| Run | Labels | Variant | F1 | AP | Fixed status | Pred/val ratio |
|---|---|---|---:|---:|---|---:|
| `20260530T004150Z_0b659aec` | current | prior weak-fold sampled artifact | `0.043435` | `0.030085` | ok | n/a |
| `20260530T004150Z_0b659aec` | current | weak-fold full-tile | `0.014940` | `0.027372` | ok | `2.659699` |
| `20260530T013234Z_c226ec6d` | villa | hard-fold baseline | `0.082057` | `0.037456` | weak | `2.554911` |
| `20260530T013457Z_1df6dddd` | villa | focal hard-fold | `0.136424` | `0.062981` | ok | `0.882155` |
| `20260530T013847Z_eb682e02` | villa | group-stratified fallback | `0.096390` | `0.044858` | ok | `1.484317` |

The focal villa-label run is the best hard-fold sampled result from this pass, but AP remains below `0.1`, so it is progress, not promotion evidence.

## LOO Outcome

No villa-label candidate met the Worker 5 LOO entry criteria (`val_f1 >= 0.42`, fixed-threshold `ok`, pred/val ratio `0.5..3.5`). The expensive 7-seed LOO pass was therefore skipped to avoid promoting or over-validating a weak sampled hard-fold run.

The three requested high-compute autoresearch cycles were executed and all paused on existing `run_full_tile_validation` planner evidence rather than launching new proposals.

## License Compliance

All borrowed villa source files include inline attribution headers. Cleaned label usage is attributed in `CREDITS.md` and this document. Generated large label/NPZ data remains outside normal git tracking under ignored `data/` paths; lightweight fold-map and rebuild-summary metadata are committed for reproducibility.

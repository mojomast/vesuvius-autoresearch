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
| `20260530T015039Z_5634be4c` | villa | focal, positive patches `0.30` | `0.042781` | `0.029655` | ok | `2.554911` |
| `20260530T015259Z_077d05f6` | villa | focal, group-stratified | `0.019669` | `0.031259` | ok | `2.554911` |
| `20260530T015539Z_a5410bc2` | villa | focal, base channels `16` | `0.054670` | `0.031924` | ok | `2.554911` |
| `20260530T020142Z_4e5cb820` | villa | focal, seed `11001` | `0.142771` | `0.064540` | ok | `2.554911` |
| `20260530T020403Z_5883d17b` | villa | focal, seed `11018` | `0.032896` | `0.026957` | weak | `2.554911` |
| `20260530T020633Z_b6dbe53e` | villa | focal, positive patches `0.60` | `0.057402` | `0.046106` | ok | `2.731593` |
| `20260530T023312Z_26f70635` | villa | focal, seed `11073` | `0.171957` | `0.096917` | ok | `2.554911` |
| `20260530T030713Z_af36e8c5` | villa | focal, seed `11071` | `0.185276` | `0.098751` | ok | `1.526962` |
| `20260530T032140Z_386d0c85` | villa | focal, seed `11071`, LR `0.0006` | `0.198718` | `0.106733` | ok | `1.902428` |
| `20260530T032140Z_d4d69b0f` | villa | focal, seed `11071`, 20 epochs | `0.199885` | `0.121940` | ok | `1.841176` |
| `20260530T024329Z_530c2561` | villa | focal ensemble `11001,11045,11073` | `0.130830` | `0.059997` | ok | `1.038905` |

Seed `11071` with the same focal recipe and 20 epochs is now the best hard-fold sampled result from this pass, improving AP from seed `11001`'s `0.064540` to `0.121940` and F1 from `0.142771` to `0.199885`. This clears the sampled AP `0.1` target, but F1 remains far below the LOO entry threshold, so it is progress, not promotion evidence.

The targeted follow-up batch falsified three simple hypotheses for this fold: lower positive patch pressure (`0.30`), higher positive patch pressure (`0.60`), group-stratified batches, and wider base channels all underperformed the original focal recipe. The dominant useful signal is seed sensitivity under the same focal/hard-mining setup; local seeds around `11073` found `11071`, and refining that seed with lower LR or longer training pushed sampled AP above `0.1`.

Built-in probability ensembling did not help on the sampled hard fold: the `11001,11045,11073` ensemble reached AP `0.059997`, below all strong single seeds. The current best path is seed `11071` with 20 epochs, not ensemble averaging.

## LOO Outcome

No villa-label candidate met the Worker 5 LOO entry criteria (`val_f1 >= 0.42`, fixed-threshold `ok`, pred/val ratio `0.5..3.5`). The expensive 7-seed LOO pass was therefore skipped to avoid promoting or over-validating a weak sampled hard-fold run.

The three requested high-compute autoresearch cycles were executed and all paused on existing `run_full_tile_validation` planner evidence rather than launching new proposals.

## License Compliance

All borrowed villa source files include inline attribution headers. Cleaned label usage is attributed in `CREDITS.md` and this document. Generated large label/NPZ data remains outside normal git tracking under ignored `data/` paths; lightweight fold-map and rebuild-summary metadata are committed for reproducibility.

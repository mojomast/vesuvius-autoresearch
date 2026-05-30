# Credits

## ScrollPrize/villa

- Components: `src/autoresearch/villa_samplers.py` with `StatefulShuffledSampler` and `GroupStratifiedBatchSampler`.
- Source: https://github.com/ScrollPrize/villa/blob/main/ink-detection/samplers.py
- License: MIT License.
- Original authors: Youssef Nader, Luke Farritor, Julian Schilliger.
- Changes: copied sampler behavior with local type annotations and integrated as opt-in torch DataLoader samplers.

- Components: `src/autoresearch/villa_metrics.py` with `StreamingBinarySegmentationMetrics`, `confusion_counts`, and Dice helpers.
- Source: https://github.com/ScrollPrize/villa/blob/main/ink-detection/metrics/binary_segmentation.py
- License: MIT License.
- Original authors: Youssef Nader, Luke Farritor, Julian Schilliger.
- Changes: copied streaming fixed-threshold Dice behavior, added optional mask support to `confusion_counts`, and integrated as a local fixed-threshold metric cross-check.

- Components: cleaned ink labels downloaded by `scripts/rebuild_npz_with_villa_labels.py` into `data/villa_labels/` and applied to rebuilt local NPZ label arrays under `data/npz_villa/`.
- Source: https://github.com/ScrollPrize/villa/tree/main/ink-detection/all_labels
- License: MIT License.
- Original authors: Youssef Nader, Luke Farritor, Julian Schilliger.
- Changes: labels are resized/aligned to existing local NPZ spatial metadata with nearest-neighbor alignment and used to replace label arrays without overwriting original NPZ files.

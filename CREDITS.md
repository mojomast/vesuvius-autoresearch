# Credits

## ScrollPrize/villa

Attribution statement used for borrowed code and labels: Adapted from ScrollPrize/villa (MIT License) — Youssef Nader, Luke Farritor, Julian Schilliger.

Sources were reviewed from the public `ScrollPrize/villa` repository in May 2026. Inline attribution headers are preserved in borrowed-code files; this document records component-level attribution for reviewers.

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

Cleaned labels and regenerated NPZ files are generated local data artifacts and are not redistributed by this repository. Use them only under the applicable Scroll Prize/Vesuvius data terms and the upstream Villa license notices.

# Villa Samplers Research - Agent B

Read-only sources: `experiments/runner.py`, current repo sampling reports/config evidence, and Villa `ink-detection/samplers.py` from `https://raw.githubusercontent.com/ScrollPrize/villa/main/ink-detection/samplers.py`.

## 1. Villa Sampler Behavior

`StatefulShuffledSampler` creates one seeded `torch.randperm(num_samples)` order and keeps a persistent cursor across `__iter__` calls. Each iterator yields exactly `num_samples` indices, wraps at the end, and does not reshuffle between epochs. Its main value is for training loops that intentionally stop early: the next epoch resumes at the next unseen index instead of repeating the same shuffled prefix.

`GroupStratifiedBatchSampler` takes one integer group id per sample, builds `indices_by_group`, and yields complete batches. It requires `batch_size >= n_groups` and `batch_size % n_groups == 0`, samples `batch_size / n_groups` items from every group per batch, reshuffles each group pool when exhausted, and shuffles the final mixed batch. Its main value is per-batch group balance, not just epoch-level randomization.

## 2. Current Sampling Implementation

Current NumPy sampling is pixel-level and one-shot. `_train_numpy_mlp()` flattens patch features to pixels, optionally caps to `training.max_train_pixels`, then chooses either a random pixel subset or a class-balanced subset using `training.sample_positive_fraction`. After that, each epoch uses `rng.permutation(len(yt))` and slices mini-batches. This balances individual pixels, not patches, source segments, groups, or batches.

Current torch sampling is patch-level and one-shot unless a curriculum is enabled. `_sample_patch_indices()` chooses all patches, uniform random patches, or `hard_mining`: positive patches weighted by ink fraction, high-texture zero-ink negatives, random negatives, then fill and shuffle. `_train_torch_unet()` materializes the selected/augmented arrays into a `TensorDataset` and uses `DataLoader(..., shuffle=True, generator=loader_generator, num_workers=0)`. With `sampling_curriculum`, it reruns `_sample_patch_indices()` per epoch and then still uses `shuffle=True`.

The runner has fold-safety checks for `extra_train_npzs`, but no sampler currently knows segment, source NPZ, fragment, prevalence bucket, hard-fold id, or patch group metadata. The only group-like control is indirect: prepared train/val NPZ choice and optional extra train NPZ composition.

## 3. StatefulShuffledSampler Comparison

`StatefulShuffledSampler` is not a direct replacement for the current full-epoch torch loader because the current loader already iterates through the whole selected dataset each epoch. In that case, PyTorch `shuffle=True` is fine and arguably better because it reshuffles every epoch.

It becomes useful if this repo adds batch-limited epochs, gradient-step budgets, or partial-epoch cron probes. Under those conditions, the current `shuffle=True` path can repeatedly expose the same early shuffled prefixes across epochs/runs, while Villa's persistent cursor would provide better coverage of the selected patch pool before repeats.

It does not solve class imbalance or fold imbalance by itself. It also does not change which patches are selected by `_sample_patch_indices()`, so it would not replace hard mining, positive patch fraction, or texture-based negatives.

## 4. GroupStratifiedBatchSampler Comparison

`GroupStratifiedBatchSampler` targets the most relevant gap in the current torch implementation: balanced exposure across groups inside each batch. Current hard mining can produce a useful global mix, but each mini-batch is only a random draw from the selected pool; low-prevalence or hard-fold-origin patches can be sparse or clumped.

For this repo, useful group definitions would need to be added before this sampler has value. Candidate group ids include source segment, source NPZ, positive/negative patch bucket, ink-fraction bucket, or a compact cross of segment bucket and ink bucket. Because Villa's implementation requires equal per-group counts per batch and `batch_size % n_groups == 0`, the group count must stay small. With current CPU batch sizes of `4` or `8`, segment-level stratification across many leave-one-out training segments is not practical without grouping segments into coarse buckets or increasing batch size.

The sampler oversamples smaller groups by reshuffling that group when exhausted. That is desirable for rare groups but can overfit small hard-fold/mined pools if used naively. It should be treated as a training distribution change and recorded in metrics/config signatures.

## 5. Minimal Integration Path

Minimal low-risk path is additive and torch-only:

1. Add a small local sampler module or inline classes copied/adapted from Villa with attribution.
2. Add config keys such as `training.batch_sampler: stateful_shuffle|group_stratified` and `training.group_stratify_by: ink_bucket|source_segment|source_npz`.
3. For `stateful_shuffle`, replace `DataLoader(..., shuffle=True, generator=...)` with `DataLoader(..., sampler=StatefulShuffledSampler(len(ds), seed=run_seed), shuffle=False)` only when a future `max_batches_per_epoch` or step cap is active.
4. For `group_stratified`, compute `group_indices` after patch selection and augmentation so indices align with the `TensorDataset`; use `batch_sampler=GroupStratifiedBatchSampler(group_indices, batch_size=batch_size, seed=run_seed, drop_last=True)` and omit `batch_size`/`shuffle` from the `DataLoader`.
5. Start with `ink_bucket` grouping, not segment grouping: e.g. zero-ink, low-positive, medium/high-positive. This uses labels already in memory and keeps group count compatible with batch size.
6. Record sampler name, group mode, group counts, `drop_last`, and effective batches in `model_meta.json` and metrics before using results in promotion decisions.

Do not integrate this into the NumPy logreg path first. NumPy logreg is full-batch over all flattened pixels; NumPy MLP already has a pixel-level subset and epoch permutation, and segment-aware batch sampling would require a larger refactor to retain pixel-to-patch/source metadata after flattening.

## 6. Hard-Fold Impact

Expected hard-fold impact is moderate but not a standalone fix. The current blocker is hard-fold robustness, especially `20230530172803`: LOO median F1 around `0.039621` and weak-fold full-tile F1 around `0.014940`, despite fixed-threshold eligibility in the focal family. Sampling can help exposure and gradient balance, but it cannot create missing positive signal in a held-out fold or replace calibration/full-tile checks.

`StatefulShuffledSampler` is unlikely to improve hard folds unless future runs use partial epochs or step caps. Current torch training consumes full selected datasets each epoch, so persistent cursor coverage adds little.

`GroupStratifiedBatchSampler` has the better hard-fold hypothesis. If grouped by ink fraction, each batch can consistently include sparse-positive patches instead of relying on global hard-mining proportions. If grouped by source segment or source NPZ, it can reduce dominance by easier/high-prevalence segments in multi-segment training. However, segment-level balancing with current small batch sizes is constrained, and oversampling small segment pools could increase overfit unless paired with existing augmentation and positive-rate controls.

Recommended first ablation for hard folds: keep the current best residual 2.5D hard-mining baseline (`max_train_samples=1024`, `positive_patch_fraction=0.45`, `hard_negative_fraction=0.75`, flips/TTA, focal/calibration controls), add an `ink_bucket` group-stratified batch sampler with 2 or 4 groups compatible with `batch_size=4/8`, and evaluate on leave-one-out plus full-tile weak fold before considering segment-level grouping.

## Recommendation

Adopt neither Villa sampler as a direct drop-in today. The smallest useful future integration is a torch-only, opt-in `GroupStratifiedBatchSampler` using label-derived ink buckets after patch selection. `StatefulShuffledSampler` should wait until the runner has partial-epoch/step-budget training. For hard-fold work, group-stratified batches are worth one controlled ablation, but the expected gain is gradient/exposure stability rather than a guaranteed promotion unblock.

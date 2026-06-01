# PHASE 1 RESEARCH AGENT D: Compute Profile

Scope: `run_experiment.py` with `configs/targeted_promo_focal_bce_sparse_ink.yaml` only. No training run was executed.

## Dry-run status

`run_experiment.py` does not support `--dry-run`.

Evidence:

```text
$ .venv/bin/python run_experiment.py --config configs/targeted_promo_focal_bce_sparse_ink.yaml --dry-run
usage: run_experiment.py [-h] --config CONFIG
run_experiment.py: error: unrecognized arguments: --dry-run
```

Because dry-run is unsupported and a normal invocation would train and write artifacts/DB rows, full training was intentionally avoided.

## Current config compute settings

From `configs/targeted_promo_focal_bce_sparse_ink.yaml`:

```yaml
model:
  name: residual_25d_torch_unet
  input_mode: z_offsets_as_channels
  base_channels: 8
training:
  allow_cuda: false
  batch_size: 8
  deterministic: true
  epochs: 4
  max_train_samples: 1024
  num_threads: 4
  augment_flips: true
evaluation:
  tta_flips: true
```

Important implication: this is bounded but still real training. `augment_flips: true` expands the sampled 1024 patches to 3072 training examples, and `tta_flips: true` makes validation inference 4x model forwards per batch.

## Runner inspection

Relevant `experiments/runner.py` behavior:

- `training.batch_size` is read at line 854, defaulting to 8.
- `training.num_threads` is read at line 859, defaulting to `min(4, os.cpu_count())`.
- `torch.set_num_threads(max(1, num_threads))` is called at line 891.
- `torch.set_num_interop_threads(...)` is not called.
- Training DataLoaders are created with `num_workers=0` at lines 1050 and 1052.
- Validation prediction does not use a DataLoader; it slices NumPy arrays by `batch_size` at lines 996-1007.

DataLoader conclusion: multiprocessing worker tuning is currently unavailable through config because `num_workers=0` is hard-coded. With in-memory NumPy arrays and small 64x64 patches, this is reasonable; CPU compute threads matter more than loader workers.

## Host and PyTorch environment

Commands used `.venv/bin/python`.

```text
nproc: 32
os.cpu_count(): 32
torch: 2.12.0+cu130
cuda_available: False
torch.get_num_threads(): 16
torch.get_num_interop_threads(): 32
MKL available: True
OpenMP available: True
ATen backend: OpenMP
OpenMP max threads: 16
MKL max threads: 16
OMP_NUM_THREADS: not set
MKL_NUM_THREADS: not set
```

The installed PyTorch build defaults to 16 intra-op threads and 32 inter-op threads on this 32-vCPU host. The runner overrides only intra-op threads from `training.num_threads`, so the current config lowers PyTorch intra-op compute to 4 threads while leaving inter-op at the process default.

## Safe profile results

No full training was run. The safe profile covered config load, NPZ validation/load, hard-mining sample selection, flip augmentation, DataLoader iteration, and one-batch synthetic forward/backward for the same residual 2.5D U-Net shape.

Dataset and sampling:

```text
train shape:        [7424, 3, 64, 64]
val shape:          [128, 3, 64, 64]
sampled shape:      [1024, 3, 64, 64]
augmented shape:    [3072, 3, 64, 64]
loader batches:     384 at batch_size=8
```

Timing:

```text
config load:        0.002 s
NPZ validation:     0.953 s
NPZ load/cast:      0.978 s
hard-mining sample: 0.224 s
flip augmentation:  0.081 s
DataLoader iterate: 0.091 s for 3072 samples, num_workers=0
```

Sampling metrics:

```text
positive_patches_available: 2863
negative_patches_available: 4551
selected_patches: 1024
positive_patches_selected: 461
selected_patch_positive_rate: 0.4502
patch_sampling: hard_mining
```

Synthetic one-batch forward/backward microbenchmarks, CPU only, inter-op set to 1 for the benchmark process:

```text
batch_size=8,  threads=4:  824 samples/s
batch_size=8,  threads=8:  826 samples/s
batch_size=8,  threads=16: 799 samples/s
batch_size=16, threads=4:  1015 samples/s
batch_size=16, threads=8:  1178 samples/s
batch_size=16, threads=16: 1263 samples/s
batch_size=32, threads=4:  858 samples/s
batch_size=32, threads=8:  1309 samples/s
batch_size=32, threads=16: 1456 samples/s
```

Interpretation: with the current `batch_size: 8`, moving beyond 8 threads does not help. If the run can use a larger batch, 16 threads become useful and `batch_size: 32` had the best synthetic throughput among tested safe settings.

## Recommended 16-core settings

For a 16-core CPU allocation, prefer using all 16 cores only if increasing batch size. Exact recommended settings:

```bash
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 .venv/bin/python run_experiment.py --config configs/targeted_promo_focal_bce_sparse_ink.yaml
```

Recommended config values for a 16-core run:

```yaml
training:
  batch_size: 32
  num_threads: 16
```

Keep these current values unless there is a model-quality reason to change them:

```yaml
training:
  max_train_samples: 1024
  epochs: 4
  allow_cuda: false
```

If `batch_size: 32` causes memory pressure on another host, fallback:

```yaml
training:
  batch_size: 16
  num_threads: 16
```

For the current `batch_size: 8`, the better setting is not a true 16-core utilization setting:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python run_experiment.py --config configs/targeted_promo_focal_bce_sparse_ink.yaml
```

```yaml
training:
  batch_size: 8
  num_threads: 8
```

This avoids spending 16 cores on a small batch where the microbenchmark showed no benefit.

## Notes and risks

- These are safe microbenchmarks, not full epoch timings. Real training also includes optimizer cost, loss terms, and validation TTA.
- The runner does not expose `num_workers`; it is fixed at 0.
- The runner does not control PyTorch inter-op threads. If code changes become allowed later, setting inter-op threads to 1 near the torch thread setup would reduce oversubscription risk for this single-model training loop.
- No artifacts, database rows, or model checkpoints were intentionally created by this research pass.

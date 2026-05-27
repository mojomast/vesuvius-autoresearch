# Method Summary

Vesuvius AutoResearch is a small, CPU-safe research loop for ink-detection experiments on public Vesuvius data. It is designed to make false-positive control and validation evidence visible before any promotion claim.

## Pipeline

1. Prepare real Vesuvius segment patches into validated NPZ files.
2. Train a compact baseline model or tiny Torch U-Net from a YAML config.
3. Evaluate by threshold sweep, average precision, calibration diagnostics, and positive-rate ratio.
4. Run leave-one-segment-out validation for robust candidates.
5. Run full-tile inference on candidate and weak-fold segments.
6. Use dashboard quality verdicts to select next actions such as calibration or fold-safe hard-negative mining.

## Models

Current supported model families include:

- `tiny_numpy_ink_logreg`: fast pixel classifier for sanity checks.
- `tiny_numpy_mlp`: lightweight nonlinear baseline.
- `tiny_torch_unet` and residual 2.5D U-Net variants: compact convolutional models for real patch training.

## Validation

Validation claims should prefer cross-segment or leave-one-segment-out evidence. Same-segment spatial splits are diagnostic only unless explicitly labeled as such.

Promotion requires more than sampled F1:

- Positive precision and recall.
- AP above prevalence, reported as `ap_prevalence_lift`.
- Controlled `pred_positive_rate / val_positive_rate`.
- Fixed-threshold diagnostics at `0.5`.
- Full-tile evidence on the candidate or weak fold.
- No fold leakage from held-out segments.

## Hard-Negative Mining

Full-tile inference can mine high-confidence false-positive patches into NPZ files. These are training artifacts and must remain outside git.

Fold safety is enforced through metadata: mined negatives from a segment cannot be used when that segment is held out. Add eligible mined files to `dataset.extra_train_npzs` only after checking the segment provenance.

Use the dry-run planner before retraining:

```bash
.venv/bin/python scripts/plan_hard_negative_retrain.py --pretty
```

The planner does not write mined data. It reads dashboard evidence, recommends bounded mining commands for overpredicting full-tile outputs, inventories existing `data/mined/**/*.npz` files, rejects held-out segment leakage, and previews fold-safe `dataset.extra_train_npzs` updates.

## Hallucination Controls

- Use small, local patch windows by default, usually `64 x 64` at 8 micron scale.
- Report full-tile quality, not only positive-biased sampled metrics.
- Keep threshold selection transparent: log whether it maximizes F1 or constrains positive rate.
- Preserve exact config, commit, data metadata, and command provenance for every submission candidate.

## Limitations

The current promotion candidate is a research baseline, not a proven decode. Current failures are dominated by overprediction and weak fixed-threshold behavior on full-tile outputs. Treat dashboard next actions as the source of truth for the next research step.

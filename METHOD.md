# Method Summary

Vesuvius AutoResearch is a CPU-safe, evidence-gated research loop for ink-detection experiments on public Vesuvius data. It is designed to make false-positive control and validation evidence visible in a reviewer-safe dashboard before any promotion claim.

## Pipeline

1. Prepare real Vesuvius segment patches into validated NPZ files.
2. Train a compact baseline model or tiny Torch U-Net from a YAML config.
3. Evaluate by threshold sweep, average precision, calibration diagnostics, and positive-rate ratio.
4. Run leave-one-segment-out validation for robust candidates.
5. Run full-tile inference on candidate and weak-fold segments.
6. Use dashboard quality verdicts and read-only evidence packages to select next actions such as cap calibration, hard-fold AP/separability audit, full-tile regression review, or fold-safe hard-negative mining.

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
- Controlled `pred_positive_rate / val_positive_rate`; ratios above `3.5x` block promotion.
- Fixed-threshold diagnostics at `0.5`; any present `fixed_threshold_status` other than `ok` blocks promotion.
- Full-tile evidence on the candidate and linked LOO folds, with no quality failures.
- Weak-fold full-tile evidence counts only when it is linked to the held-out LOO/candidate lineage, evaluates `evaluation_region.type: whole_segment`, and passes promotion checks for that segment.
- Sampled-only hard-fold results are diagnostic. The repeated `20230530172803` blocker must be cleared by candidate-linked LOO and full-tile evidence, not by threshold-only gate relaxation.
- No fold leakage from held-out segments.

## Hard-Negative Mining

Full-tile inference can mine high-confidence false-positive patches into NPZ files. These are training artifacts and must remain outside git.

Fold safety is enforced through metadata: mined negatives from a segment cannot be used when that segment is held out. Add eligible mined files to `dataset.extra_train_npzs` only after checking the segment provenance.

Use the dry-run planner before retraining:

```bash
.venv/bin/python scripts/plan_hard_negative_retrain.py --pretty
```

The planner does not write mined data. It reads dashboard evidence, recommends bounded mining commands for overpredicting full-tile outputs, inventories existing `data/mined/**/*.npz` files, rejects held-out segment leakage, and previews fold-safe `dataset.extra_train_npzs` updates. Mining commands use a fresh `experiments/runs/<run_id>/mining_refresh_<segment_id>` output directory and do not include `--overwrite`, so existing full-tile evidence remains intact.

Before mining or retraining, run the read-only next-move evidence package:

```bash
.venv/bin/python scripts/package_next_move_evidence.py --markdown
```

The package composes the dashboard snapshot, planner output, read-only cap comparisons, and optional full-tile artifact pairs. It recommends cap tightening only when all planner-suggested cap comparisons produce retained-cap evidence, surfaces blockers when cap reports fail, and marks embedded mining commands as artifact-writing review-only actions.

Planner output is fold-scoped. `fold_safe_extra_train_npzs_by_heldout` shows which mined NPZs are eligible or rejected for each held-out segment, while top-level `eligible_extra_train_npzs` remains the active weak-fold shortcut. Use `--base-config` and `--heldout-segment` to render a preview-only YAML config in JSON output; if the held-out override does not match the train/val NPZ paths, `config_preview.valid` is false and the YAML preview is omitted.

## Hallucination Controls

- Use small, local patch windows by default, usually `64 x 64` at 8 micron scale.
- Report full-tile quality, not only positive-biased sampled metrics.
- Keep threshold selection transparent: log whether it maximizes F1 or constrains positive rate.
- Preserve exact config, commit, data metadata, and command provenance for every submission candidate.

## Limitations

The current promotion candidate is a research baseline, not a proven decode. Current failures are dominated by hard-fold separability, overprediction risk, and weak fixed-threshold behavior on full-tile outputs. Treat dashboard next actions as the source of truth for the next research step.

Pick the strictest positive-rate cap that preserves about 95% of selected F1 and F0.5, considering intermediate caps such as `2.5x` before mining or broader retraining. If lower caps collapse F1 or F0.5, prefer fold-safe hard-negative mining or a bounded positive-rate-loss/sampling adjustment rather than another unconstrained threshold sweep.

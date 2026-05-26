# Vesuvius AutoResearch Plan, May 2026

This repo should optimize for reproducible cross-segment ink detection, not isolated leaderboard-style validation wins.

## Current External Signals

- Vesuvius Challenge data is now organized around standard segment artifacts: OME-Zarr/ Zarr arrays, meshes, masks, surface volumes, predictions, and metadata.
- Official 2026 prize criteria emphasize methods that work across scrolls, are reproducible, and include hallucination mitigation.
- Current prize guidance discourages machine-learning windows larger than about 0.5 x 0.5 mm, corresponding to 64 x 64 pixels for 8 um scans.
- The ScrollPrize monorepo exposes the 2023 Grand Prize ink-detection code, current data tooling, and VC3D-oriented segmentation workflows.
- Recent prize history shows continuing value in data handling, patch generation, ink-label quality, self-supervised pretraining, 3D/2.5D ink detection, and robust segment/surface workflows.

## Implications For This Repo

- Keep 64 x 64 patch experiments as the default research unit unless a specific non-submission analysis needs larger context.
- Promote configs by leave-one-segment-out median F1 first, then mean F1/AP, not by a single fold.
- Treat same-segment spatial validation as diagnostic only; it is not enough for promotion.
- Preserve train/prediction separation in every final claim and record held-out segment IDs in configs and logs.
- Prefer modular scripts that consume standard paths and emit JSONL/JSON summaries so results can be reproduced by another machine.
- Do not publish prepared data, local venvs, run artifacts, or model weights in Git; keep the repo source-first.

## Near-Term Research Agenda

1. Re-run robust candidates through `scripts/evaluate_leave_one_out.py` on the expanded fold map before promoting any config.
2. Improve data coverage by preparing additional labeled public segments only after rate limits cool down, then regenerate the fold map.
3. Add threshold-calibrated inference on full validation tiles, because per-pixel F1 on sampled patches can overstate real segment utility.
4. Add model families that are still small but closer to winning approaches: 2.5D U-Net with more z offsets, residual blocks, and optional pretrained encoders when GPU is available.
5. Add uncertainty checks: seed ensembles, test-time flip averaging, per-fold probability calibration, and prediction-rate alarms.
6. Add dataset diagnostics: positive coverage maps, train/val region plots, and segment-level metadata summaries.
7. Compare against public Grand Prize and Kaggle-style baselines before investing in larger sweeps.

## Promotion Gate

A config should be considered a robust champion only if it has:

- A committed config file.
- A leave-one-out JSONL log and summary JSON.
- No known train/validation segment overlap.
- Median F1 improvement over the current robust champion or a clear improvement in min-fold behavior.
- Comparable or improved average precision.
- A short note explaining whether it is a peak-score champion, robust champion, or diagnostic-only run.

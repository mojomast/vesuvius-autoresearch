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

1. Re-run robust candidates through `scripts/evaluate_leave_one_out.py` on the expanded fold map with repeated seeds before promoting any config.
2. Improve data coverage by preparing additional labeled public segments only after rate limits cool down, then regenerate a leakage-safe fold map.
3. Continue threshold-calibrated inference on full validation tiles, because per-pixel F1 on sampled patches can overstate real segment utility. Seed-repeat LOO now exists for `robust_calibrated_prloss_w0p03_lr0012_prratio3` and `prratio3p5`; next work should improve calibration rather than emit more local F1 micro-sweeps.
4. Add uncertainty checks: seed ensembles, test-time flip averaging, per-fold probability calibration, prediction-rate alarms, Brier score, expected calibration error, and AP/prevalence lift.
5. Add model families that are still small but closer to winning approaches: 2.5D residual U-Net with more z offsets and optional pretrained encoders when GPU is available.
6. Add dataset diagnostics: positive coverage maps, train/val region plots, and segment-level metadata summaries.
7. Compare against public Grand Prize and Kaggle-style baselines before investing in larger sweeps.

See `docs/next_best_moves_may2026.md` for command examples and promotion checks for seed-repeat LOO, safe data expansion, full-tile inference, TTA/seed ensembling, and 2.5D residual U-Net work.

## Plateau Policy

When recent robust/torch runs stop improving, AutoResearch should switch out of local exploit mode instead of emitting more near-duplicate micro-sweeps:

- Detect plateaus over the recent robust/torch window, not focused same-segment runs.
- Force proposal diversity by mutation family: optimizer, loss calibration, data sampling, model family, inference calibration, and replication.
- Avoid emitting multiple configs from the same mutation family in one plateau batch.
- Prefer promotion-aware next actions before more exploration: seed-repeat leave-one-out, then full-tile validation, then promotion review.
- Keep generated configs diagnostic-only until seed-repeat LOO and full-tile checks pass.
- Use `python3 autoresearch.py --plan --json` to inspect the next cycle without writing generated configs or launching experiments.
- When the promotion gate is ready, pause exploration by default and spend compute on promotion review, weak-fold full-tile checks, or data diagnostics.
- AutoResearch planning must consume dashboard candidate evidence, not just peak-score summaries: if a weak-fold full-tile action is available, surface that action and its paced public-directory command before proposing more experiments.
- Dashboard and audit top-level next actions should also prefer unfinished candidate evidence actions, so a plateau does not look like idle promotion readiness while weak-fold diagnostics remain missing.
- Weak-fold full-tile diagnostics should run on the seed-repeat LOO artifact that actually held out the weak fold; using a different promotion candidate artifact is diagnostic-only and can leak the target segment into training.
- Treat a ready gate as the start of review, not the end: run full-tile diagnostics on the linked LOO weakest fold before claiming Scroll Prize robustness.
- Use `scripts/evaluate_leave_one_out.py --jobs N` for seed-repeat LOO throughput only when resources allow; jobs are independent process workers and parent-only JSONL output preserves reproducibility.
- Let stale generated `configs/auto_*` reservations expire so killed exploratory configs do not permanently suppress useful ideas; completed DB runs remain reserved evidence.

## Promotion Gate

A config should be considered a robust champion only if it has:

- A committed config file.
- A leave-one-out JSONL log and summary JSON.
- No known train/validation segment overlap.
- Median F1 improvement over the current robust champion or a clear improvement in min-fold behavior.
- Comparable or improved average precision.
- Seed-repeat stability when the change affects training or model initialization; promotion summaries require at least three distinct successful seeds.
- Full-tile validation metrics when the change affects data, thresholding, or inference.
- Candidate-linked evidence: seed-repeat LOO and full-tile metrics must match the same config/run lineage as the robust candidate being promoted.
- A short note explaining whether it is a peak-score champion, robust champion, or diagnostic-only run.
- A calibration note reporting AP relative to prevalence, best threshold, fixed-threshold F1/status, pred/val positive-rate ratio, Brier score, expected calibration error, and whether threshold selection was unconstrained or positive-rate-constrained.

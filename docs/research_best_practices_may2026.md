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
6. Add dataset diagnostics: positive coverage maps, train/val region plots, segment-level metadata summaries, and validation-strip positive-coverage checks.
7. Compare against public Grand Prize and Kaggle-style baselines before investing in larger sweeps.

See `docs/next_best_moves_may2026.md` for command examples and promotion checks for seed-repeat LOO, safe data expansion, full-tile inference, TTA/seed ensembling, and 2.5D residual U-Net work.

## Data Provenance

The `promotion_checks.eligible=false` root cause for `20260529T005819Z_28af43a2` was spatial provenance, not the full-tile metric values. Prepared NPZ metadata did not record `source_segments` or `spatial_overlap_checked`, and the robust generated configs pointed at `all_segments/train.npz`, which can include the nominal validation segment. That made full-tile inference diagnostic-only once provenance was inspected.

The fix records `source_segments`, `provenance`, `spatial_overlap_checked`, `val_stride`, `z_offsets`, and spatial bounding boxes next to prepared NPZs. Combined train NPZs now preserve `train_segments`, and robust configs use leave-one-out train NPZs rather than `all_segments/train.npz`. Leave-one-out full-tile checks are eligible for any non-training held-out segment, while training-segment leakage is still rejected.

The follow-up validation-strip fix prevents tiled held-out validation regions from being empty-positive when the source segment contains ink positives. This turned the prior zero-F1 LOO collapse into measurable but still weak fold performance: `20260529T021416Z_570f6775` removed all zero precision/recall folds and reached three-seed median-over-seeds median `val_f1=0.1109`, but still did not promote because of fold-level positive-rate alarms.

## Positive-Rate Alarms

Positive-rate alarms can mean different things and should be diagnosed before changing gates. For `20260529T021416Z_570f6775`, the alarms were not fixed-threshold failures and not an overly strict summary threshold: the fold rows had `fixed_threshold_status=ok`, but selected unconstrained `best_f1` because no threshold satisfied the configured positive-rate band. Threshold CSV inspection showed cliffs where one threshold floods and the next threshold collapses recall.

Do not clear these alarms by relaxing the promotion summary alone. First try model-side probability-scale fixes such as tighter positive-rate loss tolerance, cap-aware training, calibration loss changes, or segment-balanced sampling. If a post-hoc fallback would choose near-zero-recall thresholds, keep the run diagnostic-only.

## Seed Selection

Seed-repeat LOO is a robustness test, not a seed-shopping mechanism. `seed=15050` repeatedly flooded small validation strips on `20230530212931` and `20230531121653`, but replacement seed `15073` introduced zero precision/recall failures for `20260529T024258Z_8b44032d`. Treat both as evidence of probability-scale instability rather than proof that a single seed should be excluded.

The current standard seed set remains `11001,11018,15050` unless a candidate-specific evidence package explicitly justifies a replacement. If replacing a seed for diagnosis, document the original failure, the replacement seed behavior, and whether median-over-seeds and worst-fold metrics still satisfy promotion gates.

## Results To Date

Post-fix positive-rate calibration batch, run on 2026-05-29:

| run_id | val_f1 | AP | pred_pos_rate | LOO median val_f1 | full-tile val_f1 | full-tile AP | promotion_ready | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `20260529T005748Z_7d9b2b4d` | 0.3447 | 0.2420 | 0.3148 | not run | not run | not run | not assessed | diagnostic |
| `20260529T005804Z_12a24594` | 0.3887 | 0.2473 | 0.4969 | not run | not run | not run | not assessed | unsafe ratio |
| `20260529T005819Z_28af43a2` | 0.3740 | 0.2598 | 0.3839 | 0.1747 | 0.2187 | 0.1463 | true | do not promote |
| `20260529T011645Z_20f27abf` | 0.3783 | 0.2621 | 0.3839 | not run | not run | not run | not assessed | diagnostic |
| `20260529T013717Z_27879bc4` | 0.4291 | 0.3506 | 0.4161 | 0.0766 | 0.2129 | 0.1423 | false | diagnostic |
| `20260529T014355Z_5fc7c1ca` | 0.4258 | 0.2638 | 0.4153 | 0.0766 | 0.2168 | 0.1434 | false | do not promote |
| `20260529T021416Z_570f6775` | 0.4071 | 0.2998 | 0.4149 | 0.1109 | 0.2279 | 0.1509 | false | do not promote |
| `20260529T023543Z_021a01b0` | 0.3912 | 0.2959 | 0.3462 | 0.1045 | not run | not run | false | do not promote |
| `20260529T024258Z_8b44032d` | 0.4110 | 0.2756 | 0.4881 | 0.0681 | 0.2363 | 0.1508 | false | do not promote |

The selected LOO candidate was `20260529T005819Z_28af43a2` because its pred/val positive-rate ratio was below 3.5. Its seed-repeat LOO summary reported mean AP `0.1196` and worst fold `20230522181603` at `val_f1=0.0426`. Full-tile inference improved the validation-segment comparison against the previous contract-compliant sampled DB row (`val_f1=0.2187` vs `0.1462`, AP `0.1463` vs `0.1151`) and kept both full-tile pred/val ratios below `3.5`, but promotion was rejected because `promotion_checks.eligible=false` on both full-tile segments due `spatial-same-segment` artifact provenance.

Post-decision sweeps confirmed the planner now exercises both requested axes: `20260529T011256Z_375c2944` tested `max_pred_positive_rate_ratio=2.5`, and `20260529T011645Z_20f27abf` tested `positive_rate_loss_tolerance=0.01` with `max_pred_positive_rate_ratio=3.0`. Treat these as diagnostic until LOO/full-tile promotion lineage is available.

Clean-provenance retrains confirmed the original eligibility blocker is fixed. `20260529T014355Z_5fc7c1ca` excludes both key full-tile segments from training and has `promotion_checks.eligible=true` for `20230520175435` and `20230522181603`. It still should not promote: LOO `promotion_ready=false`, worst fold `20230530172803` has `val_f1=0.0`, and fixed-threshold diagnostics are weak.

Validation-strip regeneration fixed the empty-positive held-out strips for `20230530172803`, `20230601193301`, and `20230611014200`. The retrained candidate `20260529T021416Z_570f6775` improved LOO median-over-seeds median to `0.1109` with no zero precision/recall folds, and full-tile checks remained eligible. It still should not promote because three-seed LOO reported positive-rate alarms on `20230522181603`, `20230522215721`, and `20230530212931`.

Positive-rate tightening with `20260529T023543Z_021a01b0` reduced alarms but did not pass promotion: the two-seed check cleared alarms but median-over-seeds median fell to `0.0954`, and the three-seed check recovered median `0.1045` while still alarming on `20230530212931:seed=15050` and `20230531121653:seed=15050`. Diagnostic full-tile for `20260529T022151Z_93da433a` reached unconstrained `val_f1=0.2562`, but pred/val `6.01` makes it diagnostic-only; under a `3.0x` cap it would not beat `570f6775`.

The `8b44032d` prratio3.5 diagnostic improved eligible full-tile `20230520175435` to `val_f1=0.2363`, but LOO with `11001,11018,15073` failed badly: median-over-seeds median `0.0681`, worst fold `0.0133`, and zero precision/recall on two `15073` folds. The `caa81174` prratio3 diagnostic was safer on full-tile pred/val (`2.97`) but did not beat `570f6775` (`val_f1=0.2266` vs `0.2279`).

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
- Dashboard quality verdict next actions should be handled before promotion review: `review` means inspect full-tile quality evidence, and `fail` means remediate the full-tile quality failure before promotion.
- Weak-fold full-tile diagnostics should run on the seed-repeat LOO artifact that actually held out the weak fold; using a different promotion candidate artifact is diagnostic-only and can leak the target segment into training.
- Treat a ready gate as the start of review, not the end: run full-tile diagnostics on the linked LOO weakest fold before claiming Scroll Prize robustness.
- Use `scripts/evaluate_leave_one_out.py --jobs N` for seed-repeat LOO throughput only when resources allow; jobs are independent process workers and parent-only JSONL output preserves reproducibility.
- Let stale generated `configs/auto_*` reservations expire so killed exploratory configs do not permanently suppress useful ideas; completed DB runs remain reserved evidence.
- Prefer evidence-backed calibration/search moves over blind local sweeps: positive-rate cap proposals should cover the `2.5-3.0` band, positive-rate loss tolerance should remain bounded, and `4096`-sample residual proposals require an explicit sample budget.
- Promote-phase sweeps include loss calibration as well as inference calibration and replication so positive-rate tolerance proposals are not starved by family-diversity rules.
- Empty-positive validation strips are a data-preparation failure, not model evidence. Regenerate the segment split or mark the fold diagnostic-only before interpreting zero-F1 LOO results.
- AutoResearch includes a balanced calibration family for the observed over-tight/over-loose gap: `max_pred_positive_rate_ratio=2.75`, `positive_rate_loss_tolerance=0.008`, and a combined candidate applying both.

## Promotion Gate

A config should be considered a robust champion only if it has:

- A committed config file.
- A leave-one-out JSONL log and summary JSON.
- No known train/validation segment overlap.
- Median F1 improvement over the current robust champion or a clear improvement in min-fold behavior.
- Comparable or improved average precision.
- Seed-repeat stability when the change affects training or model initialization; promotion summaries require at least three distinct successful seeds.
- Full-tile validation metrics when the change affects data, thresholding, or inference.
- Passing or reviewed full-tile quality verdicts; failing quality verdicts block promotion when full-tile quality metrics are present.
- Candidate-linked evidence: seed-repeat LOO and full-tile metrics must match the same config/run lineage as the robust candidate being promoted.
- A short note explaining whether it is a peak-score champion, robust champion, or diagnostic-only run.
- A calibration note reporting AP relative to prevalence, best threshold, fixed-threshold F1/status, pred/val positive-rate ratio, Brier score, expected calibration error, and whether threshold selection was unconstrained or positive-rate-constrained.

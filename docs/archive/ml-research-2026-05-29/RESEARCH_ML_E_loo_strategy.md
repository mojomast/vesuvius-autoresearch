# ML-E LOO Strategy Research

## Scope

Read-only evidence reviewed: `autoresearch.py`, `scripts/evaluate_leave_one_out.py`, `IMPROVEMENTS.md`, `METHOD.md`, `DATA.md`, relevant `configs/*`, all discovered `logs/**/*.summary.json` and `logs/**/*.jsonl` LOO/cap artifacts, promotion-evidence markdown logs, the read-only next-move evidence package, and `experiments/experiments.db`.

## LOO Evidence Count And Status

- Discovered LOO-like summary files: 52.
- Total requested LOO fold/seed evaluations across those summaries: 860.
- Total successful LOO fold/seed evaluations: 860.
- Total execution-level LOO failures: 0.
- `promotion_ready=true`: 17 summaries.
- `promotion_ready=false`: 33 summaries.
- Missing/legacy `promotion_ready`: 2 summaries (`logs/evaluate_leave_one_out_dice035_expanded.summary.json`, `logs/evaluate_leave_one_out_dry_run.summary.json`).

Important nuance: `scripts/evaluate_leave_one_out.py` sets summary-level `promotion_ready` from execution validity, zero precision/recall, positive-rate alarms, and sufficient seed repeats. It does not enforce the downstream worst-fold `val_f1 >= 0.04` gate used by promotion evidence. Therefore `logs/20260529T043342Z_478e28aa_3seed_loo.summary.json` can be `promotion_ready=true` while still being blocked by worst-fold F1 in `logs/promotion_evidence_20260529T043342Z_478e28aa.md`.

## Current Best Full-Tile Candidate And Blocker

- Read-only next-move package reports current candidate: `20260529T180703Z_0e0005c8`.
- Recommended next action: run linked seed-repeat LOO.
- Promotion gate ready: `false`.
- Candidate sampled metrics from `experiments/experiments.db`: `val_f1=0.4375`, `val_f05=0.3712`, `average_precision=0.3264`, `ap_prevalence_lift=2.335`, `pred_positive_rate / val_positive_rate = 2.836`, `fixed_threshold_status=weak`, `fixed_threshold_f1=0.0`.
- Candidate config lineage: `tiny_torch_unet`, `multi_segment_robust_expanded_tta_ensemble`, held out `20230520175435`, `tta_flips=true`, `max_train_samples=1024`.
- Immediate blocker: missing linked seed-repeat LOO for the candidate. Secondary blocker likely remains fixed-threshold weakness unless full LOO/full-tile evidence proves the selected threshold path is acceptable.

Best recent lineage with stronger promotion evidence:

- `20260529T043342Z_478e28aa` using `configs/residual25d_cap250_stride32_allval.yaml` has 3-seed LOO `promotion_ready=true`, median-over-seeds median `val_f1=0.1880`, and no LOO promotion warnings.
- Full-tile evidence on `20230520175435` passes primary threshold (`val_f1=0.2286`, AP `0.1629`, pred/val `2.43`, eligible `true`).
- Blockers: worst LOO fold `20230530172803` is `0.0336`, which is `0.0064` below the downstream `0.04` gate; full-tile evidence on `20230522181603` is diagnostic-only/ineligible for that artifact lineage.

## Folds Failing LOO Or Promotion Gates

Using the downstream `val_f1 >= 0.04` weak-fold gate plus explicit LOO warnings:

| Evidence | `promotion_ready` | Failing fold/seed | Fold F1 | Shortfall or warning |
| --- | --- | --- | ---: | --- |
| `logs/20260529T043342Z_478e28aa_3seed_loo.summary.json` | true | `20230530172803` | 0.0336 | `0.0064` below `0.04` |
| `logs/20260529T050630Z_c3d8317a_3seed_loo.summary.json` | false | `20230522181603` | 0.0216 | `0.0184` below `0.04` |
| `logs/20260529T050630Z_c3d8317a_3seed_loo.summary.json` | false | `20230530172803` | 0.0373 | `0.0027` below `0.04` |
| `logs/20260529T050630Z_c3d8317a_3seed_loo.summary.json` | false | `20230522215721:seed=11045` | 0.1768 | positive-rate alarm, pred/val `6.00` |
| `logs/20260529T035202Z_8f01550f_seedrepeat_loo.summary.json` | false | `20230530172803` | 0.0066 | `0.0334` below `0.04`; zero precision/recall on seeds `11001`, `11018` |
| `logs/20260529T051459Z_112aebf1_2seed_loo.summary.json` | false | `20230522181603` | 0.0158 | `0.0242` below `0.04`; only 2/3 seeds |
| `logs/20260529T051459Z_112aebf1_2seed_loo.summary.json` | false | `20230530172803` | 0.0326 | `0.0074` below `0.04`; only 2/3 seeds |
| `logs/20260529T051459Z_112aebf1_2seed_loo.summary.json` | false | `20230530212931:seed=11018` | 0.1574 | positive-rate alarm, pred/val `4.52`; only 2/3 seeds |

Stable historical weak folds are `20230522181603` and `20230530172803`. Positive-rate cliff folds are `20230522215721` and `20230530212931` under some cap/tolerance settings.

## Minimum Evidence Needed To Pass All Gates

- Linked 3-seed LOO for the active candidate or replacement candidate using at least seeds `11001,11018,11045` or the established `11001,11018,15050` set.
- All folds must have 3 successful seed repeats, no zero precision/recall, no positive-rate alarm outside `0.1x..3.5x`, no threshold edge cases, and no fixed-threshold failure severe enough to block promotion.
- Downstream weak-fold gate should be explicit: every per-fold mean `val_f1 >= 0.04`, with focus on `20230522181603` and `20230530172803`.
- Candidate-linked full-tile evidence for the validation segment and weak LOO folds, all with `evaluation_region.type: whole_segment`, `promotion_checks.eligible=true`, pred/val ratio <= `3.5`, positive precision/recall, AP above prevalence, and fixed-threshold status acceptable.
- Full-tile lineage must be fold-safe: no artifact trained on the segment being used as held-out full-tile evidence.

## Targeted LOO Experiment Plan

Primary plan:

- Base config: `configs/residual25d_cap250_stride32_allval.yaml` lineage, because it already produced the cleanest 3-seed LOO (`478e28aa`) and passed validation-segment full-tile eligibility.
- Seed set: `11001,11018,11045` first, because `478e28aa` passed LOO warnings with that set; follow with `11001,11018,15050` only if seed robustness is questioned.
- Fold exclusions to run: dual-heldout lineage excluding `20230520175435` plus each weak/evidence segment in separate runs, prioritizing `20230520175435+20230530172803` and `20230520175435+20230522181603`.
- Likely promotion path: train lineage-correct artifacts for the exact full-tile evidence segments, then run LOO and full-tile on the linked held-out folds. `478e28aa` is only `0.0064` F1 short on `20230530172803`, so the target is a small weak-fold lift without reopening positive-rate alarms.

Secondary plan:

- Base configs: `configs/robust_balanced_prratio_experiment.yaml` and `configs/robust_balanced_prratio_strict_tolerance_experiment.yaml`.
- Seed set: `11001,11018,11045`.
- Fold exclusions: keep dual-heldout `20230520175435_20230522181603` as configured, then add a parallel dual-heldout variant for `20230520175435_20230530172803`.
- Rationale: these templates explicitly target the alarm/weak folds, but existing nightly single-seed summaries failed mainly from insufficient seed repeats and occasional positive-rate alarms. Only promote them after a full 3-seed linked LOO clears alarms.

Avoid as first priority:

- Unconstrained cap loosening above `3.0`; c3d/112 diagnostics show positive-rate cliffs on `20230522215721` and `20230530212931`.
- Treating focused-pair high sampled F1 rows as promotion candidates; the current top DB row (`20260529T182028Z_8718e6cb`) is not LOO/full-tile-ready and has `fixed_threshold_status=weak`.

## Alternative LOO Evaluation Strategies

- Two-stage LOO: run a cheap sentinel set on weak folds (`20230522181603`, `20230530172803`, `20230522215721`, `20230530212931`) before full 8-fold x 3-seed LOO.
- Promotion-linked dual-heldout LOO: for full-tile promotion candidates, require the training NPZ to exclude both the validation full-tile segment and the weak-fold full-tile segment.
- Gate-separated summaries: keep script-level `promotion_ready` but add a separate downstream `promotion_gate_ready` that enforces worst-fold F1 and full-tile lineage eligibility.
- Recomputed-cap LOO: continue cap recomputation sweeps in the `2.5..3.0` band, because cap sweep summaries found promotion-ready behavior with worst-fold `20230522181603` F1 up to `0.0594` at cap `3.0`.
- Fold-family reporting: report low-ink folds separately from positive-rate-cliff folds so a single mean/median does not hide the failure mode.

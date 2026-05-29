# Promotion Evidence: 20260529T043342Z_478e28aa

Decision: do not promote.

## Segment 20230530172803 Diagnosis

- Classification: CASE 1, low-ink validation strip.
- Original validation strip: 128 patches, `positive_rate=0.00877`, `val_stride=64`.
- Fixed validation prep: `--val-tiled --val-stride 32 --val-samples 0`.
- Fixed validation strip: 1,633 patches, `positive_rate=0.01805`.

## LOO Results

- Config: `configs/residual25d_cap250_stride32_allval.yaml`.
- Run: `20260529T043342Z_478e28aa`.
- 2-seed LOO: `logs/20260529T043342Z_478e28aa_2seed_loo.summary.json`.
- 3-seed LOO: `logs/20260529T043342Z_478e28aa_3seed_loo.summary.json`.
- 3-seed `promotion_ready=true`.
- Median-over-seeds median `val_f1=0.1880`.
- No zero precision/recall folds.
- Worst fold: `20230530172803`, `val_f1=0.0336`.

## Full-Tile Results

| Segment | val_f1 | AP | pred/val | eligible | note |
| --- | ---: | ---: | ---: | --- | --- |
| `20230520175435` | 0.2286 | 0.1629 | 2.43 | true | passes primary full-tile threshold |
| `20230522181603` | 0.1793 | 0.1185 | 2.43 | false | diagnostic-only; artifact trained on this segment |

## Gate Outcome

- `promotion_ready=true` from 3-seed LOO.
- Full-tile `20230520175435` exceeds `0.2168` and is eligible.
- Promotion blocked by worst fold `0.0336 < 0.04`.
- Promotion blocked by ineligible full-tile evidence on `20230522181603` for this artifact lineage.

Next step: if pursuing this residual path, train a dual-heldout/lineage-correct candidate using the fixed fold data, then re-run full-tile checks on both held-out segments.

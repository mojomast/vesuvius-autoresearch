# PHASE 1 RESEARCH AGENT F: Villa Data Coverage

Source: GitHub API listing for `https://api.github.com/repos/ScrollPrize/villa/contents/ink-detection/all_labels?ref=main` and local fold-map files in this repository.

## Villa label inventory

The Villa `ink-detection/all_labels` listing contains 46 label files. Segment IDs below are derived by removing `_inklabels` and the file extension.

| Segment ID | Label file | Size bytes |
|---|---:|---:|
| 20230520175435 | `20230520175435_inklabels.png` | 313637 |
| 20230522181603 | `20230522181603_inklabels.png` | 381922 |
| 20230522215721 | `20230522215721_inklabels.png` | 258416 |
| 20230530164535 | `20230530164535_inklabels.png` | 48896 |
| 20230530172803 | `20230530172803_inklabels.png` | 56966 |
| 20230530212931 | `20230530212931_inklabels.png` | 169211 |
| 20230531121653 | `20230531121653_inklabels.png` | 28683 |
| 20230531193658 | `20230531193658_inklabels.png` | 255102 |
| 20230601193301 | `20230601193301_inklabels.png` | 62205 |
| 20230611014200 | `20230611014200_inklabels.png` | 39648 |
| 20230620230617 | `20230620230617_inklabels.png` | 361723 |
| 20230620230619 | `20230620230619_inklabels.png` | 396720 |
| 20230701020044 | `20230701020044_inklabels.png` | 706428 |
| 20230702185753 | `20230702185753_inklabels.png` | 3298099 |
| 20230813_real_1 | `20230813_real_1_inklabels.png` | 66449 |
| 20230820203112 | `20230820203112_inklabels.png` | 104741 |
| 20230826170124 | `20230826170124_inklabels.png` | 250231 |
| 20230827161847 | `20230827161847_inklabels.png` | 308391 |
| 20230901184804 | `20230901184804_inklabels.png` | 152657 |
| 20230902141231 | `20230902141231_inklabels.png` | 210422 |
| 20230903193206 | `20230903193206_inklabels.png` | 291480 |
| 20230904020426 | `20230904020426_inklabels.png` | 145823 |
| 20230904135535 | `20230904135535_inklabels.png` | 256626 |
| 20230905134255 | `20230905134255_inklabels.png` | 247846 |
| 20230909121925 | `20230909121925_inklabels.png` | 200786 |
| 20230929220924 | `20230929220924_inklabels.png` | 2677598 |
| 20230929220926 | `20230929220926_inklabels.png` | 5037938 |
| 20231001164029 | `20231001164029_inklabels.png` | 376758 |
| 20231004222109 | `20231004222109_inklabels.png` | 172262 |
| 20231005123333 | `20231005123333_inklabels.png` | 1826132 |
| 20231005123336 | `20231005123336_inklabels.png` | 1516813 |
| 20231007101615 | `20231007101615_inklabels.png` | 1596628 |
| 20231012085431 | `20231012085431_inklabels.png` | 252627 |
| 20231012173610 | `20231012173610_inklabels.png` | 1066017 |
| 20231012184420 | `20231012184420_inklabels.png` | 1052879 |
| 20231012184421 | `20231012184421_inklabels.png` | 2835103 |
| 20231012184423 | `20231012184423_inklabels.png` | 1002249 |
| 20231016151000 | `20231016151000_inklabels.png` | 1120389 |
| 20231022170900 | `20231022170900_inklabels.png` | 1673981 |
| 20231022170901 | `20231022170901_inklabels.tiff` | 4294144 |
| 20231031143850 | `20231031143850_inklabels.png` | 335984 |
| 20231106155350 | `20231106155350_inklabels.png` | 661861 |
| 20231106155351 | `20231106155351_inklabels.png` | 1054370 |
| 20231210121321 | `20231210121321_inklabels.png` | 7454992 |
| recto | `recto_inklabels.png` | 883164 |
| verso | `verso_inklabels.png` | 707384 |

Confirmed: `20230530172803_inklabels.png` is listed at 56966 bytes.

## Local fold-map inventory

Exact local fold-map used by the project: `data/real_cross_folds_expanded_combined/fold_map.json`.

Fold-map IDs:

| Fold-map ID |
|---|
| 20230520175435 |
| 20230522181603 |
| 20230522215721 |
| 20230530172803 |
| 20230530212931 |
| 20230531121653 |
| 20230601193301 |
| 20230611014200 |

Related local synthetic map: `data/real_cross_folds_expanded_combined/fold_map_synthetic.json` contains `20230520175435` and `20230827161847`. This file is not the primary `fold_map.json`, but it shows one additional Villa-labeled segment already used in a synthetic/local validation context.

## Coverage comparison

Using the exact primary `fold_map.json`:

| Category | Count | IDs |
|---|---:|---|
| Villa labels | 46 | See inventory above |
| Primary fold-map IDs | 8 | 20230520175435, 20230522181603, 20230522215721, 20230530172803, 20230530212931, 20230531121653, 20230601193301, 20230611014200 |
| Overlap | 8 | 20230520175435, 20230522181603, 20230522215721, 20230530172803, 20230530212931, 20230531121653, 20230601193301, 20230611014200 |
| In primary fold map but absent from Villa labels | 0 | None |
| Villa labels missing from primary fold map | 38 | 20230530164535, 20230531193658, 20230620230617, 20230620230619, 20230701020044, 20230702185753, 20230813_real_1, 20230820203112, 20230826170124, 20230827161847, 20230901184804, 20230902141231, 20230903193206, 20230904020426, 20230904135535, 20230905134255, 20230909121925, 20230929220924, 20230929220926, 20231001164029, 20231004222109, 20231005123333, 20231005123336, 20231007101615, 20231012085431, 20231012173610, 20231012184420, 20231012184421, 20231012184423, 20231016151000, 20231022170900, 20231022170901, 20231031143850, 20231106155350, 20231106155351, 20231210121321, recto, verso |

Including the related synthetic map as local prior coverage:

| Category | Count | IDs |
|---|---:|---|
| Union of local fold-map-like IDs | 9 | 20230520175435, 20230522181603, 20230522215721, 20230530172803, 20230530212931, 20230531121653, 20230601193301, 20230611014200, 20230827161847 |
| Union overlap with Villa labels | 9 | Same 9 IDs |
| Villa labels missing from local fold-map-like coverage | 37 | All Villa IDs above except the 9 local IDs |

## Sparse-label implication

The current primary fold map covers only 8 of 46 Villa-labeled entries, or about 17.4% of the available label files. The confirmed 56966-byte size for `20230530172803_inklabels.png` is small relative to most larger Villa label files and matches its observed role as a sparse/weak fold: it should remain in leave-one-out validation, but metrics on this fold should be treated as high-variance and prevalence-sensitive rather than representative of full data coverage.

This coverage gap means present experiments are likely dominated by a narrow early-segment distribution. Model selection can overfit to the eight covered folds, under-sample later/larger labeled surfaces, and understate robustness across recto/verso-style labels, TIFF label format, and high-area label masks.

## Recommended data expansion strategy

Prioritize explicit fold-map expansion from the Villa label inventory while preserving leakage controls: every newly added labeled segment should become its own held-out fold, and its training NPZ must exclude that held-out segment completely.

Suggested order:

1. Add nearby missing 2023-05/2023-06 segments first to minimize format/domain surprises: `20230530164535`, `20230531193658`, `20230620230617`, `20230620230619`.
2. Add medium/large dated segments next for coverage diversity and stronger training signal: `20230701020044`, `20230702185753`, `20230820203112`, `20230826170124`, `20230901184804`, `20230902141231`, `20230903193206`.
3. Add the large late-2023 group in batches with per-segment LOO monitoring: `20230929220924`, `20230929220926`, `20231005123333`, `20231005123336`, `20231007101615`, `20231012173610`, `20231012184420`, `20231012184421`, `20231012184423`, `20231016151000`, `20231022170900`, `20231022170901`, `20231106155350`, `20231106155351`, `20231210121321`.
4. Treat `recto`, `verso`, `20230813_real_1`, and `20231022170901` as special cases until loader/path handling and label semantics are verified, because they are non-standard IDs or file format/domain variants.
5. Keep `20230530172803` in validation, but evaluate it with AP/prevalence diagnostics and avoid using that fold alone as a promotion blocker unless failures replicate across larger folds.

Bottom line: expand from 8 primary folds toward the full 46-label inventory through reproducible per-segment fold maps, not by pooling labels blindly into a single train/validation split.

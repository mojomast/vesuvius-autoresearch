# RESEARCH VILLA A Label Delta

## Scope

Phase 1 Research Agent A compared current prepared NPZ labels against Villa cleaned labels for every segment ID present in `data/real_cross_folds_expanded_combined/fold_map.json`. The focal hard fold is `20230530172803`.

## Fold Map Entries Involving 20230530172803

| fold | train_npz | val_npz |
| --- | --- | --- |
| 20230530172803 | data/real_cross_folds_expanded_combined/leaveout_20230530172803/train.npz | data/real_cross_folds_v2/segment_20230530172803/val.npz |

## Current NPZ Files Involving 20230530172803

| path |
| --- |
| data/real_cross_folds_expanded_combined/leaveout_20230530172803/train.npz |
| data/real_cross_folds_v2/segment_20230530172803/train.npz |
| data/real_cross_folds_v2/segment_20230530172803/val.npz |

## Methodology

- Loaded current labels from each `data/real_cross_folds_v2/segment_<id>/{train,val}.npz` file with `.venv/bin/python`; labels are `[N, 1, 64, 64]` binary-compatible arrays.

- Downloaded Villa labels from `https://raw.githubusercontent.com/ScrollPrize/villa/main/ink-detection/all_labels/{segment_id}_inklabels.png` for each fold-map segment ID.

- Converted PNG labels to binary with threshold `>127`, then resized to each NPZ metadata `image_shape` using nearest-neighbor when the Villa raw shape differed.

- For tiled validation splits, reconstructed deterministic tile origins from `region_x`, `patch_size`, and `stride`, then compared Villa crops to NPZ label patches.

- For positive-mixture training splits, NPZ origins were not stored, so origins were replayed with the available source inklabel URL and preparation default seed `13`; these train rows are origin-proxy diagnostics, while tiled validation rows are coordinate-deterministic.

- Metrics use binary patch pixels: total pixels, positive pixels/rate for current NPZ and Villa crops, IoU, Villa-positive/current-negative count (`villa-not-ours`), and current-positive/Villa-negative count (`ours-not-villa`).

## Shapes And Alignment

| segment | split | npz | labels shape | aligned spatial shape | villa raw shape |
| --- | --- | --- | --- | --- | --- |
| 20230520175435 | train | data/real_cross_folds_v2/segment_20230520175435/train.npz | 2048x1x64x64 | 1468x3957 | 3072x7936 |
| 20230520175435 | val | data/real_cross_folds_v2/segment_20230520175435/val.npz | 128x1x64x64 | 1468x3957 | 3072x7936 |
| 20230522181603 | train | data/real_cross_folds_v2/segment_20230522181603/train.npz | 2048x1x64x64 | 2652x7280 | 5376x14592 |
| 20230522181603 | val | data/real_cross_folds_v2/segment_20230522181603/val.npz | 128x1x64x64 | 2652x7280 | 5376x14592 |
| 20230522215721 | train | data/real_cross_folds_v2/segment_20230522215721/train.npz | 768x1x64x64 | 2800x2202 | 5632x4608 |
| 20230522215721 | val | data/real_cross_folds_v2/segment_20230522215721/val.npz | 174x1x64x64 | 2800x2202 | 5632x4608 |
| 20230530172803 | train | data/real_cross_folds_v2/segment_20230530172803/train.npz | 1024x1x64x64 | 2330x1648 | 4864x3328 |
| 20230530172803 | val | data/real_cross_folds_v2/segment_20230530172803/val.npz | 1633x1x64x64 | 2330x1648 | 4864x3328 |
| 20230530212931 | train | data/real_cross_folds_v2/segment_20230530212931/train.npz | 768x1x64x64 | 1917x1171 | 3840x2560 |
| 20230530212931 | val | data/real_cross_folds_v2/segment_20230530212931/val.npz | 60x1x64x64 | 1917x1171 | 3840x2560 |
| 20230531121653 | train | data/real_cross_folds_v2/segment_20230531121653/train.npz | 768x1x64x64 | 1172x1174 | 2560x2560 |
| 20230531121653 | val | data/real_cross_folds_v2/segment_20230531121653/val.npz | 36x1x64x64 | 1172x1174 | 2560x2560 |
| 20230601193301 | train | data/real_cross_folds_v2/segment_20230601193301/train.npz | 1024x1x64x64 | 1957x3435 | 4096x6912 |
| 20230601193301 | val | data/real_cross_folds_v2/segment_20230601193301/val.npz | 128x1x64x64 | 1957x3435 | 4096x6912 |
| 20230611014200 | train | data/real_cross_folds_v2/segment_20230611014200/train.npz | 1024x1x64x64 | 1615x1358 | 3328x2816 |
| 20230611014200 | val | data/real_cross_folds_v2/segment_20230611014200/val.npz | 128x1x64x64 | 1615x1358 | 3328x2816 |

## Label Delta Metrics

| segment | split | sampling | patches | pixels | ours + | ours rate | villa + | villa rate | IoU | villa-not-ours | ours-not-villa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20230520175435 | train | positive_mixture | 2048 | 8388608 | 2727997 | 0.325203 | 2597709 | 0.309671 | 0.2706 | 1463652 | 1593940 |
| 20230520175435 | val | tiled | 128 | 524288 | 73282 | 0.139774 | 74963 | 0.142981 | 0.0789 | 64121 | 62440 |
| 20230522181603 | train | positive_mixture | 2048 | 8388608 | 2671192 | 0.318431 | 2619902 | 0.312317 | 0.3572 | 1227353 | 1278643 |
| 20230522181603 | val | tiled | 128 | 524288 | 7521 | 0.014345 | 8202 | 0.015644 | 0.0000 | 8202 | 7521 |
| 20230522215721 | train | positive_mixture | 768 | 3145728 | 243226 | 0.077319 | 279816 | 0.088951 | 0.0325 | 263352 | 226762 |
| 20230522215721 | val | tiled | 174 | 712704 | 25460 | 0.035723 | 25052 | 0.035151 | 0.9840 | 0 | 408 |
| 20230530172803 | train | positive_mixture | 1024 | 4194304 | 956003 | 0.227929 | 955356 | 0.227775 | 0.6594 | 195848 | 196495 |
| 20230530172803 | val | tiled | 1633 | 6688768 | 120700 | 0.018045 | 117646 | 0.017589 | 0.9747 | 0 | 3054 |
| 20230530212931 | train | positive_mixture | 768 | 3145728 | 291373 | 0.092625 | 279578 | 0.088875 | 0.0410 | 257078 | 268873 |
| 20230530212931 | val | tiled | 60 | 245760 | 12007 | 0.048857 | 10258 | 0.041740 | 0.8543 | 0 | 1749 |
| 20230531121653 | train | positive_mixture | 768 | 3145728 | 259412 | 0.082465 | 309845 | 0.098497 | 0.0532 | 281111 | 230678 |
| 20230531121653 | val | tiled | 36 | 147456 | 23062 | 0.156399 | 23062 | 0.156399 | 1.0000 | 0 | 0 |
| 20230601193301 | train | positive_mixture | 1024 | 4194304 | 1365359 | 0.325527 | 1365200 | 0.325489 | 0.8760 | 90166 | 90325 |
| 20230601193301 | val | tiled | 128 | 524288 | 14095 | 0.026884 | 21406 | 0.040829 | 0.0000 | 21406 | 14095 |
| 20230611014200 | train | positive_mixture | 1024 | 4194304 | 1336191 | 0.318573 | 1302105 | 0.310446 | 0.5238 | 395252 | 429338 |
| 20230611014200 | val | tiled | 128 | 524288 | 43471 | 0.082914 | 33665 | 0.064211 | 0.0432 | 30472 | 40278 |

## IoU Below 0.85

| segment | split | sampling | patches | pixels | ours + | ours rate | villa + | villa rate | IoU | villa-not-ours | ours-not-villa |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20230520175435 | train | positive_mixture | 2048 | 8388608 | 2727997 | 0.325203 | 2597709 | 0.309671 | 0.2706 | 1463652 | 1593940 |
| 20230520175435 | val | tiled | 128 | 524288 | 73282 | 0.139774 | 74963 | 0.142981 | 0.0789 | 64121 | 62440 |
| 20230522181603 | train | positive_mixture | 2048 | 8388608 | 2671192 | 0.318431 | 2619902 | 0.312317 | 0.3572 | 1227353 | 1278643 |
| 20230522181603 | val | tiled | 128 | 524288 | 7521 | 0.014345 | 8202 | 0.015644 | 0.0000 | 8202 | 7521 |
| 20230522215721 | train | positive_mixture | 768 | 3145728 | 243226 | 0.077319 | 279816 | 0.088951 | 0.0325 | 263352 | 226762 |
| 20230530172803 | train | positive_mixture | 1024 | 4194304 | 956003 | 0.227929 | 955356 | 0.227775 | 0.6594 | 195848 | 196495 |
| 20230530212931 | train | positive_mixture | 768 | 3145728 | 291373 | 0.092625 | 279578 | 0.088875 | 0.0410 | 257078 | 268873 |
| 20230531121653 | train | positive_mixture | 768 | 3145728 | 259412 | 0.082465 | 309845 | 0.098497 | 0.0532 | 281111 | 230678 |
| 20230601193301 | val | tiled | 128 | 524288 | 14095 | 0.026884 | 21406 | 0.040829 | 0.0000 | 21406 | 14095 |
| 20230611014200 | train | positive_mixture | 1024 | 4194304 | 1336191 | 0.318573 | 1302105 | 0.310446 | 0.5238 | 395252 | 429338 |
| 20230611014200 | val | tiled | 128 | 524288 | 43471 | 0.082914 | 33665 | 0.064211 | 0.0432 | 30472 | 40278 |

## Hard Fold Materiality

Not material for the hard fold by reliable validation comparison: the fold-map hard validation NPZ has IoU >= 0.85 against Villa, and the hard fold's leaveout training NPZ excludes 20230530172803.

- `train` IoU `0.6594` with `392,343` changed patch pixels; ours positive rate `0.227929`, Villa positive rate `0.227775`.
- `val` IoU `0.9747` with `3,054` changed patch pixels; ours positive rate `0.018045`, Villa positive rate `0.017589`.

The `20230530172803` validation split is the fold-map heldout hard fold and is coordinate-verified. Its Villa label differs only by current-positive/Villa-negative pixels in this patch set (`ours-not-villa=3054`, `villa-not-ours=0`), so the cleaned label delta does not explain the hard fold's weak validation result. The `20230530172803` positive-mixture train split is listed for completeness but is not in the leaveout training set for that same hard fold and lacks persisted patch coordinates.


## Train-Origin Proxy Notes

| segment | split | note |
| --- | --- | --- |
| 20230520175435 | train | train origins reconstructed because NPZ stores no patch coordinates; source-label replay/current comparison IoU=0.270552, so train rows are origin-proxy diagnostics rather than fully coordinate-verified deltas |
| 20230522181603 | train | train origins reconstructed because NPZ stores no patch coordinates; source-label replay/current comparison IoU=0.357197, so train rows are origin-proxy diagnostics rather than fully coordinate-verified deltas |
| 20230522215721 | train | train origins reconstructed because NPZ stores no patch coordinates; source-label replay/current comparison IoU=0.032500, so train rows are origin-proxy diagnostics rather than fully coordinate-verified deltas |
| 20230530172803 | train | train origins reconstructed because NPZ stores no patch coordinates; source-label replay/current comparison IoU=0.659380, so train rows are origin-proxy diagnostics rather than fully coordinate-verified deltas |
| 20230530212931 | train | train origins reconstructed because NPZ stores no patch coordinates; source-label replay/current comparison IoU=0.041025, so train rows are origin-proxy diagnostics rather than fully coordinate-verified deltas |
| 20230531121653 | train | train origins reconstructed because NPZ stores no patch coordinates; source-label replay/current comparison IoU=0.053160, so train rows are origin-proxy diagnostics rather than fully coordinate-verified deltas |
| 20230601193301 | train | train origins reconstructed because NPZ stores no patch coordinates; source-label replay/current comparison IoU=0.875996, so train rows are origin-proxy diagnostics rather than fully coordinate-verified deltas |
| 20230611014200 | train | train origins reconstructed because NPZ stores no patch coordinates; source-label replay/current comparison IoU=0.523756, so train rows are origin-proxy diagnostics rather than fully coordinate-verified deltas |

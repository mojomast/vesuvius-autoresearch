# Competitive Landscape: Vesuvius AutoResearch Progress Prize

Research date: 2026-05-28

## Executive Summary

Vesuvius AutoResearch sits in a narrow tooling niche: autonomous, local, auditable experiment steering for ink-detection research. Public 2025-2026 Vesuvius repositories include strong model, surface, review, and workflow projects, but GitHub search found only one clearly comparable AutoResearch-style Vesuvius project: `jonmarrs/vesuvius-autoresearch`, branded `bountyhunter`. That project is more ambitious and GPU/model-search oriented; this repository is more conservative, CPU-safe, validation-gated, and focused on avoiding leakage and false-positive promotion.

The current public state of the art for ink detection is still dominated by manual/model pipelines rather than autonomous search tools. The strongest public ink-detection model lineage remains the 2023 Kaggle/Grand Prize family and subsequent 3D/volumetric and title-prize systems. Public 2025-2026 tooling prizes have mostly rewarded segmentation, unwrapping, data handling, review, and surface-detection infrastructure rather than automated hyperparameter or AutoML loops for ink detection.

Metric comparability is weak. Kaggle 2023 used a hidden test leaderboard and competition score; newer ScrollPrize work often reports AUC, AP, F0.5, F1, Dice, or qualitative reading success on different fragments, scrolls, crops, or full-tile regions. Public sources fetched for this review did not expose a reliable 2024-2026 ink-detection leaderboard with comparable fragment-patch F1 scores. Any SOTA claim for this submission should therefore be framed as tooling contribution and validation discipline, not as best ink model performance.

## Search Scope

GitHub searches used or approximated:

- `"vesuvius" "ink detection" "autoresearch" OR "hyperparameter" in:readme pushed:>2025-01-01`
- `"scrollprize" "experiment" "search" language:python pushed:>2025-01-01`
- `vesuvius ink detection pushed:>2025-01-01`
- `vesuvius hyperparameter pushed:>2025-01-01`
- `vesuvius automl pushed:>2025-01-01`
- `scrollprize pushed:>2025-01-01`

Primary public sources:

- ScrollPrize winners page: https://scrollprize.org/winners
- ScrollPrize/villa repository: https://github.com/ScrollPrize/villa
- Kaggle 2023 ink-detection winners linked from ScrollPrize: https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection
- `ainatersol/Vesuvius-InkDetection`: https://github.com/ainatersol/Vesuvius-InkDetection
- `ryanchesler/3d-ink-detection`: https://github.com/ryanchesler/3d-ink-detection
- `mvrcii/vesuvius_first_title_prize`: https://github.com/mvrcii/vesuvius_first_title_prize
- `jgcarrasco/dino-ink-detection`: https://github.com/jgcarrasco/dino-ink-detection
- `lschlessinger1/vesuvius-patch-agg-analysis`: https://github.com/lschlessinger1/vesuvius-patch-agg-analysis
- `ciscoriordan/mednext-vs-umamba-scroll`: https://github.com/ciscoriordan/mednext-vs-umamba-scroll
- `maxliebscher/scroll-review-tooling`: https://github.com/maxliebscher/scroll-review-tooling
- `jonmarrs/vesuvius-autoresearch`: https://github.com/jonmarrs/vesuvius-autoresearch

## Competing Or Adjacent Tools

| Project | Date signal | Category | What it does | Relevance to AutoResearch |
| --- | --- | --- | --- | --- |
| `jonmarrs/vesuvius-autoresearch` / `bountyhunter` | Created 2026, pushed 2026 | Direct competitor | Autonomous research loop for Vesuvius, sampling architectures, losses, augmentations, and hyperparameters; integrates TimeSformer, ResNet3D-101, Inception-I3D, topology metrics, calibration baselines, and GPU training. | Most direct competitor. More aggressive and GPU/model-search oriented. Differentiation for this repo must be validation safety, CPU-safe reproducibility, promotion gates, full-tile checks, and one-change interpretability. |
| `mojomast/vesuvius-autoresearch` | Created 2026, pushed 2026 | This submission | Minimal continuous experiment pipeline with real-data-only execution, bounded one-change proposals, SQLite experiment records, LOO/full-tile promotion evidence, positive-rate alarms, dashboard, and no synthetic fallback. | Differentiated by conservative reviewability and false-positive controls rather than model scale. |
| `mvrcii/vesuvius_first_title_prize` | 2025 | Model/prize winner | First Title Prize winner using MiniUNETR, 3D chunks, manual annotations, ignore masks, and rapid retraining for Scroll 5 title detection. | Strong model evidence, not an AutoML/search framework. Shows high-performing teams optimize data quality and annotation loops more than blind hyperparameter search. |
| `ryanchesler/3d-ink-detection` | 2024, still relevant | Model/inference tooling | 3D U-Net ink detector trained on mapped 2D labels, sparse-label masking, validation distance exclusion, and sliding-window scroll inference. | Strong adjacent ink model pipeline. Competes as model infrastructure, not autonomous experiment steering. |
| `ainatersol/Vesuvius-InkDetection` | 2023, still public baseline | Kaggle winner model | 2023 Kaggle winning ensemble: 3D CNN/3D U-Net/UNETR features flattened to 2D SegFormer; 9-model ensemble; trained on 3 A6000 GPUs. | SOTA model lineage for fragment competition. AutoResearch is not currently competitive on model performance. |
| `jgcarrasco/dino-ink-detection` | 2024 | Unsupervised ink detection | DINOv2/PCA experiments to reveal visually obvious crackle without labels. | Differentiated as label-free exploratory detector, not experiment automation. |
| `lschlessinger1/vesuvius-patch-agg-analysis` and `sherafatia/vesuvius_patch_agg_demo` | 2025 | Analysis/tooling | Quantitative/qualitative patch aggregation comparison using AP and F0.5 across averaging, Gaussian, cropping, and Hanning methods. | Adjacent to patch-level evaluation and inference quality. Not autonomous search. |
| `ciscoriordan/mednext-vs-umamba-scroll` | 2026 | Surface/model benchmark | Benchmarks MedNeXt-L and U-Mamba for surface detection, reports official Kaggle surface metric, downstream ink AUC comparisons, and corrected audit trail. | Strong benchmark/reporting practice. It targets surface detection, not ink AutoResearch, but raises the standard for metric transparency. |
| `maxliebscher/scroll-review-tooling` | 2026 | Review infrastructure | Blind-first review-gate toolkit for validating manifests, responses, second checks, no-claim dossiers, and local dashboards. | Complementary: review safety and handoff gates. Not a model/search tool. |
| ScrollPrize/villa | 2024-2026 | Official infrastructure | Monorepo with `vesuvius`, `vesuvius-c`, `foundation`, `crackle-viewer`, `ink-detection`, VC3D, and other official/community tools. | Official platform. AutoResearch should integrate with or cite this ecosystem rather than position as replacing it. |

## Prize Winners: Infrastructure Versus Models

The ScrollPrize winners page shows a broad split.

Infrastructure/tooling winners include:

- 2026 Open Source: Scroll Slab Viewer for surface-detection challenge review.
- 2025 Open Source: VC3D stability improvements, volume masking pipeline, neuroglancer-mini, ball-spring/data handling, stress metrics.
- 2024-2025 Open Source: vesuvius-gui, Khartes, Volume Cartographer/VC3D improvements, segmentation toolkits, zarr tools, web volume viewers, patch aggregation analysis, data streaming, mesh/flattening/segmentation tooling.
- 2023 Open Source and Segmentation Tooling: Volume Cartographer, Khartes, Segment Viewer, Crackle Viewer, VA Sheet Tracer, VolumeAnnotate, scroll viewers, ilastik integration, and related data/viewer tools.

Model/prize-result winners include:

- 2023 Kaggle Ink Detection: fragment-based ink model winners led by `ryches`.
- 2023 First Letters & Ink and Grand Prize: reading/ink breakthroughs by Luke Farritor, Youssef Nader, Casey Handmer, and Grand Prize teams.
- 2024 Open Source model-ish awards: 3D ink detection, DINO/self-supervised ink detection, volumetric segmentation labels/models, segmentation models, self-supervised pretraining.
- 2025 First Title Prize: MiniUNETR title detector by Marcel Roth and Micha Nowak.
- 2026 Kaggle Surface Detection: surface model winners, not ink detection.

The relevant pattern for this submission: ScrollPrize does reward infrastructure and tooling, but the winning tooling generally improves data access, visualization, segmentation, review, or reproducibility. A Vesuvius AutoResearch submission should therefore emphasize concrete reviewer utility, reproducibility, and safety gates, not only the fact that a loop is autonomous.

## Current Best-Known Metrics

### 2023 Kaggle Ink Detection

The ScrollPrize winners page identifies the top 10 Kaggle Ink Detection winners and links the leaderboard/writeups. The winning public code reports a 9-model ensemble combining 3D CNNs, 3D U-Nets, UNETR, and SegFormer, trained on multiple A6000 GPUs. Kaggle pages were fetchable only as JavaScript shells in this environment, so exact live leaderboard rows/scores were not reliably extractable unauthenticated.

Important metric caveat: the Kaggle ink-detection competition score is commonly discussed as the competition's hidden-test segmentation score, not necessarily plain F1 as used by this repository. Public writeups and follow-up repositories also emphasize F0.5, AP, thresholding, and qualitative transfer behavior. Treat Kaggle 2023 as the best-known fragment-model baseline, but do not compare this repository's local `val_f1` directly to Kaggle leaderboard scores.

### 2024-2026 Public Ink Metrics

No public 2024-2026 ink-detection leaderboard with comparable fragment-patch F1 scores was found in the searched sources. Public projects report heterogeneous metrics:

- `mvrcii/vesuvius_first_title_prize`: prize-winning title detection result; README emphasizes MiniUNETR, ignore masks, 3D chunks, 1-hour training from scratch on RTX 4090, and successful title identification, but does not expose a comparable fragment-patch F1 leaderboard.
- `ryanchesler/3d-ink-detection`: provides 3D ink detector training/inference and qualitative scroll outputs; README discusses sparse-label masking and validation exclusion but does not publish a single comparable F1 headline.
- `jgcarrasco/dino-ink-detection`: unsupervised DINO/PCA visual detector; qualitative examples, no comparable F1 headline.
- `lschlessinger1/vesuvius-patch-agg-analysis`: evaluates patch aggregation using AP and F0.5, but this is aggregation analysis rather than a new public SOTA ink detector score.
- `ciscoriordan/mednext-vs-umamba-scroll`: reports downstream ink AUC for surface choices: mean v4 ink AUC at bruniss hand segmentation `0.7902`, at MedNeXt surface `0.7602`, and at d058 surface `0.5083` across three crops. This is useful downstream evidence but is AUC on surface choice, not fragment-patch F1.

### This Repository's Current Metrics

Local README state reports substantially lower experimental scores than public SOTA model systems. Current cited examples include:

- Three-seed median-over-seeds median `val_f1=0.1109` for strip-fix candidate `20260529T021416Z_570f6775`, blocked by positive-rate alarms.
- Worst-fold `val_f1=0.0179` for tightened candidate `20260529T023543Z_021a01b0`, not promotion-ready.
- Eligible full-tile F1 `0.2363` for prratio3.5 candidate `20260529T024258Z_8b44032d`, but it failed LOO.

These are not SOTA ink-detection scores. They are evidence that the tool records and blocks weak or unsafe candidates rather than over-claiming them.

## Differentiation Assessment

Clear differentiation:

- Conservative autonomy: one-change proposal generation, parameter bounds, pending-config dedupe, and plan mode make the loop inspectable.
- Safety gates: promotion requires LOO/full-tile evidence, positive precision/recall, AP/prevalence lift, positive-rate ratio control, fixed-threshold diagnostics, and no fold leakage.
- Real-data-only execution: missing real NPZs fail loudly instead of silently falling back to fake data.
- Reviewability: SQLite experiment records, metrics JSON, threshold CSVs, run summaries, dashboards, and reproducibility docs are first-class outputs.
- CPU-safe baseline: useful for unattended cron and reviewer reproduction even without large GPU resources.

Weak or non-differentiated areas:

- Not a best model: current local F1 evidence is far below the public 2023 Kaggle/Grand Prize and 2025 title-prize model lineages.
- Not the only AutoResearch project: `jonmarrs/vesuvius-autoresearch` is a direct public competitor with a broader GPU architecture/loss/augmentation search loop.
- Not a replacement for annotation/data-quality loops: 2025 title-prize and 2023/2024 ink winners show that manual labels, ignore masks, surface quality, and domain-specific preprocessing remain central.
- Limited public metric comparability: lack of a current public ink F1 leaderboard makes any broad SOTA comparison source-limited.

Bottom line: this repository is differentiated as a claim-safe experiment-governance tool for Vesuvius ink detection, not as a SOTA model submission. The strongest prize framing is: it reduces wasted and unsafe research cycles by making autonomous exploration reproducible, bounded, and promotion-gated on real Vesuvius data.

## Source-Limited Conclusions

- Public GitHub search did not find many Vesuvius-specific AutoML/autonomous search tools beyond the two `vesuvius-autoresearch` repositories.
- Public winners data strongly supports infrastructure/tooling prize precedent, but most rewarded tooling is not hyperparameter search.
- Exact Kaggle 2023 leaderboard scores could not be extracted from unauthenticated fetched pages in this environment; use Kaggle directly for final numeric claims if needed.
- No comparable 2024-2026 public fragment-patch F1 leaderboard was found. Public projects should be compared by task, metric, and validation scope, not collapsed into one SOTA number.

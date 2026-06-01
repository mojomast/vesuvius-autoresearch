# Vesuvius AutoResearch - Open Source Tooling Prize Submission

## Submission Positioning

Vesuvius AutoResearch is submitted as open-source tooling for the Scroll Prize ecosystem: a reusable experiment-governance, validation, and review system for Vesuvius ink-detection research.

Official references:

- Open prizes: <https://scrollprize.org/prizes>
- Data and citation policy: <https://scrollprize.org/data>
- Getting started and community tooling: <https://scrollprize.org/get_started>
- Community projects: <https://scrollprize.org/community_projects>

## What This Is

Vesuvius AutoResearch is an evidence-gated ML research automation system for Vesuvius ink-detection experiments. It proposes bounded experiment changes, runs them against Vesuvius-style prepared segment data, records metrics and provenance, and blocks promotion unless candidates pass validation gates visible in a reviewer-safe dashboard.

The submission claim is intentionally narrow: this is reusable experiment-governance and validation tooling, not a state-of-the-art ink model and not a text discovery claim.

## Prize Alignment

| Requirement theme | Repository support |
| --- | --- |
| Specific Vesuvius challenge | Reduces overfitting, leakage, and false-positive promotion in ink-detection research. |
| Clear implementation path | `README.md`, `REPRODUCE.md`, `METHOD.md`, and `docs/dashboard.md` document install, smoke test, real-data preparation, experiments, LOO, full-tile checks, and dashboard review. |
| Demonstration of use | Synthetic quickstart runs without protected data; real-data commands support public Scroll segment Zarr and local fragment-style folders. |
| Significant advantage | Turns ad hoc experiment steering into auditable proposal, metric, provenance, fold-safety, and dashboard gates. |
| Comprehensive documentation | Active docs cover method, data policy, reproduction, dashboard safety, calibration, artifacts, weights, credits, and citation. |
| Usage examples | Command examples cover smoke tests, real-data preparation, one-run training, LOO dry-runs, full-tile inference, dashboard snapshots, and hard-negative planning. |
| Community format awareness | Current support ingests public segment Zarr through preparation scripts and local TIFF/PNG folders; direct OME-Zarr/broader Zarr adapters are documented future work, not claimed as complete. |
| Modular integration | `harness/` exposes a `ResearchHarness` abstraction; dashboard snapshots use a stable JSON contract; command inventory is allowlisted. |
| Open source and license | Code is MIT licensed. Raw Vesuvius data, generated NPZs, predictions, and weights are excluded from git and remain subject to Scroll Prize data terms. |

## Tooling Value

- Problem: Vesuvius ink experiments are easy to overfit, leak, or overclaim because sampled validation F1 can diverge from full-tile behavior.
- Contribution: bounded autonomous proposal generation, strict metric contracts, candidate-linked LOO/full-tile evidence gates, fold-safe hard-negative planning, provenance checks, and a dashboard that makes blocker evidence visible.
- Inputs: prepared NPZs, public Scroll segment Zarr data converted by scripts, and local fragment-style `surface_volume/*.tif` folders with labels/masks where available.
- Outputs: SQLite experiment rows, config JSON, metrics JSON, threshold CSVs, dashboard snapshot JSON, audit reports, and external artifact manifests.
- Safety: dashboard execution is read-only by default, command execution is allowlisted by command ID, artifact-writing commands remain copy-only, and promotion evidence must be candidate-linked.

## Standard Inputs, Outputs, And Integration

| Area | Current support | Notes |
| --- | --- | --- |
| Public Scroll segment Zarr | Supported through preparation scripts | Converts labeled segment data into local NPZ training/evaluation files. |
| Local fragment folders | Supported | Expects `surface_volume/*.tif`, `inklabels.png`, and optional `mask.png`. |
| Prepared training format | Supported NPZ | `images [N,C,H,W]`, `labels [N,1,H,W]`, plus metadata sidecars. |
| Experiment outputs | Supported JSON/CSV/SQLite | Designed for audit, dashboard display, and reproducibility. |
| Dashboard output | Supported JSON snapshot | `vesuvius-dashboard/v1` contract for standalone or host dashboards. |
| Direct OME-Zarr / broader Zarr adapters | Future work | The repo does not claim complete direct community-format coverage yet. |

## Differentiation

This repository does not replace ScrollPrize/villa, VC3D viewers, segmentation tools, or strong ink model training code. Its value is the governance layer around experimentation: bounded changes, duplicate avoidance, metric contracts, strict promotion gates, no-overlap policy, candidate-linked evidence, full-tile quality checks, and a reviewer-safe dashboard.

The closest manual workflow is running experiments, LOO scripts, full-tile inference, and artifact inspection by hand. AutoResearch makes that loop repeatable and reviewable, while failing closed when data paths, evidence links, quality verdicts, or promotion conditions are missing.

## Dashboard Value

The dashboard is the main review surface. It shows which run is being considered, why it is or is not promotable, which fold or segment is blocking progress, whether a full-tile artifact is eligible, and what command should be run next. It surfaces evidence that otherwise lives across SQLite rows, JSON metrics, threshold CSVs, LOO summaries, and local artifacts.

The default dashboard is read-only. Interactive controls require explicit runtime flags and token/private-route configuration; the backend executes only allowlisted non-artifact-writing commands and rejects arbitrary shell input. Optional Agent Chat and visual analysis are documented in `docs/dashboard.md`, but they are not required for the prize use case.

## Reviewer Checklist

- Read `README.md`, `METHOD.md`, `DATA.md`, and `REPRODUCE.md`.
- Confirm license and attribution in `LICENSE`, `CREDITS.md`, `CITATION.cff`, and `VILLA_INTEGRATION.md`.
- Run the no-download synthetic smoke test in `REPRODUCE.md`.
- Run `.venv/bin/python -m pytest tests/`.
- Export or view the dashboard snapshot with `.venv/bin/python scripts/export_dashboard_snapshot.py --pretty` or `.venv/bin/python run_dashboard.py`.
- If real data is available locally, prepare public segment data and run one baseline experiment as documented in `REPRODUCE.md`.
- Treat generated `data/**`, `experiments/**`, `logs/**`, weights, NPZs, NPYs, and full-tile outputs as local artifacts that should not be committed.

## Quick Start

```bash
git clone https://github.com/mojomast/vesuvius-autoresearch.git
cd vesuvius-autoresearch
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
.venv/bin/python scripts/generate_synthetic_data.py
.venv/bin/python scripts/setup_data.py --data-dir ./data
.venv/bin/python autoresearch.py --plan --json
```

The synthetic data follows the runner schema: `images` are `[N,C,H,W]` and `labels` are `[N,1,H,W]`. Metadata marks synthetic provenance so demo artifacts cannot be confused with real scroll evidence.

## Verification Status

Last local verification during this documentation update:

```bash
.venv/bin/python -m pytest tests/
```

The exact passing count changes as tests are added; use the command above and the GitHub Actions badge in `README.md` as the maintained verification path.

## Research Results And Limitations

Local development history includes 1,600+ recorded experiment rows in local generated SQLite databases; those generated databases are not shipped in git. The public repository ships the code, schemas, configs, dashboards, and documentation needed to reproduce the workflow.

The ink detector itself is not a champion model. Recent autonomous research improved sampled validation and produced candidate-linked LOO evidence, but repository-level submission review remains blocked by the hard `20230530172803` fold. One script-level LOO summary reported `promotion_ready=true`, but the same evidence had `worst_fold_val_f1=0.03575020275750203`, below the stricter hard-fold floor used for submission review. This distinction is intentional: script-level execution readiness does not equal a prize-facing ink claim.

The current bottleneck is hard-fold separability, architecture, and calibration robustness, not the experiment-governance infrastructure.

## Work Completed

- Added a reproducible synthetic data generator and quickstart path so reviewers can run the loop without downloading real Vesuvius data first.
- Prepared and documented the ScrollPrize Community Projects PR: <https://github.com/ScrollPrize/villa/pull/991>.
- Added CI/unit tests, metric contracts, parameter bounds, provenance checks, promotion evidence packaging, full-tile inference checks, and dashboard-driven review.
- Added CPU-safe residual 2.5D configs and evidence logs, while preserving the rule that sampled-only wins remain diagnostic.
- Archived dated planning and audit notes under `docs/archive/` so active docs point to current dashboard, calibration, cost-tier, and reproducibility guidance.

## How To Extend This

The harness layer is documented in [`harness/README.md`](harness/README.md). To adapt the loop, implement `ResearchHarness` for a different model, data format, surface-detection task, annotation-quality loop, or scroll-processing workflow while keeping the same bounded proposal, experiment recording, and promotion-gating lifecycle.

Useful next integrations:

- Direct OME-Zarr/Zarr adapters rather than conversion-first preparation.
- Stronger 2.5D/3D model families behind the same promotion gates.
- Additional full-scroll or cross-scroll evaluation harnesses.
- Community-contributed proposal families for augmentations, calibration, and data-quality loops.

## Links

- Repo: <https://github.com/mojomast/vesuvius-autoresearch>
- Harness docs: [`harness/README.md`](harness/README.md)
- Dashboard safety and review workflow: [`docs/dashboard.md`](docs/dashboard.md)
- Calibration and thresholding: [`docs/calibration_and_thresholding.md`](docs/calibration_and_thresholding.md)
- Historical research archive: [`docs/archive/README.md`](docs/archive/README.md)

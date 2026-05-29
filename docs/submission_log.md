# Prize Submission Log

Date: 2026-05-28

Status: pending manual Google Form submission by repository owner.

Reason: the official May 2026 Progress Prize submission method is an interactive Google Form that requires a submitter email address, full legal/name identity, and terms acceptance. This environment can fetch the form fields but cannot complete an authenticated/personal Google Form submission on the owner's behalf.

## Submission Method

- Prize: Vesuvius Challenge May 2026 Progress Prize
- Deadline: 11:59pm Pacific, May 31, 2026
- Form: <https://forms.gle/LrpQmSAqdwGpTczLA>
- Community Projects PR required by form: submitted at <https://github.com/ScrollPrize/villa/pull/991>

## Ready-To-Paste Form Content

### URL to open source / publicly available contribution

<https://github.com/mojomast/vesuvius-autoresearch>

Recommended additional URLs:

- <https://github.com/mojomast/vesuvius-autoresearch/blob/main/SUBMISSION.md>
- <https://github.com/mojomast/vesuvius-autoresearch/blob/main/README.md>
- <https://github.com/mojomast/vesuvius-autoresearch/blob/main/harness/README.md>
- <https://github.com/ScrollPrize/villa/pull/991>

### Short Description

Vesuvius AutoResearch is an open-source autonomous ML research loop for Vesuvius ink-detection experiments. It proposes bounded experiment changes, runs them against Vesuvius-style segment data, records metrics and provenance, and blocks promotion unless candidates pass evidence-gated validation such as leave-one-out seed-repeat checks, full-tile inference, positive-rate alarms, fixed-threshold diagnostics, and eligibility checks.

The contribution substantially increases the probability of reading complete scrolls by making ink-detection research more reproducible and less prone to overfit or false-positive promotion. It includes `MetricContract` enforcement across 1,419+ recorded experiments, `PARAM_BOUNDS` clamping, a `VesuviusHarness` abstraction for reusable scroll research loops, GitHub Actions CI with 226 passing tests, a synthetic data generator for a no-download demo, and documentation that honestly records both successful infrastructure and current model limitations.

This is not submitted as a champion ink model. The best eligible tiled result so far is `val_f1=0.2363`, and promotion remains blocked by cross-seed generalization failures. The prize claim is the reusable infrastructure: a complete autonomous experiment-governance system that the Vesuvius community can run, inspect, and extend for safer model, data, and calibration research.

### Team Description

Suggested owner-editable text:

Individual submission by mojomast. The repository and documentation were prepared as an open-source Vesuvius Challenge tooling contribution.

## Verification Completed

- `python3 scripts/generate_synthetic_data.py --n-train 256 --n-val 64`: passed.
- `python3 scripts/setup_data.py --data-dir ./data`: passed.
- `python3 autoresearch.py --plan --json`: passed and returned valid plan JSON.
- `python3 autoresearch.py --plan --json | python3 -c "..."`: passed with `DEMO OK`.
- `python3 -m pytest tests/ --tb=short`: 226 passed.

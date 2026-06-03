# Autonomous Self-Improvement

AutoResearch treats each generated config as an attributed proposal, not an anonymous sweep.

## Ledger

The SQLite ledger lives in `experiments/experiments.db` and contains:

- `hypotheses`: durable research hypotheses and expected effects.
- `proposals`: generated configs with `proposal_id`, `hypothesis_id`, `arm_id`, signatures, changed path, cost tier, and status.
- `proposal_results`: completed, deduped, and failed proposal outcomes.
- `diagnoses`: structured blocker diagnoses for candidates.
- `evidence_packages`: written next-move package paths.

## Proposal Metadata

Generated configs include `autoresearch.proposal_id`, `hypothesis_id`, `arm_id`, `mutation_family`, `changed_path`, `search_signature`, `config_signature`, `cost_tier`, and `promotion_required`. Proposal IDs are metadata and are excluded from runtime dedupe signatures.

## Failure Memory

Failed subprocess attempts are recorded in `proposal_results` with return code, error class, error tail, failure log path, and duration. Failed configs are kept for triage instead of being deleted.

## Stale Causes

Guard and overnight logs are parsed into normalized no-progress causes such as `load_guard`, `mem_guard`, `promotion_evidence_required`, `signature_exhausted`, and `timeout`. Dashboard and audit reports show the latest cause and counts.

## Evidence Packages

`scripts/package_next_move_evidence.py --markdown` remains stdout-only. Use `--output-dir logs/evidence_packages` when a durable JSON/Markdown handoff is required; written packages are recorded in the ledger.

## Safety

Promotion remains dashboard-gated. Ledger attribution, method-family arms, and evidence packages improve research memory and triage, but they do not relax linked seed-repeat LOO, full-tile, positive-rate, fixed-threshold, hard-fold, or quality gates.

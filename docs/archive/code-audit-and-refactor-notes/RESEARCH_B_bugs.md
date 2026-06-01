# Research B: Bugs & Safety

Findings implemented first because they affect autonomous execution safety.

- `_RunHistory` keyed derived caches by `id(runs)`. This can return stale strategy/ranking/manual-promotion results after list mutation or object id reuse. Fix: content-based keys from sorted `run_id` tuples.
- `_LINKED_LOO_SUMMARY_CACHE*` globals were unsynchronized and returned the mutable cached list. Fix: guard with `threading.Lock` and return shallow copies.
- `subprocess.run(..., check=True)` existed for baseline and generated proposal experiments. Failures release the OS lock via context manager, but generated configs remain and can reserve a failed signature. Fix: wrap calls, clean generated config on launch failure, and rely on context manager for lock release.
- `_promotion_ready_payload()` swallowed `Exception` and let exploration continue when dashboard readiness failed. Fix: log exceptions and support `AUTORESEARCH_FAIL_ON_SNAPSHOT_ERROR=1` to fail closed with an error payload.
- There was no boot-time assertion that numeric `SIGNATURE_DEFAULTS` respect `PARAM_BOUNDS`. Fix: validate on import.
- `--plan` pruned generated configs and pruning happened before lock acquisition. Fix: plan is read-only and pruning occurs only after lock acquisition.
- Generated config writes were direct writes to final paths. Fix: write to a temp file and atomically replace.
- Lock file was opened with `w`, truncating diagnostics before lock acquisition. Fix: use `a+`.

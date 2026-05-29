# Ops Audit

There are 583 `configs/auto_*.yaml` files. Timestamp-derived growth is about 8.69 configs/hour overall, with a burst of about 21.45 configs/hour on 2026-05-26 and about 2.43 configs/hour in the latest observed window.

Unix locking uses `fcntl.flock(... LOCK_EX | LOCK_NB)` and exits safely on lock acquisition errors. Windows fallback in `autoresearch.py` catches `ImportError`, `AttributeError`, and `OSError` from `msvcrt.locking(...)` and then `pass`es, allowing execution without an effective lock.

`_linked_loo_summary_ready` rescans, stats, sorts, reads, and parses `logs/*summary.json` on each call. With many candidate runs and summary files this becomes repeated `O(C * S log S)` work. Cache the result or build an index once per cycle.

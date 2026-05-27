#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dashboard.mining import build_hard_negative_plan, resolve_safe_config_path
from research_dashboard.snapshot import build_snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run fold-safe hard-negative mining and retrain plan")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root; default is this checkout")
    parser.add_argument("--snapshot-json", default=None, help="Optional dashboard snapshot JSON to read instead of building one")
    parser.add_argument("--max-commands", type=int, default=6, help="Maximum mining commands to emit")
    parser.add_argument("--ratio-threshold", type=float, default=2.0, help="Pred/label positive-rate ratio that should trigger mining")
    parser.add_argument("--target-config", default="configs/robust_hard_negative_prratio2p0_followup.yaml", help="Config path to preview dataset.extra_train_npzs patch for")
    parser.add_argument("--heldout-segment", default=None, help="Override the held-out segment for fold-safe eligibility/config preview")
    parser.add_argument("--base-config", default=None, help="Optional YAML config to render as a patched preview in JSON output; never writes files")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args(argv)

    root = Path(args.repo_root).expanduser().resolve()
    if args.snapshot_json:
        snapshot = json.loads(Path(args.snapshot_json).expanduser().read_text())
    else:
        snapshot = build_snapshot(root)
    base_config = resolve_safe_config_path(root, args.base_config) if args.base_config else None
    plan = build_hard_negative_plan(root, snapshot, max_commands=args.max_commands, ratio_threshold=args.ratio_threshold, target_config=args.target_config, heldout_segment=args.heldout_segment, base_config=base_config)
    print(json.dumps(plan, indent=2 if args.pretty else None, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

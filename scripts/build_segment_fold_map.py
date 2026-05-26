#!/usr/bin/env python3
"""Build leave-one-segment-out fold maps from local prepared segment NPZs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from combine_segment_npzs import combine


def _segment_id(path: Path) -> str:
    return path.name.removeprefix("segment_")


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build local leave-one-segment-out fold map")
    parser.add_argument("--segments-root", default="data/real_cross_folds_v2")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--fold-map-out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.segments_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    segments = []
    for seg_dir in sorted(root.glob("segment_*")):
        train = seg_dir / "train.npz"
        val = seg_dir / "val.npz"
        if train.exists() and val.exists():
            segments.append((_segment_id(seg_dir), train, val))
    if len(segments) < 2:
        raise ValueError(f"need at least two prepared segments under {root}")

    fold_map = {}
    for heldout, _train, val in segments:
        train_inputs = [train for segment_id, train, _val in segments if segment_id != heldout]
        out_train = output_root / f"leaveout_{heldout}" / "train.npz"
        combine(train_inputs, out_train, split="train", dry_run=args.dry_run)
        fold_map[heldout] = {"train_npz": _display(out_train), "val_npz": _display(val)}

    if not args.dry_run:
        fold_map_out = Path(args.fold_map_out).expanduser()
        fold_map_out.parent.mkdir(parents=True, exist_ok=True)
        fold_map_out.write_text(json.dumps(fold_map, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"dry_run": args.dry_run, "folds": len(fold_map), "fold_map": fold_map}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

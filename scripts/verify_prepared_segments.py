#!/usr/bin/env python3
"""Verify local prepared segment NPZ coverage without downloading data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _segment_id(path: Path) -> str:
    return path.name.removeprefix("segment_")


def _check_npz(path: Path, patch_size: int | None) -> dict[str, Any]:
    with np.load(path) as data:
        missing = {"images", "labels"} - set(data.files)
        if missing:
            raise ValueError(f"missing keys: {sorted(missing)}")
        images = data["images"]
        labels = data["labels"]
        if images.ndim != 4 or labels.ndim != 4:
            raise ValueError(f"expected 4D images/labels, got {images.shape} and {labels.shape}")
        if images.shape[0] != labels.shape[0] or images.shape[-2:] != labels.shape[-2:]:
            raise ValueError(f"shape mismatch: {images.shape} vs {labels.shape}")
        if patch_size is not None and tuple(images.shape[-2:]) != (patch_size, patch_size):
            raise ValueError(f"expected patch_size={patch_size}, got {images.shape[-2:]}")
        return {
            "path": str(path),
            "samples": int(images.shape[0]),
            "shape": [int(x) for x in images.shape],
            "positive_rate": float((labels > 0.5).mean()),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify local prepared segment train/val NPZs")
    parser.add_argument("--segments-root", default="data/real_cross_folds_v2")
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    root = Path(args.segments_root).expanduser()
    rows = []
    ok = True
    for seg_dir in sorted(root.glob("segment_*")):
        if not seg_dir.is_dir():
            continue
        row: dict[str, Any] = {"segment_id": _segment_id(seg_dir), "segment_dir": str(seg_dir)}
        for split in ("train", "val"):
            path = seg_dir / f"{split}.npz"
            if not path.exists():
                row[split] = {"ok": False, "error": "missing"}
                ok = False
                continue
            try:
                row[split] = {"ok": True, **_check_npz(path, args.patch_size)}
            except Exception as exc:
                row[split] = {"ok": False, "error": repr(exc), "path": str(path)}
                ok = False
        rows.append(row)
    summary = {"segments_root": str(root), "segments_checked": len(rows), "ok": ok, "segments": rows}
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        Path(args.output_json).expanduser().write_text(text)
    print(text, end="")
    return 0 if ok and rows else 1


if __name__ == "__main__":
    raise SystemExit(main())

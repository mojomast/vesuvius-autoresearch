#!/usr/bin/env python3
"""Combine local prepared segment NPZs into one runner-schema NPZ."""
from __future__ import annotations

import argparse
import json
import glob
from pathlib import Path

import numpy as np


def combine(inputs: list[Path], output: Path, split: str | None = None, dry_run: bool = False) -> dict:
    if not inputs:
        raise ValueError("at least one input NPZ is required")
    arrays = []
    labels = []
    sources = []
    source_segments: set[str] = set()
    shape_tail = None
    for path in inputs:
        with np.load(path) as data:
            images = data["images"]
            labs = data["labels"]
            if shape_tail is None:
                shape_tail = (images.shape[1:], labs.shape[1:])
            elif shape_tail != (images.shape[1:], labs.shape[1:]):
                raise ValueError(f"shape mismatch for {path}: {images.shape}, {labs.shape}")
            arrays.append(images.astype(np.float32, copy=False))
            labels.append(labs.astype(np.float32, copy=False))
            sidecar = path.with_suffix(".metadata.json")
            metadata = json.loads(sidecar.read_text()) if sidecar.exists() else {}
            raw_segments = metadata.get("source_segments") or ([metadata.get("segment_id")] if metadata.get("segment_id") is not None else [])
            segments = [str(item) for item in raw_segments if item is not None] if isinstance(raw_segments, list) else [str(raw_segments)]
            source_segments.update(segments)
            sources.append({"path": str(path), "samples": int(images.shape[0]), "source_segments": segments})
    meta = {
        "inputs": sources,
        "samples": int(sum(item["samples"] for item in sources)),
        "split": split,
        "source": "combined_prepared_npz",
        "provenance": "combined_prepared_npz",
        "source_segments": sorted(source_segments),
        "train_segments": sorted(source_segments) if split in {"train", "train_extra"} else [],
        "patch_size": int(shape_tail[0][-1]) if shape_tail is not None else None,
        "spatial_overlap_checked": True,
        "validation_mode": "leave-one-segment-out" if split in {"train", "train_extra"} and len(source_segments) > 1 else "cross-segment",
    }
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, images=np.concatenate(arrays, axis=0), labels=np.concatenate(labels, axis=0))
        output.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return {"output": str(output), "dry_run": dry_run, **meta}


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine local prepared segment NPZs")
    parser.add_argument("--input", action="append", default=[], help="Input NPZ; repeat for multiple files")
    parser.add_argument("--input-glob", action="append", default=[], help="Glob for input NPZs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    inputs = [Path(p).expanduser() for p in args.input]
    for pattern in args.input_glob:
        inputs.extend(Path(p).expanduser() for p in sorted(glob.glob(pattern)))
    result = combine(inputs, Path(args.output).expanduser(), args.split, args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run overlapping full-tile inference for a trained torch U-Net artifact."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.tile_inference import parse_offsets, run_full_tile_inference, self_test


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer and evaluate a public labeled Vesuvius segment as one stitched tile")
    parser.add_argument("--artifact", help="Run artifact directory containing config.json and model.pt, or a config JSON path")
    parser.add_argument("--segment-id", help="Public labeled segment ID")
    parser.add_argument("--output-dir", help="Output directory for probability_map.npy, metrics.json, and metrics_by_threshold.csv")
    parser.add_argument("--catalog-source", choices=["public-directory", "installed-catalog", "merged"], default="public-directory")
    parser.add_argument("--level", default="1", help="Zarr resolution level")
    parser.add_argument("--z-offsets", default="0", help="Comma-separated z offsets from the middle layer")
    parser.add_argument("--patch-size", type=int, default=None, help="Patch size; defaults to config dataset.patch_size or 64")
    parser.add_argument("--stride", type=int, default=None, help="Patch stride; defaults to patch_size/2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda")
    parser.add_argument("--public-retry-count", type=int, default=0, help="Retry public-directory fetches after HTTP 429 responses")
    parser.add_argument("--public-retry-delay-sec", type=float, default=0.0, help="Sleep this many seconds between public-directory 429 retries")
    parser.add_argument("--public-chunk-delay-sec", type=float, default=0.0, help="Opt-in sleep after each public Zarr chunk read; 0 keeps the fast direct layer read")
    parser.add_argument("--public-chunk-retry-count", type=int, default=0, help="Retry individual public Zarr chunk reads after HTTP 429 responses")
    parser.add_argument("--public-chunk-retry-delay-sec", type=float, default=0.0, help="Sleep this many seconds between public Zarr chunk 429 retries")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--self-test", action="store_true", help="Run a lightweight synthetic tiling/evaluation self-test")
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return 0
    missing = [name for name in ["artifact", "segment_id", "output_dir"] if getattr(args, name) is None]
    if missing:
        parser.error("missing required arguments unless --self-test is used: " + ", ".join("--" + x.replace("_", "-") for x in missing))
    result = run_full_tile_inference(
        artifact=Path(args.artifact),
        segment_id=args.segment_id,
        output_dir=Path(args.output_dir),
        level=args.level,
        z_offsets=parse_offsets(args.z_offsets),
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size,
        device=args.device,
        catalog_source=args.catalog_source,
        overwrite=args.overwrite,
        public_retry_count=args.public_retry_count,
        public_retry_delay_sec=args.public_retry_delay_sec,
        public_chunk_delay_sec=args.public_chunk_delay_sec,
        public_chunk_retry_count=args.public_chunk_retry_count,
        public_chunk_retry_delay_sec=args.public_chunk_retry_delay_sec,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare labeled real Vesuvius segment patches from the official catalog.

This script reads the real segment surface-volume Zarr and aligned inklabels PNG,
then writes train/validation NPZs in the runner schema.  It uses a spatial split
over the x-axis so validation patches come from a held-out region of the segment.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np


PUBLIC_SEGMENT_BASE = "https://dl.ash2txt.org/other/dev/scrolls/1/segments/54keV_7.91um/"


def _catalog_segments() -> list[dict]:
    from vesuvius.data.volume import list_files  # type: ignore

    catalog = list_files()
    found: list[dict] = []
    for scroll_id, energies in catalog.items():
        for energy, resolutions in (energies or {}).items():
            for resolution, entry in (resolutions or {}).items():
                segments = (entry or {}).get("segments", {}) if isinstance(entry, dict) else {}
                for segment_id, url in segments.items():
                    parent_url = os.path.dirname(str(url).rstrip("/"))
                    found.append({
                        "source": "vesuvius_segment_zarr",
                        "segment_id": str(segment_id),
                        "scroll_id": str(scroll_id),
                        "energy": str(energy),
                        "resolution": str(resolution),
                        "zarr_url": str(url),
                        "inklabels_url": f"{parent_url}/{segment_id}_inklabels.png",
                    })
    return found


def _public_labeled_segments(base_url: str = PUBLIC_SEGMENT_BASE) -> list[dict]:
    with urllib.request.urlopen(base_url, timeout=30) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    zarr_ids = set(re.findall(r'href="(\d+)\.zarr/"', html))
    label_ids = set(re.findall(r'href="(\d+)_inklabels\.png"', html))
    found = []
    for segment_id in sorted(zarr_ids & label_ids):
        found.append({
            "source": "vesuvius_public_segment_zarr",
            "segment_id": segment_id,
            "scroll_id": "1",
            "energy": "54",
            "resolution": "7.91",
            "zarr_url": f"{base_url.rstrip('/')}/{segment_id}.zarr/",
            "inklabels_url": f"{base_url.rstrip('/')}/{segment_id}_inklabels.png",
        })
    return found


def _available_segments(source: str) -> list[dict]:
    if source == "installed-catalog":
        return _catalog_segments()
    if source == "public-directory":
        return _public_labeled_segments()
    merged = {item["segment_id"]: item for item in _catalog_segments()}
    merged.update({item["segment_id"]: item for item in _public_labeled_segments()})
    return sorted(merged.values(), key=lambda item: item["segment_id"])


def _segment_meta(segment_id: str, source: str, catalog: list[dict] | None = None) -> dict:
    for item in (catalog if catalog is not None else _available_segments(source)):
        if item["segment_id"] == str(segment_id):
            return item
    raise ValueError(f"Segment {segment_id} not found in Vesuvius {source}")


def _load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    segments = data.get("segments", data.get("labeled_segments", data)) if isinstance(data, dict) else data
    if not isinstance(segments, list):
        raise ValueError("manifest must be a list or contain segments/labeled_segments")
    return [dict(item) for item in segments]


def _read_label(label_url: str) -> np.ndarray:
    import fsspec
    from PIL import Image

    with fsspec.open(label_url, mode="rb") as fh:
        img = Image.open(BytesIO(fh.read())).convert("L")
        return (np.asarray(img) > 0).astype(np.float32)


def _parse_offsets(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _is_rate_limit_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if getattr(current, "status", None) == 429 or getattr(current, "code", None) == 429:
            return True
        if "429" in str(current) and "Too Many Requests" in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def _read_chunk_with_retries(arr, key: tuple, retry_count: int, retry_delay_sec: float) -> np.ndarray:
    attempts = max(1, int(retry_count) + 1)
    delay = max(0.0, float(retry_delay_sec))
    for attempt in range(attempts):
        try:
            return np.asarray(arr[key], dtype=np.float32)
        except Exception as exc:
            if attempt >= attempts - 1 or not _is_rate_limit_error(exc):
                raise
            if delay > 0:
                time.sleep(delay)
    raise RuntimeError("unreachable chunk retry state")


def _chunk_spans(length: int, chunk: int):
    step = max(1, int(chunk))
    for start in range(0, int(length), step):
        yield start, min(start + step, int(length))


def _chunk_aligned_layers(arr, z_indices: list[int], chunk_delay_sec: float, retry_count: int, retry_delay_sec: float) -> list[np.ndarray]:
    if len(arr.shape) < 3:
        return [_read_chunk_with_retries(arr, (z,), retry_count, retry_delay_sec) for z in z_indices]
    chunks = getattr(arr, "chunks", None) or arr.shape
    z_chunk = int(chunks[-3]) if len(chunks) >= 3 else int(arr.shape[-3])
    y_chunk = int(chunks[-2]) if len(chunks) >= 2 else int(arr.shape[-2])
    x_chunk = int(chunks[-1]) if len(chunks) >= 1 else int(arr.shape[-1])
    z_len, h, w = int(arr.shape[-3]), int(arr.shape[-2]), int(arr.shape[-1])
    out = np.empty((len(z_indices), h, w), dtype=np.float32)
    delay = max(0.0, float(chunk_delay_sec))
    for z0, z1 in _chunk_spans(z_len, z_chunk):
        members = [(out_idx, z) for out_idx, z in enumerate(z_indices) if z0 <= z < z1]
        if not members:
            continue
        for y0, y1 in _chunk_spans(h, y_chunk):
            for x0, x1 in _chunk_spans(w, x_chunk):
                block = _read_chunk_with_retries(arr, (slice(z0, z1), slice(y0, y1), slice(x0, x1)), retry_count, retry_delay_sec)
                for out_idx, z in members:
                    out[out_idx, y0:y1, x0:x1] = block[z - z0]
                if delay > 0:
                    time.sleep(delay)
    return [out[idx] for idx in range(out.shape[0])]


def _open_layers(url: str, level: str, offsets: list[int], public_chunk_delay_sec: float = 0.0, public_chunk_retry_count: int = 0, public_chunk_retry_delay_sec: float = 0.0) -> tuple[np.ndarray, list[int]]:
    import fsspec
    import zarr

    root = zarr.open(fsspec.get_mapper(url), mode="r")
    arr = root[level]
    center = arr.shape[0] // 2
    z_indices = [int(np.clip(center + offset, 0, arr.shape[0] - 1)) for offset in offsets]
    if public_chunk_delay_sec <= 0 and public_chunk_retry_count <= 0 and public_chunk_retry_delay_sec <= 0:
        layers = [np.asarray(arr[z], dtype=np.float32) for z in z_indices]
    else:
        layers = _chunk_aligned_layers(arr, z_indices, public_chunk_delay_sec, public_chunk_retry_count, public_chunk_retry_delay_sec)
    return np.stack(layers, axis=0), z_indices


def _align_label(label: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    if label.shape == image_shape:
        return label
    y_scale = max(1, int(round(label.shape[0] / image_shape[0])))
    x_scale = max(1, int(round(label.shape[1] / image_shape[1])))
    aligned = label[::y_scale, ::x_scale]
    return aligned[:image_shape[0], :image_shape[1]]


def _normalize(image: np.ndarray) -> np.ndarray:
    out = np.empty_like(image, dtype=np.float32)
    for idx in range(image.shape[0] if image.ndim == 3 else 1):
        src = image[idx] if image.ndim == 3 else image
        lo, hi = np.quantile(src, [0.01, 0.99])
        normalized = np.clip((src - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
        if image.ndim == 3:
            out[idx] = normalized
        else:
            out = normalized.astype(np.float32)
    return out


def _patch_mean(label: np.ndarray, yy: int, xx: int, patch_size: int) -> float:
    return float(label[yy:yy + patch_size, xx:xx + patch_size].mean())


def _origins(label: np.ndarray, region: tuple[int, int], patch_size: int, samples: int, positive_fraction: float, seed: int, negative_max_positive_rate: float = 0.001):
    rng = np.random.default_rng(seed)
    h, w = label.shape[:2]
    x0, x1 = region
    x1 = min(x1, w)
    if x1 - x0 < patch_size or h < patch_size:
        raise ValueError(f"Region {region} with image shape {(h, w)} is too small for patch_size={patch_size}")
    positive = np.argwhere(label[:, x0:x1] > 0.5)
    for _ in range(samples):
        if len(positive) and rng.random() < positive_fraction:
            y, local_x = positive[int(rng.integers(0, len(positive)))]
            x = int(local_x + x0)
            yy = int(np.clip(y - rng.integers(0, patch_size), 0, h - patch_size))
            xx = int(np.clip(x - rng.integers(0, patch_size), x0, x1 - patch_size))
        else:
            yy = xx = None
            for _attempt in range(32):
                cy = int(rng.integers(0, h - patch_size + 1))
                cx = int(rng.integers(x0, x1 - patch_size + 1))
                if _patch_mean(label, cy, cx, patch_size) <= negative_max_positive_rate:
                    yy, xx = cy, cx
                    break
            if yy is None or xx is None:
                yy = int(rng.integers(0, h - patch_size + 1))
                xx = int(rng.integers(x0, x1 - patch_size + 1))
        yield yy, xx


def _tiled_origins(label: np.ndarray, region: tuple[int, int], patch_size: int, stride: int, limit: int, seed: int | None = None) -> list[tuple[int, int]]:
    h, w = label.shape[:2]
    x0, x1 = region
    x1 = min(x1, w)
    origins = []
    for yy in range(0, h - patch_size + 1, stride):
        for xx in range(x0, x1 - patch_size + 1, stride):
            origins.append((yy, xx))
    if limit and len(origins) > limit:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(origins))[:limit]
        origins = [origins[int(i)] for i in indices]
    return origins


def _write_npz(output: Path, image: np.ndarray, label: np.ndarray, region: tuple[int, int], split: str, patch_size: int, samples: int, positive_fraction: float, seed: int, meta: dict, tiled: bool, stride: int, negative_max_positive_rate: float) -> None:
    images = []
    labels = []
    origins = _tiled_origins(label, region, patch_size, stride, samples, seed) if tiled else list(_origins(label, region, patch_size, samples, positive_fraction, seed, negative_max_positive_rate))
    for yy, xx in origins:
        images.append(image[:, yy:yy + patch_size, xx:xx + patch_size])
        labels.append(label[yy:yy + patch_size, xx:xx + patch_size][None, :, :])
    images_arr = np.stack(images).astype(np.float32)
    labels_arr = np.stack(labels).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, images=images_arr, labels=labels_arr)
    out_meta = {
        **meta,
        "split": split,
        "region_x": [int(region[0]), int(region[1])],
        "patch_size": patch_size,
        "requested_samples": samples,
        "samples": int(len(origins)),
        "actual_samples": int(len(origins)),
        "positive_fraction": positive_fraction,
        "positive_rate": float(labels_arr.mean()),
        "positive_patch_fraction": float((labels_arr.reshape(labels_arr.shape[0], -1).mean(axis=1) > 0.0).mean()),
        "patch_positive_rate_quantiles": [float(x) for x in np.quantile(labels_arr.reshape(labels_arr.shape[0], -1).mean(axis=1), [0.0, 0.25, 0.5, 0.75, 1.0])],
        "negative_max_positive_rate": negative_max_positive_rate if not tiled else None,
        "sampling": "tiled" if tiled else "positive_mixture",
        "stride": stride if tiled else None,
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(out_meta, indent=2, sort_keys=True))


def _prepare_segment(segment_id: str, output_dir: Path, level: str, patch_size: int, train_samples: int, val_samples: int, train_positive_fraction: float, val_positive_fraction: float, seed: int, source: str, z_offsets: list[int], val_tiled: bool, val_stride: int, negative_max_positive_rate: float, catalog: list[dict] | None = None, public_chunk_delay_sec: float = 0.0, public_chunk_retry_count: int = 0, public_chunk_retry_delay_sec: float = 0.0) -> dict:
    meta = _segment_meta(segment_id, source, catalog)
    image, z_indices = _open_layers(meta["zarr_url"], level, z_offsets, public_chunk_delay_sec, public_chunk_retry_count, public_chunk_retry_delay_sec)
    image = _normalize(image)
    label_raw = _read_label(meta["inklabels_url"])
    label = _align_label(label_raw, image.shape[-2:])
    if float(label.mean()) <= 0.0:
        raise RuntimeError(f"Inklabels are empty after alignment for segment {segment_id}: {meta['inklabels_url']}")

    h, w = label.shape[:2]
    image = image[:, :h, :w]
    gap = max(patch_size, w // 50)
    split_x = int(w * 0.7)
    train_region = (0, max(patch_size, split_x - gap // 2))
    val_region = (min(w - patch_size, split_x + gap // 2), w)
    out_meta = {
        **meta,
        "zarr_level": str(level),
        "image_shape": [int(h), int(w)],
        "image_channels": int(image.shape[0]),
        "z_offsets": z_offsets,
        "z_indices": z_indices,
        "raw_label_shape": [int(label_raw.shape[0]), int(label_raw.shape[1])],
        "label_positive_rate": float(label.mean()),
        "validation_mode": "spatial-same-segment",
        "public_chunk_delay_sec": public_chunk_delay_sec,
        "public_chunk_retry_count": public_chunk_retry_count,
        "public_chunk_retry_delay_sec": public_chunk_retry_delay_sec,
    }
    _write_npz(output_dir / "train.npz", image, label, train_region, "train", patch_size, train_samples, train_positive_fraction, seed, out_meta, False, patch_size, negative_max_positive_rate)
    _write_npz(output_dir / "val.npz", image, label, val_region, "val", patch_size, val_samples, val_positive_fraction, seed + 1, out_meta, val_tiled, val_stride, negative_max_positive_rate)
    return {"train_npz": str((output_dir / "train.npz").resolve()), "val_npz": str((output_dir / "val.npz").resolve()), **out_meta}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare real Vesuvius segment train/val NPZs")
    parser.add_argument("--segment-id", action="append", default=None, help="Segment ID to prepare; repeat for multiple segments")
    parser.add_argument("--catalog-source", choices=["public-directory", "installed-catalog", "merged"], default="public-directory")
    parser.add_argument("--all-labeled", action="store_true", help="Prepare every labeled segment exposed by the installed catalog")
    parser.add_argument("--list-labeled", action="store_true", help="Print catalog segment candidates and exit")
    parser.add_argument("--from-manifest", default=None, help="Read segment catalog from a prior manifest JSON instead of querying catalogs")
    parser.add_argument("--manifest-out", default=None, help="Write selected segment manifest JSON and exit after normal planning/preparation")
    parser.add_argument("--max-new-segments", type=int, default=None, help="Maximum number of not-yet-existing segments to prepare")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse segment output dirs that already contain train.npz and val.npz")
    parser.add_argument("--request-delay-sec", type=float, default=0.0, help="Sleep before each segment fetch/preparation")
    parser.add_argument("--segment-delay-sec", type=float, default=0.0, help="Sleep after each prepared segment")
    parser.add_argument("--output-dir", default="data/real/segment_20230827161847")
    parser.add_argument("--output-root", default="data/real", help="Root for multi-segment output directories")
    parser.add_argument("--level", default="1", help="Zarr resolution level; 1 is smaller/faster than full level 0")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--val-samples", type=int, default=128)
    parser.add_argument("--positive-fraction", type=float, default=0.5, help="Default positive sampling fraction for both splits")
    parser.add_argument("--train-positive-fraction", type=float, default=None)
    parser.add_argument("--val-positive-fraction", type=float, default=None)
    parser.add_argument("--z-offsets", default="0", help="Comma-separated z offsets from the middle layer, e.g. -4,-2,0,2,4")
    parser.add_argument("--val-tiled", action="store_true", help="Use uniform tiled validation sampling instead of positive oversampling")
    parser.add_argument("--val-stride", type=int, default=None)
    parser.add_argument("--negative-max-positive-rate", type=float, default=0.001, help="Prefer negative training patches with at most this ink fraction")
    parser.add_argument("--public-chunk-delay-sec", type=float, default=0.0, help="Opt-in sleep after each public Zarr chunk read; 0 keeps the fast direct layer read")
    parser.add_argument("--public-chunk-retry-count", type=int, default=0, help="Retry individual public Zarr chunk reads after HTTP 429 responses")
    parser.add_argument("--public-chunk-retry-delay-sec", type=float, default=0.0, help="Sleep this many seconds between public Zarr chunk 429 retries")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    catalog = _load_manifest(Path(args.from_manifest).expanduser()) if args.from_manifest else _available_segments(args.catalog_source)
    if args.list_labeled:
        print(json.dumps({"labeled_segments": catalog, "count": len(catalog)}, indent=2, sort_keys=True))
        return
    segment_ids = [item["segment_id"] for item in catalog] if args.all_labeled else (args.segment_id or ["20230827161847"])
    z_offsets = _parse_offsets(args.z_offsets)
    train_positive_fraction = args.positive_fraction if args.train_positive_fraction is None else args.train_positive_fraction
    val_positive_fraction = args.positive_fraction if args.val_positive_fraction is None else args.val_positive_fraction
    val_stride = args.patch_size if args.val_stride is None else args.val_stride
    results = []
    new_segments = 0
    for segment_id in segment_ids:
        out = Path(args.output_dir) if len(segment_ids) == 1 and not args.all_labeled else Path(args.output_root) / f"segment_{segment_id}"
        train_npz = out / "train.npz"
        val_npz = out / "val.npz"
        if args.skip_existing and train_npz.exists() and val_npz.exists():
            meta = _segment_meta(str(segment_id), args.catalog_source, catalog)
            results.append({"train_npz": str(train_npz.resolve()), "val_npz": str(val_npz.resolve()), "segment_id": str(segment_id), "reused": True, **meta})
            continue
        if args.max_new_segments is not None and new_segments >= args.max_new_segments:
            results.append({"segment_id": str(segment_id), "skipped": True, "reason": "max-new-segments reached"})
            continue
        if args.request_delay_sec > 0:
            time.sleep(args.request_delay_sec)
        results.append(_prepare_segment(str(segment_id), out, args.level, args.patch_size, args.train_samples, args.val_samples, train_positive_fraction, val_positive_fraction, args.seed, args.catalog_source, z_offsets, args.val_tiled, val_stride, args.negative_max_positive_rate, catalog, args.public_chunk_delay_sec, args.public_chunk_retry_count, args.public_chunk_retry_delay_sec))
        new_segments += 1
        if args.segment_delay_sec > 0:
            time.sleep(args.segment_delay_sec)
    validation_plan = {
        "mode": "cross-segment" if len({r["segment_id"] for r in results if not r.get("skipped")}) > 1 else "spatial-same-segment",
        "warning": None if len({r["segment_id"] for r in results if not r.get("skipped")}) > 1 else "Only one labeled real segment is available; proper cross-segment/cross-scroll validation is blocked.",
        "segments": results,
    }
    if args.manifest_out:
        Path(args.manifest_out).expanduser().write_text(json.dumps({"segments": [item for item in catalog if item["segment_id"] in {str(s) for s in segment_ids}]}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(validation_plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

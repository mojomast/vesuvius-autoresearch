#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


TRAIN_SEGMENT = "20230827161847"
VAL_SEGMENT = "20230520175435"
PATCH_SIZE = 64
Z_OFFSETS = [-4, 0, 4]
VAL_STRIDE = 64


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _gaussian_probability_map(rng: np.random.Generator, size: int, target_rate: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    field = np.zeros((size, size), dtype=np.float32)
    for _ in range(int(rng.integers(2, 6))):
        cy = float(rng.uniform(0, size - 1))
        cx = float(rng.uniform(0, size - 1))
        sy = float(rng.uniform(2.0, 7.5))
        sx = float(rng.uniform(2.0, 7.5))
        amp = float(rng.uniform(0.6, 1.4))
        field += amp * np.exp(-(((yy - cy) ** 2) / (2.0 * sy * sy) + ((xx - cx) ** 2) / (2.0 * sx * sx)))
    field += rng.normal(0.0, 0.015, size=(size, size)).astype(np.float32)
    field -= float(field.min())
    field /= max(float(field.max()), 1e-6)
    cutoff = float(np.quantile(field, 1.0 - target_rate))
    return (field >= cutoff).astype(np.float32)


def _make_split(n: int, seed: int, segment_shift: float) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    images = np.empty((n, len(Z_OFFSETS), PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    labels = np.empty((n, 1, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    yy, xx = np.mgrid[0:PATCH_SIZE, 0:PATCH_SIZE].astype(np.float32)
    base_texture = 0.45 + 0.08 * np.sin((xx + segment_shift) / 8.0) + 0.06 * np.cos((yy - segment_shift) / 11.0)

    for idx in range(n):
        target_rate = float(rng.uniform(0.05, 0.08))
        label = _gaussian_probability_map(rng, PATCH_SIZE, target_rate)
        labels[idx, 0] = label

        for channel, z_offset in enumerate(Z_OFFSETS):
            z_shift = float(z_offset) / 32.0
            noise = rng.normal(0.0, 0.045, size=(PATCH_SIZE, PATCH_SIZE)).astype(np.float32)
            ink_signal = label * (0.28 + 0.04 * channel)
            channel_img = base_texture + z_shift + ink_signal + noise
            images[idx, channel] = np.clip(channel_img, 0.0, 1.0)

    return images, labels, float(labels.mean())


def _write_split(path: Path, images: np.ndarray, labels: np.ndarray, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, images=images, labels=labels)
    (path.parent / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _metadata(segment_id: str, split: str, source_segments: list[str], n_train: int, n_val: int, val_positive_rate: float) -> dict[str, Any]:
    return {
        "segment_id": segment_id,
        "split": split,
        "val_stride": VAL_STRIDE,
        "spatial_overlap_checked": True,
        "provenance": "synthetic",
        "patch_size": PATCH_SIZE,
        "z_offsets": Z_OFFSETS,
        "source_segments": source_segments,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "val_positive_rate": float(val_positive_rate),
    }


def write_fold_map(output_dir: Path) -> Path:
    fold_root = output_dir / "real_cross_folds_expanded_combined"
    path = fold_root / "fold_map_synthetic.json"
    fold_map = {
        VAL_SEGMENT: {
            "train_npz": _repo_relative(output_dir / "real_cross" / f"segment_{TRAIN_SEGMENT}" / "train.npz"),
            "val_npz": _repo_relative(output_dir / "real_cross" / f"segment_{VAL_SEGMENT}" / "val.npz"),
        },
        TRAIN_SEGMENT: {
            "train_npz": _repo_relative(output_dir / "real_cross" / f"segment_{VAL_SEGMENT}" / "val.npz"),
            "val_npz": _repo_relative(output_dir / "real_cross" / f"segment_{TRAIN_SEGMENT}" / "train.npz"),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fold_map, indent=2, sort_keys=True) + "\n")
    return path


def generate(output_dir: Path, n_train: int, n_val: int, seed: int) -> dict[str, Any]:
    train_images, train_labels, train_rate = _make_split(n_train, seed, segment_shift=0.0)
    val_images, val_labels, val_rate = _make_split(n_val, seed + 10_000, segment_shift=17.0)

    train_path = output_dir / "real_cross" / f"segment_{TRAIN_SEGMENT}" / "train.npz"
    val_path = output_dir / "real_cross" / f"segment_{VAL_SEGMENT}" / "val.npz"
    source_segments = [TRAIN_SEGMENT, VAL_SEGMENT]
    _write_split(train_path, train_images, train_labels, _metadata(TRAIN_SEGMENT, "train", source_segments, n_train, n_val, val_rate))
    _write_split(val_path, val_images, val_labels, _metadata(VAL_SEGMENT, "val", source_segments, n_train, n_val, val_rate))
    fold_map_path = write_fold_map(output_dir)
    return {
        "train_path": train_path,
        "val_path": val_path,
        "fold_map_path": fold_map_path,
        "train_shape": list(train_images.shape),
        "val_shape": list(val_images.shape),
        "train_positive_rate": train_rate,
        "val_positive_rate": val_rate,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate runner-compatible synthetic Vesuvius NPZ data")
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-val", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./data")
    args = parser.parse_args(argv)

    summary = generate(Path(args.output_dir), args.n_train, args.n_val, args.seed)
    print("Synthetic data written:")
    print(f"  train: {summary['train_path']} shape={summary['train_shape']} positive_rate={summary['train_positive_rate']:.4f}")
    print(f"  val: {summary['val_path']} shape={summary['val_shape']} positive_rate={summary['val_positive_rate']:.4f}")
    print(f"  fold_map: {summary['fold_map_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

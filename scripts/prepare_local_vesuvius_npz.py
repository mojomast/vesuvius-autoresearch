#!/usr/bin/env python3
"""Prepare real local Vesuvius fragment/segment data into runner NPZ schema.

Supported input layouts:
- Kaggle-style fragment: surface_volume/*.tif + inklabels.png (+ optional mask.png)
- Segment-style folder: layers/*.tif + *_inklabels.png or inklabels.png

Output schema:
- images: [N, 1, patch_size, patch_size] float32 in [0, 1]
- labels: [N, 1, patch_size, patch_size] float32 binary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def _load_image(path: Path) -> np.ndarray:
    try:
        import tifffile  # type: ignore
        if path.suffix.lower() in {".tif", ".tiff"}:
            return np.asarray(tifffile.imread(path))
    except Exception:
        pass
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install pillow and tifffile to read local Vesuvius image layouts") from exc
    return np.asarray(Image.open(path))


def _find_label(root: Path) -> Path:
    candidates = [root / "inklabels.png", root / f"{root.name}_inklabels.png"]
    candidates.extend(sorted(root.glob("*inklabel*.png")))
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No ink label PNG found under {root}")


def _load_volume(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str]:
    if (root / "surface_volume").is_dir():
        layer_dir = root / "surface_volume"
        kind = "fragment_surface_volume"
    elif (root / "layers").is_dir():
        layer_dir = root / "layers"
        kind = "segment_layers"
    else:
        raise FileNotFoundError(f"Expected surface_volume/ or layers/ under {root}")
    layers = sorted([*layer_dir.glob("*.tif"), *layer_dir.glob("*.tiff")])
    if not layers:
        raise FileNotFoundError(f"No TIFF layers found under {layer_dir}")
    mid = layers[len(layers) // 2]
    image = _load_image(mid).astype(np.float32)
    label = (_load_image(_find_label(root)) > 0).astype(np.float32)
    mask_path = root / "mask.png"
    mask = (_load_image(mask_path) > 0).astype(np.float32) if mask_path.exists() else None
    if image.shape[:2] != label.shape[:2]:
        raise ValueError(f"Image/label shape mismatch: {image.shape} vs {label.shape}")
    return image, label, mask, kind


def _normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    lo, hi = np.quantile(image, [0.01, 0.99])
    return np.clip((image - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)


def _sample_patch_origins(label: np.ndarray, mask: np.ndarray | None, patch_size: int, samples: int, positive_fraction: float, seed: int) -> Iterable[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    h, w = label.shape[:2]
    if h < patch_size or w < patch_size:
        raise ValueError(f"Patch size {patch_size} exceeds data shape {(h, w)}")
    positive = np.argwhere(label > 0.5)
    valid_mask = mask > 0.5 if mask is not None else np.ones_like(label, dtype=bool)
    for _ in range(samples):
        want_positive = len(positive) and rng.random() < positive_fraction
        if want_positive:
            y, x = positive[int(rng.integers(0, len(positive)))]
            yy = int(np.clip(y - rng.integers(0, patch_size), 0, h - patch_size))
            xx = int(np.clip(x - rng.integers(0, patch_size), 0, w - patch_size))
        else:
            for _attempt in range(100):
                yy = int(rng.integers(0, h - patch_size + 1))
                xx = int(rng.integers(0, w - patch_size + 1))
                if valid_mask[yy:yy + patch_size, xx:xx + patch_size].mean() > 0.5:
                    break
        yield yy, xx


def prepare(root: Path, output: Path, split: str, patch_size: int, samples: int, positive_fraction: float, seed: int) -> None:
    image, label, mask, source = _load_volume(root)
    image = _normalize(image)
    images = []
    labels = []
    for yy, xx in _sample_patch_origins(label, mask, patch_size, samples, positive_fraction, seed):
        images.append(image[yy:yy + patch_size, xx:xx + patch_size][None, :, :])
        labels.append(label[yy:yy + patch_size, xx:xx + patch_size][None, :, :])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, images=np.stack(images).astype(np.float32), labels=np.stack(labels).astype(np.float32))
    meta = {
        "source": source,
        "root": str(root.resolve()),
        "split": split,
        "patch_size": patch_size,
        "samples": samples,
        "positive_fraction": positive_fraction,
        "positive_rate": float(np.stack(labels).mean()),
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local Vesuvius TIFF/PNG data into runner NPZ schema")
    parser.add_argument("--root", required=True, help="Fragment/segment folder containing surface_volume/ or layers/")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--split", default="train")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--positive-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    prepare(Path(args.root).expanduser(), Path(args.output).expanduser(), args.split, args.patch_size, args.samples, args.positive_fraction, args.seed)


if __name__ == "__main__":
    main()

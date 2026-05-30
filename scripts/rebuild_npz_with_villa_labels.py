#!/usr/bin/env python3
"""Rebuild prepared NPZ labels using ScrollPrize/villa cleaned ink labels.

Cleaned labels adapted from ScrollPrize/villa (MIT License) - Youssef Nader,
Luke Farritor, Julian Schilliger.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from data.vesuvius_data import validate_prepared_npz
from scripts.prepare_vesuvius_segment_npz import _align_label, _origins, _read_label, _tiled_origins


ROOT = Path(__file__).resolve().parents[1]
VILLA_BASE_URL = "https://raw.githubusercontent.com/ScrollPrize/villa/main/ink-detection/all_labels"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def download_villa_label(segment_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{segment_id}_inklabels.png"
    if out.exists():
        return out
    url = f"{VILLA_BASE_URL}/{segment_id}_inklabels.png"
    with urllib.request.urlopen(url, timeout=60) as resp:
        out.write_bytes(resp.read())
    return out


def load_villa_label(path: Path, image_shape: tuple[int, int]) -> np.ndarray:
    raw = np.asarray(Image.open(path).convert("L")) > 127
    aligned = _align_label(raw.astype(np.float32), image_shape)
    return (aligned > 0.5).astype(np.float32)


def compute_iou(ours: np.ndarray, villa: np.ndarray) -> dict[str, float | int]:
    ours_b = ours > 0.5
    villa_b = villa > 0.5
    intersection = int(np.logical_and(ours_b, villa_b).sum())
    union = int(np.logical_or(ours_b, villa_b).sum())
    return {
        "iou": float(intersection / union) if union else 1.0,
        "villa_not_ours": int(np.logical_and(villa_b, ~ours_b).sum()),
        "ours_not_villa": int(np.logical_and(ours_b, ~villa_b).sum()),
        "intersection": intersection,
        "union": union,
    }


def _origins_for_npz(meta: dict[str, Any], current_label: np.ndarray, sample_count: int) -> list[tuple[int, int]]:
    patch_size = int(meta.get("patch_size") or 64)
    region_x_raw = meta.get("region_x") or [0, current_label.shape[1]]
    region = (int(region_x_raw[0]), int(region_x_raw[1]))
    sampling = str(meta.get("sampling") or "")
    if sampling == "tiled":
        stride = int(meta.get("stride") or meta.get("val_stride") or patch_size)
        return _tiled_origins(current_label, region, patch_size, stride, 0)[:sample_count]
    return list(_origins(
        current_label,
        region,
        patch_size,
        sample_count,
        float(meta.get("positive_fraction") or 0.5),
        13 if str(meta.get("split")) == "train" else 14,
        float(meta.get("negative_max_positive_rate") if meta.get("negative_max_positive_rate") is not None else 0.001),
    ))


def rebuild_segment_npz(segment_id: str, split: str, source_npz: Path, villa_label_path: Path, output_npz: Path) -> dict[str, Any]:
    prepared = validate_prepared_npz(source_npz, split=split, patch_size=64)
    meta = dict(prepared.get("metadata") or {})
    with np.load(source_npz) as data:
        images = data["images"].astype(np.float32)
        original_labels = data["labels"].astype(np.float32)
    image_shape = tuple(int(x) for x in meta.get("image_shape", original_labels.shape[-2:]))
    current_full_label = _align_label(_read_label(str(meta["inklabels_url"])), image_shape)
    villa_full_label = load_villa_label(villa_label_path, image_shape)
    origins = _origins_for_npz(meta, current_full_label, images.shape[0])
    if len(origins) != images.shape[0]:
        raise RuntimeError(f"{source_npz}: reconstructed {len(origins)} origins for {images.shape[0]} samples")
    patch_size = int(original_labels.shape[-1])
    rebuilt = np.stack([villa_full_label[yy:yy + patch_size, xx:xx + patch_size][None, :, :] for yy, xx in origins]).astype(np.float32)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, images=images, labels=rebuilt)
    iou = compute_iou(original_labels, rebuilt)
    out_meta = {
        **meta,
        "villa_label_source": "ScrollPrize/villa all_labels",
        "villa_label_path": str(villa_label_path),
        "original_npz": str(source_npz),
        "positive_rate": float(rebuilt.mean()),
        "original_positive_rate": float(original_labels.mean()),
        "villa_label_iou": iou["iou"],
        "villa_not_ours": iou["villa_not_ours"],
        "ours_not_villa": iou["ours_not_villa"],
    }
    output_npz.with_suffix(".metadata.json").write_text(json.dumps(out_meta, indent=2, sort_keys=True) + "\n")
    return {
        "segment_id": segment_id,
        "split": split,
        "source_npz": str(source_npz.relative_to(ROOT)),
        "output_npz": str(output_npz.relative_to(ROOT)),
        "samples": int(images.shape[0]),
        "pixels": int(original_labels.size),
        "original_positive_rate": float(original_labels.mean()),
        "villa_positive_rate": float(rebuilt.mean()),
        **iou,
    }


def rebuild_combined_leaveout(fold_id: str, fold_map: dict[str, Any], segment_outputs: dict[str, dict[str, Path]], output_npz: Path) -> dict[str, Any]:
    image_parts = []
    label_parts = []
    inputs = []
    for segment_id in sorted(fold_map):
        if segment_id == fold_id:
            continue
        train_npz = segment_outputs[segment_id]["train"]
        with np.load(train_npz) as data:
            image_parts.append(data["images"].astype(np.float32))
            label_parts.append(data["labels"].astype(np.float32))
        meta_path = train_npz.with_suffix(".metadata.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        inputs.append({"path": str(train_npz.relative_to(ROOT)), "samples": int(image_parts[-1].shape[0]), "source_segments": [segment_id], "villa_label_iou": meta.get("villa_label_iou")})
    images = np.concatenate(image_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, images=images, labels=labels)
    meta = {
        "source": "combined_villa_label_npz",
        "provenance": "combined_villa_label_npz",
        "split": "train",
        "heldout_segment": fold_id,
        "train_segments": [item["source_segments"][0] for item in inputs],
        "source_segments": [item["source_segments"][0] for item in inputs],
        "inputs": inputs,
        "patch_size": int(labels.shape[-1]),
        "samples": int(labels.shape[0]),
        "positive_rate": float(labels.mean()),
        "validation_mode": "leave-one-segment-out",
        "villa_label_source": "ScrollPrize/villa all_labels",
    }
    output_npz.with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return {"fold_id": fold_id, "output_npz": str(output_npz.relative_to(ROOT)), "samples": int(labels.shape[0]), "positive_rate": float(labels.mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild NPZ labels from ScrollPrize/villa cleaned labels")
    parser.add_argument("--fold-map", default="data/real_cross_folds_expanded_combined/fold_map.json")
    parser.add_argument("--label-dir", default="data/villa_labels")
    parser.add_argument("--output-root", default="data/npz_villa")
    parser.add_argument("--fold-map-out", default="data/fold_map_villa_labels.json")
    args = parser.parse_args()

    fold_map_path = _repo_path(args.fold_map)
    fold_map = _load_json(fold_map_path)
    label_dir = _repo_path(args.label_dir)
    output_root = _repo_path(args.output_root)
    segment_outputs: dict[str, dict[str, Path]] = {}
    segment_summaries = []
    for segment_id, paths in sorted(fold_map.items()):
        villa_path = download_villa_label(segment_id, label_dir)
        segment_outputs[segment_id] = {}
        for split in ("train", "val"):
            source_npz = _repo_path(f"data/real_cross_folds_v2/segment_{segment_id}/{split}.npz")
            output_npz = output_root / f"segment_{segment_id}" / f"{split}.npz"
            segment_outputs[segment_id][split] = output_npz
            segment_summaries.append(rebuild_segment_npz(segment_id, split, source_npz, villa_path, output_npz))

    combined_summaries = []
    villa_fold_map: dict[str, dict[str, str]] = {}
    for fold_id in sorted(fold_map):
        combined_npz = output_root / f"leaveout_{fold_id}" / "train.npz"
        combined_summaries.append(rebuild_combined_leaveout(fold_id, fold_map, segment_outputs, combined_npz))
        villa_fold_map[fold_id] = {
            "train_npz": str(combined_npz.relative_to(ROOT)),
            "val_npz": str(segment_outputs[fold_id]["val"].relative_to(ROOT)),
        }
    _write_json(_repo_path(args.fold_map_out), villa_fold_map)
    summary = {"segments": segment_summaries, "combined_leaveouts": combined_summaries, "fold_map": args.fold_map_out}
    _write_json(output_root / "rebuild_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

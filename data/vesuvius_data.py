"""Thin data wrapper around the official villa/vesuvius library.

The public villa monorepo exposes the Python package `vesuvius`:
- `vesuvius.list_files()` lists scroll -> energy -> resolution -> volume/segments.
- `vesuvius.Volume(type="segment", segment_id=..., download_only=True)` fetches segment metadata
  and, when available, the companion `inklabel` PNG/array.

This module intentionally fails loudly when real data is unavailable.  Historical synthetic
fallback code was removed from the active workflow so research runs cannot accidentally train
on fake data.
"""
from __future__ import annotations

import json
import math
import os
import hashlib
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

DEFAULT_SCROLLS = ["1", "2"]
DEFAULT_SPLITS = {"train": ["1"], "val": ["2"]}
PUBLIC_SEGMENT_BASE = "https://dl.ash2txt.org/other/dev/scrolls/1/segments/54keV_7.91um/"


def _stable_seed(*parts: Any) -> int:
    key = "|".join(str(part) for part in parts)
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)


def _try_import_vesuvius():
    try:
        import vesuvius  # type: ignore
        return vesuvius
    except Exception:
        return None


def _public_labeled_segments() -> List[Dict[str, Any]]:
    try:
        with urllib.request.urlopen(PUBLIC_SEGMENT_BASE, timeout=20) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        zarr_ids = set(re.findall(r'href="(\d+)\.zarr/"', html))
        label_ids = set(re.findall(r'href="(\d+)_inklabels\.png"', html))
        return [{
            "scroll_id": "1",
            "segment_id": segment_id,
            "energy": "54",
            "resolution": "7.91",
            "zarr_url": f"{PUBLIC_SEGMENT_BASE}{segment_id}.zarr/",
            "inklabels_url": f"{PUBLIC_SEGMENT_BASE}{segment_id}_inklabels.png",
        } for segment_id in sorted(zarr_ids & label_ids)]
    except Exception:
        return []


def get_dataset_summary() -> Dict[str, Any]:
    """Return available scrolls/regions/splits using villa when possible.

    Shape is stable for the rest of this repo:
    {source, scrolls: [{scroll_id, energies, resolutions, segments}], splits}
    """
    vesuvius = _try_import_vesuvius()
    if vesuvius is None:
        return {
            "source": "unavailable",
            "reason": "official vesuvius package is not importable",
            "scrolls": [],
            "splits": {},
        }

    try:
        list_files = getattr(vesuvius, "list_files", None)
        if list_files is None:
            from vesuvius.data.volume import list_files as volume_list_files  # type: ignore
            list_files = volume_list_files
        files = list_files()
        scrolls: List[Dict[str, Any]] = []
        labeled_segments: List[Dict[str, Any]] = []
        for scroll_id, energies in sorted(files.items(), key=lambda kv: str(kv[0])):
            energy_keys = list((energies or {}).keys())
            resolutions = []
            segments = []
            for energy, res_map in (energies or {}).items():
                for res, entry in (res_map or {}).items():
                    resolutions.append(str(res))
                    seg_map = (entry or {}).get("segments", {}) if isinstance(entry, dict) else {}
                    segments.extend(str(seg_id) for seg_id in seg_map.keys())
                    for seg_id, zarr_url in seg_map.items():
                        parent = str(zarr_url).rstrip("/").rsplit("/", 1)[0]
                        labeled_segments.append({
                            "scroll_id": str(scroll_id),
                            "segment_id": str(seg_id),
                            "energy": str(energy),
                            "resolution": str(res),
                            "zarr_url": str(zarr_url),
                            "inklabels_url": f"{parent}/{seg_id}_inklabels.png",
                        })
            scrolls.append({
                "scroll_id": str(scroll_id),
                "energies": [str(e) for e in energy_keys],
                "resolutions": sorted(set(resolutions)),
                "segments": sorted(set(segments)),
            })
        all_ids = [s["scroll_id"] for s in scrolls] or DEFAULT_SCROLLS
        splits = {"train": all_ids[:1], "val": all_ids[1:2] or all_ids[:1]}
        public_labeled = _public_labeled_segments()
        if public_labeled:
            by_id = {item["segment_id"]: item for item in labeled_segments}
            by_id.update({item["segment_id"]: item for item in public_labeled})
            labeled_segments = sorted(by_id.values(), key=lambda item: item["segment_id"])
        validation_setup = {
            "mode": "cross-segment-available" if len(labeled_segments) > 1 else "spatial-same-segment",
            "warning": None if len(labeled_segments) > 1 else "Only one official labeled segment is exposed by the installed vesuvius catalog; cross-segment/cross-scroll validation is blocked.",
            "labeled_segment_count": len(labeled_segments),
        }
        return {"source": "vesuvius", "scrolls": scrolls, "splits": splits, "labeled_segments": labeled_segments, "validation_setup": validation_setup}
    except Exception as exc:
        return {
            "source": "unavailable",
            "reason": f"vesuvius.list_files failed: {exc}",
            "scrolls": [],
            "splits": {},
        }


def _write_npz(path: Path, images: np.ndarray, labels: np.ndarray, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.savez_compressed(tmp, images=images.astype(np.float32), labels=labels.astype(np.float32))
    # np.savez appends .npz if suffix is not .npz; handle both possibilities idempotently.
    actual = tmp if tmp.exists() else Path(str(tmp) + ".npz")
    actual.replace(path)
    (path.parent / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def validate_prepared_npz(path: str | os.PathLike[str], split: str | None = None, patch_size: int | None = None) -> Dict[str, Any]:
    """Validate a prepared real-data NPZ against the runner schema.

    Required schema: ``images`` and ``labels`` float-compatible arrays shaped
    ``[N, 1, H, W]``.  This lets real Vesuvius ingestion fail loudly instead of
    silently falling back to synthetic data.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Prepared NPZ not found: {resolved}")
    with np.load(resolved) as data:
        keys = set(data.files)
        missing = {"images", "labels"} - keys
        if missing:
            raise ValueError(f"Prepared NPZ missing keys: {sorted(missing)}")
        images = data["images"]
        labels = data["labels"]
        if images.ndim != 4 or images.shape[1] < 1:
            raise ValueError(f"Expected [N,C,H,W] arrays with C>=1, got {images.shape}")
        if labels.ndim != 4 or labels.shape[1] != 1:
            raise ValueError(f"Expected labels shaped [N,1,H,W], got {labels.shape}")
        if images.shape[0] != labels.shape[0] or images.shape[-2:] != labels.shape[-2:]:
            raise ValueError(f"images/labels sample or spatial mismatch: {images.shape} vs {labels.shape}")
        if images.shape[0] < 1:
            raise ValueError("Prepared NPZ must contain at least one sample")
        if patch_size is not None and (images.shape[-2] != int(patch_size) or images.shape[-1] != int(patch_size)):
            raise ValueError(f"Expected patch_size {patch_size}, got {images.shape[-2:]}")
        if not np.isfinite(images).all():
            raise ValueError("Prepared NPZ images contain non-finite values")
        if not np.isfinite(labels).all():
            raise ValueError("Prepared NPZ labels contain non-finite values")
        label_min = float(labels.min())
        label_max = float(labels.max())
        sidecar = resolved.with_suffix(".metadata.json")
        metadata: Dict[str, Any] = {}
        if sidecar.exists():
            try:
                metadata = json.loads(sidecar.read_text())
            except Exception as exc:
                metadata = {"metadata_error": str(exc)}
        return {
            "path": str(resolved),
            "source": metadata.get("source", "prepared_npz"),
            "split": split,
            "samples": int(images.shape[0]),
            "patch_size": int(images.shape[-1]),
            "shape": [int(x) for x in images.shape],
            "positive_rate": float((labels > 0.5).mean()),
            "label_min": label_min,
            "label_max": label_max,
            "metadata": metadata,
        }


def _try_prepare_with_vesuvius(scroll_id: str, split: str, out_dir: Path, patch_size: int, max_samples: int) -> Optional[Dict[str, Any]]:
    return None


def prepare_training_subset(scroll_id: str, split: str, out_dir: str, patch_size: int = 32, max_samples: int = 64, force: bool = False) -> Dict[str, Any]:
    """Prepare a small local subset and return metadata including the `.npz` path.

    Idempotent: if the requested subset exists with matching metadata, it is reused.
    """
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"scroll_{scroll_id}_{split}.npz"
    meta_path = out / "metadata.json"
    if not force and path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("scroll_id") == str(scroll_id) and meta.get("split") == split and int(meta.get("patch_size", -1)) == int(patch_size) and int(meta.get("samples", -1)) == int(max_samples):
                return {"path": str(path), **meta, "reused": True}
        except Exception:
            pass
    prepared = _try_prepare_with_vesuvius(scroll_id, split, out, patch_size, max_samples)
    if prepared is not None:
        return prepared
    raise RuntimeError(
        "Real Vesuvius data could not be prepared automatically. Provide dataset.train_npz "
        "and dataset.val_npz, or run scripts/prepare_vesuvius_segment_npz.py first."
    )

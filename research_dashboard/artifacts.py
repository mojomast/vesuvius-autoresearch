from __future__ import annotations

import json
import base64
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from .quality import decoded_output_quality, metrics_quality_verdict


def _under(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def list_artifact_files(artifact_dir: str | None, limit: int = 24) -> list[dict[str, Any]]:
    if not artifact_dir:
        return []
    root = Path(artifact_dir).expanduser()
    if not root.exists() or not root.is_dir():
        return []
    files = []
    candidates = [p for p in root.rglob("*") if p.is_file()]
    candidates.sort(key=lambda p: (0 if p.name == "probability_map.npy" else 1, -p.stat().st_mtime))
    for path in candidates:
        stat = path.stat()
        rel_path = str(path.relative_to(root))
        item = {"name": path.name, "relative_path": rel_path, "path": str(path), "size_bytes": stat.st_size, "modified_at": stat.st_mtime, "kind": path.suffix.lstrip(".") or "file"}
        if path.name == "probability_map.npy":
            metrics_path = path.parent / "metrics.json"
            if metrics_path.exists():
                try:
                    metrics = json.loads(metrics_path.read_text(errors="replace"))
                    item["quality_verdict"] = metrics_quality_verdict(metrics)
                except Exception:
                    pass
        files.append(item)
        if len(files) >= limit:
            break
    return files


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _rgb_png_data_url(rgb: np.ndarray) -> str:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].astype(np.uint8).tobytes() for y in range(h))
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(raw, 9))
    png += _png_chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _downsample_2d(arr: np.ndarray, max_side: int = 384) -> np.ndarray:
    h, w = arr.shape
    scale = max(h / max_side, w / max_side, 1.0)
    if scale <= 1.0:
        return arr
    yy = np.linspace(0, h - 1, max(1, int(round(h / scale)))).astype(np.int64)
    xx = np.linspace(0, w - 1, max(1, int(round(w / scale)))).astype(np.int64)
    return arr[np.ix_(yy, xx)]


def _probability_map_preview(path: Path) -> dict[str, Any]:
    arr = np.load(path, allow_pickle=False)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D probability map, got shape {list(arr.shape)}")
    probs = arr.astype(np.float32)
    sample = _downsample_2d(np.clip(probs, 0.0, 1.0))
    heat = np.zeros((*sample.shape, 3), dtype=np.uint8)
    heat[..., 0] = np.clip(sample * 255.0, 0, 255).astype(np.uint8)
    heat[..., 1] = np.clip(np.sqrt(sample) * 220.0, 0, 255).astype(np.uint8)
    heat[..., 2] = np.clip((1.0 - sample) * 90.0, 0, 255).astype(np.uint8)

    threshold = 0.5
    metrics_path = path.parent / "metrics.json"
    metrics: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(errors="replace"))
            threshold = float(metrics.get("best_threshold", metrics.get("fixed_threshold", threshold)))
        except Exception:
            metrics = {}
    map_quality = decoded_output_quality(probs, threshold)
    metric_quality = metrics_quality_verdict(metrics)
    mask = np.where(sample >= threshold, 230, 12).astype(np.uint8)
    mask_rgb = np.stack([mask // 6, mask, mask], axis=-1)
    return {
        "shape": [int(probs.shape[0]), int(probs.shape[1])],
        "rendered_shape": [int(sample.shape[0]), int(sample.shape[1])],
        "min": float(np.min(probs)),
        "mean": float(np.mean(probs)),
        "p95": float(np.quantile(probs, 0.95)),
        "max": float(np.max(probs)),
        "threshold": threshold,
        "pred_positive_rate": float((probs >= threshold).mean()),
        "metrics": {key: metrics.get(key) for key in ("val_f1", "tile_f1", "average_precision", "val_positive_rate", "fixed_threshold_f1", "threshold_selection") if key in metrics},
        "map_quality": map_quality,
        "quality_verdict": metric_quality,
        "heatmap_data_url": _rgb_png_data_url(heat),
        "mask_data_url": _rgb_png_data_url(mask_rgb),
    }


def preview_artifact(project_root: Path, artifact_path: str) -> dict[str, Any]:
    runs_root = (project_root / "experiments" / "runs").resolve()
    path = Path(artifact_path).expanduser().resolve()
    if not _under(path, runs_root):
        raise ValueError("artifact path must be under experiments/runs")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("artifact not found")
    stat = path.stat()
    out: dict[str, Any] = {"name": path.name, "path": str(path), "size_bytes": stat.st_size, "modified_at": stat.st_mtime, "kind": path.suffix.lstrip(".") or "file", "preview": None, "truncated": False}
    suffix = path.suffix.lower()
    if suffix == ".json":
        out["preview"] = json.loads(path.read_text(errors="replace"))
        return out
    if suffix == ".npy":
        try:
            out["kind"] = "npy"
            out["preview"] = _probability_map_preview(path)
            return out
        except Exception as exc:
            out["preview_error"] = f"Failed to decode NumPy artifact: {exc}"
            return out
    if suffix in {".txt", ".log", ".md", ".yaml", ".yml", ".csv"}:
        text = path.read_text(errors="replace")
        out["preview"] = text[:24000]
        out["truncated"] = len(text) > 24000
        return out
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        try:
            data = path.read_bytes()
            encoded = base64.b64encode(data).decode("utf-8")
            mime = f"image/{suffix.lstrip('.')}"
            if mime == "image/jpg":
                mime = "image/jpeg"
            out["preview"] = f"data:{mime};base64,{encoded}"
            return out
        except Exception as exc:
            out["preview_error"] = f"Failed to load image preview: {exc}"
            return out
    out["preview_error"] = "Preview unavailable for binary or unsupported artifact type"
    return out

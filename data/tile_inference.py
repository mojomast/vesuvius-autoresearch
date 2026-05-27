"""Full-tile inference helpers for trained torch U-Net run artifacts."""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def parse_offsets(raw: str | Iterable[int]) -> list[int]:
    if isinstance(raw, str):
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    return [int(x) for x in raw]


def _validate_tiling_args(patch_size: int, stride: int, batch_size: int | None = None) -> None:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive")


def _is_rate_limit_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if getattr(current, "status", None) == 429 or getattr(current, "code", None) == 429:
            return True
        if "429" in str(current) and "Too Many Requests" in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def load_public_segment(segment_id: str, level: str, z_offsets: list[int], catalog_source: str = "public-directory", retry_count: int = 0, retry_delay_sec: float = 0.0, public_chunk_delay_sec: float = 0.0, public_chunk_retry_count: int = 0, public_chunk_retry_delay_sec: float = 0.0) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Open a public labeled segment and return normalized image [C,H,W] plus label [H,W]."""
    from scripts.prepare_vesuvius_segment_npz import _align_label, _normalize, _open_layers, _read_label, _segment_meta

    attempts = max(1, int(retry_count) + 1)
    delay = max(0.0, float(retry_delay_sec))
    meta = _segment_meta(segment_id, catalog_source)
    for attempt in range(attempts):
        try:
            image, z_indices = _open_layers(meta["zarr_url"], str(level), z_offsets, public_chunk_delay_sec, public_chunk_retry_count, public_chunk_retry_delay_sec)
            image = _normalize(image)
            raw_label = _read_label(meta["inklabels_url"])
            break
        except Exception as exc:
            if attempt >= attempts - 1 or not _is_rate_limit_error(exc):
                raise
            if delay > 0:
                time.sleep(delay)
    label = _align_label(raw_label, image.shape[-2:])
    h, w = label.shape[:2]
    image = image[:, :h, :w]
    out_meta = {
        **meta,
        "zarr_level": str(level),
        "z_offsets": z_offsets,
        "z_indices": z_indices,
        "image_shape": [int(h), int(w)],
        "image_channels": int(image.shape[0]),
        "raw_label_shape": [int(raw_label.shape[0]), int(raw_label.shape[1])],
        "label_positive_rate": float(label.mean()),
        "public_chunk_delay_sec": public_chunk_delay_sec,
        "public_chunk_retry_count": public_chunk_retry_count,
        "public_chunk_retry_delay_sec": public_chunk_retry_delay_sec,
    }
    return image.astype(np.float32), label.astype(np.float32), out_meta


def tile_origins(height: int, width: int, patch_size: int = 64, stride: int = 32) -> list[tuple[int, int]]:
    _validate_tiling_args(int(patch_size), int(stride))
    if height < patch_size or width < patch_size:
        raise ValueError(f"Patch size {patch_size} exceeds image shape {(height, width)}")

    def starts(length: int) -> list[int]:
        vals = list(range(0, length - patch_size + 1, stride))
        last = length - patch_size
        if not vals or vals[-1] != last:
            vals.append(last)
        return vals

    return [(yy, xx) for yy in starts(height) for xx in starts(width)]


def blending_window(patch_size: int, floor: float = 0.05) -> np.ndarray:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    one = np.hanning(patch_size).astype(np.float32)
    if not np.any(one):
        one = np.ones(patch_size, dtype=np.float32)
    one = np.maximum(one, float(floor))
    win = np.outer(one, one).astype(np.float32)
    return win / max(float(win.max()), 1e-6)


def build_tiny_unet(in_channels: int, base_channels: int):
    try:
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("Full-tile torch U-Net inference requires PyTorch") from exc

    class ConvBlock(nn.Module):
        def __init__(self, cin: int, cout: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class TinyUNet(nn.Module):
        def __init__(self, cin: int, channels: int):
            super().__init__()
            self.enc = ConvBlock(cin, channels)
            self.pool = nn.MaxPool2d(2)
            self.mid = ConvBlock(channels, channels * 2)
            self.up = nn.ConvTranspose2d(channels * 2, channels, 2, stride=2)
            self.dec = ConvBlock(channels * 2, channels)
            self.out = nn.Conv2d(channels, 1, 1)

        def forward(self, x):
            skip = self.enc(x)
            x = self.up(self.mid(self.pool(skip)))
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            import torch

            return self.out(self.dec(torch.cat([x, skip], dim=1)))

    return TinyUNet(in_channels, base_channels)


def build_residual_25d_unet(in_channels: int, base_channels: int):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("Full-tile torch U-Net inference requires PyTorch") from exc

    class ResidualBlock(nn.Module):
        def __init__(self, cin: int, cout: int):
            super().__init__()
            groups = max(g for g in range(1, min(8, cout) + 1) if cout % g == 0)
            self.proj = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
            self.net = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(groups, cout), nn.SiLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(groups, cout),
            )
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            return self.act(self.net(x) + self.proj(x))

    class Residual25DUNet(nn.Module):
        def __init__(self, cin: int, channels: int):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(cin, channels, 1), nn.SiLU(inplace=True))
            self.enc1 = ResidualBlock(channels, channels)
            self.pool1 = nn.MaxPool2d(2)
            self.enc2 = ResidualBlock(channels, channels * 2)
            self.pool2 = nn.MaxPool2d(2)
            self.mid = ResidualBlock(channels * 2, channels * 4)
            self.up2 = nn.ConvTranspose2d(channels * 4, channels * 2, 2, stride=2)
            self.dec2 = ResidualBlock(channels * 4, channels * 2)
            self.up1 = nn.ConvTranspose2d(channels * 2, channels, 2, stride=2)
            self.dec1 = ResidualBlock(channels * 2, channels)
            self.out = nn.Conv2d(channels, 1, 1)

        def forward(self, x):
            x = self.stem(x)
            s1 = self.enc1(x)
            s2 = self.enc2(self.pool1(s1))
            x = self.up2(self.mid(self.pool2(s2)))
            if x.shape[-2:] != s2.shape[-2:]:
                x = F.interpolate(x, size=s2.shape[-2:], mode="bilinear", align_corners=False)
            x = self.dec2(torch.cat([x, s2], dim=1))
            x = self.up1(x)
            if x.shape[-2:] != s1.shape[-2:]:
                x = F.interpolate(x, size=s1.shape[-2:], mode="bilinear", align_corners=False)
            return self.out(self.dec1(torch.cat([x, s1], dim=1)))

    return Residual25DUNet(in_channels, base_channels)


def _build_model(model_name: str, in_channels: int, base_channels: int):
    if model_name == "tiny_torch_unet":
        return build_tiny_unet(in_channels, base_channels)
    if model_name in {"residual_25d_torch_unet", "torch_residual_25d_unet"}:
        return build_residual_25d_unet(in_channels, base_channels)
    raise ValueError(f"Unsupported full-tile model: {model_name}")


def load_torch_unet_artifact(artifact_or_config: Path, in_channels: int, device: str = "cpu") -> tuple[list[Any], dict[str, Any], Path, dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Full-tile torch U-Net inference requires PyTorch") from exc

    path = artifact_or_config.expanduser()
    artifact_dir = path if path.is_dir() else path.parent
    config_path = artifact_dir / "config.json" if path.is_dir() else path
    model_path = artifact_dir / "model.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"No config JSON found at {config_path}")
    cfg = json.loads(config_path.read_text())
    meta_path = artifact_dir / "model_meta.json"
    model_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    model_name = str(model_meta.get("model_name") or cfg.get("model", {}).get("name"))
    base = int(cfg.get("model", {}).get("base_channels", 8))
    model_names = model_meta.get("model_paths") or ["model.pt"]
    models = []
    for name in model_names:
        model_file = artifact_dir / name
        if not model_file.exists():
            raise FileNotFoundError(f"No torch model artifact found at {model_file}")
        model = _build_model(model_name, in_channels, base)
        try:
            state = torch.load(model_file, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(model_file, map_location=device)
        model.load_state_dict(state)
        model.to(torch.device(device))
        model.eval()
        models.append(model)
    return models, cfg, artifact_dir, model_meta


def stitch_probabilities_from_predictor(predict_batch: Any, image: np.ndarray, patch_size: int = 64, stride: int = 32, batch_size: int = 8) -> tuple[np.ndarray, dict[str, Any]]:
    _validate_tiling_args(int(patch_size), int(stride), int(batch_size))
    if image.ndim != 3:
        raise ValueError(f"Expected image [C,H,W], got shape {image.shape}")
    _, h, w = image.shape
    origins = tile_origins(h, w, patch_size, stride)
    window = blending_window(patch_size)
    weighted = np.zeros((h, w), dtype=np.float32)
    weights = np.zeros((h, w), dtype=np.float32)
    for start in range(0, len(origins), batch_size):
        batch_origins = origins[start:start + batch_size]
        patches = np.stack([image[:, yy:yy + patch_size, xx:xx + patch_size] for yy, xx in batch_origins]).astype(np.float32)
        probs = np.asarray(predict_batch(patches), dtype=np.float32)
        if probs.shape != (len(batch_origins), patch_size, patch_size):
            raise ValueError(f"Predictor returned shape {probs.shape}, expected {(len(batch_origins), patch_size, patch_size)}")
        for prob, (yy, xx) in zip(probs, batch_origins):
            weighted[yy:yy + patch_size, xx:xx + patch_size] += prob.astype(np.float32) * window
            weights[yy:yy + patch_size, xx:xx + patch_size] += window
    prob_map = weighted / np.maximum(weights, 1e-6)
    return prob_map.astype(np.float32), {
        "patch_size": int(patch_size),
        "stride": int(stride),
        "batch_size": int(batch_size),
        "patches": int(len(origins)),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
    }


def stitch_probabilities(models: Any, image: np.ndarray, patch_size: int = 64, stride: int = 32, batch_size: int = 8, device: str = "cpu", tta_flips: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Full-tile torch U-Net inference requires PyTorch") from exc

    dev = torch.device(device)
    model_list = list(models) if isinstance(models, (list, tuple)) else [models]

    def predict_batch(patches: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xb = torch.from_numpy(patches).to(dev)
            pred_sum = None
            pred_count = 0
            for model in model_list:
                probs = torch.sigmoid(model(xb))
                pred_sum = probs if pred_sum is None else pred_sum + probs
                pred_count += 1
                if tta_flips:
                    pred_sum = pred_sum + torch.flip(torch.sigmoid(model(torch.flip(xb, dims=[3]))), dims=[3])
                    pred_sum = pred_sum + torch.flip(torch.sigmoid(model(torch.flip(xb, dims=[2]))), dims=[2])
                    pred_sum = pred_sum + torch.flip(torch.sigmoid(model(torch.flip(xb, dims=[2, 3]))), dims=[2, 3])
                    pred_count += 3
            return (pred_sum / max(pred_count, 1)).detach().cpu().numpy()[:, 0]

    prob_map, meta = stitch_probabilities_from_predictor(predict_batch, image, patch_size, stride, batch_size)
    meta.update({"ensemble_size": int(len(model_list)), "tta_flips": bool(tta_flips)})
    return prob_map, meta


def _binary_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    pred = probs >= threshold
    truth = labels > 0.5
    tp = float((pred & truth).sum())
    fp = float((pred & ~truth).sum())
    fn = float((~pred & truth).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    beta2 = 0.25
    f05 = (1.0 + beta2) * precision * recall / max(beta2 * precision + recall, 1e-9)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f05": float(f05),
        "f1": float(f1),
        "pred_positive_rate": float(pred.mean()),
    }


def _average_precision(probs: np.ndarray, labels: np.ndarray) -> float:
    positives = float((labels > 0.5).sum())
    if positives <= 0:
        return 0.0
    order = np.argsort(-probs)
    y = (labels[order] > 0.5).astype(np.float32)
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1, dtype=np.float32)
    return float((precision * y).sum() / positives)


def evaluate_probability_map(prob_map: np.ndarray, label: np.ndarray, fixed_threshold: float = 0.5, eval_cfg: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, float]]]:
    eval_cfg = eval_cfg or {}
    if prob_map.ndim != 2:
        raise ValueError(f"Expected probability map [H,W], got shape {prob_map.shape}")
    if label.ndim != 2:
        raise ValueError(f"Expected label [H,W], got shape {label.shape}")
    if prob_map.shape != label.shape:
        raise ValueError(f"Probability map shape {prob_map.shape} does not match label shape {label.shape}")
    probs = prob_map.reshape(-1).astype(np.float32)
    labels = label.reshape(-1).astype(np.float32)
    score_quantiles = np.quantile(probs, np.linspace(0.001, 0.999, 160, dtype=np.float32))
    thresholds = sorted(set(float(x) for x in np.concatenate([
        np.linspace(0.02, 0.95, 48, dtype=np.float32),
        score_quantiles.astype(np.float32),
        np.asarray([fixed_threshold], dtype=np.float32),
    ])))
    rows = [_binary_metrics(probs, labels, threshold) for threshold in thresholds]
    label_positive_rate = float((labels > 0.5).mean())
    max_ratio = eval_cfg.get("max_pred_positive_rate_ratio")
    min_ratio = eval_cfg.get("min_pred_positive_rate_ratio")
    constrained_rows = rows
    if max_ratio is not None or min_ratio is not None:
        max_ratio_value = float(max_ratio) if max_ratio is not None else float("inf")
        min_ratio_value = float(min_ratio) if min_ratio is not None else 0.0
        constrained_rows = [row for row in rows if min_ratio_value <= row["pred_positive_rate"] / max(label_positive_rate, 1e-12) <= max_ratio_value] or rows
    target_rate_raw = eval_cfg.get("target_pred_positive_rate", eval_cfg.get("positive_rate_loss_target"))
    if isinstance(target_rate_raw, str) and target_rate_raw.lower().strip() in {"auto", "auto_val", "val"}:
        target_rate = label_positive_rate
    elif isinstance(target_rate_raw, str) and target_rate_raw.lower().strip() in {"auto_train", "train"}:
        train_rate = eval_cfg.get("train_positive_rate")
        target_rate = float(train_rate) if train_rate is not None else None
    elif target_rate_raw is None:
        target_rate = None
    else:
        target_rate = float(target_rate_raw)

    def threshold_key(row: dict[str, float], metric: str) -> tuple[float, float, float, float]:
        target_distance = abs(row["pred_positive_rate"] - target_rate) if target_rate is not None else 0.0
        return (row[metric], -target_distance, row["precision"], -row["pred_positive_rate"])

    best_f1 = max(constrained_rows, key=lambda row: threshold_key(row, "f1"))
    best_f05 = max(rows, key=lambda row: threshold_key(row, "f05"))
    fixed = _binary_metrics(probs, labels, fixed_threshold)
    eps = 1e-7
    loss = float(-np.mean(labels * np.log(probs + eps) + (1.0 - labels) * np.log(1.0 - probs + eps)))
    metrics = {
        "tile_loss": loss,
        "tile_f1": float(best_f1["f1"]),
        "tile_f05": float(best_f05["f05"]),
        "val_loss": loss,
        "val_f1": float(best_f1["f1"]),
        "val_f05": float(best_f05["f05"]),
        "best_threshold": float(best_f1["threshold"]),
        "precision": float(best_f1["precision"]),
        "recall": float(best_f1["recall"]),
        "fixed_threshold": float(fixed_threshold),
        "fixed_threshold_f1": float(fixed["f1"]),
        "fixed_threshold_precision": float(fixed["precision"]),
        "fixed_threshold_recall": float(fixed["recall"]),
        "average_precision": _average_precision(probs, labels),
        "label_positive_rate": label_positive_rate,
        "val_positive_rate": label_positive_rate,
        "pred_positive_rate": float(best_f1["pred_positive_rate"]),
        "threshold_selection": "positive_rate_constrained" if constrained_rows is not rows else "best_f1",
        "max_pred_positive_rate_ratio": max_ratio,
        "min_pred_positive_rate_ratio": min_ratio,
        "target_pred_positive_rate": target_rate,
        "prob_min": float(np.min(probs)),
        "prob_mean": float(np.mean(probs)),
        "prob_p95": float(np.quantile(probs, 0.95)),
        "prob_max": float(np.max(probs)),
        "pixels": int(labels.size),
    }
    return metrics, rows


def write_outputs(output_dir: Path, prob_map: np.ndarray, metrics: dict[str, Any], threshold_rows: list[dict[str, float]], overwrite: bool = False) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prob_path = output_dir / "probability_map.npy"
    metrics_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics_by_threshold.csv"
    paths = [prob_path, metrics_path, csv_path]
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Refusing to overwrite existing output files without overwrite=True: " + ", ".join(str(path) for path in existing))
    np.save(prob_path, prob_map.astype(np.float32))
    metrics_path.write_text(json.dumps(_jsonable(metrics), indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["threshold", "precision", "recall", "f05", "f1", "pred_positive_rate"])
        writer.writeheader()
        writer.writerows(threshold_rows)
    return {"probability_map": str(prob_path), "metrics_json": str(metrics_path), "threshold_csv": str(csv_path)}


def _promotion_checks(cfg: dict[str, Any], segment_id: str) -> dict[str, Any]:
    setup = cfg.get("validation_setup") if isinstance(cfg.get("validation_setup"), dict) else {}
    resolved = cfg.get("resolved_data") if isinstance(cfg.get("resolved_data"), dict) else {}
    train_meta = resolved.get("train", {}).get("metadata", {}) if isinstance(resolved.get("train"), dict) else {}
    val_meta = resolved.get("val", {}).get("metadata", {}) if isinstance(resolved.get("val"), dict) else {}
    train_segment = str(setup.get("train_segment_id") or train_meta.get("segment_id") or "")
    val_segment = str(setup.get("val_segment_id") or val_meta.get("segment_id") or "")
    mode = str(setup.get("mode") or cfg.get("dataset", {}).get("validation_mode") or "unknown")
    warnings = []
    if train_segment and train_segment == str(segment_id):
        warnings.append("inference segment matches training segment; treat as diagnostic-only")
    if mode not in {"cross-segment", "cross-scroll", "leave-one-segment-out"}:
        warnings.append(f"validation mode is {mode}; promotion requires held-out segment evidence")
    if val_segment and val_segment != str(segment_id):
        warnings.append("inference segment does not match recorded validation segment")
    return {
        "eligible": not warnings,
        "warnings": warnings,
        "validation_mode": mode,
        "train_segment_id": train_segment or None,
        "val_segment_id": val_segment or None,
        "inference_segment_id": str(segment_id),
        "validation_setup": setup,
        "resolved_data": resolved,
    }


def run_full_tile_inference(artifact: Path, segment_id: str, output_dir: Path, level: str = "1", z_offsets: list[int] | None = None, patch_size: int | None = None, stride: int | None = None, batch_size: int = 8, device: str = "cpu", catalog_source: str = "public-directory", overwrite: bool = False, public_retry_count: int = 0, public_retry_delay_sec: float = 0.0, public_chunk_delay_sec: float = 0.0, public_chunk_retry_count: int = 0, public_chunk_retry_delay_sec: float = 0.0) -> dict[str, Any]:
    offsets = z_offsets if z_offsets is not None else [0]
    image, label, segment_meta = load_public_segment(segment_id, level, offsets, catalog_source, public_retry_count, public_retry_delay_sec, public_chunk_delay_sec, public_chunk_retry_count, public_chunk_retry_delay_sec)
    models, cfg, artifact_dir, model_meta = load_torch_unet_artifact(artifact, image.shape[0], device)
    patch = int(cfg.get("dataset", {}).get("patch_size", 64) if patch_size is None else patch_size)
    step = int(max(1, patch // 2) if stride is None else stride)
    _validate_tiling_args(patch, step, int(batch_size))
    fixed_threshold = float(cfg.get("evaluation", {}).get("threshold", 0.5))
    tta_flips = bool(model_meta.get("tta_flips", cfg.get("evaluation", {}).get("tta_flips", False)))
    prob_map, tile_meta = stitch_probabilities(models, image, patch, step, batch_size, device, tta_flips)
    tile_eval_cfg = {**cfg.get("evaluation", {})}
    if cfg.get("training", {}).get("positive_rate_loss_target") is not None:
        tile_eval_cfg.setdefault("positive_rate_loss_target", cfg.get("training", {}).get("positive_rate_loss_target"))
    train_meta = cfg.get("resolved_data", {}).get("train", {}) if isinstance(cfg.get("resolved_data"), dict) else {}
    if isinstance(train_meta, dict) and train_meta.get("positive_rate") is not None:
        tile_eval_cfg.setdefault("train_positive_rate", train_meta.get("positive_rate"))
    metrics, threshold_rows = evaluate_probability_map(prob_map, label, fixed_threshold, tile_eval_cfg)
    metrics.update({
        "model_name": str(model_meta.get("model_name") or cfg.get("model", {}).get("name")),
        "artifact_dir": str(artifact_dir),
        "model_meta": model_meta,
        "segment": segment_meta,
        "tile_inference": tile_meta,
        "evaluation_region": {
            "type": "whole_segment",
            "segment_id": str(segment_id),
            "shape": [int(label.shape[0]), int(label.shape[1])],
        },
        "inference_provenance": {
            "artifact": str(artifact),
            "artifact_dir": str(artifact_dir),
            "catalog_source": str(catalog_source),
            "device": str(device),
            "level": str(level),
            "z_offsets": [int(x) for x in offsets],
            "public_retry_count": int(public_retry_count),
            "public_retry_delay_sec": float(public_retry_delay_sec),
            "public_chunk_delay_sec": float(public_chunk_delay_sec),
            "public_chunk_retry_count": int(public_chunk_retry_count),
            "public_chunk_retry_delay_sec": float(public_chunk_retry_delay_sec),
            "overwrite": bool(overwrite),
        },
        "promotion_checks": _promotion_checks(cfg, segment_id),
    })
    outputs = write_outputs(output_dir, prob_map, metrics, threshold_rows, overwrite=overwrite)
    return {"metrics": metrics, "outputs": outputs}


def self_test() -> dict[str, Any]:
    image = np.zeros((1, 96, 80), dtype=np.float32)
    image[:, 24:72, 20:60] = 1.0
    label = image[0].copy()

    def predict_batch(patches: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip((patches.mean(axis=1) - 0.5) * 8.0, -30.0, 30.0)))

    prob_map, tile_meta = stitch_probabilities_from_predictor(predict_batch, image, patch_size=32, stride=16, batch_size=3)
    metrics, rows = evaluate_probability_map(prob_map, label, fixed_threshold=0.5)
    if prob_map.shape != label.shape:
        raise AssertionError(f"probability map shape mismatch: {prob_map.shape} vs {label.shape}")
    if not math.isclose(float(prob_map.min()), float(prob_map[0, 0]), rel_tol=0.0, abs_tol=1e-6):
        raise AssertionError("stitched probability map contains unexpected uncovered pixels")
    if metrics["fixed_threshold_f1"] < 0.99:
        raise AssertionError(f"self-test F1 too low: {metrics['fixed_threshold_f1']}")
    return {"tile_meta": tile_meta, "metrics": metrics, "threshold_rows": len(rows)}


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2, sort_keys=True))

"""Small, dependency-light experiment runner for ink detection.

Uses NumPy logistic regression over compact image features.  This is deliberately fast and
cron-safe; the module boundary makes it easy to swap `TinyInkModel` for a PyTorch U-Net when
that environment is available.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml

from data.vesuvius_data import prepare_training_subset, validate_prepared_npz

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "experiments" / "experiments.db"
RUNS_DIR = ROOT / "experiments" / "runs"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path): return str(obj)
    if isinstance(obj, (np.floating, np.integer)): return obj.item()
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_jsonable(v) for v in obj]
    return obj


def load_config(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r") as f:
        if str(path).endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f)


def _resolve_repo_path(raw: str | os.PathLike[str]) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else ROOT / path


def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                config_json TEXT NOT NULL,
                main_metric REAL NOT NULL,
                secondary_metrics_json TEXT NOT NULL,
                artifact_dir TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_metric ON experiments(main_metric)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_timestamp ON experiments(timestamp)")


def _features(images: np.ndarray, depth: int) -> np.ndarray:
    # images: [N,C,H,W].  Pixel-wise classifier with local, channel, and texture features.
    feats = []
    for channel in range(images.shape[1]):
        x = images[:, channel]
        feats.extend([x.reshape(-1), np.ones(x.size, dtype=np.float32) if channel == 0 else np.zeros(x.size, dtype=np.float32)])
        if depth >= 2:
            up = np.roll(x, 1, axis=1); down = np.roll(x, -1, axis=1); left = np.roll(x, 1, axis=2); right = np.roll(x, -1, axis=2)
            mean4 = (up + down + left + right) / 4.0
            feats += [mean4.reshape(-1), (x - mean4).reshape(-1)]
        if depth >= 3:
            gy, gx = np.gradient(x, axis=(1, 2))
            lap = np.roll(x, 1, axis=1) + np.roll(x, -1, axis=1) + np.roll(x, 1, axis=2) + np.roll(x, -1, axis=2) - 4.0 * x
            feats += [gx.reshape(-1), gy.reshape(-1), (gx * gx + gy * gy).reshape(-1), lap.reshape(-1)]
    if depth >= 2 and images.shape[1] > 1:
        center = images[:, images.shape[1] // 2]
        mean_depth = images.mean(axis=1)
        std_depth = images.std(axis=1)
        feats += [mean_depth.reshape(-1), (center - mean_depth).reshape(-1), std_depth.reshape(-1), (images[:, -1] - images[:, 0]).reshape(-1)]
    return np.stack(feats, axis=1).astype(np.float32)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _binary_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (probs >= threshold).astype(np.float32)
    tp = float(((pred == 1) & (labels == 1)).sum())
    fp = float(((pred == 1) & (labels == 0)).sum())
    fn = float(((pred == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    beta2 = 0.5 * 0.5
    f05 = (1 + beta2) * precision * recall / max(beta2 * precision + recall, 1e-9)
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


def _resolve_pos_weight(raw: Any, labels: np.ndarray) -> float:
    if isinstance(raw, str) and raw.lower() == "auto":
        pos = max(float((labels > 0.5).sum()), 1.0)
        neg = max(float((labels <= 0.5).sum()), 1.0)
        return float(min(np.sqrt(neg / pos), 10.0))
    return float(raw)


def _positive_rate_excess(pred_rate: float, target_rate: float, tolerance: float) -> float:
    return max(0.0, abs(float(pred_rate) - float(target_rate)) - float(tolerance))


def _resolve_positive_rate_target(raw: Any, train_labels: np.ndarray) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        label = raw.lower().strip()
        if label in {"auto", "auto_train", "train"}:
            return float(np.clip((train_labels.reshape(-1) > 0.5).mean(), 1e-6, 0.5))
        raise ValueError("training.positive_rate_loss_target must be numeric, auto, or auto_train")
    target = float(raw)
    if not 0.0 <= target <= 1.0:
        raise ValueError("training.positive_rate_loss_target must be in [0, 1]")
    return target


def _sample_patch_indices(images: np.ndarray, labels: np.ndarray, max_samples: int, seed: int, train_cfg: Dict[str, Any]) -> tuple[np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    count = images.shape[0]
    if max_samples <= 0 or count <= max_samples:
        return np.arange(count, dtype=np.int64), {"patch_sampling": "all", "selected_patches": int(count)}
    strategy = str(train_cfg.get("patch_sampling", train_cfg.get("sampling_strategy", "random")))
    if strategy != "hard_mining":
        return rng.choice(count, size=max_samples, replace=False), {"patch_sampling": "random", "selected_patches": int(max_samples)}

    ink_fraction = labels.reshape(count, -1).mean(axis=1)
    positive_idx = np.flatnonzero(ink_fraction > float(train_cfg.get("positive_patch_min_ink", 0.001)))
    negative_idx = np.flatnonzero(ink_fraction <= float(train_cfg.get("negative_patch_max_ink", 0.0)))
    positive_fraction = float(train_cfg.get("positive_patch_fraction", 0.5))
    hard_negative_fraction = float(train_cfg.get("hard_negative_fraction", 0.5))
    positive_take = min(len(positive_idx), int(round(max_samples * positive_fraction)))
    negative_take = max_samples - positive_take
    hard_negative_take = min(len(negative_idx), int(round(negative_take * hard_negative_fraction)))
    random_negative_take = max(0, negative_take - hard_negative_take)

    chosen_parts = []
    if positive_take:
        weights = ink_fraction[positive_idx].astype(np.float64)
        weights = weights / max(float(weights.sum()), 1e-12)
        chosen_parts.append(rng.choice(positive_idx, size=positive_take, replace=False, p=weights))
    if hard_negative_take:
        # High local variation negatives are more likely to resemble papyrus crackle/ink texture.
        center = images[:, images.shape[1] // 2].astype(np.float32)
        gy, gx = np.gradient(center, axis=(1, 2))
        texture_score = (gx * gx + gy * gy).reshape(count, -1).mean(axis=1)
        hard_pool = negative_idx[np.argsort(texture_score[negative_idx])[-max(hard_negative_take * 3, hard_negative_take):]]
        hard_negative_take = min(hard_negative_take, len(hard_pool))
        if hard_negative_take:
            chosen_parts.append(rng.choice(hard_pool, size=hard_negative_take, replace=False))
    if random_negative_take and len(negative_idx):
        already = set(np.concatenate(chosen_parts).tolist()) if chosen_parts else set()
        pool = np.asarray([i for i in negative_idx if int(i) not in already], dtype=np.int64)
        take = min(random_negative_take, len(pool))
        if take:
            chosen_parts.append(rng.choice(pool, size=take, replace=False))
    chosen = np.concatenate(chosen_parts) if chosen_parts else np.array([], dtype=np.int64)
    if len(chosen) < max_samples:
        remaining_pool = np.asarray([i for i in range(count) if int(i) not in set(chosen.tolist())], dtype=np.int64)
        fill = min(max_samples - len(chosen), len(remaining_pool))
        if fill:
            chosen = np.concatenate([chosen, rng.choice(remaining_pool, size=fill, replace=False)])
    rng.shuffle(chosen)
    return chosen.astype(np.int64), {
        "patch_sampling": "hard_mining",
        "selected_patches": int(len(chosen)),
        "positive_patches_available": int(len(positive_idx)),
        "negative_patches_available": int(len(negative_idx)),
        "positive_patches_selected": int(sum(ink_fraction[chosen] > 0.001)),
        "selected_patch_positive_rate": float((labels[chosen].reshape(len(chosen), -1).mean(axis=1) > 0.001).mean()) if len(chosen) else 0.0,
    }


def _pixel_metrics_from_probs(probs: np.ndarray, labels: np.ndarray, train_labels: np.ndarray, threshold: float, artifact_dir: Path, extra: Dict[str, Any], eval_cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    eval_cfg = eval_cfg or {}
    pv = probs.reshape(-1).astype(np.float32)
    yv = labels.reshape(-1).astype(np.float32)
    yt = train_labels.reshape(-1).astype(np.float32)
    eps = 1e-7
    val_loss = float(-np.mean(yv * np.log(pv + eps) + (1 - yv) * np.log(1 - pv + eps)))
    fixed = _binary_metrics(pv, yv, threshold)
    score_quantiles = np.quantile(pv, np.linspace(0.001, 0.999, 160, dtype=np.float32))
    sweep_thresholds = sorted(set(float(x) for x in np.concatenate([
        np.linspace(0.02, 0.95, 48, dtype=np.float32),
        score_quantiles.astype(np.float32),
        np.asarray([threshold], dtype=np.float32),
    ])))
    threshold_rows = [_binary_metrics(pv, yv, t) for t in sweep_thresholds]
    val_positive_rate = float((yv > 0.5).mean())
    max_ratio = eval_cfg.get("max_pred_positive_rate_ratio")
    min_ratio = eval_cfg.get("min_pred_positive_rate_ratio")
    constrained_rows = threshold_rows
    if max_ratio is not None or min_ratio is not None:
        max_ratio_value = float(max_ratio) if max_ratio is not None else float("inf")
        min_ratio_value = float(min_ratio) if min_ratio is not None else 0.0
        constrained_rows = [row for row in threshold_rows if min_ratio_value <= row["pred_positive_rate"] / max(val_positive_rate, 1e-12) <= max_ratio_value] or threshold_rows
    target_rate_raw = eval_cfg.get("target_pred_positive_rate", eval_cfg.get("positive_rate_loss_target"))
    if isinstance(target_rate_raw, str) and target_rate_raw.lower().strip() in {"auto", "auto_val", "val"}:
        target_rate = val_positive_rate
    elif target_rate_raw is None:
        target_rate = None
    else:
        target_rate = float(target_rate_raw)

    def threshold_key(row: Dict[str, float], metric: str) -> tuple[float, float, float, float]:
        target_distance = abs(row["pred_positive_rate"] - target_rate) if target_rate is not None else 0.0
        return (row[metric], -target_distance, row["precision"], -row["pred_positive_rate"])

    best_f1_row = max(constrained_rows, key=lambda row: threshold_key(row, "f1"))
    best_f05_row = max(threshold_rows, key=lambda row: threshold_key(row, "f05"))
    metrics = {
        "val_loss": val_loss,
        "val_f1": float(best_f1_row["f1"]),
        "val_f05": float(best_f05_row["f05"]),
        "best_threshold": float(best_f1_row["threshold"]),
        "precision": float(best_f1_row["precision"]),
        "recall": float(best_f1_row["recall"]),
        "fixed_threshold": threshold,
        "fixed_threshold_f1": float(fixed["f1"]),
        "fixed_threshold_precision": float(fixed["precision"]),
        "fixed_threshold_recall": float(fixed["recall"]),
        "average_precision": _average_precision(pv, yv),
        "train_positive_rate": float((yt > 0.5).mean()),
        "val_positive_rate": val_positive_rate,
        "pred_positive_rate": float(best_f1_row["pred_positive_rate"]),
        "threshold_selection": "positive_rate_constrained" if constrained_rows is not threshold_rows else "best_f1",
        "max_pred_positive_rate_ratio": max_ratio,
        "min_pred_positive_rate_ratio": min_ratio,
        "target_pred_positive_rate": target_rate,
        "prob_min": float(np.min(pv)),
        "prob_mean": float(np.mean(pv)),
        "prob_p95": float(np.quantile(pv, 0.95)),
        "prob_max": float(np.max(pv)),
        **extra,
    }
    csv_lines = ["threshold,precision,recall,f05,f1,pred_positive_rate"]
    for row in threshold_rows:
        csv_lines.append(",".join(str(row[key]) for key in ["threshold", "precision", "recall", "f05", "f1", "pred_positive_rate"]))
    (artifact_dir / "metrics_by_threshold.csv").write_text("\n".join(csv_lines) + "\n")
    (artifact_dir / "run_summary.md").write_text(
        "# Run Summary\n\n"
        f"- Main validation loss: {val_loss:.6f}\n"
        f"- Best F1: {metrics['val_f1']:.6f} at threshold {metrics['best_threshold']:.3f}\n"
        f"- Precision/recall at best F1: {metrics['precision']:.6f} / {metrics['recall']:.6f}\n"
        f"- Average precision: {metrics['average_precision']:.6f}\n"
        f"- Label positive rate train/val: {metrics['train_positive_rate']:.6f} / {metrics['val_positive_rate']:.6f}\n"
        f"- Probability max/p95/mean: {metrics['prob_max']:.6f} / {metrics['prob_p95']:.6f} / {metrics['prob_mean']:.6f}\n"
    )
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def _validation_setup(train_meta: Dict[str, Any], val_meta: Dict[str, Any]) -> Dict[str, Any]:
    train_data = train_meta.get("metadata", {}) if isinstance(train_meta.get("metadata"), dict) else {}
    val_data = val_meta.get("metadata", {}) if isinstance(val_meta.get("metadata"), dict) else {}
    train_scroll = str(train_data.get("scroll_id") or train_meta.get("scroll_id") or "?")
    val_scroll = str(val_data.get("scroll_id") or val_meta.get("scroll_id") or "?")
    train_segment = str(train_data.get("segment_id") or train_meta.get("segment_id") or "?")
    val_segment = str(val_data.get("segment_id") or val_meta.get("segment_id") or "?")
    train_segments_raw = train_data.get("train_segments") or train_meta.get("train_segments") or []
    train_segments = {str(segment) for segment in train_segments_raw if segment is not None} if isinstance(train_segments_raw, list) else set()
    heldout_segment = str(train_data.get("heldout_segment") or train_meta.get("heldout_segment") or "?")
    if train_scroll != "?" and val_scroll != "?" and train_scroll != val_scroll:
        mode = "cross-scroll"
        warning = None
    elif train_segments and val_segment != "?" and val_segment not in train_segments:
        mode = "leave-one-segment-out" if heldout_segment in {"?", val_segment} else "cross-segment"
        warning = None
    elif train_segments and val_segment != "?" and val_segment in train_segments:
        mode = "spatial-same-segment"
        warning = "Validation segment is present in train_segments metadata; this is not held-out validation."
    elif train_segment != "?" and val_segment != "?" and train_segment != val_segment:
        mode = "cross-segment"
        warning = None
    else:
        mode = "spatial-same-segment"
        warning = "Validation is a spatial split within one labeled segment, not cross-segment/cross-scroll."
    return {
        "mode": mode,
        "warning": warning,
        "train_scroll_id": train_scroll,
        "val_scroll_id": val_scroll,
        "train_segment_id": train_segment,
        "val_segment_id": val_segment,
        "train_segments": sorted(train_segments),
        "heldout_segment": heldout_segment,
    }


def _train_logreg(train_npz: str, val_npz: str, cfg: Dict[str, Any], artifact_dir: Path) -> Dict[str, Any]:
    tr = np.load(train_npz); va = np.load(val_npz)
    Xtr_img, ytr_img = tr["images"], tr["labels"]
    Xva_img, yva_img = va["images"], va["labels"]
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    lr = float(train_cfg.get("learning_rate", 0.2))
    epochs = int(train_cfg.get("epochs", 4))
    depth = int(model_cfg.get("depth", 2))
    l2 = float(train_cfg.get("weight_decay", 0.0))
    pos_weight_raw = train_cfg.get("pos_weight", 2.0)
    threshold = float(cfg.get("evaluation", {}).get("threshold", 0.5))

    X = _features(Xtr_img, depth)
    y = ytr_img.reshape(-1).astype(np.float32)
    pos_weight = _resolve_pos_weight(pos_weight_raw, y)
    Xv = _features(Xva_img, depth)
    yv = yva_img.reshape(-1).astype(np.float32)
    w = np.zeros(X.shape[1], dtype=np.float32)
    prior = np.clip(float((y > 0.5).mean()), 1e-4, 1.0 - 1e-4)
    # Feature column 1 is the constant intercept emitted by _features for channel 0.
    w[1] = np.log(prior / (1.0 - prior))
    sample_w = np.where(y > 0.5, pos_weight, 1.0).astype(np.float32)
    losses = []
    for epoch in range(epochs):
        p = _sigmoid(X @ w)
        grad = (X.T @ ((p - y) * sample_w)) / max(1, X.shape[0]) + l2 * w
        w -= lr * grad.astype(np.float32)
        eps = 1e-7
        loss = -np.mean(sample_w * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
        losses.append(float(loss))

    pv = _sigmoid(Xv @ w)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.save(artifact_dir / "weights.npy", w)
    return _pixel_metrics_from_probs(pv, yv, y, threshold, artifact_dir, {
        "model_name": "tiny_numpy_ink_logreg",
        "pos_weight_resolved": pos_weight,
        "initial_prior": float(prior),
        "train_loss_last": losses[-1] if losses else None,
        "epochs": epochs,
    }, cfg.get("evaluation", {}))


def _standardize_features(train_x: np.ndarray, val_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0).astype(np.float32)
    std = np.maximum(train_x.std(axis=0).astype(np.float32), 1e-6)
    return (train_x - mean) / std, (val_x - mean) / std, mean, std


def _train_numpy_mlp(train_npz: str, val_npz: str, cfg: Dict[str, Any], artifact_dir: Path) -> Dict[str, Any]:
    tr = np.load(train_npz); va = np.load(val_npz)
    Xtr_img, ytr_img = tr["images"], tr["labels"]
    Xva_img, yva_img = va["images"], va["labels"]
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    depth = int(model_cfg.get("depth", 3))
    hidden = int(model_cfg.get("hidden_units", 24))
    lr = float(train_cfg.get("learning_rate", 0.01))
    epochs = int(train_cfg.get("epochs", 8))
    batch_size = int(train_cfg.get("batch_size", 8192))
    max_pixels = int(train_cfg.get("max_train_pixels", 600000))
    sample_positive_fraction = train_cfg.get("sample_positive_fraction", None)
    l2 = float(train_cfg.get("weight_decay", 0.0001))
    seed = int(train_cfg.get("seed", 1337))
    pos_weight_raw = train_cfg.get("pos_weight", "auto")
    threshold = float(cfg.get("evaluation", {}).get("threshold", 0.5))

    X = _features(Xtr_img, depth)
    y = ytr_img.reshape(-1).astype(np.float32)
    Xv = _features(Xva_img, depth)
    yv = yva_img.reshape(-1).astype(np.float32)
    X, Xv, feat_mean, feat_std = _standardize_features(X, Xv)
    pos_weight = _resolve_pos_weight(pos_weight_raw, y)
    rng = np.random.default_rng(seed)
    if max_pixels > 0 and len(y) > max_pixels:
        if sample_positive_fraction is None:
            chosen = rng.choice(len(y), size=max_pixels, replace=False)
        else:
            pos_idx = np.flatnonzero(y > 0.5)
            neg_idx = np.flatnonzero(y <= 0.5)
            pos_take = min(len(pos_idx), int(round(max_pixels * float(sample_positive_fraction))))
            neg_take = max_pixels - pos_take
            chosen = np.concatenate([
                rng.choice(pos_idx, size=pos_take, replace=False) if pos_take else np.array([], dtype=np.int64),
                rng.choice(neg_idx, size=min(len(neg_idx), neg_take), replace=False),
            ])
            rng.shuffle(chosen)
        Xt = X[chosen]
        yt = y[chosen]
    else:
        Xt, yt = X, y

    scale1 = np.sqrt(2.0 / max(1, Xt.shape[1]))
    W1 = rng.normal(0.0, scale1, size=(Xt.shape[1], hidden)).astype(np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    W2 = rng.normal(0.0, 0.01, size=(hidden,)).astype(np.float32)
    prior = np.clip(float((yt > 0.5).mean()), 1e-4, 1.0 - 1e-4)
    b2 = float(np.log(prior / (1.0 - prior)))
    losses = []
    for _epoch in range(epochs):
        order = rng.permutation(len(yt))
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            xb = Xt[idx]
            yb = yt[idx]
            h = np.tanh(xb @ W1 + b1)
            logits = h @ W2 + b2
            p = _sigmoid(logits)
            sw = np.where(yb > 0.5, pos_weight, 1.0).astype(np.float32)
            err = ((p - yb) * sw) / max(1, len(yb))
            grad_W2 = h.T @ err + l2 * W2
            grad_b2 = float(err.sum())
            dh = err[:, None] * W2[None, :] * (1.0 - h * h)
            grad_W1 = xb.T @ dh + l2 * W1
            grad_b1 = dh.sum(axis=0)
            W2 -= lr * grad_W2.astype(np.float32)
            b2 -= lr * grad_b2
            W1 -= lr * grad_W1.astype(np.float32)
            b1 -= lr * grad_b1.astype(np.float32)
            eps = 1e-7
            epoch_losses.append(float(-np.mean(sw * (yb * np.log(p + eps) + (1 - yb) * np.log(1 - p + eps)))))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else None)

    probs = np.empty(len(yv), dtype=np.float32)
    for start in range(0, len(yv), 250000):
        xb = Xv[start:start + 250000]
        probs[start:start + len(xb)] = _sigmoid(np.tanh(xb @ W1 + b1) @ W2 + b2)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifact_dir / "mlp_weights.npz", W1=W1, b1=b1, W2=W2, b2=np.asarray([b2], dtype=np.float32), feature_mean=feat_mean, feature_std=feat_std)
    return _pixel_metrics_from_probs(probs, yv, y, threshold, artifact_dir, {
        "model_name": "tiny_numpy_mlp",
        "hidden_units": hidden,
        "max_train_pixels": max_pixels,
        "sample_positive_fraction": sample_positive_fraction,
        "trained_pixels": int(len(yt)),
        "pos_weight_resolved": pos_weight,
        "initial_prior": float(prior),
        "train_loss_last": losses[-1] if losses else None,
        "epochs": epochs,
    }, cfg.get("evaluation", {}))


def _train_torch_unet(train_npz: str, val_npz: str, cfg: Dict[str, Any], artifact_dir: Path) -> Dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError("model.name=tiny_torch_unet or residual_25d_torch_unet requires PyTorch installed in the project venv") from exc

    tr = np.load(train_npz); va = np.load(val_npz)
    Xtr_img, ytr_img = tr["images"].astype(np.float32), tr["labels"].astype(np.float32)
    Xva_img, yva_img = va["images"].astype(np.float32), va["labels"].astype(np.float32)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})
    model_name = str(model_cfg.get("name", "tiny_torch_unet"))
    base = int(model_cfg.get("base_channels", 8))
    epochs = int(train_cfg.get("epochs", 3))
    batch_size = int(train_cfg.get("batch_size", 8))
    lr = float(train_cfg.get("learning_rate", 0.001))
    weight_decay = float(train_cfg.get("weight_decay", 0.0001))
    max_train_samples = int(train_cfg.get("max_train_samples", 0))
    seed = int(train_cfg.get("seed", 1337))
    num_threads = int(train_cfg.get("num_threads", min(4, os.cpu_count() or 1)))
    dice_loss_weight = float(train_cfg.get("dice_loss_weight", 0.0))
    tversky_loss_weight = float(train_cfg.get("tversky_loss_weight", 0.0))
    tversky_alpha = float(train_cfg.get("tversky_alpha", 0.3))
    tversky_beta = float(train_cfg.get("tversky_beta", 0.7))
    focal_tversky_gamma = float(train_cfg.get("focal_tversky_gamma", 1.0))
    positive_rate_loss_weight = float(train_cfg.get("positive_rate_loss_weight", 0.0))
    positive_rate_loss_tolerance = float(train_cfg.get("positive_rate_loss_tolerance", 0.0))
    if positive_rate_loss_weight < 0.0:
        raise ValueError("training.positive_rate_loss_weight must be non-negative")
    if positive_rate_loss_tolerance < 0.0:
        raise ValueError("training.positive_rate_loss_tolerance must be non-negative")
    positive_rate_loss_target = _resolve_positive_rate_target(train_cfg.get("positive_rate_loss_target"), ytr_img)
    augment_flips = bool(train_cfg.get("augment_flips", False))
    tta_flips = bool(eval_cfg.get("tta_flips", eval_cfg.get("test_time_flips", False)))
    raw_seeds = train_cfg.get("seeds", None)
    ensemble_seeds = [int(s) for s in raw_seeds] if raw_seeds is not None else [seed]
    if not ensemble_seeds:
        raise ValueError("training.seeds must contain at least one seed when provided")
    pos_weight = _resolve_pos_weight(train_cfg.get("pos_weight", "auto"), ytr_img.reshape(-1))
    threshold = float(eval_cfg.get("threshold", 0.5))
    torch.set_num_threads(max(1, num_threads))
    device = torch.device("cuda" if bool(train_cfg.get("allow_cuda", False)) and torch.cuda.is_available() else "cpu")

    def seed_everything(run_seed: int) -> None:
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)
        if bool(train_cfg.get("deterministic", True)):
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    if max_train_samples > 0 and Xtr_img.shape[0] > max_train_samples:
        idx, sampling_metrics = _sample_patch_indices(Xtr_img, ytr_img, max_train_samples, seed, train_cfg)
        Xtr_img = Xtr_img[idx]
        ytr_img = ytr_img[idx]
    else:
        sampling_metrics = {"patch_sampling": "all", "selected_patches": int(Xtr_img.shape[0])}
    if augment_flips:
        Xtr_img = np.concatenate([Xtr_img, Xtr_img[:, :, :, ::-1], Xtr_img[:, :, ::-1, :]], axis=0).copy()
        ytr_img = np.concatenate([ytr_img, ytr_img[:, :, :, ::-1], ytr_img[:, :, ::-1, :]], axis=0).copy()

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
            return self.out(self.dec(torch.cat([x, skip], dim=1)))

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

    def build_model():
        if model_name == "tiny_torch_unet":
            return TinyUNet(Xtr_img.shape[1], base).to(device)
        if model_name in {"residual_25d_torch_unet", "torch_residual_25d_unet"}:
            return Residual25DUNet(Xtr_img.shape[1], base).to(device)
        raise ValueError(f"Unknown torch model.name: {model_name}")

    ds = TensorDataset(torch.from_numpy(Xtr_img), torch.from_numpy(ytr_img))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))

    def predict(model):
        model.eval()
        batch_probs = []
        with torch.no_grad():
            for start in range(0, Xva_img.shape[0], batch_size):
                x_np = Xva_img[start:start + batch_size]
                xb = torch.from_numpy(x_np).to(device)
                pred_sum = torch.sigmoid(model(xb))
                pred_count = 1
                if tta_flips:
                    pred_sum = pred_sum + torch.flip(torch.sigmoid(model(torch.flip(xb, dims=[3]))), dims=[3])
                    pred_sum = pred_sum + torch.flip(torch.sigmoid(model(torch.flip(xb, dims=[2]))), dims=[2])
                    pred_sum = pred_sum + torch.flip(torch.sigmoid(model(torch.flip(xb, dims=[2, 3]))), dims=[2, 3])
                    pred_count = 4
                batch_probs.append((pred_sum / pred_count).detach().cpu().numpy())
        return np.concatenate(batch_probs, axis=0)

    all_probs = []
    model_paths = []
    losses_by_seed = []
    prior_losses_by_seed = []
    soft_rates_by_seed = []
    def dice_loss(logits, target):
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probs * target).sum(dim=dims)
        denom = probs.sum(dim=dims) + target.sum(dim=dims)
        return (1.0 - ((2.0 * intersection + 1.0) / (denom + 1.0))).mean()

    def tversky_loss(logits, target):
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        tp = (probs * target).sum(dim=dims)
        fp = (probs * (1.0 - target)).sum(dim=dims)
        fn = ((1.0 - probs) * target).sum(dim=dims)
        score = (tp + 1.0) / (tp + tversky_alpha * fp + tversky_beta * fn + 1.0)
        loss = 1.0 - score
        if focal_tversky_gamma != 1.0:
            loss = loss.pow(focal_tversky_gamma)
        return loss.mean()

    for ensemble_idx, run_seed in enumerate(ensemble_seeds):
        seed_everything(run_seed)
        model = build_model()
        loader_generator = torch.Generator()
        loader_generator.manual_seed(run_seed)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=loader_generator, num_workers=0)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        losses = []
        prior_losses = []
        soft_rates = []
        for _epoch in range(epochs):
            model.train()
            epoch_losses = []
            epoch_prior_losses = []
            epoch_soft_rates = []
            for xb, yb in loader:
                xb = xb.to(device); yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                if positive_rate_loss_weight and positive_rate_loss_target is not None:
                    pred_rate = torch.sigmoid(logits).mean()
                    excess = torch.relu(torch.abs(pred_rate - positive_rate_loss_target) - positive_rate_loss_tolerance)
                    prior_loss = positive_rate_loss_weight * excess.pow(2)
                    loss = loss + prior_loss
                    epoch_prior_losses.append(float(prior_loss.detach().cpu()))
                    epoch_soft_rates.append(float(pred_rate.detach().cpu()))
                if dice_loss_weight:
                    loss = loss + dice_loss_weight * dice_loss(logits, yb)
                if tversky_loss_weight:
                    loss = loss + tversky_loss_weight * tversky_loss(logits, yb)
                loss.backward()
                opt.step()
                epoch_losses.append(float(loss.detach().cpu()))
            losses.append(float(np.mean(epoch_losses)) if epoch_losses else None)
            prior_losses.append(float(np.mean(epoch_prior_losses)) if epoch_prior_losses else 0.0)
            soft_rates.append(float(np.mean(epoch_soft_rates)) if epoch_soft_rates else None)
        losses_by_seed.append(losses)
        prior_losses_by_seed.append(prior_losses)
        soft_rates_by_seed.append(soft_rates)
        all_probs.append(predict(model))
        suffix = "" if len(ensemble_seeds) == 1 else f"_seed{run_seed}"
        model_path = artifact_dir / f"model{suffix}.pt"
        model_paths.append(model_path.name)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        if ensemble_idx == 0 and model_path.name != "model.pt":
            torch.save(model.state_dict(), artifact_dir / "model.pt")

    pv = np.mean(np.stack(all_probs, axis=0), axis=0).reshape(-1)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "model_meta.json").write_text(json.dumps({
        "model_name": model_name,
        "base_channels": base,
        "device": str(device),
        "ensemble_seeds": ensemble_seeds,
        "model_paths": model_paths,
        "tta_flips": tta_flips,
        "torch_version": torch.__version__,
        "train_samples": int(Xtr_img.shape[0]),
        "batch_size": batch_size,
        "positive_rate_loss_weight": positive_rate_loss_weight,
        "positive_rate_loss_target": positive_rate_loss_target,
        "positive_rate_loss_tolerance": positive_rate_loss_tolerance,
    }, indent=2, sort_keys=True))
    metrics_eval_cfg = {**eval_cfg}
    if positive_rate_loss_target is not None:
        metrics_eval_cfg.setdefault("positive_rate_loss_target", positive_rate_loss_target)
    return _pixel_metrics_from_probs(pv, yva_img.reshape(-1), ytr_img.reshape(-1), threshold, artifact_dir, {
        "model_name": model_name,
        "base_channels": base,
        "pos_weight_resolved": pos_weight,
        "train_loss_last": losses_by_seed[-1][-1] if losses_by_seed and losses_by_seed[-1] else None,
        "epochs": epochs,
        "train_samples_used": int(Xtr_img.shape[0]),
        "ensemble_size": len(ensemble_seeds),
        "ensemble_seeds": ensemble_seeds,
        "dice_loss_weight": dice_loss_weight,
        "tversky_loss_weight": tversky_loss_weight,
        "tversky_alpha": tversky_alpha,
        "tversky_beta": tversky_beta,
        "focal_tversky_gamma": focal_tversky_gamma,
        "positive_rate_loss_weight": positive_rate_loss_weight,
        "positive_rate_loss_target": positive_rate_loss_target,
        "positive_rate_loss_tolerance": positive_rate_loss_tolerance,
        "positive_rate_loss_last": prior_losses_by_seed[-1][-1] if prior_losses_by_seed and prior_losses_by_seed[-1] else None,
        "train_soft_positive_rate_last": soft_rates_by_seed[-1][-1] if soft_rates_by_seed and soft_rates_by_seed[-1] else None,
        "train_soft_positive_rate_error_last": abs(soft_rates_by_seed[-1][-1] - positive_rate_loss_target) if positive_rate_loss_target is not None and soft_rates_by_seed and soft_rates_by_seed[-1] and soft_rates_by_seed[-1][-1] is not None else None,
        "augment_flips": augment_flips,
        "tta_flips": tta_flips,
        **sampling_metrics,
        "device": str(device),
    }, metrics_eval_cfg)


def run_experiment(config_path: str | os.PathLike[str], db_path: Path = DB_PATH) -> Dict[str, Any]:
    cfg = load_config(config_path)
    dataset = cfg.setdefault("dataset", {})
    data_root = Path(dataset.get("prepared_root", ROOT / "data" / "prepared")).expanduser()
    patch_size = int(dataset.get("patch_size", 32))
    max_samples = int(dataset.get("max_samples_per_split", 48))
    train_scroll = str(dataset.get("train_scroll_id", "1"))
    val_scroll = str(dataset.get("val_scroll_id", "2" if train_scroll != "2" else "1"))
    train_npz = dataset.get("train_npz")
    val_npz = dataset.get("val_npz")
    if train_npz or val_npz:
        if not train_npz or not val_npz:
            raise ValueError("dataset.train_npz and dataset.val_npz must be provided together")
        train_meta = validate_prepared_npz(_resolve_repo_path(train_npz), split="train", patch_size=patch_size)
        val_meta = validate_prepared_npz(_resolve_repo_path(val_npz), split="val", patch_size=patch_size)
    else:
        train_meta = prepare_training_subset(train_scroll, "train", str(data_root / f"scroll_{train_scroll}_train_ps{patch_size}_n{max_samples}"), patch_size, max_samples)
        val_meta = prepare_training_subset(val_scroll, "val", str(data_root / f"scroll_{val_scroll}_val_ps{patch_size}_n{max_samples}"), patch_size, max_samples)
    cfg["resolved_data"] = {"train": train_meta, "val": val_meta}
    cfg["validation_setup"] = _validation_setup(train_meta, val_meta)
    raw = json.dumps(_jsonable(cfg), sort_keys=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + hashlib.sha1(raw.encode()).hexdigest()[:8]
    artifact_dir = RUNS_DIR / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "config.json").write_text(json.dumps(_jsonable(cfg), indent=2, sort_keys=True))
    model_name = cfg.get("model", {}).get("name", "tiny_numpy_ink_logreg")
    if model_name == "tiny_numpy_ink_logreg":
        metrics = _train_logreg(train_meta["path"], val_meta["path"], cfg, artifact_dir)
    elif model_name == "tiny_numpy_mlp":
        metrics = _train_numpy_mlp(train_meta["path"], val_meta["path"], cfg, artifact_dir)
    elif model_name in {"tiny_torch_unet", "residual_25d_torch_unet", "torch_residual_25d_unet"}:
        metrics = _train_torch_unet(train_meta["path"], val_meta["path"], cfg, artifact_dir)
    else:
        raise ValueError(f"Unknown model.name: {model_name}")
    main_name = cfg.get("evaluation", {}).get("main_metric", "val_loss")
    main_metric = float(metrics[main_name])
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO experiments(run_id,timestamp,config_json,main_metric,secondary_metrics_json,artifact_dir) VALUES(?,?,?,?,?,?)",
            (run_id, datetime.now(timezone.utc).isoformat(), raw, main_metric, json.dumps(metrics, sort_keys=True), str(artifact_dir)),
        )
    return {"run_id": run_id, "main_metric": main_metric, "metrics": metrics, "artifact_dir": str(artifact_dir), "db_path": str(db_path)}

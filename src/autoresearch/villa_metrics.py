# Adapted from ScrollPrize/villa (MIT License)
# Original authors: Youssef Nader, Luke Farritor, Julian Schilliger
# Source: https://github.com/ScrollPrize/villa/blob/main/ink-detection/metrics/binary_segmentation.py
# Changes: added optional mask support to confusion_counts for local unit tests and parity checks.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


def _safe_div(numer: torch.Tensor, denom: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return numer / (denom + eps)


@dataclass
class ConfusionCounts:
    tp: torch.Tensor
    fp: torch.Tensor
    fn: torch.Tensor


def confusion_counts(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> ConfusionCounts:
    """Compute TP/FP/FN for boolean tensors, using float64 reductions."""
    preds = preds.bool()
    targets = targets.bool()
    if mask is not None:
        mask = mask.bool()
        if mask.shape != targets.shape:
            raise ValueError(f"mask/targets shape mismatch: {tuple(mask.shape)} vs {tuple(targets.shape)}")
        preds = preds[mask]
        targets = targets[mask]
    tp = (preds & targets).sum(dtype=torch.float64)
    fp = (preds & ~targets).sum(dtype=torch.float64)
    fn = (~preds & targets).sum(dtype=torch.float64)
    return ConfusionCounts(tp=tp, fp=fp, fn=fn)


def dice_from_counts(c: ConfusionCounts) -> torch.Tensor:
    return _safe_div(2.0 * c.tp, 2.0 * c.tp + c.fp + c.fn)


class StreamingBinarySegmentationMetrics:
    """Streaming fixed-threshold Dice/F1 for binary segmentation logits."""

    def __init__(self, *, threshold: float = 0.5, device: torch.device | None = None) -> None:
        if not (0.0 <= float(threshold) <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.threshold = float(threshold)
        self._device = device
        self.reset(device=device)

    @property
    def device(self) -> torch.device:
        return self._device if self._device is not None else torch.device("cpu")

    def reset(self, *, device: torch.device | None = None) -> None:
        if device is not None:
            self._device = device
        dev = self.device
        self._counts = ConfusionCounts(
            tp=torch.zeros((), device=dev, dtype=torch.float64),
            fp=torch.zeros((), device=dev, dtype=torch.float64),
            fn=torch.zeros((), device=dev, dtype=torch.float64),
        )

    def update(self, *, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        if logits.shape != targets.shape:
            raise ValueError(f"logits/targets shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}")
        logits = logits.detach()
        targets = targets.detach()
        if mask is not None:
            mask = mask.detach().bool()
            if mask.shape != targets.shape:
                raise ValueError(f"mask/targets shape mismatch: {tuple(mask.shape)} vs {tuple(targets.shape)}")
            logits = logits[mask]
            targets = targets[mask]
        if targets.numel() == 0:
            return
        targets_bool = targets.to(dtype=torch.float32) >= 0.5
        probs = torch.sigmoid(logits).to(dtype=torch.float32)
        preds = probs >= float(self.threshold)
        batch_counts = confusion_counts(preds, targets_bool)
        self._counts.tp += batch_counts.tp
        self._counts.fp += batch_counts.fp
        self._counts.fn += batch_counts.fn

    def compute(self) -> Dict[str, torch.Tensor]:
        return {"dice": dice_from_counts(self._counts)}

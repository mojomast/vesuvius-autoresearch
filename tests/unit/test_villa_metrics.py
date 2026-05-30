import logging

import pytest
import torch

from experiments.runner import _warn_metric_discrepancy
from src.autoresearch.villa_metrics import StreamingBinarySegmentationMetrics, confusion_counts


# Metrics adapted from ScrollPrize/villa: https://github.com/ScrollPrize/villa/blob/main/ink-detection/metrics/binary_segmentation.py


def test_streaming_binary_segmentation_metrics_accumulates() -> None:
    metrics = StreamingBinarySegmentationMetrics(threshold=0.5)
    metrics.update(logits=torch.tensor([10.0, -10.0]), targets=torch.tensor([1.0, 0.0]))
    metrics.update(logits=torch.tensor([10.0, -10.0]), targets=torch.tensor([0.0, 1.0]))
    assert float(metrics.compute()["dice"]) == pytest.approx(0.5)


def test_confusion_counts_handles_masked_pixels() -> None:
    counts = confusion_counts(
        torch.tensor([True, True, False, False]),
        torch.tensor([True, False, True, False]),
        mask=torch.tensor([True, False, True, False]),
    )
    assert float(counts.tp) == 1.0
    assert float(counts.fp) == 0.0
    assert float(counts.fn) == 1.0


def test_confusion_counts_rejects_mask_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="mask/targets shape mismatch"):
        confusion_counts(torch.tensor([True]), torch.tensor([True]), mask=torch.ones((1, 1), dtype=torch.bool))


def test_streaming_metrics_rejects_invalid_inputs_and_empty_mask() -> None:
    with pytest.raises(ValueError, match="threshold"):
        StreamingBinarySegmentationMetrics(threshold=2.0)
    metrics = StreamingBinarySegmentationMetrics(threshold=0.5)
    with pytest.raises(ValueError, match="logits/targets shape mismatch"):
        metrics.update(logits=torch.ones(2), targets=torch.ones(1))
    with pytest.raises(ValueError, match="mask/targets shape mismatch"):
        metrics.update(logits=torch.ones(1), targets=torch.ones(1), mask=torch.ones((1, 1), dtype=torch.bool))
    metrics.update(logits=torch.ones(1), targets=torch.ones(1), mask=torch.zeros(1, dtype=torch.bool))
    assert float(metrics.compute()["dice"]) == 0.0


def test_metric_discrepancy_warning_fires(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="experiments.runner")
    _warn_metric_discrepancy(0.1, 0.2)
    assert "METRIC DISCREPANCY: local=0.1000 villa=0.2000" in caplog.text

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from experiments.runner import _torch_combo_loss, _torch_focal_bce_loss


def test_focal_bce_loss_downweights_easy_examples() -> None:
    target = torch.tensor([[[[1.0, 0.0]]]])
    easy_logits = torch.tensor([[[[4.0, -4.0]]]])
    hard_logits = torch.tensor([[[[0.1, -0.1]]]])

    assert _torch_focal_bce_loss(easy_logits, target, alpha=0.25, gamma=2.0) < _torch_focal_bce_loss(hard_logits, target, alpha=0.25, gamma=2.0)


def test_combo_loss_combines_bce_and_dice_terms() -> None:
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    logits = torch.tensor([[[[1.5, -1.5], [-0.5, 0.5]]]])

    bce_only = _torch_combo_loss(logits, target, bce_weight=1.0, dice_weight=0.0)
    dice_only = _torch_combo_loss(logits, target, bce_weight=0.0, dice_weight=1.0)
    mixed = _torch_combo_loss(logits, target, bce_weight=0.45, dice_weight=0.55)

    assert mixed.item() == pytest.approx(0.45 * bce_only.item() + 0.55 * dice_only.item())

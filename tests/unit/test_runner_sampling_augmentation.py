from __future__ import annotations

import numpy as np

from experiments.runner import _apply_training_augmentation, _curriculum_sampling_strategy


def test_rotation_augmentation_rotates_images_and_labels() -> None:
    images = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    labels = (images > 1).astype(np.float32)

    np.random.seed(1)
    aug_images, aug_labels = _apply_training_augmentation(images, labels, augment_rotation=True)

    assert aug_images.shape == images.shape
    assert aug_labels.shape == labels.shape
    np.testing.assert_array_equal(aug_images[0], np.rot90(images, k=1, axes=(-2, -1))[0])
    np.testing.assert_array_equal(aug_labels[0], np.rot90(labels, k=1, axes=(-2, -1))[0])


def test_rotation_augmentation_disabled_returns_original_batch() -> None:
    images = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    labels = (images > 1).astype(np.float32)

    aug_images, aug_labels = _apply_training_augmentation(images, labels, augment_rotation=False)

    np.testing.assert_array_equal(aug_images, images)
    np.testing.assert_array_equal(aug_labels, labels)


def test_sampling_curriculum_switches_to_hard_mining() -> None:
    curriculum = {"initial": "random", "final": "hard_mining", "switch_epoch": 2}

    assert _curriculum_sampling_strategy(curriculum, 0, 6) == "random"
    assert _curriculum_sampling_strategy(curriculum, 2, 6) == "hard_mining"
    assert _curriculum_sampling_strategy("warmup_then_hard", 1, 4) == "random"
    assert _curriculum_sampling_strategy("warmup_then_hard", 2, 4) == "hard_mining"


def test_sampling_curriculum_disabled_does_not_transition() -> None:
    assert _curriculum_sampling_strategy(None, 2, 4) is None

from __future__ import annotations

import pytest
import yaml

from src.autoresearch.schemas import ExperimentConfig, load_typed_config


def test_experiment_config_defaults_are_runtime_dict() -> None:
    cfg = ExperimentConfig()
    raw = cfg.to_runtime_dict()
    assert raw["model"]["name"] == "tiny_numpy_ink_logreg"
    assert raw["training"]["epochs"] == 5
    assert raw["evaluation"]["threshold"] == 0.5


def test_dataset_train_val_pair_required() -> None:
    with pytest.raises(ValueError, match="train_npz and dataset.val_npz"):
        ExperimentConfig.model_validate({"dataset": {"train_npz": "train.npz"}})


def test_training_seeds_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="training.seeds"):
        ExperimentConfig.model_validate({"training": {"seeds": []}})


def test_load_typed_config_validates_yaml(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"model": {"name": "tiny_numpy_mlp"}, "training": {"epochs": 3}}))
    cfg = load_typed_config(path)
    assert cfg.model.name == "tiny_numpy_mlp"
    assert cfg.training.epochs == 3

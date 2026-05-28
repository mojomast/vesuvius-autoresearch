#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
PIVOT_CONFIGS = (
    "robust_multisegment_dice035_expanded.yaml",
    "robust_calibrated_prloss_w0p03_lr0012_prratio3_seed11018.yaml",
    "robust_tta_seed_ensemble.yaml",
    "residual_25d_torch_unet_cpu.yaml",
)


def _rel(path: Path) -> str:
    """Return a repo-relative path string for generated configs."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _display(path: Path) -> str:
    """Return a readable path relative to the repo when possible."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write YAML with stable key order and parent directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _base_config(data_dir: Path) -> dict[str, Any]:
    """Build the focused-pair baseline config for prepared Vesuvius NPZs."""
    return {
        "dataset": {
            "research_scope": "focused_pair",
            "scope_note": "One train segment and one held-out validation segment; add folds only through explicit configs.",
            "train_npz": _rel(data_dir / "real_cross" / "segment_20230827161847" / "train.npz"),
            "val_npz": _rel(data_dir / "real_cross" / "segment_20230520175435" / "val.npz"),
            "patch_size": 64,
            "validation_mode": "cross-segment",
        },
        "model": {"name": "tiny_numpy_ink_logreg", "depth": 3},
        "training": {"epochs": 5, "learning_rate": 0.12, "weight_decay": 0.0001, "pos_weight": "auto"},
        "evaluation": {"main_metric": "val_f1", "threshold": 0.50},
        "outputs": {"runs_dir": "experiments/runs"},
    }


def _pivot_configs(data_dir: Path) -> dict[str, dict[str, Any]]:
    """Build robust pivot configs that unlock the torch-first search path."""
    expanded_train = _rel(data_dir / "real_cross_folds_expanded_combined" / "all_segments" / "train.npz")
    expanded_val = _rel(data_dir / "real_cross_folds_v2" / "segment_20230520175435" / "val.npz")
    focused_train = _rel(data_dir / "real_cross" / "segment_20230827161847_pf0p15_n2048" / "train.npz")
    focused_val = _rel(data_dir / "real_cross" / "segment_20230520175435" / "val.npz")
    common_torch = {
        "model": {"name": "tiny_torch_unet", "base_channels": 8},
        "training": {"epochs": 5, "batch_size": 8, "learning_rate": 0.0012, "weight_decay": 0.0001, "pos_weight": "auto", "max_train_samples": 0, "allow_cuda": False},
        "evaluation": {"main_metric": "val_f1", "threshold": 0.50},
        "outputs": {"runs_dir": "experiments/runs"},
    }
    return {
        "robust_multisegment_dice035_expanded.yaml": {
            **common_torch,
            "dataset": {"research_scope": "multi_segment_robust_expanded", "train_npz": expanded_train, "val_npz": expanded_val, "patch_size": 64, "validation_mode": "leave-one-segment-out"},
            "training": {**common_torch["training"], "dice_loss_weight": 0.35},
            "autoresearch": {"scope_policy": "expanded_multi_segment_leave_one_out"},
        },
        "robust_calibrated_prloss_w0p03_lr0012_prratio3_seed11018.yaml": {
            **common_torch,
            "dataset": {"research_scope": "multi_segment_robust_expanded_prratio3_seed11018", "train_npz": expanded_train, "val_npz": expanded_val, "patch_size": 64, "validation_mode": "leave-one-segment-out"},
            "training": {**common_torch["training"], "seed": 11018, "positive_rate_loss_weight": 0.03, "positive_rate_loss_target": "auto_train", "positive_rate_loss_tolerance": 0.02},
            "evaluation": {"main_metric": "val_f1", "threshold": 0.50, "max_pred_positive_rate_ratio": 3.0, "min_pred_positive_rate_ratio": 0.1, "target_pred_positive_rate": "auto_val"},
            "autoresearch": {"scope_policy": "expanded_multi_segment_leave_one_out"},
        },
        "robust_tta_seed_ensemble.yaml": {
            **common_torch,
            "dataset": {"research_scope": "multi_segment_robust_expanded_tta_ensemble", "train_npz": expanded_train, "val_npz": expanded_val, "patch_size": 64, "validation_mode": "leave-one-segment-out"},
            "training": {**common_torch["training"], "seeds": [11001, 11013, 11029], "deterministic": True},
            "evaluation": {"main_metric": "val_f1", "threshold": 0.50, "tta_flips": True},
            "autoresearch": {"scope_policy": "expanded_multi_segment_leave_one_out"},
        },
        "residual_25d_torch_unet_cpu.yaml": {
            "dataset": {"research_scope": "focused_pair_residual_25d_cpu", "train_npz": focused_train, "val_npz": focused_val, "patch_size": 64, "validation_mode": "cross-segment"},
            "model": {"name": "residual_25d_torch_unet", "base_channels": 8},
            "training": {"epochs": 5, "batch_size": 4, "learning_rate": 0.0012, "weight_decay": 0.0001, "pos_weight": "auto", "max_train_samples": 512, "deterministic": True, "augment_flips": True, "allow_cuda": False},
            "evaluation": {"main_metric": "val_f1", "threshold": 0.50, "tta_flips": True},
            "autoresearch": {"scope_policy": "focused_pair_only"},
            "outputs": {"runs_dir": "experiments/runs"},
        },
    }


def validate_config_schema(config: dict[str, Any]) -> None:
    """Validate the minimal config schema required by the runner and autoresearch."""
    required = (("dataset", "train_npz"), ("dataset", "val_npz"), ("dataset", "patch_size"), ("model", "name"), ("training", "epochs"), ("training", "learning_rate"), ("training", "weight_decay"), ("training", "pos_weight"), ("evaluation", "main_metric"), ("evaluation", "threshold"), ("outputs", "runs_dir"))
    for section, key in required:
        if section not in config or key not in config[section]:
            raise ValueError(f"missing required config key: {section}.{key}")


def generate_configs(data_dir: Path, configs_dir: Path | None = None) -> list[Path]:
    """Generate baseline and pivot configs for a prepared data directory."""
    configs_dir = CONFIGS if configs_dir is None else configs_dir
    configs = {"baseline.yaml": _base_config(data_dir), **_pivot_configs(data_dir)}
    written: list[Path] = []
    for name, config in configs.items():
        validate_config_schema(config)
        path = configs_dir / name
        _write_yaml(path, config)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    """Generate local data-path configs and print a setup summary."""
    parser = argparse.ArgumentParser(description="Set up Vesuvius AutoResearch data configs")
    parser.add_argument("--data-dir", default="./data", help="Prepared Vesuvius NPZ data directory")
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir)
    print("Data download/symlink step: TODO requires ScrollPrize/public mirror access; expected prepared NPZ paths will be written into configs.")
    written = generate_configs(data_dir)
    print(f"Generated {len(written)} config file(s):")
    for path in written:
        print(f"  {_display(path)}")
    print("Success: configs generated and schema-validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

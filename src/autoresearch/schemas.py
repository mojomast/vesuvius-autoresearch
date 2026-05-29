from __future__ import annotations
# mypy: ignore-errors

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PathLike = str | Path


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = "tiny_numpy_ink_logreg"
    input_mode: str | None = None
    depth: int = Field(default=2, ge=1, le=3)
    hidden_units: int | None = Field(default=24, ge=1)
    base_channels: int | None = Field(default=None, ge=1)


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    research_scope: str | None = None
    scope_note: str | None = None
    train_npz: PathLike | None = None
    val_npz: PathLike | None = None
    extra_train_npzs: list[PathLike] = Field(default_factory=list)
    prepared_root: PathLike | None = None
    patch_size: int = Field(default=32, ge=1)
    max_samples_per_split: int = Field(default=48, ge=1)
    train_scroll_id: str = "1"
    val_scroll_id: str | None = None
    validation_mode: str | None = None
    z_offsets: list[int] | None = None

    @field_validator("extra_train_npzs", mode="before")
    @classmethod
    def _normalize_extra_train_npzs(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, (str, Path)):
            return [value]
        return value

    @model_validator(mode="after")
    def _require_train_val_pair(self) -> "DatasetConfig":
        if bool(self.train_npz) ^ bool(self.val_npz):
            raise ValueError("dataset.train_npz and dataset.val_npz must be provided together")
        return self


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    epochs: int = Field(default=5, ge=1)
    batch_size: int | None = Field(default=None, ge=1)
    learning_rate: float = Field(default=0.2, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    pos_weight: float | Literal["auto"] = "auto"
    max_train_samples: int | None = Field(default=None, ge=0)
    max_train_pixels: int | None = Field(default=600000, ge=0)
    sample_positive_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    seed: int = 1337
    seeds: list[int] | None = None
    deterministic: bool | None = None
    allow_cuda: bool = False
    num_threads: int | None = Field(default=None, ge=1)
    augment_flips: bool = False
    sampling_strategy: str | None = None
    patch_sampling: str | None = None
    hard_negative_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    positive_patch_min_ink: float = Field(default=0.001, ge=0.0, le=1.0)
    negative_patch_max_ink: float = Field(default=0.0, ge=0.0, le=1.0)
    positive_patch_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    dice_loss_weight: float = Field(default=0.0, ge=0.0)
    positive_rate_loss_weight: float = Field(default=0.0, ge=0.0)
    positive_rate_loss_tolerance: float = Field(default=0.0, ge=0.0)
    positive_rate_loss_target: float | Literal["auto", "auto_train", "train"] | None = None
    tversky_loss_weight: float = Field(default=0.0, ge=0.0)
    tversky_alpha: float = Field(default=0.3, ge=0.0, le=1.0)
    tversky_beta: float = Field(default=0.7, ge=0.0, le=1.0)
    focal_tversky_gamma: float = Field(default=1.0, ge=0.0)

    @field_validator("seeds")
    @classmethod
    def _seeds_non_empty(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and not value:
            raise ValueError("training.seeds must contain at least one seed when provided")
        return value


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    main_metric: str = "val_loss"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    tta_flips: bool = False
    test_time_flips: bool | None = None
    calibration_bins: int = Field(default=15, ge=1)
    max_pred_positive_rate_ratio: float | None = Field(default=None, gt=0.0)
    min_pred_positive_rate_ratio: float | None = Field(default=None, ge=0.0)
    target_pred_positive_rate: float | Literal["auto", "auto_val", "val"] | None = None

    @model_validator(mode="after")
    def _ratios_are_ordered(self) -> "EvaluationConfig":
        if self.max_pred_positive_rate_ratio is not None and self.min_pred_positive_rate_ratio is not None and self.min_pred_positive_rate_ratio > self.max_pred_positive_rate_ratio:
            raise ValueError("evaluation.min_pred_positive_rate_ratio must be <= max_pred_positive_rate_ratio")
        return self


class AutoresearchConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    scope_policy: str | None = None
    parent_reason: str | None = None
    parent_recent_winner: str | None = None
    search_signature: list[Any] | None = None
    intent: str | None = None
    run_profile: str | None = None
    promotable: bool | None = None
    proposal_status: str | None = None
    changed_path: str | None = None
    mutation_family: str | None = None
    cost_tier: str | None = None
    strategy_phase: str | None = None
    promotion_required: list[str] = Field(default_factory=list)
    promotion_action_id: str | None = None
    heldout_segment: str | None = None
    cron_safety: dict[str, Any] | None = None


class OutputsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    runs_dir: PathLike = "experiments/runs"


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    autoresearch: AutoresearchConfig = Field(default_factory=AutoresearchConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)
    resolved_data: dict[str, Any] | None = None
    validation_setup: dict[str, Any] | None = None

    def to_runtime_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python", exclude_none=True)


def load_typed_config(path: str | Path) -> ExperimentConfig:
    from experiments.runner import load_config

    return ExperimentConfig.model_validate(load_config(path))


__all__ = ["ExperimentConfig", "load_typed_config"]

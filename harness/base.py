from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """Proposed experiment config and explanatory metadata."""

    name: str
    config: dict[str, Any]
    reason: str = ""


@dataclass(frozen=True)
class RunRecord:
    """Historical experiment record consumed by a research harness."""

    run_id: str
    config: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    main_metric: float | None = None
    artifact_dir: str | None = None


@dataclass(frozen=True)
class RunResult:
    """Result returned by a harness experiment evaluation."""

    run_id: str
    metrics: dict[str, Any]
    main_metric: float
    artifact_dir: str | None = None


class ResearchHarness(ABC):
    """Abstract interface for pluggable autoresearch harnesses."""

    @abstractmethod
    def propose_next_experiment(self, history: list[RunRecord]) -> ExperimentConfig:
        """Return the next experiment to evaluate."""

    @abstractmethod
    def evaluate_experiment(self, config: ExperimentConfig) -> RunResult:
        """Evaluate an experiment config and return its result."""

    @abstractmethod
    def should_promote(self, result: RunResult, history: list[RunRecord]) -> bool:
        """Return whether a result satisfies harness promotion criteria."""

    @abstractmethod
    def on_promotion(self, config: ExperimentConfig, result: RunResult) -> None:
        """Handle promotion side effects for a successful result."""

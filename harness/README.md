# Research Harness

`ResearchHarness` defines the minimal lifecycle for an autoresearch loop: propose an experiment, evaluate it, decide whether to promote it, and handle promotion side effects.

`VesuviusHarness` is a thin adapter over the existing `autoresearch.py` functions. New ScrollPrize harnesses should subclass `ResearchHarness`, define typed history/result conversion, and keep promotion gates explicit so dashboard and cron automation can inspect decisions.

## Quick Start

Implement `ResearchHarness` for a new research loop by returning typed configs and results while keeping promotion decisions explicit:

```python
from harness import ExperimentConfig, ResearchHarness, RunRecord, RunResult


class CustomHarness(ResearchHarness):
    def propose_next_experiment(self, history: list[RunRecord]) -> ExperimentConfig:
        return ExperimentConfig(
            name="custom.yaml",
            config={"model": {"name": "custom"}},
            reason="try custom baseline",
        )

    def evaluate_experiment(self, config: ExperimentConfig) -> RunResult:
        return RunResult(run_id="custom-1", metrics={"val_f1": 0.1}, main_metric=0.1)

    def should_promote(self, result: RunResult, history: list[RunRecord]) -> bool:
        return result.main_metric >= 0.5

    def on_promotion(self, config: ExperimentConfig, result: RunResult) -> None:
        return None
```

## Method Reference

| Method | Signature | Purpose |
| --- | --- | --- |
| `propose_next_experiment` | `(history: list[RunRecord]) -> ExperimentConfig` | Select the next config to evaluate from prior run history. |
| `evaluate_experiment` | `(config: ExperimentConfig) -> RunResult` | Execute or adapt the experiment runner and return metrics. |
| `should_promote` | `(result: RunResult, history: list[RunRecord]) -> bool` | Decide whether the result satisfies promotion criteria. |
| `on_promotion` | `(config: ExperimentConfig, result: RunResult) -> None` | Run any side effects after a promotion decision. |

## Extension Pattern

Keep harnesses thin: convert local run history into `RunRecord`, call the domain-specific runner, convert outputs into `RunResult`, and leave storage or dashboard-specific rendering outside the harness. `VesuviusHarness` shows how to wrap existing AutoResearch functions without changing their behavior.

## Deployment Note

Warning: AutoResearch initializes the harness before pruning configs or planning work. A broken harness import prevents AutoResearch from running at all; test harness imports before deploying changes.

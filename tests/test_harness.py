from __future__ import annotations

from harness import ExperimentConfig, ResearchHarness, RunRecord, RunResult
from harness.vesuvius_harness import VesuviusHarness


class MockHarness(ResearchHarness):
    def propose_next_experiment(self, history: list[RunRecord]) -> ExperimentConfig:
        return ExperimentConfig(name="mock.yaml", config={"model": {"name": "mock"}}, reason="test")

    def evaluate_experiment(self, config: ExperimentConfig) -> RunResult:
        return RunResult(run_id="run-1", metrics={"val_f1": 1.0}, main_metric=1.0)

    def should_promote(self, result: RunResult, history: list[RunRecord]) -> bool:
        return result.main_metric >= 1.0

    def on_promotion(self, config: ExperimentConfig, result: RunResult) -> None:
        return None


def test_research_harness_interface_with_mock():
    harness = MockHarness()
    config = harness.propose_next_experiment([])
    result = harness.evaluate_experiment(config)

    assert config.name == "mock.yaml"
    assert harness.should_promote(result, []) is True


def test_vesuvius_harness_instantiates():
    harness = VesuviusHarness()

    assert harness.harness_type == "vesuvius"
    assert harness.autoresearch is not None

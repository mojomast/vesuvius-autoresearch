from __future__ import annotations

import importlib
import subprocess
import sys
from types import ModuleType
from typing import Any

from .base import ExperimentConfig, ResearchHarness, RunRecord, RunResult


class VesuviusHarness(ResearchHarness):
    """ResearchHarness adapter over the existing `autoresearch.py` workflow."""

    harness_type = "vesuvius"

    def __init__(self, autoresearch_module: ModuleType | None = None) -> None:
        """Create a harness using the provided autoresearch module or import it."""
        self.autoresearch = autoresearch_module or importlib.import_module("autoresearch")

    def propose_next_experiment(self, history: list[RunRecord]) -> ExperimentConfig:
        """Return the first current AutoResearch proposal for compatibility."""
        runs = [self._run_record_to_dict(item) for item in history]
        base = self.best_base_config(runs)
        proposals, _source_action = self.promotion_or_fallback_proposals(runs, base, None, 1)
        if not proposals:
            raise RuntimeError("no experiment proposals available")
        name, config, reason = proposals[0]
        return ExperimentConfig(name=name, config=config, reason=reason)

    def evaluate_experiment(self, config: ExperimentConfig) -> RunResult:
        """Run the existing experiment entrypoint for a generated config file."""
        path = self.autoresearch.CONFIGS / config.name
        self.autoresearch._dump_config_with_comment(path, config.config, config.reason)
        subprocess.run([sys.executable, "run_experiment.py", "--config", str(path)], cwd=self.autoresearch.ROOT, check=True)
        run = self.autoresearch._recent_runs(limit=1)[0]
        return RunResult(run_id=str(run["run_id"]), metrics=run["metrics"], main_metric=float(run["main_metric"]), artifact_dir=run.get("artifact_dir"))

    def should_promote(self, result: RunResult, history: list[RunRecord]) -> bool:
        """Delegate promotion eligibility to the existing promotion gate."""
        run = {"run_id": result.run_id, "metrics": result.metrics, "main_metric": result.main_metric, "artifact_dir": result.artifact_dir, "config": {}}
        eligible, _warnings = self.autoresearch._promotion_gate(run)
        return eligible

    def on_promotion(self, config: ExperimentConfig, result: RunResult) -> None:
        """No-op hook; existing promotion automation records evidence separately."""
        return None

    def best_base_config(self, runs: list[dict[str, Any]]) -> dict[str, Any]:
        """Return the existing AutoResearch best base config."""
        return self.autoresearch._best_base_config(runs)

    def promotion_ready_payload(self) -> dict[str, Any] | None:
        """Return current dashboard promotion-ready payload if any."""
        return self.autoresearch._promotion_ready_payload()

    def promotion_phase_manual_action(self, runs: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Return existing promotion action payload for plateaued runs."""
        return self.autoresearch._promotion_phase_manual_action(runs)

    def promotion_or_fallback_proposals(self, runs: list[dict[str, Any]], base: dict[str, Any], ready_payload: dict[str, Any] | None, count: int):
        """Return existing promotion-action or exploration proposals."""
        return self.autoresearch._promotion_or_fallback_proposals(runs, base, ready_payload, count)

    @staticmethod
    def _run_record_to_dict(record: RunRecord) -> dict[str, Any]:
        """Convert a typed RunRecord into the legacy dict shape."""
        return {"run_id": record.run_id, "config": record.config, "metrics": record.metrics, "main_metric": record.main_metric, "artifact_dir": record.artifact_dir}

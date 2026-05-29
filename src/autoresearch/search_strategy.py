from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

Config = dict[str, Any]
Run = dict[str, Any]
Candidate = tuple[tuple[str, ...], Any, str]
Proposal = tuple[str, Config, str]


@dataclass(frozen=True)
class SearchContext:
    proposal_count: int
    strategy_phase: str | None = None
    required_families: set[str] | None = None
    allow_expensive: bool = False
    scope_policy: str = "focused_pair_only"
    lock_to_baseline_scope: bool = True
    name_index_offset: int = 0


class SearchStrategy(Protocol):
    name: str

    def order_candidates(self, candidates: Sequence[Candidate], runs: Sequence[Run]) -> list[Candidate]:
        ...


@dataclass
class HeuristicSearchStrategy:
    name: str = "heuristic"

    def order_candidates(self, candidates: Sequence[Candidate], runs: Sequence[Run]) -> list[Candidate]:
        return list(candidates)


@dataclass
class OptunaBayesianSearchStrategy:
    name: str = "bayesian"

    def order_candidates(self, candidates: Sequence[Candidate], runs: Sequence[Run]) -> list[Candidate]:
        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError("AUTORESEARCH_SEARCH_STRATEGY=bayesian requires installing the optional search extra: pip install -e '.[search]'") from exc
        if not candidates:
            return []
        sampler = optuna.samplers.TPESampler(seed=20260529, multivariate=True, group=True)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        for run in runs:
            metrics = run.get("metrics", {}) if isinstance(run.get("metrics"), dict) else {}
            value = float(metrics.get("val_f1") or run.get("main_metric") or 0.0)
            trial = optuna.trial.create_trial(params={"candidate_index": 0}, distributions={"candidate_index": optuna.distributions.IntDistribution(0, len(candidates) - 1)}, value=value)
            try:
                study.add_trial(trial)
            except ValueError:
                continue
        trial = study.ask({"candidate_index": optuna.distributions.IntDistribution(0, len(candidates) - 1)})
        first = int(trial.params.get("candidate_index", 0)) % len(candidates)
        ordered = list(candidates)
        return ordered[first:] + ordered[:first]


def strategy_from_env(name: str | None) -> SearchStrategy:
    normalized = (name or "heuristic").strip().lower()
    if normalized in {"", "heuristic", "random", "current", "default"}:
        return HeuristicSearchStrategy()
    if normalized in {"bayesian", "optuna", "tpe"}:
        return OptunaBayesianSearchStrategy()
    raise ValueError(f"unsupported AUTORESEARCH_SEARCH_STRATEGY={name!r}; expected heuristic or bayesian")


__all__ = ["Candidate", "Config", "HeuristicSearchStrategy", "OptunaBayesianSearchStrategy", "Proposal", "Run", "SearchContext", "SearchStrategy", "strategy_from_env"]

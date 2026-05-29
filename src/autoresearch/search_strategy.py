from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
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
    storage_path: Path = Path("logs/optuna.db")
    study_name: str = "autoresearch:default"
    multi_objective: bool = False
    last_trial_metadata: dict[str, Any] = field(default_factory=dict)

    def _load_optuna(self) -> Any:  # pragma: no cover - exercised by integration plan validation when Optuna is installed.
        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError("AUTORESEARCH_SEARCH_STRATEGY=bayesian requires installing the optional search extra: pip install -e '.[search]'") from exc
        return optuna

    def _study(self, optuna: Any, scope: str, model_name: str) -> Any:  # pragma: no cover
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.study_name = f"autoresearch:{'bayesian_multi' if self.multi_objective else 'bayesian'}:{scope}:{model_name}"
        storage = f"sqlite:///{self.storage_path}"
        if self.multi_objective:
            sampler = optuna.samplers.NSGAIISampler(seed=20260529)
            return optuna.create_study(study_name=self.study_name, storage=storage, load_if_exists=True, directions=["maximize", "maximize", "maximize"], sampler=sampler)
        sampler = optuna.samplers.TPESampler(seed=20260529, multivariate=True, group=True)
        return optuna.create_study(study_name=self.study_name, storage=storage, load_if_exists=True, direction="maximize", sampler=sampler)

    @staticmethod
    def _metric_objectives(run: Run) -> float | list[float]:  # pragma: no cover
        metrics = run.get("metrics", {}) if isinstance(run.get("metrics"), dict) else {}
        val_f1 = float(metrics.get("val_f1") or run.get("main_metric") or 0.0)
        ap = float(metrics.get("average_precision") or 0.0)
        pred_rate = metrics.get("pred_positive_rate")
        val_rate = metrics.get("val_positive_rate")
        ratio = float(pred_rate) / max(float(val_rate), 1e-12) if pred_rate is not None and val_rate is not None else 1.0
        calibration = 1.0 - min(abs(ratio - 1.0), 1.0)
        return [val_f1, ap, calibration]

    @staticmethod
    def _quality_score(run: Run) -> float:  # pragma: no cover
        metrics = run.get("metrics", {}) if isinstance(run.get("metrics"), dict) else {}
        val_f1 = float(metrics.get("val_f1") or run.get("main_metric") or 0.0)
        ap = float(metrics.get("average_precision") or 0.0)
        f05 = float(metrics.get("val_f05") or 0.0)
        pred_rate = metrics.get("pred_positive_rate")
        val_rate = metrics.get("val_positive_rate")
        penalty = 0.0
        if pred_rate is not None and val_rate is not None:
            ratio = float(pred_rate) / max(float(val_rate), 1e-12)
            penalty = min(abs(ratio - 1.0), 0.5) * 0.05
        return val_f1 + 0.25 * ap + 0.10 * f05 - penalty

    @staticmethod
    def _config_param(cfg: Config, path: tuple[str, ...]) -> Any:  # pragma: no cover
        value: Any = cfg
        for key in path:
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
        return value

    @staticmethod
    def _set_config_param(cfg: Config, path: tuple[str, ...], value: Any) -> None:  # pragma: no cover
        current = cfg
        for key in path[:-1]:
            current = current.setdefault(key, {})
        current[path[-1]] = value

    def _backfill_trials(self, optuna: Any, study: Any, runs: Sequence[Run], distributions: dict[str, Any], paths_by_name: dict[str, tuple[str, ...]]) -> None:  # pragma: no cover
        known_run_ids = {trial.user_attrs.get("run_id") for trial in getattr(study, "trials", []) if trial.user_attrs.get("run_id")}
        for run in runs:
            run_id = str(run.get("run_id") or "")
            cfg = run.get("config", {}) if isinstance(run.get("config"), dict) else {}
            if not run_id or run_id in known_run_ids:
                continue
            params: dict[str, Any] = {}
            trial_distributions: dict[str, Any] = {}
            for name, path in paths_by_name.items():
                value = self._config_param(cfg, path)
                if isinstance(value, bool) or value is None or isinstance(value, str):
                    continue
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                params[name] = value
                trial_distributions[name] = distributions[name]
            if not params:
                continue
            kwargs: dict[str, Any] = {"params": params, "distributions": trial_distributions, "user_attrs": {"run_id": run_id}}
            if self.multi_objective:
                kwargs["values"] = self._metric_objectives(run)
            else:
                kwargs["value"] = self._quality_score(run)
            try:
                study.add_trial(optuna.trial.create_trial(**kwargs))
            except ValueError:
                continue

    def propose_candidate(self, base: Config, candidates: Sequence[Candidate], runs: Sequence[Run], context: SearchContext, param_bounds: dict[tuple[str, ...], tuple[float, float]]) -> Candidate | None:  # pragma: no cover
        if not candidates:
            return None
        optuna = self._load_optuna()
        scope = str(base.get("autoresearch", {}).get("scope_policy") or base.get("dataset", {}).get("research_scope") or context.scope_policy)
        model_name = str(base.get("model", {}).get("name") or "unknown")
        study = self._study(optuna, scope, model_name)
        candidate_paths = [path for path, _value, _reason in candidates if path in param_bounds]
        if not candidate_paths:
            return None
        paths_by_name = {"__".join(path): path for path in candidate_paths}
        distributions = {name: optuna.distributions.FloatDistribution(float(param_bounds[path][0]), float(param_bounds[path][1])) for name, path in paths_by_name.items()}
        self._backfill_trials(optuna, study, runs, distributions, paths_by_name)
        trial = study.ask(distributions)
        params = getattr(trial, "params", {}) or {}
        if not params:
            return None
        best_name, raw_value = next(iter(params.items()))
        path = paths_by_name[best_name]
        low, high = param_bounds[path]
        value: Any = min(max(float(raw_value), float(low)), float(high))
        current_value = self._config_param(base, path)
        if isinstance(current_value, int) and not isinstance(current_value, bool):
            value = int(round(value))
        else:
            value = round(value, 6)
        self.last_trial_metadata = {
            "search_strategy": "bayesian_multi" if self.multi_objective else "bayesian",
            "optuna_study_name": getattr(study, "study_name", self.study_name),
            "optuna_trial_number": getattr(trial, "number", None),
            "optuna_storage": str(self.storage_path),
        }
        return (path, value, f"Optuna ask/tell proposal for {'.'.join(path)}={value}")

    def observe(self, run: Run) -> None:  # pragma: no cover
        optuna = self._load_optuna()
        cfg = copy.deepcopy(run.get("config", {}) if isinstance(run.get("config"), dict) else {})
        scope = str(cfg.get("autoresearch", {}).get("scope_policy") or cfg.get("dataset", {}).get("research_scope") or "default")
        model_name = str(cfg.get("model", {}).get("name") or "unknown")
        study = self._study(optuna, scope, model_name)
        trial_number = cfg.get("autoresearch", {}).get("optuna_trial_number")
        if trial_number is None:
            return
        try:
            trial = next(t for t in study.trials if t.number == int(trial_number))
        except StopIteration:
            return
        if getattr(trial, "state", None) and str(trial.state).endswith("COMPLETE"):
            return
        if self.multi_objective:
            study.tell(int(trial_number), self._metric_objectives(run))
        else:
            study.tell(int(trial_number), self._quality_score(run))

    def order_candidates(self, candidates: Sequence[Candidate], runs: Sequence[Run]) -> list[Candidate]:
        optuna = self._load_optuna()
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
    if normalized in {"bayesian_multi", "optuna_multi", "nsga2"}:
        return OptunaBayesianSearchStrategy(name="bayesian_multi", multi_objective=True)
    raise ValueError(f"unsupported AUTORESEARCH_SEARCH_STRATEGY={name!r}; expected heuristic, bayesian, or bayesian_multi")


__all__ = ["Candidate", "Config", "HeuristicSearchStrategy", "OptunaBayesianSearchStrategy", "Proposal", "Run", "SearchContext", "SearchStrategy", "strategy_from_env"]

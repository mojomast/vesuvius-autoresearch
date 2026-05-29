from __future__ import annotations

import pytest

from src.autoresearch.search_strategy import HeuristicSearchStrategy, strategy_from_env


def test_heuristic_strategy_preserves_candidate_order() -> None:
    candidates = [(("training", "epochs"), 3, "epochs"), (("training", "seed"), 1354, "seed")]
    assert HeuristicSearchStrategy().order_candidates(candidates, []) == candidates


def test_strategy_from_env_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        strategy_from_env("unknown")


def test_bayesian_strategy_errors_without_optuna(monkeypatch) -> None:
    strategy = strategy_from_env("bayesian")

    def blocked_import(name, *args, **kwargs):
        if name == "optuna":
            raise ImportError("missing optuna")
        return __import__(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked_import)
    with pytest.raises(RuntimeError, match="requires installing"):
        strategy.order_candidates([(("training", "epochs"), 3, "epochs")], [])

"""AutoResearch package facade."""

from .schemas import ExperimentConfig, load_typed_config
from .cli import main
from .search_strategy import SearchContext, strategy_from_env

__all__ = ["ExperimentConfig", "SearchContext", "load_typed_config", "main", "strategy_from_env"]

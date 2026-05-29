from __future__ import annotations

import src.autoresearch as package
from src.autoresearch import cache, cli, config_io, promotion, proposals, strategy


def test_package_facade_exports_main_and_schema() -> None:
    assert callable(package.main)
    assert package.ExperimentConfig().model.name == "tiny_numpy_ink_logreg"


def test_transitional_modules_export_legacy_boundaries() -> None:
    assert config_io.PARAM_BOUNDS[("training", "epochs")] == (2.0, 20.0)
    assert proposals._clamp_param(("training", "epochs"), 99) == 20
    assert promotion.validate_metric_contract is not None
    assert cache._RunHistory is not None
    assert strategy.PIVOT_CONFIGS
    assert callable(cli.main)

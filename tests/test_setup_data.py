from __future__ import annotations

import yaml

from scripts.setup_data import PIVOT_CONFIGS, generate_configs, main, validate_config_schema


def test_generate_configs_writes_baseline_and_pivots(tmp_path):
    data_dir = tmp_path / "data"
    configs_dir = tmp_path / "configs"

    written = generate_configs(data_dir, configs_dir=configs_dir)

    assert {path.name for path in written} == {"baseline.yaml", *PIVOT_CONFIGS}
    for path in written:
        config = yaml.safe_load(path.read_text())
        validate_config_schema(config)
        assert config["dataset"]["train_npz"]
        assert config["dataset"]["val_npz"]


def test_setup_data_main_prints_success(tmp_path, monkeypatch, capsys):
    configs_dir = tmp_path / "configs"
    monkeypatch.setattr("scripts.setup_data.CONFIGS", configs_dir)

    assert main(["--data-dir", str(tmp_path / "data")]) == 0

    output = capsys.readouterr().out
    assert "Success" in output
    assert (configs_dir / "baseline.yaml").exists()

from __future__ import annotations

import autoresearch


def test_main_fails_closed_when_harness_import_breaks(monkeypatch, capsys):
    prune_called = False

    def broken_harness():
        raise ImportError("broken harness")

    def prune_configs():
        nonlocal prune_called
        prune_called = True
        return 0

    monkeypatch.setattr(autoresearch, "_research_harness", broken_harness)
    monkeypatch.setattr(autoresearch, "_prune_stale_configs", prune_configs)
    monkeypatch.setattr("sys.argv", ["autoresearch.py", "--plan"])

    assert autoresearch.main() == 1
    assert prune_called is False
    assert "Failed to initialize research harness: broken harness" in capsys.readouterr().err

from __future__ import annotations

import importlib
from typing import Any


def legacy() -> Any:
    return importlib.import_module("autoresearch")


__all__ = ["legacy"]

from __future__ import annotations

from ._legacy import legacy


def main() -> int:
    return int(legacy().main())


__all__ = ["main"]

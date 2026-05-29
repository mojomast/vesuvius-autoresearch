from __future__ import annotations

from ._legacy import legacy

_m = legacy()

globals().update({name: getattr(_m, name) for name in dir(_m) if not name.startswith("__")})

__all__ = [name for name in globals() if not name.startswith("__")]

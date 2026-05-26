from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_configs(project_root: Path) -> list[dict[str, Any]]:
    cfg_dir = project_root / "configs"
    if not cfg_dir.exists():
        return []
    paths = list(cfg_dir.glob("*.yaml")) + list(cfg_dir.glob("*.yml")) + list(cfg_dir.glob("*.json"))
    out = []
    for path in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            raw = json.loads(path.read_text()) if path.suffix == ".json" else yaml.safe_load(path.read_text())
            summary = raw if isinstance(raw, dict) else {}
        except Exception as exc:
            summary = {"_error": str(exc)}
        out.append({"name": path.name, "path": f"configs/{path.name}", "modified_at": path.stat().st_mtime, "summary": summary})
    return out

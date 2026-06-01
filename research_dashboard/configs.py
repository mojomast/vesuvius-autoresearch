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
    existing_paths: list[tuple[float, Path]] = []
    for path in paths:
        try:
            existing_paths.append((path.stat().st_mtime, path))
        except FileNotFoundError:
            continue
    for modified_at, path in sorted(existing_paths, key=lambda item: item[0], reverse=True):
        try:
            raw = json.loads(path.read_text()) if path.suffix == ".json" else yaml.safe_load(path.read_text())
            summary = raw if isinstance(raw, dict) else {}
        except FileNotFoundError:
            continue
        except Exception as exc:
            summary = {"_error": str(exc)}
        out.append({"name": path.name, "path": f"configs/{path.name}", "modified_at": modified_at, "summary": summary})
    return out

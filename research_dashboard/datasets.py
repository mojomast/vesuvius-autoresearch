from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def dataset_summary() -> dict[str, Any]:
    try:
        from data.vesuvius_data import get_dataset_summary
        return get_dataset_summary()
    except Exception as exc:
        return {"source": "summary_error", "error": str(exc), "scrolls": [], "splits": {}}


def prepared_datasets(project_root: Path, limit: int = 80) -> list[dict[str, Any]]:
    roots = [project_root / "data" / "prepared", project_root / "data" / "real", project_root / "data" / "real_cross", project_root / "data" / "real_cross_folds", project_root / "data" / "real_cross_folds_v2", project_root / "data" / "real_cross_folds_expanded_combined"]
    meta_paths = []
    for root in roots:
        if root.exists():
            meta_paths.extend(root.glob("**/metadata.json"))
            meta_paths.extend(root.glob("**/*.metadata.json"))
    items = []
    for meta_path in sorted(set(meta_paths), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            metadata = json.loads(meta_path.read_text())
        except Exception as exc:
            metadata = {"metadata_error": str(exc)}
        npz = Path(str(meta_path)[:-len(".metadata.json")] + ".npz") if meta_path.name.endswith(".metadata.json") else next(meta_path.parent.glob("*.npz"), None)
        items.append({"path": _display(project_root, npz or meta_path), "metadata": metadata})
    return items


def fold_maps(project_root: Path, limit: int = 20) -> list[dict[str, Any]]:
    out = []
    data_root = project_root / "data"
    if not data_root.exists():
        return out
    for path in sorted(data_root.glob("**/fold_map.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            data = json.loads(path.read_text())
            folds = len(data) if isinstance(data, dict) else 0
        except Exception:
            folds = 0
        out.append({"path": _display(project_root, path), "folds": folds, "modified_at": path.stat().st_mtime})
    return out


def _display(project_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)

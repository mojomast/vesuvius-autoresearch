from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _under(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def list_artifact_files(artifact_dir: str | None, limit: int = 8) -> list[dict[str, Any]]:
    if not artifact_dir:
        return []
    root = Path(artifact_dir).expanduser()
    if not root.exists() or not root.is_dir():
        return []
    files = []
    for path in sorted((p for p in root.iterdir() if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = path.stat()
        files.append({"name": path.name, "path": str(path), "size_bytes": stat.st_size, "modified_at": stat.st_mtime, "kind": path.suffix.lstrip(".") or "file"})
        if len(files) >= limit:
            break
    return files


def preview_artifact(project_root: Path, artifact_path: str) -> dict[str, Any]:
    runs_root = (project_root / "experiments" / "runs").resolve()
    path = Path(artifact_path).expanduser().resolve()
    if not _under(path, runs_root):
        raise ValueError("artifact path must be under experiments/runs")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("artifact not found")
    stat = path.stat()
    out: dict[str, Any] = {"name": path.name, "path": str(path), "size_bytes": stat.st_size, "modified_at": stat.st_mtime, "kind": path.suffix.lstrip(".") or "file", "preview": None, "truncated": False}
    suffix = path.suffix.lower()
    if suffix == ".json":
        out["preview"] = json.loads(path.read_text(errors="replace"))
        return out
    if suffix in {".txt", ".log", ".md", ".yaml", ".yml", ".csv"}:
        text = path.read_text(errors="replace")
        out["preview"] = text[:24000]
        out["truncated"] = len(text) > 24000
        return out
    out["preview_error"] = "Preview unavailable for binary or unsupported artifact type"
    return out

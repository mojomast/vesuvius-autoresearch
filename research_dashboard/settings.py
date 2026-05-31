from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


SETTINGS_SCHEMA_VERSION = "vesuvius-dashboard-settings/v1"
SNAPSHOT_SCHEMA_VERSION = "vesuvius-dashboard-settings-snapshot/v1"

DEFAULT_SETTINGS: dict[str, Any] = {
    "poll_seconds": 15,
    "decoded_gallery_limit": 12,
    "agent_provider": "hermes",
    "agent_base_url": "http://127.0.0.1:8766/api/agent/chat",
    "agent_model": "",
    "visual_analysis_enabled": False,
    "visual_analysis_provider": "hermes",
    "visual_analysis_base_url": "http://127.0.0.1:8766/api/agent/chat",
    "visual_analysis_model": "",
}

SETTING_REGISTRY: dict[str, dict[str, Any]] = {
    "poll_seconds": {"type": int, "min": 2, "max": 300, "description": "Dashboard auto-refresh interval in seconds."},
    "decoded_gallery_limit": {"type": int, "min": 1, "max": 50, "description": "Maximum decoded outputs shown in the gallery."},
    "agent_provider": {"type": str, "max_len": 64, "description": "Default advisory agent provider label."},
    "agent_base_url": {"type": str, "max_len": 512, "url": True, "description": "Default advisory agent endpoint."},
    "agent_model": {"type": str, "max_len": 128, "description": "Default advisory agent model name."},
    "visual_analysis_enabled": {"type": bool, "description": "Allow token-authenticated visual analysis requests for decoded outputs."},
    "visual_analysis_provider": {"type": str, "max_len": 64, "description": "Default visual-analysis provider label."},
    "visual_analysis_base_url": {"type": str, "max_len": 512, "url": True, "description": "Default visual-analysis endpoint."},
    "visual_analysis_model": {"type": str, "max_len": 128, "description": "Default visual-analysis model name."},
}

SECRET_MARKERS = ("api_key", "apikey", "secret", "token", "password", "bearer")


def settings_dir(project_root: Path) -> Path:
    return project_root / ".dashboard"


def settings_path(project_root: Path) -> Path:
    return settings_dir(project_root) / "settings.json"


def snapshots_dir(project_root: Path) -> Path:
    return settings_dir(project_root) / "settings_snapshots"


def audit_path(project_root: Path) -> Path:
    return settings_dir(project_root) / "settings_audit.jsonl"


def settings_registry() -> dict[str, dict[str, Any]]:
    return {
        key: {k: v for k, v in meta.items() if k != "type"} | {"type": meta["type"].__name__}
        for key, meta in SETTING_REGISTRY.items()
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _new_version(values: dict[str, Any], *, actor: str, reason: str) -> str:
    return _stable_hash({"values": values, "actor": actor, "reason": reason, "time": _now(), "pid": os.getpid()})


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def _append_audit(project_root: Path, event: dict[str, Any]) -> None:
    path = audit_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"created_at": _now(), **event}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True, default=str) + "\n")


def _default_envelope() -> dict[str, Any]:
    values = dict(DEFAULT_SETTINGS)
    return {
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "version": _stable_hash({"defaults": values, "schema": SETTINGS_SCHEMA_VERSION}),
        "updated_at": None,
        "updated_by": "default",
        "values": values,
    }


def _contains_secret_marker(key: str, value: Any) -> bool:
    text = f"{key} {value}".lower()
    return any(marker in text for marker in SECRET_MARKERS)


def _validate_url(value: str, key: str) -> None:
    if not value:
        return
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{key} must be an http(s) URL")
    if parsed.username or parsed.password:
        raise ValueError(f"{key} must not contain embedded credentials")
    if _contains_secret_marker(key, parsed.query):
        raise ValueError(f"{key} must not contain secret-like query parameters")


def validate_settings_values(values: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(values, dict):
        raise ValueError("settings values must be an object")
    unknown = sorted(set(values) - set(SETTING_REGISTRY))
    if unknown:
        raise ValueError(f"unknown dashboard setting(s): {', '.join(unknown)}")
    merged = dict(DEFAULT_SETTINGS)
    merged.update(values)
    for key, meta in SETTING_REGISTRY.items():
        value = merged[key]
        expected = meta["type"]
        if expected is bool:
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean")
        elif expected is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
            if value < meta["min"] or value > meta["max"]:
                raise ValueError(f"{key} must be between {meta['min']} and {meta['max']}")
        elif expected is str:
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            if len(value) > meta["max_len"]:
                raise ValueError(f"{key} is too long")
            if _contains_secret_marker(key, value):
                raise ValueError(f"{key} must not contain secret-like material")
            if meta.get("url"):
                _validate_url(value, key)
    return merged


def load_dashboard_settings(project_root: Path) -> dict[str, Any]:
    path = settings_path(project_root)
    if not path.exists():
        return _default_envelope()
    data = json.loads(path.read_text(errors="replace"))
    if not isinstance(data, dict):
        raise ValueError("dashboard settings file must contain an object")
    values = validate_settings_values(data.get("values", {}))
    return {
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "version": str(data.get("version") or _stable_hash(values)),
        "updated_at": data.get("updated_at"),
        "updated_by": str(data.get("updated_by") or "unknown"),
        "values": values,
    }


def redact_dashboard_settings(envelope: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": envelope.get("schema_version", SETTINGS_SCHEMA_VERSION),
        "version": envelope.get("version"),
        "updated_at": envelope.get("updated_at"),
        "updated_by": envelope.get("updated_by"),
        "values": dict(envelope.get("values", {})),
        "registry": settings_registry(),
        "secret_policy": "API keys and tokens are never stored in dashboard settings; use environment variables or session-only UI fields.",
    }


def _snapshot_id(envelope: dict[str, Any], reason: str) -> str:
    created_at = _now()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = _stable_hash({"settings": envelope, "reason": reason, "created_at": created_at, "pid": os.getpid()})
    return f"{stamp}_{digest}"


def create_settings_snapshot(project_root: Path, *, actor: str, reason: str) -> dict[str, Any]:
    envelope = load_dashboard_settings(project_root)
    snap_id = _snapshot_id(envelope, reason)
    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "id": snap_id,
        "created_at": _now(),
        "actor": actor,
        "reason": reason,
        "settings": envelope,
    }
    _atomic_write_json(snapshots_dir(project_root) / f"{snap_id}.json", snapshot)
    _append_audit(project_root, {"event": "snapshot", "actor": actor, "reason": reason, "snapshot_id": snap_id, "settings_version": envelope.get("version")})
    return redact_settings_snapshot(snapshot)


def redact_settings_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": snapshot.get("schema_version", SNAPSHOT_SCHEMA_VERSION),
        "id": snapshot.get("id"),
        "created_at": snapshot.get("created_at"),
        "actor": snapshot.get("actor"),
        "reason": snapshot.get("reason"),
        "settings": redact_dashboard_settings(snapshot.get("settings", _default_envelope())),
    }


def list_settings_snapshots(project_root: Path, limit: int = 50) -> list[dict[str, Any]]:
    root = snapshots_dir(project_root)
    if not root.exists():
        return []
    items = []
    for path in sorted(root.glob("*.json"), key=lambda p: p.name, reverse=True)[: max(0, limit)]:
        try:
            data = json.loads(path.read_text(errors="replace"))
            if isinstance(data, dict):
                items.append(redact_settings_snapshot(data))
        except Exception:
            continue
    return items


def apply_dashboard_settings_patch(project_root: Path, patch: dict[str, Any], *, actor: str, reason: str, base_version: str | None = None) -> dict[str, Any]:
    if not isinstance(patch, dict):
        raise ValueError("settings patch must be an object")
    for key, value in patch.items():
        if _contains_secret_marker(key, value):
            raise ValueError(f"{key} must not contain secret-like material")
    current = load_dashboard_settings(project_root)
    if base_version and base_version != current.get("version"):
        raise ValueError("settings version changed; refresh before applying this patch")
    before = create_settings_snapshot(project_root, actor=actor, reason=f"pre-change: {reason or 'dashboard settings update'}")
    values = dict(current.get("values", {}))
    values.update(patch)
    values = validate_settings_values(values)
    updated = {
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "version": _new_version(values, actor=actor, reason=reason),
        "updated_at": _now(),
        "updated_by": actor,
        "values": values,
    }
    _atomic_write_json(settings_path(project_root), updated)
    _append_audit(project_root, {"event": "apply", "actor": actor, "reason": reason, "base_version": current.get("version"), "new_version": updated["version"], "pre_change_snapshot_id": before.get("id"), "keys": sorted(patch)})
    return {"ok": True, "settings": redact_dashboard_settings(updated), "pre_change_snapshot": before}


def rollback_settings_snapshot(project_root: Path, snapshot_id: str, *, actor: str, reason: str) -> dict[str, Any]:
    if not snapshot_id or "/" in snapshot_id or ".." in snapshot_id:
        raise ValueError("invalid snapshot id")
    path = snapshots_dir(project_root) / f"{snapshot_id}.json"
    if not path.exists():
        raise FileNotFoundError("settings snapshot not found")
    target = json.loads(path.read_text(errors="replace"))
    if not isinstance(target, dict) or not isinstance(target.get("settings"), dict):
        raise ValueError("settings snapshot is malformed")
    current = create_settings_snapshot(project_root, actor=actor, reason=f"pre-rollback: {reason or snapshot_id}")
    values = validate_settings_values(target["settings"].get("values", {}))
    restored = {
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "version": _new_version(values, actor=actor, reason=f"rollback:{snapshot_id}"),
        "updated_at": _now(),
        "updated_by": actor,
        "values": values,
    }
    _atomic_write_json(settings_path(project_root), restored)
    _append_audit(project_root, {"event": "rollback", "actor": actor, "reason": reason, "restored_snapshot_id": snapshot_id, "new_version": restored["version"], "pre_rollback_snapshot_id": current.get("id")})
    return {"ok": True, "settings": redact_dashboard_settings(restored), "pre_rollback_snapshot": current, "restored_snapshot_id": snapshot_id}

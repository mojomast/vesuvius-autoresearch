from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _rel_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def resolve_safe_config_path(project_root: Path, value: str | Path) -> Path:
    root = project_root.expanduser().resolve()
    raw = Path(value).expanduser()
    path = raw if raw.is_absolute() else root / raw
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"base config must resolve inside project root: {value}") from exc
    if not resolved.exists():
        raise FileNotFoundError(f"base config not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"base config must be a file: {resolved}")
    if resolved.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"base config must be YAML (.yaml or .yml): {resolved}")
    return resolved


def _safe_ratio(left: Any, right: Any) -> float | None:
    try:
        denominator = float(right)
        if abs(denominator) < 1e-9:
            return None
        ratio = float(left) / denominator
    except (TypeError, ValueError):
        return None
    return ratio if ratio == ratio and ratio not in {float("inf"), float("-inf")} else None


def _metadata_segments(meta: dict[str, Any]) -> set[str]:
    segments: set[str] = set()
    for key in ("segment_id", "mined_segment_id", "source_segment_id"):
        value = meta.get(key)
        if value is not None:
            segments.add(str(value))
    for key in ("forbidden_heldout_segments", "source_segments", "train_segments"):
        value = meta.get(key)
        if isinstance(value, list):
            segments.update(str(item) for item in value if item is not None)
    parent = meta.get("parent_full_tile")
    if isinstance(parent, dict) and parent.get("segment_id") is not None:
        segments.add(str(parent["segment_id"]))
    return {item for item in segments if item and item != "?"}


def mined_npz_inventory(project_root: Path) -> dict[str, Any]:
    mined: list[dict[str, Any]] = []
    for path in sorted((project_root / "data" / "mined").glob("**/*.npz")):
        meta_path = path.with_suffix(".metadata.json")
        metadata: dict[str, Any] = {}
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text())
            except Exception as exc:
                metadata = {"metadata_error": str(exc)}
        warnings: list[str] = []
        npz_samples = None
        pixel_positive_rate = None
        positive_patch_rate = None
        try:
            with np.load(path) as data:
                labels = data["labels"]
                npz_samples = int(labels.shape[0])
                pixel_positive_rate = float((labels > 0.5).mean()) if labels.size else 0.0
                positive_patch_rate = float(((labels > 0.5).reshape(labels.shape[0], -1).mean(axis=1) > 0.001).mean()) if labels.shape[0] else 0.0
        except Exception as exc:
            warnings.append(f"schema_error:{exc}")
        segment_id = str(metadata.get("mined_segment_id") or metadata.get("segment_id") or "") or None
        forbidden = list(metadata.get("forbidden_heldout_segments") or ([segment_id] if segment_id else []))
        samples = metadata.get("samples") or metadata.get("actual_samples") or npz_samples
        provenance = sorted(_metadata_segments(metadata))
        if not meta_path.exists():
            warnings.append("missing_metadata")
        if metadata.get("metadata_error"):
            warnings.append("metadata_parse_error")
        if not provenance:
            warnings.append("missing_provenance_segment")
        if not samples:
            warnings.append("zero_samples")
        if pixel_positive_rate is not None and pixel_positive_rate > 0.001:
            warnings.append("positive_contaminated")
        reject_prefixes = ("schema_error", "metadata_parse_error", "zero_samples", "missing_provenance_segment")
        status = "eligible" if not warnings else ("reject" if any(any(reason.startswith(prefix) for reason in warnings) for prefix in reject_prefixes) else "review")
        mined.append({
            "path": str(path),
            "relative_path": _rel_path(path, project_root),
            "metadata_path": str(meta_path) if meta_path.exists() else None,
            "segment_id": segment_id,
            "samples": samples,
            "npz_samples": npz_samples,
            "pixel_positive_rate": pixel_positive_rate,
            "positive_patch_rate": positive_patch_rate,
            "candidate_patches": metadata.get("candidate_patches"),
            "threshold": metadata.get("threshold") or metadata.get("mining_threshold"),
            "source": metadata.get("source"),
            "parent_artifact_dir": metadata.get("artifact_dir") or metadata.get("parent_artifact_dir"),
            "parent_full_tile_output_dir": metadata.get("parent_full_tile_output_dir"),
            "forbidden_heldout_segments": [str(item) for item in forbidden if item is not None],
            "provenance_segments": provenance,
            "eligibility_status": status,
            "eligibility_warnings": warnings,
        })
    return {"count": len(mined), "mined_npzs": mined}


def _snapshot_fold_ids(snapshot: dict[str, Any]) -> list[str]:
    evidence = (snapshot.get("research_summary") or {}).get("candidate_evidence") or {}
    ids: set[str] = set()
    loo = evidence.get("loo") if isinstance(evidence, dict) else {}
    if isinstance(loo, dict) and loo.get("worst_fold_id"):
        ids.add(str(loo["worst_fold_id"]))
    loo_full = evidence.get("loo_full_tile") if isinstance(evidence, dict) else {}
    if isinstance(loo_full, dict):
        ids.update(str(item) for item in loo_full.get("segments_covered") or [] if item)
        for item in loo_full.get("evidence") or []:
            if isinstance(item, dict) and item.get("segment_id"):
                ids.add(str(item["segment_id"]))
    for fold_map in ((snapshot.get("datasets") or {}).get("fold_maps") or []):
        if isinstance(fold_map, dict):
            ids.update(str(item) for item in fold_map.get("heldout_segments") or [] if item)
    return sorted(ids)


def _fold_eligibility_map(inventory: dict[str, Any], fold_ids: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for heldout in fold_ids:
        eligible: list[str] = []
        rejected: list[dict[str, str]] = []
        for item in inventory.get("mined_npzs") or []:
            path = str(item.get("relative_path") or item.get("path") or "")
            forbidden = {str(value) for value in item.get("forbidden_heldout_segments") or []} | {str(value) for value in item.get("provenance_segments") or []}
            status = str(item.get("eligibility_status") or "review")
            if status != "eligible":
                rejected.append({"relative_path": path, "reason": status})
            elif heldout in forbidden:
                rejected.append({"relative_path": path, "reason": f"mined from/provenance includes held-out segment {heldout}"})
            else:
                eligible.append(path)
        out[str(heldout)] = {"eligible_extra_train_npzs": eligible, "rejected_extra_train_npzs": rejected}
    return out


def build_fold_safe_config_preview(base_config: Path, project_root: Path, heldout_segment: str | None, extra_train_npzs: list[str]) -> dict[str, Any]:
    cfg = yaml.safe_load(base_config.read_text())
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected mapping config: {base_config}")
    patched = json.loads(json.dumps(cfg, sort_keys=True, default=str))
    patched.setdefault("dataset", {})["extra_train_npzs"] = list(extra_train_npzs)
    if heldout_segment:
        patched.setdefault("autoresearch", {})["heldout_segment"] = str(heldout_segment)
    dataset = patched.get("dataset", {}) if isinstance(patched.get("dataset"), dict) else {}
    warnings: list[str] = []
    if heldout_segment and any(str(dataset.get(key) or "").find(str(heldout_segment)) < 0 for key in ("train_npz", "val_npz") if dataset.get(key)):
        warnings.append("heldout_segment_override_does_not_rewrite_train_val_npz_paths")
    valid = not warnings
    return {
        "base_config": _rel_path(base_config, project_root),
        "heldout_segment": heldout_segment,
        "mutation": False,
        "valid": valid,
        "warnings": warnings,
        "patched_config_yaml": yaml.safe_dump(patched, sort_keys=False) if valid else None,
    }


def _candidate_tiles(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = (((snapshot.get("research_summary") or {}).get("candidate_evidence") or {}) if isinstance(snapshot, dict) else {})
    raw_tiles: list[dict[str, Any]] = []
    for section in ("full_tile",):
        data = evidence.get(section) if isinstance(evidence, dict) else None
        if isinstance(data, dict):
            raw_tiles.extend(item for item in data.get("evidence") or [] if isinstance(item, dict))
    loo = evidence.get("loo_full_tile") if isinstance(evidence, dict) else None
    if isinstance(loo, dict):
        raw_tiles.extend(item for item in loo.get("evidence") or [] if isinstance(item, dict))
    tiles_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []
    for tile in raw_tiles:
        segment = str(tile.get("segment_id") or tile.get("heldout_segment") or "")
        metrics_path = Path(str(tile.get("path") or ""))
        artifact = str(metrics_path.parent.parent) if metrics_path.name == "metrics.json" else ""
        key = (artifact, segment)
        if key not in tiles_by_key:
            order.append(key)
            tiles_by_key[key] = tile
            continue
        current = tiles_by_key[key]
        current_has_risk = isinstance(current.get("threshold_risk_summary"), dict) and bool(current.get("threshold_risk_summary"))
        tile_has_risk = isinstance(tile.get("threshold_risk_summary"), dict) and bool(tile.get("threshold_risk_summary"))
        if tile_has_risk and not current_has_risk:
            tiles_by_key[key] = tile
    return [tiles_by_key[key] for key in order]


def _tile_needs_mining(tile: dict[str, Any], ratio_threshold: float) -> bool:
    actions = tile.get("quality_next_actions") if isinstance(tile.get("quality_next_actions"), list) else []
    if any(action.get("id") == "mine_hard_negatives" for action in actions if isinstance(action, dict)):
        return True
    ratio = _safe_ratio(tile.get("pred_positive_rate"), tile.get("val_positive_rate"))
    fixed_status = str(tile.get("fixed_threshold_status") or "")
    quality = tile.get("quality_verdict") if isinstance(tile.get("quality_verdict"), dict) else {}
    reasons = {str(item) for item in quality.get("reasons") or tile.get("quality_reasons") or []}
    fixed_weak = fixed_status in {"weak", "failed", "zero_f1"} or "fixed_threshold_weak" in reasons or float(tile.get("fixed_threshold_f1") or 0.0) <= 0.0
    return bool(ratio is not None and ratio >= ratio_threshold and fixed_weak)


def _threshold_risk_decision(tile: dict[str, Any]) -> dict[str, Any]:
    summary = tile.get("threshold_risk_summary") if isinstance(tile.get("threshold_risk_summary"), dict) else {}
    selected = summary.get("selected") if isinstance(summary.get("selected"), dict) else {}
    selected_f1 = float(selected.get("f1") or tile.get("val_f1") or 0.0)
    selected_ratio = selected.get("pred_to_val_ratio") or _safe_ratio(tile.get("pred_positive_rate"), tile.get("val_positive_rate"))
    if not summary or selected_f1 <= 0.0:
        return {"action": "review_threshold_risk", "reason": "missing_threshold_risk_summary", "selected_pred_to_val_ratio": selected_ratio}

    def kept_fraction(key: str) -> float | None:
        row = summary.get(key) if isinstance(summary.get(key), dict) else None
        if not row:
            return None
        return float(row.get("f1") or 0.0) / max(selected_f1, 1e-12)

    keep2 = kept_fraction("best_under_prratio2p0")
    keep3 = kept_fraction("best_under_prratio3p0")
    if keep2 is not None and keep2 >= 0.90:
        return {"action": "tighten_positive_rate_cap", "target_max_pred_positive_rate_ratio": 2.0, "reason": "prratio2_preserves_selected_f1", "selected_pred_to_val_ratio": selected_ratio, "f1_retained_fraction": keep2}
    if keep3 is not None and keep3 >= 0.90:
        return {"action": "tighten_positive_rate_cap", "target_max_pred_positive_rate_ratio": 3.0, "reason": "prratio3_preserves_selected_f1", "selected_pred_to_val_ratio": selected_ratio, "f1_retained_fraction": keep3}
    if summary.get("cap_binding") or (selected_ratio is not None and float(selected_ratio) >= 2.0):
        return {"action": "mine_hard_negatives", "reason": "lower_ratio_caps_reduce_selected_f1", "selected_pred_to_val_ratio": selected_ratio, "f1_retained_under_prratio2": keep2, "f1_retained_under_prratio3": keep3}
    return {"action": "review_threshold_risk", "reason": "no_ratio_pressure", "selected_pred_to_val_ratio": selected_ratio, "f1_retained_under_prratio2": keep2, "f1_retained_under_prratio3": keep3}


def build_hard_negative_plan(project_root: Path, snapshot: dict[str, Any], *, max_commands: int = 6, ratio_threshold: float = 2.0, target_config: str = "configs/robust_hard_negative_prratio2p0_followup.yaml", heldout_segment: str | None = None, base_config: Path | None = None) -> dict[str, Any]:
    evidence = (snapshot.get("research_summary") or {}).get("candidate_evidence") or {}
    candidate_run_id = evidence.get("candidate_run_id")
    heldout_segment = heldout_segment or str((evidence.get("loo") or {}).get("worst_fold_id") or "") or None
    inventory = mined_npz_inventory(project_root)
    fold_ids = sorted({*(_snapshot_fold_ids(snapshot)), *([str(heldout_segment)] if heldout_segment else [])})
    fold_map = _fold_eligibility_map(inventory, fold_ids)

    mine_commands: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    seen_outputs: set[str] = set()
    for tile in _candidate_tiles(snapshot):
        segment_id = tile.get("segment_id") or tile.get("heldout_segment")
        if not segment_id:
            continue
        decision = {**_threshold_risk_decision(tile), "segment_id": str(segment_id)}
        decisions.append(decision)
        explicit_mine = any(action.get("id") == "mine_hard_negatives" for action in tile.get("quality_next_actions") or [] if isinstance(action, dict))
        if not _tile_needs_mining(tile, ratio_threshold) or (decision.get("action") == "tighten_positive_rate_cap" and not explicit_mine):
            continue
        metrics_path = Path(str(tile.get("path") or ""))
        existing_output_dir = metrics_path.parent if metrics_path.name == "metrics.json" else None
        if existing_output_dir is None:
            continue
        artifact = existing_output_dir.parent
        run_label = artifact.name[:16] if artifact.name else str(candidate_run_id or "candidate")[:16]
        mine_output = f"data/mined/{run_label}_{segment_id}_hard_negatives.npz"
        if mine_output in seen_outputs:
            continue
        seen_outputs.add(mine_output)
        output_dir = artifact / f"mining_refresh_{segment_id}"
        command = [
            ".venv/bin/python", "scripts/infer_full_tile.py",
            "--artifact", _rel_path(Path(str(artifact)), project_root),
            "--segment-id", str(segment_id),
            "--output-dir", _rel_path(output_dir, project_root),
            "--catalog-source", "public-directory",
            "--level", "1",
            "--z-offsets=-4,0,4",
            "--patch-size", "64",
            "--stride", "32",
            "--batch-size", "8",
            "--device", "cpu",
            "--mine-output", mine_output,
            "--mine-max-patches", "512",
            "--mine-max-label-positive-rate", "0.001",
        ]
        if len(mine_commands) < max_commands:
            mine_commands.append({
                "id": f"mine_{segment_id}",
                "segment_id": str(segment_id),
                "reason": "full_tile_false_positive_overprediction",
                "pred_to_val_ratio": _safe_ratio(tile.get("pred_positive_rate"), tile.get("val_positive_rate")),
                "threshold_risk_decision": decision,
                "fixed_threshold_status": tile.get("fixed_threshold_status"),
                "mine_output": mine_output,
                "command": command,
                "command_text": " ".join(command),
                "writes_artifacts": True,
                "safe_to_execute_from_dashboard": False,
            })

    eligible = []
    rejected = []
    all_forbidden_segments: set[str] = set()
    for item in inventory["mined_npzs"]:
        forbidden = {str(value) for value in item.get("forbidden_heldout_segments") or []} | {str(value) for value in item.get("provenance_segments") or []}
        all_forbidden_segments.update(forbidden)
        status = str(item.get("eligibility_status") or "review")
        if status != "eligible":
            rejected.append({"relative_path": item["relative_path"], "reason": status})
        elif heldout_segment and heldout_segment in forbidden:
            rejected.append({"relative_path": item["relative_path"], "reason": f"mined from/provenance includes held-out segment {heldout_segment}"})
        else:
            eligible.append(item["relative_path"])

    patches = []
    if eligible:
        patches.append({
            "target_config": target_config,
            "mutation": False,
            "json_path": "dataset.extra_train_npzs",
            "new_value": eligible,
            "heldout_segment": heldout_segment,
            "preconditions": [
                {"json_path": "autoresearch.heldout_segment", "equals": heldout_segment},
                {"json_path": "autoresearch.heldout_segment", "must_not_be_in": sorted(all_forbidden_segments)}
            ],
        })

    config_preview = None
    if base_config is not None:
        config_preview = build_fold_safe_config_preview(base_config, project_root, heldout_segment, eligible)

    return {
        "schema": "vesuvius.hard_negative_retrain_plan.v1",
        "dry_run": True,
        "candidate_run_id": candidate_run_id,
        "heldout_segment": heldout_segment,
        "ratio_threshold": ratio_threshold,
        "mined_inventory": inventory,
        "mine_commands": mine_commands,
        "calibration_mining_decisions": decisions,
        "fold_safe_config_patches": patches,
        "fold_safe_extra_train_npzs_by_heldout": fold_map,
        "config_preview": config_preview,
        "eligible_extra_train_npzs": eligible,
        "rejected_extra_train_npzs": rejected,
        "artifact_policy": "commands may write data/mined and experiments/runs artifacts; do not commit generated NPZ/NPY/PT/DB/log outputs",
        "next_step": "Run one mine command, add only eligible_extra_train_npzs to a fold-safe config, then validate with full-tile quality verdicts.",
    }

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _rel_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


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
        segment_id = str(metadata.get("mined_segment_id") or metadata.get("segment_id") or "") or None
        forbidden = list(metadata.get("forbidden_heldout_segments") or ([segment_id] if segment_id else []))
        mined.append({
            "path": str(path),
            "relative_path": _rel_path(path, project_root),
            "metadata_path": str(meta_path) if meta_path.exists() else None,
            "segment_id": segment_id,
            "samples": metadata.get("samples") or metadata.get("actual_samples"),
            "candidate_patches": metadata.get("candidate_patches"),
            "threshold": metadata.get("threshold") or metadata.get("mining_threshold"),
            "source": metadata.get("source"),
            "parent_artifact_dir": metadata.get("artifact_dir") or metadata.get("parent_artifact_dir"),
            "parent_full_tile_output_dir": metadata.get("parent_full_tile_output_dir"),
            "forbidden_heldout_segments": [str(item) for item in forbidden if item is not None],
            "provenance_segments": sorted(_metadata_segments(metadata)),
        })
    return {"count": len(mined), "mined_npzs": mined}


def _candidate_tiles(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = (((snapshot.get("research_summary") or {}).get("candidate_evidence") or {}) if isinstance(snapshot, dict) else {})
    tiles: list[dict[str, Any]] = []
    for section in ("full_tile",):
        data = evidence.get(section) if isinstance(evidence, dict) else None
        if isinstance(data, dict):
            tiles.extend(item for item in data.get("evidence") or [] if isinstance(item, dict))
    loo = evidence.get("loo_full_tile") if isinstance(evidence, dict) else None
    if isinstance(loo, dict):
        tiles.extend(item for item in loo.get("evidence") or [] if isinstance(item, dict))
    return tiles


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


def build_hard_negative_plan(project_root: Path, snapshot: dict[str, Any], *, max_commands: int = 6, ratio_threshold: float = 2.0, target_config: str = "configs/robust_hard_negative_prratio2p0_followup.yaml") -> dict[str, Any]:
    evidence = (snapshot.get("research_summary") or {}).get("candidate_evidence") or {}
    candidate_run_id = evidence.get("candidate_run_id")
    heldout_segment = str((evidence.get("loo") or {}).get("worst_fold_id") or "") or None
    inventory = mined_npz_inventory(project_root)

    mine_commands: list[dict[str, Any]] = []
    seen_outputs: set[str] = set()
    for tile in _candidate_tiles(snapshot):
        segment_id = tile.get("segment_id") or tile.get("heldout_segment")
        if not segment_id or not _tile_needs_mining(tile, ratio_threshold):
            continue
        metrics_path = Path(str(tile.get("path") or ""))
        output_dir = metrics_path.parent if metrics_path.name == "metrics.json" else None
        if output_dir is None:
            continue
        artifact = output_dir.parent
        run_label = artifact.name[:16] if artifact.name else str(candidate_run_id or "candidate")[:16]
        mine_output = f"data/mined/{run_label}_{segment_id}_hard_negatives.npz"
        if mine_output in seen_outputs:
            continue
        seen_outputs.add(mine_output)
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
            "--overwrite",
        ]
        mine_commands.append({
            "id": f"mine_{segment_id}",
            "segment_id": str(segment_id),
            "reason": "full_tile_false_positive_overprediction",
            "pred_to_val_ratio": _safe_ratio(tile.get("pred_positive_rate"), tile.get("val_positive_rate")),
            "fixed_threshold_status": tile.get("fixed_threshold_status"),
            "mine_output": mine_output,
            "command": command,
            "command_text": " ".join(command),
            "writes_artifacts": True,
            "safe_to_execute_from_dashboard": False,
        })
        if len(mine_commands) >= max_commands:
            break

    eligible = []
    rejected = []
    all_forbidden_segments: set[str] = set()
    for item in inventory["mined_npzs"]:
        forbidden = {str(value) for value in item.get("forbidden_heldout_segments") or []} | {str(value) for value in item.get("provenance_segments") or []}
        all_forbidden_segments.update(forbidden)
        if heldout_segment and heldout_segment in forbidden:
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

    return {
        "schema": "vesuvius.hard_negative_retrain_plan.v1",
        "dry_run": True,
        "candidate_run_id": candidate_run_id,
        "heldout_segment": heldout_segment,
        "ratio_threshold": ratio_threshold,
        "mined_inventory": inventory,
        "mine_commands": mine_commands,
        "fold_safe_config_patches": patches,
        "eligible_extra_train_npzs": eligible,
        "rejected_extra_train_npzs": rejected,
        "artifact_policy": "commands may write data/mined and experiments/runs artifacts; do not commit generated NPZ/NPY/PT/DB/log outputs",
        "next_step": "Run one mine command, add only eligible_extra_train_npzs to a fold-safe config, then validate with full-tile quality verdicts.",
    }

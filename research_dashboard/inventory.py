from __future__ import annotations

from pathlib import Path
from typing import Any


def _command_item(project_root: Path, item_id: str, title: str, command: list[str], description: str, required_paths: list[str], tags: list[str], safe_to_execute: bool = False) -> dict[str, Any]:
    return {
        "id": item_id,
        "title": title,
        "kind": "command",
        "description": description,
        "command": command,
        "command_text": " ".join(command),
        "working_dir": ".",
        "required_paths": required_paths,
        "available": all((project_root / path).exists() for path in required_paths),
        "safe_to_execute_from_dashboard": safe_to_execute,
        "tags": tags,
    }


def build_inventory(project_root: Path) -> dict[str, Any]:
    python = ".venv/bin/python" if (project_root / ".venv" / "bin" / "python").exists() else "python3"
    features = [
        _command_item(
            project_root,
            "seed_repeat_loo_dry_run",
            "Seed-repeat LOO dry-run",
            [python, "scripts/evaluate_leave_one_out.py", "--base-config", "configs/robust_multisegment_dice035_expanded.yaml", "--fold-map", "data/real_cross_folds_expanded_combined/fold_map.json", "--output-jsonl", "/tmp/opencode/vesuvius_loo_seed_dry_run.jsonl", "--summary-json", "/tmp/opencode/vesuvius_loo_seed_dry_run.summary.json", "--seeds", "1,2", "--dry-run"],
            "Plans repeated leave-one-segment-out folds without launching training.",
            ["scripts/evaluate_leave_one_out.py", "configs/robust_multisegment_dice035_expanded.yaml"],
            ["validation", "dry-run", "promotion-gate"],
        ),
        _command_item(
            project_root,
            "verify_prepared_segments",
            "Verify prepared segments",
            [python, "scripts/verify_prepared_segments.py", "--segments-root", "data/real_cross_folds_v2", "--patch-size", "64"],
            "Checks local prepared NPZ schema and metadata before fold promotion.",
            ["scripts/verify_prepared_segments.py"],
            ["data", "validation"],
        ),
        _command_item(
            project_root,
            "build_fold_map_dry_run",
            "Build fold map dry-run",
            [python, "scripts/build_segment_fold_map.py", "--segments-root", "data/real_cross_folds_v2", "--output-root", "/tmp/opencode/vesuvius_folds_dry_run", "--fold-map-out", "/tmp/opencode/vesuvius_folds_dry_run/fold_map.json", "--dry-run"],
            "Validates leave-one-segment-out fold construction without writing repo artifacts.",
            ["scripts/build_segment_fold_map.py"],
            ["data", "dry-run"],
        ),
        _command_item(
            project_root,
            "full_tile_self_test",
            "Full-tile self-test",
            [python, "scripts/infer_full_tile.py", "--self-test"],
            "Runs synthetic stitched-tile inference checks before promotion full-tile runs.",
            ["scripts/infer_full_tile.py", "data/tile_inference.py"],
            ["inference", "test"],
        ),
        _command_item(
            project_root,
            "residual_25d_smoke",
            "Residual 2.5D smoke command",
            [python, "run_experiment.py", "--config", "configs/residual_25d_torch_unet_cpu.yaml"],
            "CPU smoke for the residual 2.5D U-Net candidate; this writes run artifacts.",
            ["run_experiment.py", "configs/residual_25d_torch_unet_cpu.yaml"],
            ["model", "torch", "writes-artifacts"],
        ),
        _command_item(
            project_root,
            "robust_tta_seed_ensemble",
            "Robust TTA seed-ensemble config",
            [python, "run_experiment.py", "--config", "configs/robust_tta_seed_ensemble.yaml"],
            "Champion-candidate config using TTA and seed ensembling; budget before running.",
            ["run_experiment.py", "configs/robust_tta_seed_ensemble.yaml"],
            ["model", "ensemble", "writes-artifacts"],
        ),
        _command_item(
            project_root,
            "safe_data_expansion_one_segment",
            "Safe data expansion, one segment",
            [python, "scripts/prepare_vesuvius_segment_npz.py", "--all-labeled", "--catalog-source", "public-directory", "--output-root", "data/real_cross_folds_v3", "--skip-existing", "--max-new-segments", "1", "--request-delay-sec", "10", "--segment-delay-sec", "60", "--level", "1", "--patch-size", "64", "--train-samples", "768", "--positive-fraction", "0.15", "--negative-max-positive-rate", "0.001", "--z-offsets=-4,0,4", "--val-tiled", "--val-stride", "64", "--val-samples", "0"],
            "Adds at most one labeled public segment with conservative delays.",
            ["scripts/prepare_vesuvius_segment_npz.py"],
            ["data", "network", "writes-artifacts"],
        ),
    ]
    champion_names = [
        "robust_multisegment_dice05_all.yaml",
        "robust_multisegment_dice035_expanded.yaml",
        "residual_25d_torch_unet_cpu.yaml",
        "robust_tta_seed_ensemble.yaml",
        "next_best_moves_robust_template.yaml",
    ]
    return {
        "features": features,
        "champion_configs": [
            {"name": name, "path": f"configs/{name}", "available": (project_root / "configs" / name).exists()}
            for name in champion_names
        ],
        "warnings": [
            "Dashboard commands are copyable by default; the standalone dashboard is read-only unless run controls are explicitly enabled.",
            "Use the repo virtualenv for PyTorch experiments when available; system python may not have torch.",
            "Do not judge champions from ignored logs, NPZs, model weights, or archived synthetic artifacts alone.",
        ],
    }

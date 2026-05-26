#!/usr/bin/env python3
"""Run and summarize leave-one-segment-out validation for a base config."""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.runner import ROOT, load_config, run_experiment


def _resolve(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if rows and all(row.get("dry_run") for row in rows):
        return {
            "folds_requested": len(rows),
            "folds_successful": len(rows),
            "folds_failed": 0,
            "dry_run": True,
        }
    successful = [row for row in rows if row.get("returncode", 0) == 0 and "val_f1" in row]
    f1 = [float(row["val_f1"]) for row in successful]
    ap = [float(row.get("average_precision", 0.0)) for row in successful]
    summary: dict[str, Any] = {
        "folds_requested": len(rows),
        "folds_successful": len(successful),
        "folds_failed": len(rows) - len(successful),
    }
    if f1:
        summary.update({
            "median_val_f1": float(statistics.median(f1)),
            "mean_val_f1": float(statistics.mean(f1)),
            "min_val_f1": float(min(f1)),
            "max_val_f1": float(max(f1)),
            "mean_average_precision": float(statistics.mean(ap)),
            "per_fold_val_f1": {row["heldout_segment"]: row["val_f1"] for row in successful},
        })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a config with leave-one-segment-out folds")
    parser.add_argument("--base-config", required=True, help="YAML/JSON config to evaluate")
    parser.add_argument("--fold-map", required=True, help="JSON map of heldout segment to train_npz/val_npz")
    parser.add_argument("--output-jsonl", required=True, help="Per-fold result JSONL path")
    parser.add_argument("--summary-json", default=None, help="Optional summary JSON path")
    parser.add_argument("--label", default=None, help="Label stored in outputs; defaults to base config stem")
    parser.add_argument("--dry-run", action="store_true", help="Write planned fold configs without running experiments")
    args = parser.parse_args()

    base_path = _resolve(args.base_config)
    fold_map_path = _resolve(args.fold_map)
    output_jsonl = _resolve(args.output_jsonl)
    summary_json = _resolve(args.summary_json) if args.summary_json else output_jsonl.with_suffix(".summary.json")
    label = args.label or base_path.stem

    base_cfg = load_config(base_path)
    fold_map = json.loads(fold_map_path.read_text())
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="vesuvius_loo_") as tmpdir:
        tmp = Path(tmpdir)
        for heldout_segment, fold in sorted(fold_map.items()):
            cfg = copy.deepcopy(base_cfg)
            cfg.pop("resolved_data", None)
            cfg.pop("validation_setup", None)
            dataset = cfg.setdefault("dataset", {})
            dataset["train_npz"] = fold["train_npz"]
            dataset["val_npz"] = fold["val_npz"]
            dataset["validation_mode"] = "leave-one-segment-out"
            cfg.setdefault("autoresearch", {})["heldout_segment"] = heldout_segment
            cfg["autoresearch"]["fold_map"] = str(fold_map_path.relative_to(ROOT) if fold_map_path.is_relative_to(ROOT) else fold_map_path)

            fold_config = tmp / f"{label}__leaveout_{heldout_segment}.yaml"
            fold_config.write_text(yaml.safe_dump(_jsonable(cfg), sort_keys=False))
            row: dict[str, Any] = {"label": label, "heldout_segment": heldout_segment, "config_path": str(fold_config)}
            if args.dry_run:
                row["returncode"] = 0
                row["dry_run"] = True
            else:
                try:
                    result = run_experiment(fold_config)
                    metrics = result["metrics"]
                    row.update({
                        "returncode": 0,
                        "run_id": result["run_id"],
                        "artifact_dir": result["artifact_dir"],
                        "val_f1": float(metrics["val_f1"]),
                        "val_f05": float(metrics["val_f05"]),
                        "average_precision": float(metrics["average_precision"]),
                        "precision": float(metrics["precision"]),
                        "recall": float(metrics["recall"]),
                        "best_threshold": float(metrics["best_threshold"]),
                        "val_positive_rate": float(metrics["val_positive_rate"]),
                        "pred_positive_rate": float(metrics["pred_positive_rate"]),
                    })
                except Exception as exc:
                    row.update({"returncode": 1, "error": repr(exc)})
            rows.append(row)
            with output_jsonl.open("a") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)

    summary = _summarize(rows)
    summary.update({"label": label, "base_config": str(base_path), "fold_map": str(fold_map_path)})
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("SUMMARY_JSON " + json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary.get("folds_failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

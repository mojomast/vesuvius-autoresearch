#!/usr/bin/env python3
from __future__ import annotations

import copy
try:
    import fcntl
except ImportError:
    fcntl = None
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from experiments.runner import DB_PATH, init_db, load_config

ROOT = Path(__file__).resolve().parent
CONFIGS = ROOT / "configs"
LOGS = ROOT / "logs"
LOCK_PATH = ROOT / "logs" / "autoresearch.lock"
BASELINE = CONFIGS / "baseline.yaml"
SCOPE_KEYS = ("train_npz", "val_npz", "validation_mode", "research_scope")
SEARCH_PATHS = (
    ("model", "name"),
    ("model", "base_channels"),
    ("model", "depth"),
    ("model", "hidden_units"),
    ("dataset", "research_scope"),
    ("dataset", "train_npz"),
    ("dataset", "val_npz"),
    ("training", "epochs"),
    ("training", "batch_size"),
    ("training", "learning_rate"),
    ("training", "weight_decay"),
    ("training", "pos_weight"),
    ("training", "max_train_samples"),
    ("training", "max_train_pixels"),
    ("training", "sample_positive_fraction"),
    ("training", "dice_loss_weight"),
    ("training", "augment_flips"),
    ("training", "seed"),
    ("evaluation", "tta_flips"),
)
SIGNATURE_DEFAULTS = {
    ("model", "name"): "tiny_numpy_ink_logreg",
    ("model", "base_channels"): None,
    ("model", "depth"): 2,
    ("model", "hidden_units"): 24,
    ("dataset", "research_scope"): None,
    ("dataset", "train_npz"): None,
    ("dataset", "val_npz"): None,
    ("training", "epochs"): 5,
    ("training", "batch_size"): None,
    ("training", "learning_rate"): 0.2,
    ("training", "weight_decay"): 0.0,
    ("training", "pos_weight"): 2.0,
    ("training", "max_train_samples"): None,
    ("training", "max_train_pixels"): 600000,
    ("training", "sample_positive_fraction"): None,
    ("training", "dice_loss_weight"): None,
    ("training", "augment_flips"): None,
    ("training", "seed"): 1337,
    ("evaluation", "tta_flips"): None,
}
PIVOT_CONFIGS = (
    "residual_25d_torch_unet_cpu.yaml",
    "robust_multisegment_dice035_expanded.yaml",
    "robust_tta_seed_ensemble.yaml",
)


def _metric_direction(cfg: Dict[str, Any]) -> int:
    metric = cfg.get("evaluation", {}).get("main_metric", "val_loss")
    return 1 if "loss" in metric.lower() else -1  # sort ascending for loss, descending otherwise


def _recent_runs(limit: int | None = None) -> List[Dict[str, Any]]:
    if limit is None:
        raw_limit = os.environ.get("AUTORESEARCH_RECENT_LIMIT", "1000")
        limit = 0 if raw_limit.lower() in {"0", "all", "none"} else int(raw_limit)
    init_db(DB_PATH)
    query = "SELECT run_id,timestamp,config_json,main_metric,secondary_metrics_json,artifact_dir FROM experiments ORDER BY timestamp DESC"
    params: tuple[Any, ...] = ()
    if limit > 0:
        query += " LIMIT ?"
        params = (limit,)
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(query, params).fetchall()
    runs = []
    for run_id, ts, cfg_json, metric, sec_json, artifact_dir in rows:
        runs.append({"run_id": run_id, "timestamp": ts, "config": json.loads(cfg_json), "main_metric": float(metric), "metrics": json.loads(sec_json), "artifact_dir": artifact_dir})
    return runs


def _best_base_config(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not runs:
        return load_config(BASELINE)
    baseline = load_config(BASELINE)
    target_metric = baseline.get("evaluation", {}).get("main_metric", "val_loss")
    baseline_dataset = baseline.get("dataset", {})
    direction = _metric_direction(baseline)
    eligible = []
    for run in runs:
        if target_metric not in run.get("metrics", {}):
            continue
        run_dataset = run.get("config", {}).get("dataset", {})
        if any(baseline_dataset.get(key) != run_dataset.get(key) for key in SCOPE_KEYS if baseline_dataset.get(key) is not None):
            continue
        eligible.append(run)
    if not eligible:
        return baseline
    ranked = sorted(eligible, key=lambda r: direction * float(r["metrics"][target_metric]))
    return _canonicalize_config(ranked[0]["config"])


def _set_nested(cfg: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    cur = cfg
    for p in path[:-1]:
        cur = cur.setdefault(p, {})
    cur[path[-1]] = value


def _get_nested(cfg: Dict[str, Any], path: Tuple[str, ...], default: Any) -> Any:
    cur: Any = cfg
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _normalize_signature_value(path: Tuple[str, ...], value: Any) -> Any:
    if path == ("model", "depth") and isinstance(value, int) and value > 3:
        return 3
    if isinstance(value, float):
        return round(value, 10)
    return value


def _canonicalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    depth = _get_nested(cfg, ("model", "depth"), None)
    if isinstance(depth, int) and depth > 3:
        _set_nested(cfg, ("model", "depth"), 3)
    return cfg


def _search_signature(cfg: Dict[str, Any]) -> Tuple[Any, ...]:
    return tuple(_normalize_signature_value(path, _get_nested(cfg, path, SIGNATURE_DEFAULTS.get(path))) for path in SEARCH_PATHS)


def _tested_signatures(runs: List[Dict[str, Any]]) -> set[Tuple[Any, ...]]:
    return {_search_signature(run.get("config", {})) for run in runs}


def _proposal_candidates(base: Dict[str, Any]) -> list[tuple[Tuple[str, ...], Any, str]]:
    model_name = str(_get_nested(base, ("model", "name"), "tiny_numpy_ink_logreg"))
    lr = float(_get_nested(base, ("training", "learning_rate"), 0.2))
    epochs = int(_get_nested(base, ("training", "epochs"), 5))
    weight_decay = float(_get_nested(base, ("training", "weight_decay"), 0.0))
    seed = int(_get_nested(base, ("training", "seed"), 1337))
    pos_weight_raw = _get_nested(base, ("training", "pos_weight"), 2.0)
    pos_weight = 2.0 if isinstance(pos_weight_raw, str) and pos_weight_raw.lower() == "auto" else float(pos_weight_raw)

    if "torch" in model_name:
        base_channels = int(_get_nested(base, ("model", "base_channels"), 8))
        batch_size = int(_get_nested(base, ("training", "batch_size"), 8))
        max_train_samples = int(_get_nested(base, ("training", "max_train_samples"), 512) or 0)
        dice = float(_get_nested(base, ("training", "dice_loss_weight"), 0.0) or 0.0)
        augment_flips = bool(_get_nested(base, ("training", "augment_flips"), False))
        tta_flips = bool(_get_nested(base, ("evaluation", "tta_flips"), False))
        bounded_samples = max_train_samples if max_train_samples > 0 else 1024
        return [
            (("training", "learning_rate"), round(max(0.0002, lr * 0.6), 6), "lower torch learning rate to test calibration on the current robust/residual base"),
            (("training", "learning_rate"), round(min(0.006, lr * 1.5), 6), "raise torch learning rate modestly to test convergence-limited behavior"),
            (("training", "dice_loss_weight"), round(max(0.0, dice - 0.15), 4), "reduce Dice weight to test whether BCE precision improves"),
            (("training", "dice_loss_weight"), round(min(0.8, dice + 0.15), 4), "increase Dice weight to test ink-recall stability"),
            (("model", "base_channels"), max(4, base_channels // 2), "smaller torch U-Net width for faster regularized CPU search"),
            (("model", "base_channels"), min(16, base_channels * 2), "larger torch U-Net width to test capacity without changing data scope"),
            (("training", "epochs"), max(2, epochs - 1), "shorter torch training to test overfit/probability inflation"),
            (("training", "epochs"), min(8, epochs + 1), "one extra torch epoch to test under-convergence"),
            (("training", "batch_size"), max(2, batch_size // 2), "smaller torch batch for noisier but possibly better CPU generalization"),
            (("training", "max_train_samples"), bounded_samples, "bound full robust training to a CPU-safe sample budget for cron exploration"),
            (("training", "augment_flips"), not augment_flips, "toggle train-time flip augmentation on this torch base"),
            (("evaluation", "tta_flips"), not tta_flips, "toggle test-time flip TTA to measure ensemble-like lift"),
            (("training", "seed"), seed + 17, "repeat torch setup with a deterministic seed change"),
        ]

    depth = int(_get_nested(base, ("model", "depth"), 2))
    hidden_units = int(_get_nested(base, ("model", "hidden_units"), 24))
    max_train_pixels = int(_get_nested(base, ("training", "max_train_pixels"), 600000))
    sample_positive_fraction = _get_nested(base, ("training", "sample_positive_fraction"), None)
    candidates = [
        (("training", "pos_weight"), round(max(0.25, pos_weight * 0.5), 4), "reduce positive weight because current best predicts positives far above the validation ink rate"),
        (("training", "pos_weight"), round(min(8.0, pos_weight * 1.25), 4), "test a modest positive-weight increase under the tempered auto-weight policy"),
        (("training", "learning_rate"), round(max(0.005, lr * 0.6), 5), "lower learning rate more aggressively to test probability calibration and plateau escape"),
        (("training", "learning_rate"), round(min(1.2, lr * 1.6), 5), "higher learning rate jump to test whether the current optimum is convergence-limited"),
        (("training", "weight_decay"), round(max(0.0, weight_decay * 0.5), 7), "lower regularization to test whether ink features are underfit"),
        (("training", "weight_decay"), round(max(1e-7, weight_decay * 3.0 if weight_decay else 0.0003), 7), "higher regularization to test whether high predicted-positive rate is overfit confidence"),
        (("training", "epochs"), max(2, epochs - 2), "fewer epochs to test whether current high probabilities come from over-training"),
        (("training", "epochs"), min(20, epochs + 3), "more epochs to test whether ranking improves with additional convergence"),
        (("model", "depth"), 2 if depth != 2 else 1, "simpler feature depth to test whether cross-segment generalization improves with fewer texture terms"),
        (("model", "depth"), 3 if depth < 3 else 1, "alternate effective feature depth; depth above 3 is avoided because current feature extractor saturates at depth 3"),
        (("training", "seed"), seed + 17, "repeat the selected setup with a different deterministic sampling/initialization seed"),
    ]
    if model_name != "tiny_numpy_mlp":
        candidates.append((("model", "name"), "tiny_numpy_mlp", "switch to the local NumPy MLP for a nonlinear baseline without using external LLM/API tokens"))
    else:
        candidates.extend([
            (("model", "hidden_units"), max(8, hidden_units // 2), "smaller MLP hidden layer to test under/overfit boundary"),
            (("model", "hidden_units"), min(96, hidden_units * 2), "larger MLP hidden layer to test nonlinear texture capacity"),
            (("training", "max_train_pixels"), min(1200000, max_train_pixels * 2), "more local pixels for the MLP while staying bounded"),
            (("training", "max_train_pixels"), max(100000, max_train_pixels // 2), "fewer local pixels for faster noise-check MLP runs"),
            (("training", "sample_positive_fraction"), 0.25 if sample_positive_fraction != 0.25 else 0.5, "adjust MLP positive sampling fraction to probe prevalence calibration"),
        ])
    return candidates


def _propose_configs(base: Dict[str, Any], runs: List[Dict[str, Any]], count: int = 3, *, scope_policy: str = "focused_pair_only", lock_to_baseline_scope: bool = True) -> List[Tuple[str, Dict[str, Any], str]]:
    """Change only 1 hyperparameter per proposal for interpretable search."""
    baseline_dataset = load_config(BASELINE).get("dataset", {})
    tested = _tested_signatures(runs)
    candidates = _proposal_candidates(base)
    # Rotate deterministically by minute slot so cron does not emit identical batches forever.
    slot = int(datetime.now(timezone.utc).strftime("%M")) // 10
    candidates = candidates[slot:] + candidates[:slot]
    proposals = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for path, value, reason in candidates:
        cfg = copy.deepcopy(base)
        cfg.pop("resolved_data", None)
        cfg.pop("validation_setup", None)
        if lock_to_baseline_scope:
            cfg["dataset"] = copy.deepcopy(baseline_dataset)
        if _get_nested(cfg, path, None) == value:
            continue
        _set_nested(cfg, path, value)
        signature = _search_signature(cfg)
        if signature in tested:
            print(f"Skipping already-tested search signature {signature}")
            continue
        cfg.setdefault("autoresearch", {})["parent_reason"] = reason
        cfg.setdefault("autoresearch", {})["scope_policy"] = scope_policy
        cfg.setdefault("autoresearch", {})["search_signature"] = list(signature)
        name = f"auto_{stamp}_{len(proposals) + 1}_{'_'.join(path)}_{str(value).replace('.', 'p')}.yaml"
        proposals.append((name, cfg, reason))
        if len(proposals) >= count:
            break
    return proposals


def _pivot_bases() -> list[tuple[str, Dict[str, Any], str]]:
    bases = []
    for name in PIVOT_CONFIGS:
        path = CONFIGS / name
        if not path.exists():
            continue
        cfg = load_config(path)
        if not (cfg.get("dataset", {}).get("train_npz") and cfg.get("dataset", {}).get("val_npz")):
            continue
        bases.append((name, _canonicalize_config(cfg), str(cfg.get("autoresearch", {}).get("scope_policy") or cfg.get("dataset", {}).get("research_scope") or "strategy_pivot")))
    return bases


def _propose_with_pivots(base: Dict[str, Any], runs: List[Dict[str, Any]], count: int) -> List[Tuple[str, Dict[str, Any], str]]:
    proposals = _propose_configs(base, runs, count=count)
    if proposals:
        return proposals
    print("Focused-pair NumPy search is exhausted; trying curated robust/torch pivot bases", flush=True)
    out: list[Tuple[str, Dict[str, Any], str]] = []
    for name, pivot, scope_policy in _pivot_bases():
        remaining = count - len(out)
        if remaining <= 0:
            break
        print(f"Trying AutoResearch pivot base {name} scope={scope_policy}", flush=True)
        out.extend(_propose_configs(pivot, runs, count=remaining, scope_policy=scope_policy, lock_to_baseline_scope=False))
    return out


def _dump_config_with_comment(path: Path, cfg: Dict[str, Any], reason: str) -> None:
    body = yaml.safe_dump(cfg, sort_keys=False)
    header = (
        "# AutoResearch generated config.\n"
        "# Hyperparameter change: " + reason + "\n"
        "# Constraint: only one small hyperparameter change from the selected best recent config.\n"
    )
    path.write_text(header + body)


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    CONFIGS.mkdir(parents=True, exist_ok=True)
    with open(LOCK_PATH, "w") as lock:
        try:
            if fcntl is not None:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                try:
                    import msvcrt
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                except (ImportError, AttributeError, OSError):
                    # Basic fallback if msvcrt isn't available or fails
                    pass
        except (BlockingIOError, PermissionError, OSError):
            print(f"{datetime.now(timezone.utc).isoformat()} another autoresearch run is active; exiting safely")
            return 0
        init_db(DB_PATH)
        runs = _recent_runs()
        if not runs:
            print("No prior runs found; executing baseline first")
            subprocess.run([sys.executable, "run_experiment.py", "--config", str(BASELINE)], cwd=ROOT, check=True)
            runs = _recent_runs()
        base = _best_base_config(runs)
        proposal_count = int(os.environ.get("AUTORESEARCH_PROPOSALS", "3"))
        print(f"AutoResearch local-only cycle: loaded {len(runs)} prior runs; proposal_count={proposal_count}; no web/LLM calls", flush=True)
        proposals = _propose_with_pivots(base, runs, count=proposal_count)
        if not proposals:
            print("No novel one-change proposals remain across focused and curated pivot bases; pause instead of repeating runs")
            return 0
        for name, cfg, reason in proposals:
            cfg_path = CONFIGS / name
            _dump_config_with_comment(cfg_path, cfg, reason)
            print(f"Running generated experiment {cfg_path.name}: {reason}", flush=True)
            subprocess.run([sys.executable, "run_experiment.py", "--config", str(cfg_path)], cwd=ROOT, check=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

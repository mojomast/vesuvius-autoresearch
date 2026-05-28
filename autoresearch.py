#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import copy
from dataclasses import dataclass, field
try:
    import fcntl
except ImportError:
    fcntl = None
import json
import os
import shlex
import sqlite3
import subprocess
import sys
import time
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
    ("model", "input_mode"),
    ("model", "base_channels"),
    ("model", "depth"),
    ("model", "hidden_units"),
    ("dataset", "research_scope"),
    ("dataset", "train_npz"),
    ("dataset", "val_npz"),
    ("dataset", "z_offsets"),
    ("training", "epochs"),
    ("training", "batch_size"),
    ("training", "learning_rate"),
    ("training", "weight_decay"),
    ("training", "pos_weight"),
    ("training", "max_train_samples"),
    ("training", "max_train_pixels"),
    ("training", "sample_positive_fraction"),
    ("training", "dice_loss_weight"),
    ("training", "positive_rate_loss_weight"),
    ("training", "tversky_loss_weight"),
    ("training", "tversky_alpha"),
    ("training", "tversky_beta"),
    ("training", "focal_tversky_gamma"),
    ("training", "augment_flips"),
    ("training", "seed"),
    ("training", "seeds"),
    ("training", "deterministic"),
    ("training", "sampling_strategy"),
    ("training", "hard_negative_fraction"),
    ("evaluation", "threshold"),
    ("evaluation", "tta_flips"),
)
SIGNATURE_DEFAULTS = {
    ("model", "name"): "tiny_numpy_ink_logreg",
    ("model", "input_mode"): None,
    ("model", "base_channels"): None,
    ("model", "depth"): 2,
    ("model", "hidden_units"): 24,
    ("dataset", "research_scope"): None,
    ("dataset", "train_npz"): None,
    ("dataset", "val_npz"): None,
    ("dataset", "z_offsets"): None,
    ("training", "epochs"): 5,
    ("training", "batch_size"): None,
    ("training", "learning_rate"): 0.2,
    ("training", "weight_decay"): 0.0,
    ("training", "pos_weight"): 2.0,
    ("training", "max_train_samples"): None,
    ("training", "max_train_pixels"): 600000,
    ("training", "sample_positive_fraction"): None,
    ("training", "dice_loss_weight"): None,
    ("training", "positive_rate_loss_weight"): None,
    ("training", "tversky_loss_weight"): None,
    ("training", "tversky_alpha"): None,
    ("training", "tversky_beta"): None,
    ("training", "focal_tversky_gamma"): None,
    ("training", "augment_flips"): None,
    ("training", "seed"): 1337,
    ("training", "seeds"): None,
    ("training", "deterministic"): None,
    ("training", "sampling_strategy"): None,
    ("training", "hard_negative_fraction"): None,
    ("evaluation", "threshold"): 0.5,
    ("evaluation", "tta_flips"): None,
}
PIVOT_CONFIGS = (
    "robust_multisegment_dice035_expanded.yaml",
    "robust_calibrated_prloss_w0p03_lr0012_prratio3_seed11018.yaml",
    "robust_tta_seed_ensemble.yaml",
    "residual_25d_torch_unet_cpu.yaml",
)


@dataclass(frozen=True)
class MetricContract:
    """Required runner-to-autoresearch metric keys and expected Python types."""

    required_metrics: Dict[str, tuple[type, ...]] = field(default_factory=lambda: {
        "val_loss": (int, float),
        "val_f1": (int, float),
        "val_f05": (int, float),
        "best_threshold": (int, float),
        "precision": (int, float),
        "recall": (int, float),
        "average_precision": (int, float),
        "ap_prevalence_lift": (int, float),
        "val_positive_rate": (int, float),
        "pred_positive_rate": (int, float),
        "fixed_threshold_status": (str,),
    })
    optional_metrics: Dict[str, tuple[type, ...]] = field(default_factory=lambda: {
        "promotion_checks": (dict,),
        "loo_promotion_ready": (bool,),
        "full_tile_promotion_ready": (bool,),
    })


METRIC_CONTRACT = MetricContract()


def validate_metric_contract(run: Dict[str, Any], contract: MetricContract = METRIC_CONTRACT) -> Dict[str, Any]:
    """Validate and normalize an experiment row loaded for autoresearch.

    Args:
        run: Experiment row with decoded `config` and `metrics` objects.
        contract: Metric keys and types expected by autoresearch.

    Returns:
        A shallow copy of `run` with optional promotion evidence defaults present.

    Raises:
        ValueError: If required row fields or metric keys are missing or mistyped.
    """
    missing = [key for key in ("run_id", "timestamp", "config", "main_metric", "metrics", "artifact_dir") if key not in run]
    if missing:
        raise ValueError(f"experiment row missing required fields: {', '.join(missing)}")
    if not isinstance(run.get("config"), dict):
        raise ValueError("experiment row config must decode to an object")
    metrics = run.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("experiment row metrics must decode to an object")
    normalized_metrics = dict(metrics)
    defaults: Dict[str, Any] = {"promotion_checks": {}, "loo_promotion_ready": False, "full_tile_promotion_ready": False}
    for key, value in defaults.items():
        normalized_metrics.setdefault(key, value)
    for key, expected in contract.required_metrics.items():
        if key not in normalized_metrics:
            raise ValueError(f"metrics missing required key: {key}")
        value = normalized_metrics[key]
        if isinstance(value, bool) or not isinstance(value, expected):
            expected_names = " or ".join(item.__name__ for item in expected)
            raise ValueError(f"metrics key {key} must be {expected_names}")
    for key, expected in contract.optional_metrics.items():
        value = normalized_metrics.get(key)
        if value is not None and not isinstance(value, expected):
            expected_names = " or ".join(item.__name__ for item in expected)
            raise ValueError(f"metrics key {key} must be {expected_names}")
    normalized = dict(run)
    normalized["main_metric"] = float(run["main_metric"])
    normalized["metrics"] = normalized_metrics
    return normalized


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
        runs.append(validate_metric_contract({"run_id": run_id, "timestamp": ts, "config": json.loads(cfg_json), "main_metric": float(metric), "metrics": json.loads(sec_json), "artifact_dir": artifact_dir}))
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
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, float):
        return round(value, 10)
    return value


def _canonicalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    depth = _get_nested(cfg, ("model", "depth"), None)
    if isinstance(depth, int) and depth > 3:
        _set_nested(cfg, ("model", "depth"), 3)
    return cfg


def _prepare_autoresearch_base(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Make curated bases safe for unattended cron while preserving their research scope."""
    cfg = _canonicalize_config(cfg)
    model_name = str(_get_nested(cfg, ("model", "name"), ""))
    if "torch" not in model_name:
        return cfg
    training = cfg.setdefault("training", {})
    max_train_samples = int(training.get("max_train_samples") or 0)
    sample_cap = int(os.environ.get("AUTORESEARCH_TORCH_MAX_TRAIN_SAMPLES", "1024"))
    if max_train_samples <= 0 or max_train_samples > sample_cap:
        training["max_train_samples"] = sample_cap
        cfg.setdefault("autoresearch", {})["cron_safety"] = {
            "max_train_samples_bound": sample_cap,
            "reason": "automatic runs use bounded torch samples; promotion runs should use explicit seed-repeat LOO configs",
        }
    if "seeds" in training and os.environ.get("AUTORESEARCH_ALLOW_SEED_ENSEMBLE", "0") != "1":
        training.pop("seeds", None)
        cfg.setdefault("autoresearch", {}).setdefault("cron_safety", {})["seed_ensemble_disabled"] = True
    return cfg


def _search_signature(cfg: Dict[str, Any]) -> Tuple[Any, ...]:
    return tuple(_normalize_signature_value(path, _get_nested(cfg, path, SIGNATURE_DEFAULTS.get(path))) for path in SEARCH_PATHS)


def _tested_signatures(runs: List[Dict[str, Any]]) -> set[Tuple[Any, ...]]:
    return {_search_signature(run.get("config", {})) for run in runs}


def _reserved_signatures(runs: List[Dict[str, Any]]) -> set[Tuple[Any, ...]]:
    signatures = _tested_signatures(runs)
    ttl_hours = float(os.environ.get("AUTORESEARCH_PENDING_CONFIG_TTL_HOURS", "24"))
    now = time.time()
    for path in CONFIGS.glob("auto_*.yaml"):
        try:
            if ttl_hours > 0 and now - path.stat().st_mtime > ttl_hours * 3600:
                continue
            signatures.add(_search_signature(load_config(path)))
        except Exception:
            continue
    return signatures


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
        prloss = float(_get_nested(base, ("training", "positive_rate_loss_weight"), 0.0) or 0.0)
        augment_flips = bool(_get_nested(base, ("training", "augment_flips"), False))
        tta_flips = bool(_get_nested(base, ("evaluation", "tta_flips"), False))
        bounded_samples = max_train_samples if max_train_samples > 0 else 1024
        candidates = [
            (("training", "learning_rate"), round(max(0.0002, lr * 0.6), 6), "lower torch learning rate to test calibration on the current robust/residual base"),
            (("training", "learning_rate"), round(min(0.006, lr * 1.5), 6), "raise torch learning rate modestly to test convergence-limited behavior"),
            (("training", "dice_loss_weight"), round(max(0.0, dice - 0.15), 4), "reduce Dice weight to test whether BCE precision improves"),
            (("training", "dice_loss_weight"), round(min(0.8, dice + 0.15), 4), "increase Dice weight to test ink-recall stability"),
            (("training", "positive_rate_loss_weight"), round(max(0.0, prloss - 0.02), 4), "reduce positive-rate loss weight to test probability calibration spread"),
            (("training", "positive_rate_loss_weight"), round(min(0.1, prloss + 0.02), 4), "increase positive-rate loss weight to tighten prediction rate toward the cap"),
            (("training", "tversky_loss_weight"), 0.15, "add a light Tversky term to test recall/precision balance on the current robust base"),
            (("training", "tversky_beta"), 0.8, "bias Tversky toward false-negative reduction for rare ink recall"),
            (("training", "sampling_strategy"), "hard_mining", "try hard-negative mining to improve precision against textured non-ink"),
            (("evaluation", "threshold"), 0.35, "evaluate a calibrated fixed threshold closer to recent swept-F1 optima"),
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
        if max_train_samples and max_train_samples < 2048 and int(os.environ.get("AUTORESEARCH_TORCH_MAX_TRAIN_SAMPLES", "1024")) >= 2048:
            candidates.append((("training", "max_train_samples"), 2048, "increase robust torch sample budget after local hyperparameter plateau"))
        return candidates

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


def _mutation_family(path: Tuple[str, ...]) -> str:
    if path in {("training", "learning_rate"), ("training", "weight_decay"), ("training", "epochs"), ("training", "batch_size")}:
        return "optimizer"
    if path in {("training", "pos_weight"), ("training", "dice_loss_weight"), ("training", "positive_rate_loss_weight"), ("training", "tversky_loss_weight"), ("training", "tversky_alpha"), ("training", "tversky_beta"), ("training", "focal_tversky_gamma")}:
        return "loss_calibration"
    if path in {("training", "sampling_strategy"), ("training", "hard_negative_fraction"), ("training", "max_train_samples"), ("training", "max_train_pixels"), ("training", "sample_positive_fraction"), ("training", "augment_flips")}:
        return "data_sampling"
    if path in {("model", "name"), ("model", "input_mode"), ("model", "base_channels"), ("model", "depth"), ("model", "hidden_units")}:
        return "model_family"
    if path in {("evaluation", "threshold"), ("evaluation", "tta_flips")}:
        return "inference_calibration"
    if path in {("training", "seed"), ("training", "seeds"), ("training", "deterministic")}:
        return "replication"
    return "other"


def _propose_configs(base: Dict[str, Any], runs: List[Dict[str, Any]], count: int = 3, *, scope_policy: str = "focused_pair_only", lock_to_baseline_scope: bool = True, name_index_offset: int = 0, required_families: set[str] | None = None, strategy_phase: str | None = None) -> List[Tuple[str, Dict[str, Any], str]]:
    """Change only 1 hyperparameter per proposal for interpretable search."""
    baseline_dataset = load_config(BASELINE).get("dataset", {})
    tested = _reserved_signatures(runs)
    candidates = _proposal_candidates(base)
    # Rotate deterministically by minute slot so cron does not emit identical batches forever.
    slot = int(datetime.now(timezone.utc).strftime("%M")) // 10
    candidates = candidates[slot:] + candidates[:slot]
    proposals = []
    used_families: set[str] = set()
    seen_batch_signatures: set[Tuple[Any, ...]] = set()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for path, value, reason in candidates:
        family = _mutation_family(path)
        if required_families and family not in required_families:
            continue
        if required_families and family in used_families:
            continue
        cfg = copy.deepcopy(base)
        cfg.pop("resolved_data", None)
        cfg.pop("validation_setup", None)
        if lock_to_baseline_scope:
            cfg["dataset"] = copy.deepcopy(baseline_dataset)
        if _get_nested(cfg, path, None) == value:
            continue
        _set_nested(cfg, path, value)
        if path == ("training", "sampling_strategy") and value == "hard_mining":
            cfg.setdefault("training", {}).setdefault("hard_negative_fraction", 0.5)
        if path == ("training", "tversky_loss_weight"):
            cfg.setdefault("training", {}).setdefault("tversky_alpha", 0.3)
            cfg.setdefault("training", {}).setdefault("tversky_beta", 0.7)
        if path == ("training", "tversky_beta"):
            cfg.setdefault("training", {}).setdefault("tversky_loss_weight", 0.15)
            cfg.setdefault("training", {})["tversky_alpha"] = round(1.0 - float(value), 4)
        signature = _search_signature(cfg)
        if signature in tested or signature in seen_batch_signatures:
            print(f"Skipping already-tested search signature {signature}")
            continue
        seen_batch_signatures.add(signature)
        autoresearch = cfg.setdefault("autoresearch", {})
        autoresearch["parent_reason"] = reason
        autoresearch["scope_policy"] = scope_policy
        autoresearch["search_signature"] = list(signature)
        autoresearch["intent"] = "cron_exploration"
        autoresearch["promotable"] = False
        autoresearch["proposal_status"] = "generated"
        autoresearch["changed_path"] = ".".join(path)
        autoresearch["mutation_family"] = family
        if strategy_phase:
            autoresearch["strategy_phase"] = strategy_phase
        autoresearch["promotion_required"] = ["seed_repeat_leave_one_out", "full_tile_validation", "promotion_checks_eligible"]
        name = f"auto_{stamp}_{name_index_offset + len(proposals) + 1}_{'_'.join(path)}_{str(value).replace('.', 'p')}.yaml"
        proposals.append((name, cfg, reason))
        used_families.add(family)
        if len(proposals) >= count:
            break
    return proposals


def _run_metric_value(run: Dict[str, Any]) -> float | None:
    cfg = run.get("config", {})
    metric_name = cfg.get("evaluation", {}).get("main_metric", "val_loss")
    metrics = run.get("metrics", {})
    value = metrics.get(metric_name, run.get("main_metric"))
    if value is None:
        return None
    return float(value)


def _scope_priority(scope: str, scope_policy: str) -> int:
    label = f"{scope} {scope_policy}"
    if "multi_segment_robust_expanded_tta_ensemble" in label:
        return 0
    if "multi_segment_robust_expanded" in label:
        return 1
    if "expanded_multi_segment" in label or "expanded_leave_one_out" in label:
        return 2
    if "multi_segment" in label:
        return 3
    if "focused_pair_residual_25d_cpu" in label:
        return 4
    return 5


def _has_oversized_ml_patch_or_window(cfg: Dict[str, Any]) -> bool:
    size_keys = {"patch_size", "window_size", "patch_shape", "window_shape"}

    def _value_exceeds_guidance(value: Any) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return float(value) > 64.0
        if isinstance(value, (list, tuple)):
            return any(_value_exceeds_guidance(item) for item in value)
        return False

    def _walk(value: Any) -> bool:
        if isinstance(value, dict):
            for key, nested in value.items():
                key_name = str(key).lower()
                if key_name in size_keys and _value_exceeds_guidance(nested):
                    return True
                if isinstance(nested, (dict, list, tuple)) and _walk(nested):
                    return True
        elif isinstance(value, (list, tuple)):
            return any(_walk(item) for item in value)
        return False

    return _walk(cfg)


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _promotion_gate(run: Dict[str, Any]) -> tuple[bool, list[str]]:
    cfg = run.get("config", {})
    metrics = run.get("metrics", {})
    warnings: list[str] = []
    validation_setup = cfg.get("validation_setup") if isinstance(cfg.get("validation_setup"), dict) else {}
    if validation_setup.get("mode") not in {"cross-segment", "cross-scroll", "leave-one-segment-out"}:
        warnings.append("validation_not_held_out")
    if metrics.get("val_f1") is None:
        warnings.append("missing_val_f1")
    if metrics.get("average_precision") is None:
        warnings.append("missing_average_precision")
    if float(metrics.get("precision") or 0.0) <= 0.0 or float(metrics.get("recall") or 0.0) <= 0.0:
        warnings.append("zero_precision_or_recall")
    pred_rate = metrics.get("pred_positive_rate")
    val_rate = metrics.get("val_positive_rate")
    if pred_rate is not None and val_rate is not None:
        ratio = float(pred_rate) / max(float(val_rate), 1e-6)
        if ratio > 3.5 or ratio < 0.1:
            warnings.append("pred_positive_rate_ratio_suspicious")
    if "fixed_threshold_status" not in metrics:
        warnings.append("missing_fixed_threshold_status")
    fixed_threshold_status = str(metrics.get("fixed_threshold_status") or "").lower()
    if fixed_threshold_status and fixed_threshold_status != "ok":
        warnings.append("fixed_threshold_status_weak")
    best_threshold = _optional_float(metrics.get("best_threshold"))
    if best_threshold is not None and (best_threshold <= 0.03 or best_threshold >= 0.94):
        warnings.append("best_threshold_at_sweep_edge")
    ap_prevalence_lift = _optional_float(metrics.get("ap_prevalence_lift"))
    if ap_prevalence_lift is not None and ap_prevalence_lift < 1.25:
        warnings.append("weak_ap_lift")
    checks = metrics.get("promotion_checks") or {}
    if isinstance(checks, dict) and checks.get("eligible") is False:
        warnings.append("promotion_checks_ineligible")
    if cfg.get("autoresearch", {}).get("cron_safety"):
        warnings.append("cron_safety_run_not_promotable")
    if _has_oversized_ml_patch_or_window(cfg):
        warnings.append("patch_size_exceeds_scroll_prize_guidance")
    scope = str(cfg.get("dataset", {}).get("research_scope") or cfg.get("autoresearch", {}).get("scope_policy") or "")
    if "multi_segment" not in scope and "leave_one_out" not in scope:
        warnings.append("not_multisegment_scope")
    return not warnings, warnings


def _run_quality_score(run: Dict[str, Any]) -> float:
    metrics = run.get("metrics", {})
    score = float(metrics.get("val_f1") or run.get("main_metric") or 0.0)
    score += 0.25 * float(metrics.get("average_precision") or 0.0)
    score += 0.10 * float(metrics.get("val_f05") or 0.0)
    pred_rate = metrics.get("pred_positive_rate")
    val_rate = metrics.get("val_positive_rate")
    if pred_rate is not None and val_rate is not None:
        ratio = float(pred_rate) / max(float(val_rate), 1e-6)
        if ratio > 3.0:
            score -= min(0.25, 0.03 * (ratio - 3.0))
        elif ratio < 0.25:
            score -= min(0.25, 0.03 * (0.25 / max(ratio, 1e-6)))
    if "fixed_threshold_status" not in metrics:
        score -= 0.03
    elif str(metrics.get("fixed_threshold_status") or "").lower() != "ok":
        score -= 0.05
    eligible, warnings = _promotion_gate(run)
    if eligible:
        score += 0.05
    else:
        score -= 0.01 * len(warnings)
    return score


def _promotion_next_action(run: Dict[str, Any]) -> str:
    eligible, warnings = _promotion_gate(run)
    metrics = run.get("metrics", {})
    if eligible and not metrics.get("loo_promotion_ready"):
        return "run_seed_repeat_leave_one_out"
    if eligible and not metrics.get("full_tile_promotion_ready"):
        return "run_full_tile_validation"
    if "pred_positive_rate_ratio_suspicious" in warnings:
        return "calibrate_prediction_rate"
    if "missing_fixed_threshold_status" in warnings:
        return "calibrate_probability_scale"
    if "fixed_threshold_status_weak" in warnings:
        return "calibrate_probability_scale"
    if "zero_precision_or_recall" in warnings:
        return "repair_precision_recall"
    if eligible:
        return "promotion_review"
    return "continue_exploration"


def _strategy_phase(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    window = int(os.environ.get("AUTORESEARCH_PLATEAU_WINDOW", "12"))
    min_delta = float(os.environ.get("AUTORESEARCH_PLATEAU_MIN_DELTA", "0.005"))
    ranked = [
        run for run in runs
        if "torch" in str(_get_nested(run.get("config", {}), ("model", "name"), ""))
        and _run_metric_value(run) is not None
    ]
    if len(ranked) < window:
        return {"plateau": False, "phase": "exploit", "required_families": set(), "next_action": "continue_exploration"}
    recent = ranked[:window]
    previous = ranked[window:]
    recent_best = max(_run_quality_score(run) for run in recent)
    previous_best = max((_run_quality_score(run) for run in previous), default=recent_best)
    plateau = recent_best <= previous_best + min_delta
    best_recent = max(recent, key=_run_quality_score)
    next_action = _promotion_next_action(best_recent)
    if plateau and next_action != "continue_exploration":
        phase = "promote"
        families = {"replication", "inference_calibration"}
    elif plateau:
        phase = "diversify"
        families = {"loss_calibration", "data_sampling", "model_family", "inference_calibration"}
    else:
        phase = "exploit"
        families = set()
    return {
        "plateau": plateau,
        "phase": phase,
        "required_families": families,
        "next_action": next_action,
        "candidate_run_id": best_recent.get("run_id"),
    }


def _repo_arg(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _shell_command(args: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in args)


def _path_tail(value: Any) -> str:
    return str(value or "").replace("\\", "/").lstrip("./")


def _same_path_tail(left: Any, right: Any) -> bool:
    left_tail = _path_tail(left)
    right_tail = _path_tail(right)
    return bool(left_tail and right_tail and (left_tail.endswith(right_tail) or right_tail.endswith(left_tail)))


def _linked_loo_summary_ready(run: Dict[str, Any]) -> bool:
    run_id = str(run.get("run_id") or "")
    artifact_config = Path(str(run.get("artifact_dir") or "")) / "config.json" if run.get("artifact_dir") else None
    for path in sorted(LOGS.glob("*summary.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            summary = json.loads(path.read_text())
        except Exception:
            continue
        if not summary.get("promotion_ready"):
            continue
        run_ids = {str(item) for item in summary.get("run_ids") or []}
        if run_id and run_id in run_ids:
            return True
        if artifact_config and _same_path_tail(summary.get("base_config"), artifact_config):
            return True
    return False


def _promotion_next_action_with_evidence(run: Dict[str, Any]) -> str:
    if _linked_loo_summary_ready(run):
        metrics = run.setdefault("metrics", {})
        if isinstance(metrics, dict):
            metrics = {**metrics, "loo_promotion_ready": True}
            run = {**run, "metrics": metrics}
    return _promotion_next_action(run)


def _manual_promotion_candidate(runs: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    candidates: list[Dict[str, Any]] = []
    for run in runs:
        cfg = run.get("config", {})
        metrics = run.get("metrics", {}) if isinstance(run.get("metrics"), dict) else {}
        model_name = str(_get_nested(cfg, ("model", "name"), ""))
        if "torch" not in model_name:
            continue
        scope = str(cfg.get("dataset", {}).get("research_scope") or cfg.get("autoresearch", {}).get("scope_policy") or "")
        if "multi_segment" not in scope and "leave_one_out" not in scope:
            continue
        if str(metrics.get("fixed_threshold_status") or "").lower() != "ok":
            continue
        if _run_metric_value(run) is None or metrics.get("average_precision") is None:
            continue
        if float(metrics.get("precision") or 0.0) <= 0.0 or float(metrics.get("recall") or 0.0) <= 0.0:
            continue
        pred_rate = metrics.get("pred_positive_rate")
        val_rate = metrics.get("val_positive_rate")
        if pred_rate is not None and val_rate is not None:
            ratio = float(pred_rate) / max(float(val_rate), 1e-6)
            if ratio > 3.5 or ratio < 0.1:
                continue
        action = _promotion_next_action_with_evidence(run)
        if action in {"run_seed_repeat_leave_one_out", "run_full_tile_validation"}:
            candidates.append(run)
    if not candidates:
        return None
    return max(candidates, key=_run_quality_score)


def _promotion_phase_manual_action(runs: List[Dict[str, Any]]) -> dict[str, Any] | None:
    """Return a candidate-linked promotion action when local sweeps should stop."""
    if os.environ.get("AUTORESEARCH_CONTINUE_AFTER_PROMOTION_ACTION", "0") == "1":
        return None
    strategy = _strategy_phase(runs)
    if not strategy.get("plateau"):
        return None
    candidate = _manual_promotion_candidate(runs)
    if not candidate:
        return None
    action = _promotion_next_action_with_evidence(candidate)
    if action not in {"run_seed_repeat_leave_one_out", "run_full_tile_validation"}:
        return None
    candidate_run_id = str(candidate.get("run_id") or "")

    reasoning = [
        "plateau_detected",
        "promotion_candidate_available",
        "stop_local_sampled_sweeps",
        "candidate_linked_evidence_required",
    ]
    command = None
    if action == "run_seed_repeat_leave_one_out":
        artifact_dir = Path(str(candidate.get("artifact_dir") or ROOT / "experiments" / "runs" / candidate_run_id))
        base_config = artifact_dir / "config.json"
        output_stem = f"{candidate_run_id}_seedrepeat_loo"
        command = _shell_command([
            ".venv/bin/python",
            "scripts/evaluate_leave_one_out.py",
            "--base-config",
            _repo_arg(base_config),
            "--fold-map",
            "data/real_cross_folds_expanded_combined/fold_map.json",
            "--output-jsonl",
            f"logs/{output_stem}.jsonl",
            "--summary-json",
            f"logs/{output_stem}.summary.json",
            "--seeds",
            "11001,11018,15050",
            "--jobs",
            os.environ.get("AUTORESEARCH_LOO_JOBS", "2"),
        ])
        reasoning.append("median_over_seeds_and_folds_required")
    else:
        reasoning.append("full_tile_command_should_come_from_dashboard_candidate_evidence")

    return {
        "status": "manual_promotion_action",
        "next_action": action,
        "candidate_run_id": candidate_run_id,
        "command": command,
        "reasoning": reasoning,
        "strategy_phase": strategy.get("phase"),
        "promotion_required": ["seed_repeat_leave_one_out", "full_tile_validation", "promotion_checks_eligible"],
        "proposals": [],
    }


def _ranked_recent_torch_bases(runs: List[Dict[str, Any]]) -> list[tuple[Dict[str, Any], str]]:
    robust_scopes = (
        "multi_segment_robust",
        "expanded_multi_segment",
        "expanded_leave_one_out",
        "focused_pair_residual_25d_cpu",
    )
    ranked: list[tuple[int, float, int, str, Dict[str, Any], str, list[str]]] = []
    seen: set[Tuple[Any, ...]] = set()
    for run in runs:
        cfg = run.get("config", {})
        model_name = str(_get_nested(cfg, ("model", "name"), ""))
        if "torch" not in model_name:
            continue
        dataset = cfg.get("dataset", {})
        scope = str(dataset.get("research_scope") or "")
        scope_policy = str(cfg.get("autoresearch", {}).get("scope_policy") or scope)
        if not any(token in scope or token in scope_policy for token in robust_scopes):
            continue
        if not (dataset.get("train_npz") and dataset.get("val_npz")):
            continue
        value = _run_metric_value(run)
        if value is None:
            continue
        prepared = _prepare_autoresearch_base(cfg)
        signature = _search_signature(prepared)
        if signature in seen:
            continue
        seen.add(signature)
        eligible, warnings = _promotion_gate(run)
        ranked.append((0 if eligible else 1, -_run_quality_score(run), _scope_priority(scope, scope_policy), str(run.get("run_id") or "unknown"), prepared, scope_policy or "recent_robust_torch_winner", warnings))
    ranked.sort(key=lambda item: (item[0], item[2], item[1]))
    limit = int(os.environ.get("AUTORESEARCH_RECENT_WINNER_BASES", "6"))
    out = []
    for eligible_rank, _score, _scope_rank, run_id, cfg, scope_policy, warnings in ranked[:limit]:
        kind = "promotion-eligible recent base" if eligible_rank == 0 else "diagnostic recent base"
        suffix = "" if not warnings else " warnings=" + ",".join(warnings)
        out.append((cfg, f"{kind} {run_id} scope={scope_policy}{suffix}"))
    return out


def _propose_from_recent_winners(runs: List[Dict[str, Any]], count: int, *, name_index_offset: int = 0, required_families: set[str] | None = None, strategy_phase: str | None = None) -> List[Tuple[str, Dict[str, Any], str]]:
    out: list[Tuple[str, Dict[str, Any], str]] = []
    for base, label in _ranked_recent_torch_bases(runs):
        remaining = count - len(out)
        if remaining <= 0:
            break
        scope_policy = str(base.get("autoresearch", {}).get("scope_policy") or base.get("dataset", {}).get("research_scope") or "recent_robust_torch_winner")
        print(f"Trying AutoResearch recent robust winner base {label}", flush=True)
        proposals = _propose_configs(base, runs, count=remaining, scope_policy=scope_policy, lock_to_baseline_scope=False, name_index_offset=name_index_offset + len(out), required_families=required_families, strategy_phase=strategy_phase)
        for name, cfg, reason in proposals:
            cfg.setdefault("autoresearch", {})["parent_recent_winner"] = label
            out.append((name, cfg, f"follow up {label}: {reason}"))
    return out


def _pivot_bases() -> list[tuple[str, Dict[str, Any], str]]:
    bases = []
    for name in PIVOT_CONFIGS:
        path = CONFIGS / name
        if not path.exists():
            continue
        cfg = load_config(path)
        if not (cfg.get("dataset", {}).get("train_npz") and cfg.get("dataset", {}).get("val_npz")):
            continue
        prepared = _prepare_autoresearch_base(cfg)
        bases.append((name, prepared, str(prepared.get("autoresearch", {}).get("scope_policy") or prepared.get("dataset", {}).get("research_scope") or "strategy_pivot")))
    return bases


def _propose_best_path(base: Dict[str, Any], runs: List[Dict[str, Any]], count: int) -> List[Tuple[str, Dict[str, Any], str]]:
    print("AutoResearch strategy: robust/torch best path first; focused NumPy is fallback only", flush=True)
    strategy = _strategy_phase(runs)
    required_families = strategy["required_families"] or None
    phase = str(strategy["phase"])
    print(f"AutoResearch strategy phase={phase} plateau={strategy['plateau']} next_action={strategy['next_action']}", flush=True)
    out: list[Tuple[str, Dict[str, Any], str]] = []
    for name, pivot, scope_policy in _pivot_bases():
        remaining = count - len(out)
        if remaining <= 0:
            break
        print(f"Trying AutoResearch best-path base {name} scope={scope_policy}", flush=True)
        out.extend(_propose_configs(pivot, runs, count=remaining, scope_policy=scope_policy, lock_to_baseline_scope=False, name_index_offset=len(out), required_families=required_families, strategy_phase=phase))
    if out:
        return out
    print("Curated robust/torch bases are exhausted; trying best recent robust/torch winners", flush=True)
    out = _propose_from_recent_winners(runs, count=count, required_families=required_families, strategy_phase=phase)
    if out:
        return out
    if os.environ.get("AUTORESEARCH_ALLOW_FOCUSED_FALLBACK", "1") != "1":
        return []
    print("Recent robust/torch winner follow-ups are exhausted; falling back to focused NumPy search", flush=True)
    return _propose_configs(base, runs, count=count, required_families=required_families, strategy_phase=phase)


def _propose_with_pivots(base: Dict[str, Any], runs: List[Dict[str, Any]], count: int) -> List[Tuple[str, Dict[str, Any], str]]:
    return _propose_best_path(base, runs, count)


def _proposal_plan(proposals: List[Tuple[str, Dict[str, Any], str]]) -> list[Dict[str, Any]]:
    plan = []
    for name, cfg, reason in proposals:
        autoresearch = cfg.get("autoresearch", {}) if isinstance(cfg.get("autoresearch"), dict) else {}
        plan.append({
            "name": name,
            "reason": reason,
            "changed_path": autoresearch.get("changed_path"),
            "mutation_family": autoresearch.get("mutation_family"),
            "strategy_phase": autoresearch.get("strategy_phase"),
            "scope_policy": autoresearch.get("scope_policy"),
            "promotable": bool(autoresearch.get("promotable")),
            "promotion_required": autoresearch.get("promotion_required", []),
            "search_signature": autoresearch.get("search_signature"),
        })
    return plan


_AUTO_ACTIONS: set[str] = {
    "calibrate_probability_scale",
    "calibrate_positive_rate",
    "improve_ranking_signal",
    "mine_hard_negatives",
    "repair_precision_recall",
}


def _generate_promotion_action_proposals(runs: List[Dict[str, Any]], ready_payload: dict[str, Any], count: int = 2) -> List[Tuple[str, Dict[str, Any], str]]:
    """Generate targeted proposals for auto-executable promotion actions."""
    action_id = ready_payload.get("action_id")
    candidate_run_id = ready_payload.get("candidate_run_id")
    candidate_run = next((r for r in runs if r.get("run_id") == candidate_run_id), None)
    if not candidate_run:
        return []

    base = _canonicalize_config(candidate_run.get("config", {}))
    model_name = str(_get_nested(base, ("model", "name"), ""))
    is_torch = "torch" in model_name
    proposals: List[Tuple[str, Dict[str, Any], str]] = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tested = _reserved_signatures(runs)
    seen: set[Tuple[Any, ...]] = set()

    def _make_proposal(path: Tuple[str, ...], value: Any, reason: str) -> Tuple[str, Dict[str, Any], str] | None:
        cfg = copy.deepcopy(base)
        cfg.pop("resolved_data", None)
        cfg.pop("validation_setup", None)
        if _get_nested(cfg, path, None) == value:
            return None
        _set_nested(cfg, path, value)
        if path == ("training", "sampling_strategy") and value == "hard_mining":
            cfg.setdefault("training", {}).setdefault("hard_negative_fraction", 0.5)
        if path == ("training", "tversky_loss_weight"):
            cfg.setdefault("training", {}).setdefault("tversky_alpha", 0.3)
            cfg.setdefault("training", {}).setdefault("tversky_beta", 0.7)
        signature = _search_signature(cfg)
        if signature in tested or signature in seen:
            return None
        seen.add(signature)
        autoresearch = cfg.setdefault("autoresearch", {})
        autoresearch["parent_reason"] = reason
        autoresearch["scope_policy"] = str(autoresearch.get("scope_policy") or base.get("dataset", {}).get("research_scope") or "promotion_action")
        autoresearch["intent"] = "promotion_action"
        autoresearch["promotion_action_id"] = action_id
        autoresearch["promotable"] = False
        autoresearch["proposal_status"] = "generated"
        autoresearch["changed_path"] = ".".join(path)
        autoresearch["mutation_family"] = _mutation_family(path)
        name = f"auto_{stamp}_promotion_{action_id}_{len(proposals)+1}_{'_'.join(path)}_{str(value).replace('.', 'p')}.yaml"
        return (name, cfg, reason)

    if action_id == "calibrate_probability_scale":
        thresholds = [0.31, 0.33, 0.35, 0.37] if is_torch else [0.33, 0.35, 0.4]
        for thresh in thresholds:
            if len(proposals) >= count:
                break
            p = _make_proposal(("evaluation", "threshold"), thresh, f"promotion action {action_id}: evaluate threshold={thresh}")
            if p:
                proposals.append(p)
        # Also try seed repeats to confirm threshold robustness
        seed = int(_get_nested(base, ("training", "seed"), 1337))
        for seed_delta in (17, 31, 53):
            if len(proposals) >= count:
                break
            p = _make_proposal(("training", "seed"), seed + seed_delta, f"promotion action {action_id}: seed repeat with threshold-aware calibration")
            if p:
                proposals.append(p)

    elif action_id == "calibrate_positive_rate":
        if is_torch:
            prloss = float(_get_nested(base, ("training", "positive_rate_loss_weight"), 0.0) or 0.0)
            values = [round(max(0.0, prloss - 0.02), 4), round(min(0.1, prloss + 0.02), 4)]
            for val in values:
                if len(proposals) >= count:
                    break
                p = _make_proposal(("training", "positive_rate_loss_weight"), val, f"promotion action {action_id}: adjust positive_rate_loss_weight to {val}")
                if p:
                    proposals.append(p)
        pos_weight = float(_get_nested(base, ("training", "pos_weight"), 2.0))
        if len(proposals) < count:
            p = _make_proposal(("training", "pos_weight"), round(max(0.25, pos_weight * 0.8), 4), f"promotion action {action_id}: reduce pos_weight to tighten positive rate")
            if p:
                proposals.append(p)

    elif action_id == "improve_ranking_signal":
        if is_torch:
            epochs = int(_get_nested(base, ("training", "epochs"), 5))
            if len(proposals) < count:
                p = _make_proposal(("training", "epochs"), min(8, epochs + 1), f"promotion action {action_id}: extra epoch to improve AP/ranking")
                if p:
                    proposals.append(p)
            lr = float(_get_nested(base, ("training", "learning_rate"), 0.001))
            if len(proposals) < count:
                p = _make_proposal(("training", "learning_rate"), round(max(0.0002, lr * 0.6), 6), f"promotion action {action_id}: lower lr to improve ranking calibration")
                if p:
                    proposals.append(p)

    elif action_id == "mine_hard_negatives":
        if is_torch and len(proposals) < count:
            p = _make_proposal(("training", "sampling_strategy"), "hard_mining", f"promotion action {action_id}: hard-negative mining")
            if p:
                proposals.append(p)

    elif action_id == "repair_precision_recall":
        pos_weight = float(_get_nested(base, ("training", "pos_weight"), 2.0))
        if len(proposals) < count:
            p = _make_proposal(("training", "pos_weight"), round(max(0.25, pos_weight * 0.75), 4), f"promotion action {action_id}: reduce pos_weight to repair precision/recall balance")
            if p:
                proposals.append(p)
        dice = float(_get_nested(base, ("training", "dice_loss_weight"), 0.0) or 0.0)
        if is_torch and len(proposals) < count:
            p = _make_proposal(("training", "dice_loss_weight"), round(min(0.8, dice + 0.15), 4), f"promotion action {action_id}: increase dice weight to improve recall")
            if p:
                proposals.append(p)

    return proposals


def _promotion_or_fallback_proposals(runs: List[Dict[str, Any]], base: Dict[str, Any], ready_payload: dict[str, Any] | None, count: int) -> tuple[List[Tuple[str, Dict[str, Any], str]], str | None]:
    """Return auto-action proposals, falling back to normal exploration if exhausted."""
    if ready_payload and ready_payload.get("action_id") in _AUTO_ACTIONS:
        action_proposals = _generate_promotion_action_proposals(runs, ready_payload, count=count)
        if action_proposals:
            return action_proposals, str(ready_payload.get("action_id"))
        print(f"Promotion gate action={ready_payload.get('action_id')} proposals exhausted; falling back to normal exploration", flush=True)
    return _propose_best_path(base, runs, count=count), None


def _promotion_ready_payload() -> dict[str, Any] | None:
    if os.environ.get("AUTORESEARCH_PAUSE_WHEN_PROMOTION_READY", "1") != "1":
        return None
    try:
        from research_dashboard.snapshot import build_snapshot
        decision = build_snapshot(ROOT).get("research_summary", {}).get("decision", {})
        gate = decision.get("promotion_gate", {}) if isinstance(decision, dict) else {}
        if gate.get("ready") is True:
            evidence = decision.get("candidate_evidence", {}) if isinstance(decision.get("candidate_evidence"), dict) else {}
            actions = decision.get("promotion_actions", []) if isinstance(decision.get("promotion_actions"), list) else []
            if not actions and isinstance(evidence.get("promotion_actions"), list):
                actions = evidence.get("promotion_actions", [])
            top_action = actions[0] if actions and isinstance(actions[0], dict) else {}
            next_action = str(top_action.get("label") or decision.get("next_action") or "Promotion gate is ready; review the promotion candidate before more exploration.")
            weak_tile = evidence.get("weak_fold_full_tile", {}) if isinstance(evidence.get("weak_fold_full_tile"), dict) else {}
            command = top_action.get("command_text")
            if not command and top_action.get("id") == "weak_fold_full_tile" and weak_tile.get("status") != "done":
                command = weak_tile.get("command_text")
            # Compressed probabilities producing no fixed-threshold positives is expected;
            # do not pause exploration for this alone.
            if top_action.get("id") == "calibrate_probability_scale":
                ft_reasons: set[str] = set()
                for item in (evidence.get("full_tile", {}).get("evidence", []) if isinstance(evidence.get("full_tile"), dict) else []):
                    if isinstance(item, dict) and item.get("fixed_threshold_failure_reason"):
                        ft_reasons.add(str(item["fixed_threshold_failure_reason"]))
                for item in (evidence.get("loo_full_tile", {}).get("evidence", []) if isinstance(evidence.get("loo_full_tile"), dict) else []):
                    if isinstance(item, dict) and item.get("fixed_threshold_failure_reason"):
                        ft_reasons.add(str(item["fixed_threshold_failure_reason"]))
                if ft_reasons == {"no_fixed_positive_predictions"}:
                    print("Promotion gate ready but fixed-threshold weakness is expected (compressed probabilities); continuing exploration.", flush=True)
                    return None
            reasoning = [
                "promotion_gate_ready",
                "pause_exploration_before_more_local_sweeps",
            ]
            if evidence.get("candidate_run_id"):
                reasoning.append("candidate_linked_evidence_available")
            if command:
                reasoning.append("use_public_directory_backoff_and_chunk_pacing")
            return {
                "status": "promotion_ready",
                "next_action": next_action,
                "candidate_run_id": evidence.get("candidate_run_id"),
                "action_id": top_action.get("id"),
                "command": command,
                "safe_to_execute_from_dashboard": top_action.get("safe_to_execute_from_dashboard"),
                "writes_artifacts": top_action.get("writes_artifacts"),
                "reasoning": reasoning,
                "promotion_actions": actions,
                "candidate_evidence": evidence,
                "proposals": [],
            }
    except Exception as exc:
        print(f"Promotion readiness check skipped: {exc}", flush=True)
    return None


def _promotion_ready_message() -> str | None:
    payload = _promotion_ready_payload()
    if not payload:
        return None
    message = str(payload.get("next_action") or "Promotion gate is ready; review the promotion candidate before more exploration.")
    if payload.get("command"):
        message += f" Command: {payload['command']}"
    return message


def _dump_config_with_comment(path: Path, cfg: Dict[str, Any], reason: str) -> None:
    body = yaml.safe_dump(cfg, sort_keys=False)
    header = (
        "# AutoResearch generated config.\n"
        "# Hyperparameter change: " + reason + "\n"
        "# Constraint: only one small hyperparameter change from the selected best recent config.\n"
    )
    path.write_text(header + body)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or plan the local AutoResearch cycle")
    parser.add_argument("--plan", action="store_true", help="Print planned proposals without writing configs or launching experiments")
    parser.add_argument("--json", action="store_true", help="Emit planning output as JSON; only valid with --plan")
    args = parser.parse_args()
    if args.json and not args.plan:
        parser.error("--json requires --plan")

    if args.plan:
        runs = _recent_runs()
        if not runs:
            print(json.dumps({"status": "needs_baseline", "proposals": []}, indent=2 if args.json else None))
            return 0
        ready_payload = _promotion_ready_payload()
        if ready_payload and ready_payload.get("action_id") not in _AUTO_ACTIONS:
            payload = ready_payload
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(f"Promotion gate is ready; pausing exploration. Next action: {payload.get('next_action')}")
                if payload.get("command"):
                    print(f"Command: {payload['command']}")
            return 0
        # Auto-execute promotion actions by generating targeted proposals
        if ready_payload and ready_payload.get("action_id") in _AUTO_ACTIONS:
            action_proposals = _generate_promotion_action_proposals(runs, ready_payload, count=int(os.environ.get("AUTORESEARCH_PROPOSALS", "3")))
            if action_proposals:
                payload = {"status": "promotion_action", "action_id": ready_payload.get("action_id"), "proposal_count": len(action_proposals), "proposals": _proposal_plan(action_proposals)}
                if args.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    print(f"Promotion gate ready with auto-executable action={ready_payload.get('action_id')}; generating targeted proposals:")
                    for item in payload["proposals"]:
                        print(f"  {item['name']}: {item['reason']} [{item.get('strategy_phase')}/{item.get('mutation_family')}]")
                return 0
        manual_payload = _promotion_phase_manual_action(runs)
        if manual_payload:
            if args.json:
                print(json.dumps(manual_payload, indent=2, sort_keys=True))
            else:
                print(f"AutoResearch promotion action required: {manual_payload.get('next_action')}")
                if manual_payload.get("command"):
                    print(f"Command: {manual_payload['command']}")
            return 0
        base = _best_base_config(runs)
        proposal_count = int(os.environ.get("AUTORESEARCH_PROPOSALS", "3"))
        if args.json:
            with contextlib.redirect_stdout(sys.stderr):
                proposals, source_action = _promotion_or_fallback_proposals(runs, base, ready_payload, proposal_count)
        else:
            proposals, source_action = _promotion_or_fallback_proposals(runs, base, ready_payload, proposal_count)
        payload = {"status": "planned", "proposal_count": len(proposals), "fallback_from_action": source_action, "proposals": _proposal_plan(proposals)}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            for item in payload["proposals"]:
                print(f"{item['name']}: {item['reason']} [{item.get('strategy_phase')}/{item.get('mutation_family')}]")
            if not proposals:
                print("No novel one-change proposals remain.")
        return 0

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
        ready_payload = _promotion_ready_payload()
        if ready_payload and ready_payload.get("action_id") not in _AUTO_ACTIONS:
            print(f"Promotion gate is ready; pausing exploration. Next action: {ready_payload.get('next_action')}", flush=True)
            if ready_payload.get("command"):
                print(f"Command: {ready_payload['command']}", flush=True)
            return 0
        manual_payload = _promotion_phase_manual_action(runs)
        if manual_payload:
            print(f"AutoResearch promotion action required: {manual_payload.get('next_action')}", flush=True)
            if manual_payload.get("command"):
                print(f"Command: {manual_payload['command']}", flush=True)
            return 0
        proposals, source_action = _promotion_or_fallback_proposals(runs, base, ready_payload, proposal_count)
        if ready_payload and source_action:
            print(f"Promotion gate ready with auto-executable action={source_action}; generating targeted proposals", flush=True)
        if not proposals:
            print("No novel one-change proposals remain across the robust best path, focused fallback, and promotion actions; pause instead of repeating runs")
            return 0
        deadline_seconds = int(os.environ.get("AUTORESEARCH_DEADLINE_SECONDS", "0") or 0)
        deadline = time.monotonic() + deadline_seconds if deadline_seconds > 0 else None
        for name, cfg, reason in proposals:
            if deadline is not None and time.monotonic() > deadline - 90:
                print("AutoResearch deadline is near; stopping before launching another experiment", flush=True)
                break
            cfg_path = CONFIGS / name
            _dump_config_with_comment(cfg_path, cfg, reason)
            print(f"Running generated experiment {cfg_path.name}: {reason}", flush=True)
            subprocess.run([sys.executable, "run_experiment.py", "--config", str(cfg_path)], cwd=ROOT, check=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import copy
from dataclasses import dataclass, field
from types import ModuleType
_fcntl: ModuleType | None
try:
    import fcntl as _fcntl
except ImportError:
    _fcntl = None
import json
import logging
import os
import shlex
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, cast

import yaml

from experiments.runner import DB_PATH, init_db, load_config
from src.autoresearch.schemas import ExperimentConfig, load_typed_config

fcntl: ModuleType | None = _fcntl

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


ROOT = Path(os.environ.get("VESUVIUS_PROJECT_ROOT", Path(__file__).resolve().parent)).resolve()
CONFIGS = Path(os.environ.get("VESUVIUS_CONFIG_DIR", ROOT / "configs"))
LOGS = Path(os.environ.get("VESUVIUS_LOG_DIR", ROOT / "logs"))
LOCK_PATH = LOGS / "autoresearch.lock"
BASELINE = Path(os.environ.get("VESUVIUS_BASELINE_CONFIG", CONFIGS / "baseline.yaml"))
LINKED_LOO_SUMMARY_CACHE_TTL_SECONDS = 300.0
DEFAULT_SEED = _env_int("AUTORESEARCH_DEFAULT_SEED", 1337)
SEED_REPEAT_DELTAS = tuple(int(part.strip()) for part in os.environ.get("AUTORESEARCH_SEED_REPEAT_DELTAS", "17,31,53").split(",") if part.strip())
DEFAULT_PROMOTION_SEEDS = os.environ.get("AUTORESEARCH_PROMOTION_SEEDS", "11001,11018,11045")
DEFAULT_LOO_JOBS = os.environ.get("AUTORESEARCH_LOO_JOBS", "2")
DEFAULT_FOLD_MAP = os.environ.get("VESUVIUS_FOLD_MAP", "data/real_cross_folds_expanded_combined/fold_map.json")
DEFAULT_PYTHON = os.environ.get("VESUVIUS_PYTHON", sys.executable)
DEFAULT_LOO_SCRIPT = os.environ.get("VESUVIUS_LOO_SCRIPT", "scripts/evaluate_leave_one_out.py")
PROMOTION_POS_RATE_RATIO_HIGH = _env_float("AUTORESEARCH_POS_RATE_RATIO_HIGH", 3.5)
PROMOTION_POS_RATE_RATIO_LOW = _env_float("AUTORESEARCH_POS_RATE_RATIO_LOW", 0.1)
PROMOTION_THRESHOLD_EDGE_LOW = _env_float("AUTORESEARCH_THRESHOLD_EDGE_LOW", 0.03)
PROMOTION_THRESHOLD_EDGE_HIGH = _env_float("AUTORESEARCH_THRESHOLD_EDGE_HIGH", 0.94)
PROMOTION_WEAK_AP_LIFT = _env_float("AUTORESEARCH_WEAK_AP_LIFT", 1.25)
QUALITY_AP_WEIGHT = _env_float("AUTORESEARCH_QUALITY_AP_WEIGHT", 0.20)
QUALITY_F05_WEIGHT = _env_float("AUTORESEARCH_QUALITY_F05_WEIGHT", 0.10)
QUALITY_CALIBRATION_PENALTY_WEIGHT = _env_float("AUTORESEARCH_QUALITY_CALIBRATION_PENALTY_WEIGHT", 0.05)
QUALITY_RATIO_HIGH = _env_float("AUTORESEARCH_QUALITY_RATIO_HIGH", 3.0)
QUALITY_RATIO_LOW = _env_float("AUTORESEARCH_QUALITY_RATIO_LOW", 0.25)
QUALITY_RATIO_PENALTY_CAP = _env_float("AUTORESEARCH_QUALITY_RATIO_PENALTY_CAP", 0.25)
QUALITY_RATIO_PENALTY_SLOPE = _env_float("AUTORESEARCH_QUALITY_RATIO_PENALTY_SLOPE", 0.03)
QUALITY_MISSING_FIXED_THRESHOLD_PENALTY = _env_float("AUTORESEARCH_QUALITY_MISSING_FIXED_THRESHOLD_PENALTY", 0.03)
QUALITY_WEAK_FIXED_THRESHOLD_PENALTY = _env_float("AUTORESEARCH_QUALITY_WEAK_FIXED_THRESHOLD_PENALTY", 0.05)
QUALITY_PROMOTION_ELIGIBLE_BONUS = _env_float("AUTORESEARCH_QUALITY_PROMOTION_ELIGIBLE_BONUS", 0.05)
QUALITY_WARNING_PENALTY = _env_float("AUTORESEARCH_QUALITY_WARNING_PENALTY", 0.01)
COST_TIER_TORCH_SAMPLE_NORMAL_MAX = _env_int("AUTORESEARCH_COST_TIER_TORCH_SAMPLE_NORMAL_MAX", 2048)
COST_TIER_NUMPY_PIXEL_MAX = _env_int("AUTORESEARCH_COST_TIER_NUMPY_PIXEL_MAX", 600000)
COST_TIER_EPOCH_NORMAL_MAX = _env_int("AUTORESEARCH_COST_TIER_EPOCH_NORMAL_MAX", 8)
_LINKED_LOO_SUMMARY_CACHE_AT = 0.0
_LINKED_LOO_SUMMARY_CACHE: list[Dict[str, Any]] | None = None
_LINKED_LOO_SUMMARY_CACHE_ROOT: Path | None = None
_LINKED_LOO_SUMMARY_CACHE_LOCK = threading.Lock()
LOGGER = logging.getLogger(__name__)
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
    ("dataset", "num_workers"),
    ("dataset", "prefetch_factor"),
    ("dataset", "pin_memory"),
    ("training", "epochs"),
    ("training", "batch_size"),
    ("training", "learning_rate"),
    ("training", "weight_decay"),
    ("training", "pos_weight"),
    ("training", "max_train_samples"),
    ("training", "max_train_pixels"),
    ("training", "sample_positive_fraction"),
    ("training", "dice_loss_weight"),
    ("training", "focal_loss_weight"),
    ("training", "focal_alpha"),
    ("training", "focal_gamma"),
    ("training", "combo_loss_weight"),
    ("training", "combo_bce_weight"),
    ("training", "combo_dice_weight"),
    ("training", "positive_rate_loss_weight"),
    ("training", "positive_rate_loss_tolerance"),
    ("training", "tversky_loss_weight"),
    ("training", "tversky_alpha"),
    ("training", "tversky_beta"),
    ("training", "focal_tversky_gamma"),
    ("training", "augment_flips"),
    ("training", "augment_rotation"),
    ("training", "seed"),
    ("training", "seeds"),
    ("training", "deterministic"),
    ("training", "sampling_strategy"),
    ("training", "stateful_sampler"),
    ("training", "group_stratified_sampling"),
    ("training", "sampling_curriculum"),
    ("training", "hard_negative_fraction"),
    ("evaluation", "threshold"),
    ("evaluation", "use_villa_metrics"),
    ("evaluation", "max_pred_positive_rate_ratio"),
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
    ("dataset", "num_workers"): 8,
    ("dataset", "prefetch_factor"): 4,
    ("dataset", "pin_memory"): False,
    ("training", "epochs"): 5,
    ("training", "batch_size"): None,
    ("training", "learning_rate"): 0.2,
    ("training", "weight_decay"): 0.0,
    ("training", "pos_weight"): 2.0,
    ("training", "max_train_samples"): None,
    ("training", "max_train_pixels"): 600000,
    ("training", "sample_positive_fraction"): None,
    ("training", "dice_loss_weight"): None,
    ("training", "focal_loss_weight"): None,
    ("training", "focal_alpha"): None,
    ("training", "focal_gamma"): None,
    ("training", "combo_loss_weight"): None,
    ("training", "combo_bce_weight"): None,
    ("training", "combo_dice_weight"): None,
    ("training", "positive_rate_loss_weight"): None,
    ("training", "positive_rate_loss_tolerance"): None,
    ("training", "tversky_loss_weight"): None,
    ("training", "tversky_alpha"): None,
    ("training", "tversky_beta"): None,
    ("training", "focal_tversky_gamma"): None,
    ("training", "augment_flips"): None,
    ("training", "augment_rotation"): None,
    ("training", "seed"): DEFAULT_SEED,
    ("training", "seeds"): None,
    ("training", "deterministic"): None,
    ("training", "sampling_strategy"): None,
    ("training", "stateful_sampler"): False,
    ("training", "group_stratified_sampling"): False,
    ("training", "sampling_curriculum"): None,
    ("training", "hard_negative_fraction"): None,
    ("evaluation", "threshold"): 0.5,
    ("evaluation", "use_villa_metrics"): True,
    ("evaluation", "max_pred_positive_rate_ratio"): None,
    ("evaluation", "tta_flips"): None,
}
PIVOT_CONFIGS = (
    "robust_multisegment_dice035_expanded.yaml",
    "robust_calibrated_prloss_w0p03_lr0012_prratio3_seed11018.yaml",
    "robust_tta_seed_ensemble.yaml",
    "residual_25d_torch_unet_cpu.yaml",
    "targeted_promo_fixed_threshold_cap275.yaml",
    "targeted_promo_fixed_threshold_prw006_cap250.yaml",
    "targeted_promo_hardfold_20230530172803_tol005.yaml",
    "targeted_promo_hardfold_dualheldout_22181603_30172803.yaml",
    "targeted_promo_prratio_strict_cap250.yaml",
    "targeted_promo_light_tversky_precision.yaml",
)
BALANCED_CALIBRATION_PATH = ("balanced_calibration",)
BALANCED_CALIBRATION_FIELDS = {
    "max_pred_positive_rate_ratio": ("evaluation", "max_pred_positive_rate_ratio"),
    "positive_rate_loss_tolerance": ("training", "positive_rate_loss_tolerance"),
    "positive_rate_loss_weight": ("training", "positive_rate_loss_weight"),
}
PARAM_BOUNDS: Dict[Tuple[str, ...], tuple[float, float]] = {
    ("training", "pos_weight"): (0.25, 25.0),
    ("training", "learning_rate"): (0.001, 0.25),
    ("training", "weight_decay"): (0.0, 0.05),
    ("evaluation", "threshold"): (0.05, 0.95),
    ("training", "epochs"): (2.0, 20.0),
    ("training", "dice_loss_weight"): (0.0, 0.8),
    ("training", "focal_loss_weight"): (0.0, 0.5),
    ("training", "focal_alpha"): (0.05, 0.95),
    ("training", "focal_gamma"): (0.5, 5.0),
    ("training", "combo_loss_weight"): (0.0, 0.8),
    ("training", "combo_bce_weight"): (0.0, 1.0),
    ("training", "combo_dice_weight"): (0.0, 1.0),
    ("training", "positive_rate_loss_weight"): (0.0, 0.1),
    ("training", "positive_rate_loss_tolerance"): (0.001, 0.02),
    ("training", "tversky_loss_weight"): (0.0, 0.5),
    ("training", "tversky_beta"): (0.1, 0.9),
    ("model", "base_channels"): (4.0, 16.0),
    ("training", "batch_size"): (2.0, 32.0),
    ("training", "max_train_samples"): (1.0, 4096.0),
    ("model", "depth"): (1.0, 3.0),
    ("model", "hidden_units"): (8.0, 96.0),
    ("dataset", "z_offsets"): (-8.0, 8.0),
    ("training", "max_train_pixels"): (100000.0, 1200000.0),
    ("training", "sample_positive_fraction"): (0.05, 0.95),
    ("training", "hard_negative_fraction"): (0.1, 0.9),
    ("evaluation", "max_pred_positive_rate_ratio"): (1.5, 3.5),
}


def _assert_signature_defaults_within_param_bounds() -> None:
    for path, (low, high) in PARAM_BOUNDS.items():
        if path not in SIGNATURE_DEFAULTS:
            continue
        value = SIGNATURE_DEFAULTS[path]
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not low <= float(value) <= high:
            dotted = ".".join(path)
            raise AssertionError(f"SIGNATURE_DEFAULTS[{dotted}]={value!r} outside PARAM_BOUNDS {low}..{high}")


_assert_signature_defaults_within_param_bounds()


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
        raw_limit = os.environ.get("AUTORESEARCH_RECENT_LIMIT", "300")
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
        try:
            runs.append(validate_metric_contract({"run_id": run_id, "timestamp": ts, "config": json.loads(cfg_json), "main_metric": float(metric), "metrics": json.loads(sec_json), "artifact_dir": artifact_dir}))
        except ValueError as exc:
            print(f"Skipping run {run_id} with invalid metric contract: {exc}", file=sys.stderr)
    return runs


class _CycleProfiler:
    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = (os.environ.get("AUTORESEARCH_PROFILE", "0") == "1") if enabled is None else enabled
        self.events: list[tuple[str, float]] = []

    @contextlib.contextmanager
    def measure(self, label: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            if self.enabled:
                self.events.append((label, elapsed))
                print(f"AUTORESEARCH_PROFILE {label} {elapsed:.3f}s", file=sys.stderr, flush=True)


class _RunHistory:
    """Per-cycle cache for decoded experiment rows and derived views."""

    def __init__(self, profiler: _CycleProfiler | None = None) -> None:
        self.profiler = profiler or _CycleProfiler(enabled=False)
        self._recent_by_limit: dict[int | None, list[Dict[str, Any]]] = {}
        self._strategy_by_runs_key: dict[tuple[str, ...], Dict[str, Any]] = {}
        self._ranked_torch_by_runs_key: dict[tuple[str, ...], list[tuple[Dict[str, Any], str]]] = {}
        self._manual_candidate_by_runs_key: dict[tuple[str, ...], Dict[str, Any] | None] = {}

    @staticmethod
    def _runs_cache_key(runs: list[Dict[str, Any]]) -> tuple[str, ...]:
        return tuple(sorted(str(run.get("run_id") or "") for run in runs))

    def recent_runs(self, limit: int | None = None) -> list[Dict[str, Any]]:
        if limit not in self._recent_by_limit:
            with self.profiler.measure("recent_runs"):
                self._recent_by_limit[limit] = _recent_runs(limit=limit)
        return self._recent_by_limit[limit]

    def invalidate(self) -> None:
        self._recent_by_limit.clear()
        self._strategy_by_runs_key.clear()
        self._ranked_torch_by_runs_key.clear()
        self._manual_candidate_by_runs_key.clear()

    def torch_runs(self, runs: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        return [run for run in runs if "torch" in str(_get_nested(run.get("config", {}), ("model", "name"), ""))]

    def promotion_eligible_runs(self, runs: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        return [run for run in runs if _promotion_gate(run)[0]]

    def strategy_phase(self, runs: list[Dict[str, Any]]) -> Dict[str, Any]:
        key = self._runs_cache_key(runs)
        if key not in self._strategy_by_runs_key:
            with self.profiler.measure("strategy_phase"):
                self._strategy_by_runs_key[key] = _strategy_phase(runs)
        return self._strategy_by_runs_key[key]

    def ranked_recent_torch_bases(self, runs: list[Dict[str, Any]]) -> list[tuple[Dict[str, Any], str]]:
        key = self._runs_cache_key(runs)
        if key not in self._ranked_torch_by_runs_key:
            with self.profiler.measure("ranked_recent_torch_bases"):
                self._ranked_torch_by_runs_key[key] = _ranked_recent_torch_bases(runs)
        return self._ranked_torch_by_runs_key[key]

    def manual_promotion_candidate(self, runs: list[Dict[str, Any]]) -> Dict[str, Any] | None:
        key = self._runs_cache_key(runs)
        if key not in self._manual_candidate_by_runs_key:
            with self.profiler.measure("manual_promotion_candidate"):
                self._manual_candidate_by_runs_key[key] = _manual_promotion_candidate(runs)
        return self._manual_candidate_by_runs_key[key]


def _best_base_config(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not runs:
        return cast(Dict[str, Any], load_config(BASELINE))
    baseline = cast(Dict[str, Any], load_config(BASELINE))
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


def _candidate_is_noop(cfg: Dict[str, Any], path: Tuple[str, ...], value: Any) -> bool:
    if path == BALANCED_CALIBRATION_PATH:
        return all(
            _get_nested(cfg, BALANCED_CALIBRATION_FIELDS[key], None) == proposed
            for key, proposed in value.items()
        )
    return bool(_get_nested(cfg, path, None) == value)


def _apply_candidate(cfg: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    if path == BALANCED_CALIBRATION_PATH:
        for key, proposed in value.items():
            _set_nested(cfg, BALANCED_CALIBRATION_FIELDS[key], proposed)
        return
    _set_nested(cfg, path, value)


def _get_nested(cfg: Dict[str, Any], path: Tuple[str, ...], default: Any) -> Any:
    cur: Any = cfg
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _clamp_param(path: Tuple[str, ...], value: Any) -> Any:
    """Clamp mutable numeric proposal values to safe autoresearch bounds."""
    bounds = PARAM_BOUNDS.get(path)
    if bounds is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    low, high = bounds
    clamped = min(high, max(low, float(value)))
    return int(clamped) if isinstance(value, int) else clamped


def _bounded_candidates(candidates: list[tuple[Tuple[str, ...], Any, str]]) -> list[tuple[Tuple[str, ...], Any, str]]:
    """Apply `PARAM_BOUNDS` to generated proposal candidates."""
    return [(path, _clamp_param(path, value), reason) for path, value, reason in candidates]


def _proposal_value_slug(value: Any) -> str:
    """Return a filesystem-safe short slug for generated proposal names."""
    if isinstance(value, list):
        text = f"{len(value)}items"
    elif isinstance(value, dict):
        text = "_".join(f"{key}_{value[key]}" for key in sorted(value))
    else:
        text = str(value).replace(".", "p")
    slug = "".join(ch if ch.isalnum() or ch in {"_", "-", "p"} else "_" for ch in text)
    return slug[:80]


COST_TIER_ORDER = {"cheap": 0, "normal": 1, "expensive": 2}


def _cost_tier_rank(tier: str) -> int:
    return COST_TIER_ORDER.get(tier, COST_TIER_ORDER["expensive"])


def _config_cost_tier(cfg: Dict[str, Any]) -> str:
    """Classify expected local runtime cost without changing experiment behavior."""
    model_name = str(_get_nested(cfg, ("model", "name"), "tiny_numpy_ink_logreg"))
    training = cast(dict[str, Any], cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {})
    evaluation = cast(dict[str, Any], cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation"), dict) else {})
    epochs = int(cast(int | float | str, training.get("epochs") or SIGNATURE_DEFAULTS[("training", "epochs")]))
    max_train_samples = int(training.get("max_train_samples") or 0)
    max_train_pixels = int(cast(int | float | str, training.get("max_train_pixels") or SIGNATURE_DEFAULTS[("training", "max_train_pixels")]))
    seeds = training.get("seeds")
    ensemble_size = len(seeds) if isinstance(seeds, list) else 1
    if ensemble_size > 1 or bool(evaluation.get("tta_flips")) or max_train_samples > COST_TIER_TORCH_SAMPLE_NORMAL_MAX or max_train_pixels > COST_TIER_NUMPY_PIXEL_MAX or epochs > COST_TIER_EPOCH_NORMAL_MAX:
        return "expensive"
    if model_name == "tiny_numpy_ink_logreg" and epochs <= 5 and max_train_pixels <= COST_TIER_NUMPY_PIXEL_MAX:
        return "cheap"
    if model_name == "tiny_numpy_mlp" and max_train_pixels <= COST_TIER_NUMPY_PIXEL_MAX and epochs <= COST_TIER_EPOCH_NORMAL_MAX:
        return "normal"
    if "torch" in model_name and (max_train_samples == 0 or max_train_samples <= COST_TIER_TORCH_SAMPLE_NORMAL_MAX) and epochs <= COST_TIER_EPOCH_NORMAL_MAX:
        return "normal"
    return "expensive"


def _max_allowed_cost_tier(allow_expensive: bool = False) -> str:
    if allow_expensive:
        return "expensive"
    tier = os.environ.get("AUTORESEARCH_MAX_COST_TIER", "normal").lower()
    return tier if tier in COST_TIER_ORDER else "normal"


def _cost_tier_allowed(tier: str, *, allow_expensive: bool = False) -> bool:
    return _cost_tier_rank(tier) <= _cost_tier_rank(_max_allowed_cost_tier(allow_expensive))


def _strategy_allows_expensive(strategy: Dict[str, Any] | None) -> bool:
    if not strategy:
        return False
    return bool(strategy.get("plateau")) or str(strategy.get("phase") or "") == "promote"


def _metadata_allows_expensive(cfg: Dict[str, Any]) -> bool:
    autoresearch = cfg.get("autoresearch", {}) if isinstance(cfg.get("autoresearch"), dict) else {}
    return str(autoresearch.get("run_profile") or "") == "promotion" or str(autoresearch.get("intent") or "") == "promotion_action"


def _normalize_signature_value(path: Tuple[str, ...], value: Any) -> Any:
    if path == ("model", "depth") and isinstance(value, int) and value > 3:
        return 3
    if isinstance(value, dict):
        return tuple((key, _normalize_signature_value(path + (str(key),), value[key])) for key in sorted(value))
    if isinstance(value, list):
        return tuple(_normalize_signature_value(path + (str(index),), item) for index, item in enumerate(value))
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


def _prune_stale_configs(max_age_hours: float = 48) -> int:
    """Delete stale generated `auto_*.yaml` configs older than the given age."""
    if max_age_hours <= 0:
        return 0
    cutoff = time.time() - max_age_hours * 3600
    deleted = 0
    for path in CONFIGS.glob("auto_*.yaml"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                deleted += 1
        except FileNotFoundError:
            continue
    return deleted


def _proposal_candidates(base: Dict[str, Any]) -> list[tuple[Tuple[str, ...], Any, str]]:
    model_name = str(_get_nested(base, ("model", "name"), "tiny_numpy_ink_logreg"))
    lr = float(_get_nested(base, ("training", "learning_rate"), 0.2))
    epochs = int(_get_nested(base, ("training", "epochs"), 5))
    weight_decay = float(_get_nested(base, ("training", "weight_decay"), 0.0))
    seed = int(_get_nested(base, ("training", "seed"), DEFAULT_SEED))
    pos_weight_raw = _get_nested(base, ("training", "pos_weight"), 2.0)
    pos_weight = 2.0 if isinstance(pos_weight_raw, str) and pos_weight_raw.lower() == "auto" else float(pos_weight_raw)

    if "torch" in model_name:
        base_channels = int(_get_nested(base, ("model", "base_channels"), 8))
        batch_size = int(_get_nested(base, ("training", "batch_size"), 8))
        max_train_samples = int(_get_nested(base, ("training", "max_train_samples"), 512) or 0)
        dice = float(_get_nested(base, ("training", "dice_loss_weight"), 0.0) or 0.0)
        prloss = float(_get_nested(base, ("training", "positive_rate_loss_weight"), 0.0) or 0.0)
        prtol = float(_get_nested(base, ("training", "positive_rate_loss_tolerance"), 0.02) or 0.02)
        pred_ratio_cap = float(_get_nested(base, ("evaluation", "max_pred_positive_rate_ratio"), 3.0) or 3.0)
        sampling_strategy = _get_nested(base, ("training", "sampling_strategy"), None)
        sampling_curriculum = _get_nested(base, ("training", "sampling_curriculum"), None)
        augment_flips = bool(_get_nested(base, ("training", "augment_flips"), False))
        augment_rotation = bool(_get_nested(base, ("training", "augment_rotation"), False))
        focal_loss_weight = float(_get_nested(base, ("training", "focal_loss_weight"), 0.0) or 0.0)
        tta_flips = bool(_get_nested(base, ("evaluation", "tta_flips"), False))
        bounded_samples = max_train_samples if max_train_samples > 0 else 1024
        candidates = [
            (("training", "learning_rate"), round(max(0.0002, lr * 0.6), 6), "lower torch learning rate to test calibration on the current robust/residual base"),
            (("training", "learning_rate"), round(min(0.006, lr * 1.5), 6), "raise torch learning rate modestly to test convergence-limited behavior"),
            (("training", "positive_rate_loss_tolerance"), round(max(0.001, prtol * 0.5), 4), "tighten positive-rate loss tolerance using recent promotion-ready residual evidence"),
            (("training", "positive_rate_loss_tolerance"), round(min(0.05, prtol * 1.5), 4), "relax positive-rate loss tolerance to recover F1 when ranking is strong"),
            (("training", "positive_rate_loss_tolerance"), 0.008, "test balanced tighter positive-rate loss tolerance from recent calibration evidence"),
            (("training", "positive_rate_loss_weight"), round(max(0.0, prloss - 0.02), 4), "reduce positive-rate loss weight to test probability calibration spread"),
            (("training", "positive_rate_loss_weight"), round(min(0.1, prloss + 0.02), 4), "increase positive-rate loss weight to tighten prediction rate toward the cap"),
            (("training", "dice_loss_weight"), round(max(0.0, dice - 0.15), 4), "reduce Dice weight to test whether BCE precision improves"),
            (("training", "dice_loss_weight"), round(min(0.8, dice + 0.15), 4), "increase Dice weight to test ink-recall stability"),
            (("training", "focal_loss_weight"), 0.1, "add a light focal BCE term to test sparse ink recall"),
            (("training", "focal_loss_weight"), 0.2, "add a stronger focal BCE term to focus sparse ink learning"),
            (("training", "tversky_loss_weight"), 0.15, "add a light Tversky term to test recall/precision balance on the current robust base"),
            (("training", "tversky_beta"), 0.8, "bias Tversky toward false-negative reduction for rare ink recall"),
            (("training", "sampling_strategy"), "hard_mining", "try hard-negative mining to improve precision against textured non-ink"),
            (("training", "sampling_curriculum"), "warmup_then_hard", "warm up with uniform sampling before switching to hard mining"),
            (("evaluation", "threshold"), 0.35, "evaluate a calibrated fixed threshold closer to recent swept-F1 optima"),
            (("evaluation", "max_pred_positive_rate_ratio"), 2.5 if pred_ratio_cap >= 3.0 else 3.0, "test positive-rate cap in the 2.5-3.0 band that retained F1 in recent cap sweeps"),
            (("evaluation", "max_pred_positive_rate_ratio"), 3.0 if pred_ratio_cap < 3.0 else 3.5, "test a slightly looser positive-rate cap when AP lift is strong but strict caps suppress F1"),
            (("evaluation", "max_pred_positive_rate_ratio"), 2.75, "test intermediate positive-rate cap between 2.5 and 3.0 for balanced calibration"),
            (BALANCED_CALIBRATION_PATH, {"max_pred_positive_rate_ratio": 2.75, "positive_rate_loss_tolerance": 0.008}, "combine intermediate positive-rate cap with tighter tolerance for balanced calibration"),
            (BALANCED_CALIBRATION_PATH, {"max_pred_positive_rate_ratio": 3.0, "positive_rate_loss_weight": 0.10}, "combine maximum positive-rate loss weight with 3.0 cap to diagnose blocking fold 20230530172803"),
            (("model", "base_channels"), max(4, base_channels // 2), "smaller torch U-Net width for faster regularized CPU search"),
            (("model", "base_channels"), min(16, base_channels * 2), "larger torch U-Net width to test capacity without changing data scope"),
            (("training", "epochs"), max(2, epochs - 1), "shorter torch training to test overfit/probability inflation"),
            (("training", "epochs"), min(8, epochs + 1), "one extra torch epoch to test under-convergence"),
            (("training", "batch_size"), max(2, batch_size // 2), "smaller torch batch for noisier but possibly better CPU generalization"),
            (("training", "max_train_samples"), bounded_samples, "bound full robust training to a CPU-safe sample budget for cron exploration"),
            (("training", "augment_flips"), not augment_flips, "toggle train-time flip augmentation on this torch base"),
            (("training", "augment_rotation"), not augment_rotation, "toggle train-time 90-degree rotation augmentation on this torch base"),
            (("evaluation", "tta_flips"), not tta_flips, "toggle test-time flip TTA to measure ensemble-like lift"),
            (("training", "seed"), seed + (SEED_REPEAT_DELTAS[0] if SEED_REPEAT_DELTAS else 17), "repeat torch setup with a deterministic seed change"),
        ]
        if focal_loss_weight > 0.0:
            candidates.extend([
                (("training", "focal_gamma"), 2.0, "use standard focal gamma for sparse ink classification"),
                (("training", "focal_gamma"), 3.0, "increase focal gamma to focus harder on confusing negatives"),
            ])
        if sampling_strategy == "hard_mining":
            candidates.append((("training", "hard_negative_fraction"), 0.85, "raise hard-negative fraction toward recent precision-oriented residual configs"))
        if sampling_curriculum:
            candidates = [candidate for candidate in candidates if candidate[0] != ("training", "sampling_curriculum")]
        sample_cap = int(os.environ.get("AUTORESEARCH_TORCH_MAX_TRAIN_SAMPLES", "1024"))
        if max_train_samples and max_train_samples < 2048 and sample_cap >= 2048:
            candidates.append((("training", "max_train_samples"), 2048, "increase robust torch sample budget after local hyperparameter plateau"))
        if max_train_samples and max_train_samples < 4096 and sample_cap >= 4096:
            candidates.append((("training", "max_train_samples"), 4096, "increase robust torch sample budget to the recent 4096-sample residual setting with stronger LOO evidence"))
        return _bounded_candidates(candidates)

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
            (("training", "seed"), seed + (SEED_REPEAT_DELTAS[0] if SEED_REPEAT_DELTAS else 17), "repeat the selected setup with a different deterministic sampling/initialization seed"),
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
    return _bounded_candidates(candidates)


def _mutation_family(path: Tuple[str, ...]) -> str:
    if path == BALANCED_CALIBRATION_PATH:
        return "balanced_calibration"
    if path in {("training", "learning_rate"), ("training", "weight_decay"), ("training", "epochs"), ("training", "batch_size")}:
        return "optimizer"
    if path in {("training", "pos_weight"), ("training", "dice_loss_weight"), ("training", "focal_loss_weight"), ("training", "focal_gamma"), ("training", "positive_rate_loss_weight"), ("training", "positive_rate_loss_tolerance"), ("training", "tversky_loss_weight"), ("training", "tversky_alpha"), ("training", "tversky_beta"), ("training", "focal_tversky_gamma")}:
        return "loss_calibration"
    if path in {("training", "sampling_strategy"), ("training", "stateful_sampler"), ("training", "group_stratified_sampling"), ("training", "sampling_curriculum"), ("training", "hard_negative_fraction"), ("training", "max_train_samples"), ("training", "max_train_pixels"), ("training", "sample_positive_fraction"), ("training", "augment_flips"), ("training", "augment_rotation")}:
        return "data_sampling"
    if path in {("model", "name"), ("model", "input_mode"), ("model", "base_channels"), ("model", "depth"), ("model", "hidden_units")}:
        return "model_family"
    if path in {("evaluation", "threshold"), ("evaluation", "max_pred_positive_rate_ratio"), ("evaluation", "tta_flips")}:
        return "inference_calibration"
    if path in {("training", "seed"), ("training", "seeds"), ("training", "deterministic")}:
        return "replication"
    return "other"


def _propose_configs(base: Dict[str, Any], runs: List[Dict[str, Any]], count: int = 3, *, scope_policy: str = "focused_pair_only", lock_to_baseline_scope: bool = True, name_index_offset: int = 0, required_families: set[str] | None = None, strategy_phase: str | None = None, allow_expensive: bool = False) -> List[Tuple[str, Dict[str, Any], str]]:
    """Change only 1 hyperparameter per proposal for interpretable search."""
    baseline_dataset = load_config(BASELINE).get("dataset", {})
    tested = _reserved_signatures(runs)
    candidates = _proposal_candidates(base)
    search_strategy_name = os.environ.get("AUTORESEARCH_SEARCH_STRATEGY", "heuristic")
    search_strategy_metadata: dict[str, Any] = {}
    search_strategy_candidate: Tuple[str, ...] | None = None
    search_strategy_label: str | None = None
    if search_strategy_name.strip().lower() not in {"", "heuristic", "random", "current", "default"}:
        from src.autoresearch.search_strategy import SearchContext, strategy_from_env
        strategy = strategy_from_env(search_strategy_name)
        search_strategy_label = getattr(strategy, "name", search_strategy_name)
        context = SearchContext(
            proposal_count=count,
            strategy_phase=strategy_phase,
            required_families=required_families,
            allow_expensive=allow_expensive,
            scope_policy=scope_policy,
            lock_to_baseline_scope=lock_to_baseline_scope,
            name_index_offset=name_index_offset,
        )
        propose_candidate = getattr(strategy, "propose_candidate", None)
        if callable(propose_candidate):
            asked = propose_candidate(base, candidates, runs, context, PARAM_BOUNDS)
            if asked is not None:
                candidates = [asked] + [candidate for candidate in candidates if candidate[0] != asked[0]]
                search_strategy_metadata = dict(getattr(strategy, "last_trial_metadata", {}) or {})
                search_strategy_candidate = asked[0]
            else:
                candidates = strategy.order_candidates(candidates, runs)
        else:
            candidates = strategy.order_candidates(candidates, runs)
    # Rotate deterministically by minute slot so cron does not emit identical batches forever.
    else:
        slot = int(datetime.now(timezone.utc).strftime("%M")) // 10
        candidates = candidates[slot:] + candidates[:slot]
    proposals: List[Tuple[str, Dict[str, Any], str]] = []
    deferred_expensive: list[tuple[str, Dict[str, Any], str, str, Tuple[Any, ...]]] = []
    used_families: set[str] = set()
    seen_batch_signatures: set[Tuple[Any, ...]] = set()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    expensive_context = allow_expensive or strategy_phase in {"promote", "diversify"} or _metadata_allows_expensive(base)
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
        if _candidate_is_noop(cfg, path, value):
            continue
        _apply_candidate(cfg, path, value)
        if path == ("training", "sampling_strategy") and value == "hard_mining":
            cfg.setdefault("training", {}).setdefault("hard_negative_fraction", 0.5)
        if path == ("training", "tversky_loss_weight"):
            cfg.setdefault("training", {}).setdefault("tversky_alpha", 0.3)
            cfg.setdefault("training", {}).setdefault("tversky_beta", 0.7)
        if path == ("training", "tversky_beta"):
            cfg.setdefault("training", {}).setdefault("tversky_loss_weight", 0.15)
            cfg.setdefault("training", {})["tversky_alpha"] = round(1.0 - float(value), 4)
        cost_tier = _config_cost_tier(cfg)
        if not _cost_tier_allowed(cost_tier, allow_expensive=expensive_context):
            print(f"Skipping {cost_tier} proposal {'.'.join(path)} under AUTORESEARCH_MAX_COST_TIER={_max_allowed_cost_tier(expensive_context)}")
            continue
        signature = _search_signature(cfg)
        if signature in tested or signature in seen_batch_signatures:
            print(f"Skipping already-tested search signature {signature}")
            continue
        autoresearch = cfg.setdefault("autoresearch", {})
        autoresearch["parent_reason"] = reason
        autoresearch["scope_policy"] = scope_policy
        autoresearch["search_signature"] = list(signature)
        autoresearch["intent"] = "cron_exploration"
        autoresearch["run_profile"] = "exploration"
        autoresearch["promotable"] = False
        autoresearch["proposal_status"] = "generated"
        autoresearch["changed_path"] = ".".join(path)
        autoresearch["mutation_family"] = family
        autoresearch["cost_tier"] = cost_tier
        if search_strategy_label:
            autoresearch.setdefault("search_strategy", search_strategy_label)
        if search_strategy_candidate == path:
            autoresearch.update(search_strategy_metadata)
        if strategy_phase:
            autoresearch["strategy_phase"] = strategy_phase
        autoresearch["promotion_required"] = ["seed_repeat_leave_one_out", "full_tile_validation", "promotion_checks_eligible"]
        name = f"auto_{stamp}_{name_index_offset + len(proposals) + 1}_{'_'.join(path)}_{_proposal_value_slug(value)}.yaml"
        if cost_tier == "expensive":
            deferred_expensive.append((name, cfg, reason, family, signature))
            continue
        seen_batch_signatures.add(signature)
        proposals.append((name, cfg, reason))
        used_families.add(family)
        if len(proposals) >= count:
            break
    for _name, cfg, reason, family, signature in deferred_expensive:
        if len(proposals) >= count:
            break
        if required_families and family in used_families:
            continue
        if signature in seen_batch_signatures:
            continue
        name = f"auto_{stamp}_{name_index_offset + len(proposals) + 1}_{cfg['autoresearch']['changed_path'].replace('.', '_')}_{_proposal_value_slug(_get_nested(cfg, tuple(cfg['autoresearch']['changed_path'].split('.')), 'combined'))}.yaml"
        seen_batch_signatures.add(signature)
        proposals.append((name, cfg, reason))
        used_families.add(family)
    proposals.sort(key=lambda item: _cost_tier_rank(item[1].get("autoresearch", {}).get("cost_tier", "expensive")))
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
        if ratio > PROMOTION_POS_RATE_RATIO_HIGH or ratio < PROMOTION_POS_RATE_RATIO_LOW:
            warnings.append("pred_positive_rate_ratio_suspicious")
    if "fixed_threshold_status" not in metrics:
        warnings.append("missing_fixed_threshold_status")
    fixed_threshold_status = str(metrics.get("fixed_threshold_status") or "").lower()
    if fixed_threshold_status and fixed_threshold_status != "ok":
        warnings.append("fixed_threshold_status_weak")
    best_threshold = _optional_float(metrics.get("best_threshold"))
    if best_threshold is not None and (best_threshold <= PROMOTION_THRESHOLD_EDGE_LOW or best_threshold >= PROMOTION_THRESHOLD_EDGE_HIGH):
        warnings.append("best_threshold_at_sweep_edge")
    ap_prevalence_lift = _optional_float(metrics.get("ap_prevalence_lift"))
    if ap_prevalence_lift is not None and ap_prevalence_lift < PROMOTION_WEAK_AP_LIFT:
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
    score += QUALITY_AP_WEIGHT * float(metrics.get("average_precision") or 0.0)
    score += QUALITY_F05_WEIGHT * float(metrics.get("val_f05") or 0.0)
    pred_rate = metrics.get("pred_positive_rate")
    val_rate = metrics.get("val_positive_rate")
    if pred_rate is not None and val_rate is not None:
        ratio = float(pred_rate) / max(float(val_rate), 1e-6)
        score -= QUALITY_CALIBRATION_PENALTY_WEIGHT * min(abs(ratio - 1.0), 0.5)
        if ratio > QUALITY_RATIO_HIGH:
            score -= min(QUALITY_RATIO_PENALTY_CAP, QUALITY_RATIO_PENALTY_SLOPE * (ratio - QUALITY_RATIO_HIGH))
        elif ratio < QUALITY_RATIO_LOW:
            score -= min(QUALITY_RATIO_PENALTY_CAP, QUALITY_RATIO_PENALTY_SLOPE * (QUALITY_RATIO_LOW / max(ratio, 1e-6)))
    if "fixed_threshold_status" not in metrics:
        score -= QUALITY_MISSING_FIXED_THRESHOLD_PENALTY
    elif str(metrics.get("fixed_threshold_status") or "").lower() != "ok":
        score -= QUALITY_WEAK_FIXED_THRESHOLD_PENALTY
    eligible, warnings = _promotion_gate(run)
    if eligible:
        score += QUALITY_PROMOTION_ELIGIBLE_BONUS
    else:
        score -= QUALITY_WARNING_PENALTY * len(warnings)
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
        families = {"replication", "inference_calibration", "loss_calibration", "balanced_calibration"}
    elif plateau:
        phase = "diversify"
        families = {"loss_calibration", "data_sampling", "model_family", "inference_calibration", "balanced_calibration"}
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


def _record_promotion_status(run_id: str, status: str, payload: Dict[str, Any] | None = None) -> None:
    """Persist automated promotion pipeline status for dashboard and audits."""
    init_db(DB_PATH)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO promotion_results(run_id, timestamp, status, payload_json) VALUES (?, ?, ?, ?)",
            (run_id, datetime.now(timezone.utc).isoformat(), status, json.dumps(payload or {}, sort_keys=True)),
        )


def _reported_promotion_outputs(stdout: str) -> dict[str, str]:
    decoder = json.JSONDecoder()
    payload = None
    text = stdout.strip()
    try:
        payload = json.loads(text)
    except Exception:
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                parsed, _end = decoder.raw_decode(text[index:])
            except Exception:
                continue
            if isinstance(parsed, dict):
                payload = parsed
                if isinstance(parsed.get("outputs"), dict):
                    break
        if payload is None:
            return {}
    if not isinstance(payload, dict):
        return {}
    outputs = payload.get("outputs") if isinstance(payload, dict) else None
    if not isinstance(outputs, dict):
        return {}
    return {str(key): str(value) for key, value in outputs.items() if value}


def _promotion_output_exists(path_text: str) -> bool:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path.exists()


def _promotion_output_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _append_promotion_reason(reasons: list[str], reason: str) -> None:
    if reason and reason not in reasons:
        reasons.append(reason)


def _promotion_payload_reasons(payload: Dict[str, Any], seed_reasons: list[str] | None = None) -> list[str]:
    reasons = list(seed_reasons or [])
    error = str(payload.get("error") or "")
    if payload.get("missing_outputs"):
        _append_promotion_reason(reasons, "reported_artifact_path_not_found")
    if payload.get("empty_outputs"):
        _append_promotion_reason(reasons, "reported_artifact_empty")
    if payload.get("missing_output_keys"):
        _append_promotion_reason(reasons, "full_tile_output_set_incomplete")
    if payload.get("outside_output_dir"):
        _append_promotion_reason(reasons, "full_tile_output_outside_output_dir")
    if payload.get("metrics_json_error"):
        _append_promotion_reason(reasons, "full_tile_metrics_json_invalid")
    if payload.get("stale_summary_json"):
        _append_promotion_reason(reasons, "summary_json_stale")
    if payload.get("returncode") not in (None, 0):
        _append_promotion_reason(reasons, "promotion_command_nonzero_exit")
    if payload.get("outputs_verified"):
        _append_promotion_reason(reasons, "reported_artifacts_verified")
    if payload.get("summary"):
        _append_promotion_reason(reasons, "summary_json_verified")
    if "no current summary_json or validated outputs" in error:
        _append_promotion_reason(reasons, "summary_json_missing")
    if "missing promotion checks" in error:
        _append_promotion_reason(reasons, "full_tile_metrics_missing_promotion_checks")
    if "timed out" in error:
        _append_promotion_reason(reasons, "promotion_command_timeout")
    if error and not reasons:
        _append_promotion_reason(reasons, "promotion_command_failed")
    return reasons


def _validate_reported_promotion_outputs(command_args: list[str], outputs: dict[str, str]) -> tuple[bool, dict[str, Any]]:
    payload: dict[str, Any] = {"outputs": outputs}
    missing_outputs = [path for path in outputs.values() if not _promotion_output_exists(path)]
    if missing_outputs:
        payload["error"] = "promotion command reported outputs that are missing"
        payload["missing_outputs"] = missing_outputs
        return False, payload
    empty_outputs = [path for path in outputs.values() if _promotion_output_path(path).is_file() and _promotion_output_path(path).stat().st_size == 0]
    if empty_outputs:
        payload["error"] = "promotion command reported empty output files"
        payload["empty_outputs"] = empty_outputs
        return False, payload
    if any(str(part).endswith("scripts/infer_full_tile.py") for part in command_args):
        required = {"metrics_json", "probability_map", "threshold_csv"}
        missing_keys = sorted(required - set(outputs))
        if missing_keys:
            payload["error"] = "full-tile promotion output set is incomplete"
            payload["missing_output_keys"] = missing_keys
            return False, payload
        try:
            output_dir = command_args[command_args.index("--output-dir") + 1]
        except (ValueError, IndexError):
            output_dir = None
        if output_dir:
            root = _promotion_output_path(output_dir).resolve()
            outside = []
            for key in required:
                path = _promotion_output_path(outputs[key]).resolve()
                if path.parent != root:
                    outside.append(outputs[key])
            if outside:
                payload["error"] = "full-tile outputs are outside the requested output directory"
                payload["outside_output_dir"] = outside
                return False, payload
        try:
            metrics = json.loads(_promotion_output_path(outputs["metrics_json"]).read_text())
        except Exception as exc:
            payload["error"] = "full-tile metrics_json is not valid JSON"
            payload["metrics_json_error"] = str(exc)
            return False, payload
        if not isinstance(metrics, dict) or "promotion_checks" not in metrics:
            payload["error"] = "full-tile metrics_json is missing promotion checks"
            return False, payload
        payload["outputs_verified"] = True
    else:
        payload["outputs_verified"] = True
    return True, payload


def _run_automated_promotion(command_args: list[str], candidate_run_id: str, summary_json: Path, timeout: int, promotion_failure_reasons: list[str] | None = None) -> dict[str, Any]:
    """Run promotion evidence and record summary or artifact-output status."""
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"promotion_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.log"
    if any(str(part).endswith("scripts/evaluate_leave_one_out.py") for part in command_args):
        try:
            summary_json = _promotion_output_path(command_args[command_args.index("--summary-json") + 1])
        except (ValueError, IndexError):
            pass
    status = "FAILED"
    payload: Dict[str, Any] = {"command": command_args, "log_file": str(log_path)}
    if any(str(part).endswith("scripts/infer_full_tile.py") for part in command_args):
        try:
            output_dir = _promotion_output_path(command_args[command_args.index("--output-dir") + 1])
        except (ValueError, IndexError):
            output_dir = None
        if output_dir is not None:
            existing_outputs = {
                "metrics_json": str(output_dir / "metrics.json"),
                "probability_map": str(output_dir / "probability_map.npy"),
                "threshold_csv": str(output_dir / "metrics_by_threshold.csv"),
            }
            if all(_promotion_output_exists(path) for path in existing_outputs.values()):
                valid_outputs, output_payload = _validate_reported_promotion_outputs(command_args, existing_outputs)
                payload.update({"returncode": 0, "reused_existing_outputs": True})
                payload.update(output_payload)
                if valid_outputs:
                    status = "SUCCEEDED_ARTIFACTS"
                    payload["promotion_failure_reasons"] = _promotion_payload_reasons(payload, promotion_failure_reasons)
                    payload["diagnostic_summary"] = {
                        "status": status,
                        "summary_json_verified": False,
                        "artifacts_verified": True,
                        "reused_existing_outputs": True,
                        "reasons": payload["promotion_failure_reasons"],
                    }
                    log_path.write_text(
                        f"$ {_shell_command(command_args)}\n\nREUSED_EXISTING_OUTPUTS\n"
                        + json.dumps(existing_outputs, indent=2, sort_keys=True)
                        + "\n\nPROMOTION_DIAGNOSTICS\n"
                        + json.dumps(payload["diagnostic_summary"], indent=2, sort_keys=True)
                        + "\n"
                    )
                    _record_promotion_status(candidate_run_id, status, payload)
                    return {"automation_status": status, "promotion_log": str(log_path), "promotion_payload": payload}
    try:
        started_at = time.time()
        completed = subprocess.run(command_args, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False)
        log_path.write_text(f"$ {_shell_command(command_args)}\n\nSTDOUT\n{completed.stdout}\n\nSTDERR\n{completed.stderr}\n")
        payload.update({"returncode": completed.returncode})
        if completed.returncode == 0:
            payload["summary_json"] = str(summary_json)
            summary_is_current = summary_json.exists() and summary_json.stat().st_mtime >= started_at
            if summary_is_current:
                summary = json.loads(summary_json.read_text())
                status = "SUCCEEDED"
                payload["summary"] = summary
            else:
                if summary_json.exists():
                    payload["stale_summary_json"] = str(summary_json)
                outputs = _reported_promotion_outputs(completed.stdout)
                if outputs:
                    valid_outputs, output_payload = _validate_reported_promotion_outputs(command_args, outputs)
                    payload.update(output_payload)
                    if valid_outputs:
                        status = "SUCCEEDED_ARTIFACTS"
                elif any(str(part).endswith(("scripts/plan_hard_negative_retrain.py", "scripts/compare_threshold_caps.py")) for part in command_args):
                    try:
                        diagnostic = json.loads(completed.stdout)
                    except Exception:
                        diagnostic = {"stdout": completed.stdout[-4000:]}
                    payload["diagnostic"] = diagnostic
                    status = "SUCCEEDED_DIAGNOSTIC"
                else:
                    payload.update({"error": "promotion command succeeded but produced no current summary_json or validated outputs", "path": str(summary_json)})
        else:
            payload["error"] = f"promotion command exited {completed.returncode}"
    except subprocess.TimeoutExpired as exc:
        log_path.write_text(f"$ {_shell_command(command_args)}\n\nTIMEOUT after {timeout}s\n{exc}\n")
        payload["error"] = f"promotion command timed out after {timeout}s"
    except Exception as exc:
        log_path.write_text(f"$ {_shell_command(command_args)}\n\nERROR\n{exc}\n")
        payload["error"] = str(exc)
    payload["promotion_failure_reasons"] = _promotion_payload_reasons(payload, promotion_failure_reasons)
    payload["diagnostic_summary"] = {
        "status": status,
        "summary_json_verified": "summary_json_verified" in payload["promotion_failure_reasons"],
        "artifacts_verified": bool(payload.get("outputs_verified")),
        "reasons": payload["promotion_failure_reasons"],
    }
    try:
        with log_path.open("a") as fh:
            fh.write("\nPROMOTION_DIAGNOSTICS\n")
            fh.write(json.dumps(payload["diagnostic_summary"], indent=2, sort_keys=True) + "\n")
    except Exception:
        pass
    _record_promotion_status(candidate_run_id, status, payload)
    return {"automation_status": status, "promotion_log": str(log_path), "promotion_payload": payload}


def _path_tail(value: Any) -> str:
    return str(value or "").replace("\\", "/").lstrip("./")


def _same_path_tail(left: Any, right: Any) -> bool:
    left_tail = _path_tail(left)
    right_tail = _path_tail(right)
    return bool(left_tail and right_tail and (left_tail.endswith(right_tail) or right_tail.endswith(left_tail)))


def _cached_loo_summaries(now: float | None = None) -> list[Dict[str, Any]]:
    """Return LOO summaries, caching filesystem scans for five minutes."""
    global _LINKED_LOO_SUMMARY_CACHE_AT, _LINKED_LOO_SUMMARY_CACHE, _LINKED_LOO_SUMMARY_CACHE_ROOT
    current = time.time() if now is None else now
    cache_root = LOGS.resolve()
    with _LINKED_LOO_SUMMARY_CACHE_LOCK:
        if _LINKED_LOO_SUMMARY_CACHE is not None and _LINKED_LOO_SUMMARY_CACHE_ROOT == cache_root and current - _LINKED_LOO_SUMMARY_CACHE_AT < LINKED_LOO_SUMMARY_CACHE_TTL_SECONDS:
            return list(_LINKED_LOO_SUMMARY_CACHE)
        summaries: list[Dict[str, Any]] = []
        for path in sorted(LOGS.glob("*summary.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                summary = json.loads(path.read_text())
            except Exception as exc:
                LOGGER.warning("Skipping malformed LOO summary %s: %s", path, exc)
                continue
            summaries.append(summary)
        _LINKED_LOO_SUMMARY_CACHE = summaries
        _LINKED_LOO_SUMMARY_CACHE_AT = current
        _LINKED_LOO_SUMMARY_CACHE_ROOT = cache_root
        return list(summaries)


def _linked_loo_summary(run: Dict[str, Any]) -> dict[str, Any] | None:
    run_id = str(run.get("run_id") or "")
    artifact_config = Path(str(run.get("artifact_dir") or "")) / "config.json" if run.get("artifact_dir") else None
    for summary in _cached_loo_summaries():
        run_ids = {str(item) for item in summary.get("run_ids") or []}
        if run_id and run_id in run_ids:
            return summary
        if artifact_config and _same_path_tail(summary.get("base_config"), artifact_config):
            return summary
    return None


def _linked_loo_summary_ready(run: Dict[str, Any]) -> bool:
    summary = _linked_loo_summary(run)
    return bool(summary and summary.get("promotion_ready"))


def _linked_loo_summary_failed(run: Dict[str, Any]) -> bool:
    summary = _linked_loo_summary(run)
    if not summary:
        return False
    return not bool(summary.get("promotion_ready"))


def _artifact_full_tile_ready(run: Dict[str, Any]) -> bool:
    artifact_dir = run.get("artifact_dir")
    if not artifact_dir:
        return False
    for path in Path(str(artifact_dir)).glob("full_tile*/metrics.json"):
        try:
            metrics = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(metrics, dict):
            continue
        checks = metrics.get("promotion_checks") if isinstance(metrics.get("promotion_checks"), dict) else {}
        region = metrics.get("evaluation_region") if isinstance(metrics.get("evaluation_region"), dict) else {}
        if checks.get("eligible") is True and region.get("type") == "whole_segment":
            return True
    return False


def _promotion_next_action_with_evidence(run: Dict[str, Any]) -> str:
    evidence: dict[str, bool] = {}
    if _linked_loo_summary_ready(run):
        evidence["loo_promotion_ready"] = True
    if _artifact_full_tile_ready(run):
        evidence["full_tile_promotion_ready"] = True
    if evidence:
        metrics = run.get("metrics", {}) if isinstance(run.get("metrics"), dict) else {}
        run = {**run, "metrics": {**metrics, **evidence}}
    return _promotion_next_action(run)


def _manual_promotion_candidate(runs: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    candidates: list[Dict[str, Any]] = []
    for run in runs:
        cfg = run.get("config", {})
        metrics = run.get("metrics", {}) if isinstance(run.get("metrics"), dict) else {}
        autoresearch_meta = cfg.get("autoresearch", {}) if isinstance(cfg.get("autoresearch"), dict) else {}
        if autoresearch_meta.get("promotable") is False:
            continue
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
            if ratio > PROMOTION_POS_RATE_RATIO_HIGH or ratio < PROMOTION_POS_RATE_RATIO_LOW:
                continue
        if _linked_loo_summary_failed(run):
            continue
        action = _promotion_next_action_with_evidence(run)
        if action in {"run_seed_repeat_leave_one_out", "run_full_tile_validation"}:
            candidates.append(run)
    if not candidates:
        return None
    return max(candidates, key=_run_quality_score)


def _promotion_phase_manual_action(runs: List[Dict[str, Any]], history: _RunHistory | None = None) -> dict[str, Any] | None:
    """Return a candidate-linked promotion action when local sweeps should stop."""
    if os.environ.get("AUTORESEARCH_CONTINUE_AFTER_PROMOTION_ACTION", "0") == "1":
        return None
    strategy = history.strategy_phase(runs) if history else _strategy_phase(runs)
    if not strategy.get("plateau"):
        return None
    candidate = history.manual_promotion_candidate(runs) if history else _manual_promotion_candidate(runs)
    if not candidate:
        return None
    action = _promotion_next_action_with_evidence(candidate)
    if action not in {"run_seed_repeat_leave_one_out", "run_full_tile_validation"}:
        return None
    candidate_run_id = str(candidate.get("run_id") or "")
    _eligible, gate_warnings = _promotion_gate(candidate)

    reasoning = [
        "plateau_detected",
        "promotion_candidate_available",
        "stop_local_sampled_sweeps",
        "candidate_linked_evidence_required",
    ]
    command = None
    command_args = None
    summary_json = None
    if action == "run_seed_repeat_leave_one_out":
        artifact_dir = Path(str(candidate.get("artifact_dir") or ROOT / "experiments" / "runs" / candidate_run_id))
        base_config = artifact_dir / "config.json"
        output_stem = f"{candidate_run_id}_seedrepeat_loo"
        command_args = [
            DEFAULT_PYTHON,
            DEFAULT_LOO_SCRIPT,
            "--base-config",
            _repo_arg(base_config),
            "--fold-map",
            DEFAULT_FOLD_MAP,
            "--output-jsonl",
            f"logs/{output_stem}.jsonl",
            "--summary-json",
            f"logs/{output_stem}.summary.json",
            "--seeds",
            DEFAULT_PROMOTION_SEEDS,
            "--jobs",
            DEFAULT_LOO_JOBS,
        ]
        summary_json = ROOT / f"logs/{output_stem}.summary.json"
        command = _shell_command(command_args)
        reasoning.append("median_over_seeds_and_folds_required")
    else:
        artifact_dir = Path(str(candidate.get("artifact_dir") or ROOT / "experiments" / "runs" / candidate_run_id))
        base_config = candidate.get("config", {}) if isinstance(candidate.get("config"), dict) else {}
        heldout_segment = str(
            _get_nested(base_config, ("autoresearch", "heldout_segment"), "")
            or _get_nested(base_config, ("resolved_data", "val", "metadata", "segment_id"), "")
        )
        if heldout_segment:
            output_dir = artifact_dir / f"full_tile_candidate_{heldout_segment}"
            command_args = [
                DEFAULT_PYTHON,
                "scripts/infer_full_tile.py",
                "--artifact",
                _repo_arg(artifact_dir),
                "--segment-id",
                heldout_segment,
                "--output-dir",
                _repo_arg(output_dir),
                "--catalog-source",
                "public-directory",
                "--level",
                "1",
                "--z-offsets=-4,0,4",
                "--patch-size",
                "64",
                "--stride",
                "32",
                "--batch-size",
                os.environ.get("AUTORESEARCH_FULL_TILE_BATCH_SIZE", "4"),
                "--device",
                "cpu",
                "--public-retry-count",
                "5",
                "--public-retry-delay-sec",
                "180",
                "--public-chunk-delay-sec",
                os.environ.get("AUTORESEARCH_FULL_TILE_CHUNK_DELAY_SEC", "0.5"),
                "--public-chunk-retry-count",
                "5",
                "--public-chunk-retry-delay-sec",
                "180",
            ]
            command = _shell_command(command_args)
            reasoning.append("candidate_heldout_segment_full_tile_validation")
        else:
            reasoning.append("full_tile_command_should_come_from_dashboard_candidate_evidence")

    payload = {
        "status": "manual_promotion_action",
        "next_action": action,
        "candidate_run_id": candidate_run_id,
        "command": command,
        "reasoning": reasoning,
        "promotion_failure_reasons": gate_warnings,
        "strategy_phase": strategy.get("phase"),
        "promotion_required": ["seed_repeat_leave_one_out", "full_tile_validation", "promotion_checks_eligible"],
        "proposals": [],
    }
    if summary_json is not None:
        payload["summary_json"] = str(summary_json)
    return payload


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


def _propose_from_recent_winners(runs: List[Dict[str, Any]], count: int, *, name_index_offset: int = 0, required_families: set[str] | None = None, strategy_phase: str | None = None, allow_expensive: bool = False, history: _RunHistory | None = None) -> List[Tuple[str, Dict[str, Any], str]]:
    out: list[Tuple[str, Dict[str, Any], str]] = []
    ranked_bases = history.ranked_recent_torch_bases(runs) if history else _ranked_recent_torch_bases(runs)
    for base, label in ranked_bases:
        remaining = count - len(out)
        if remaining <= 0:
            break
        scope_policy = str(base.get("autoresearch", {}).get("scope_policy") or base.get("dataset", {}).get("research_scope") or "recent_robust_torch_winner")
        print(f"Trying AutoResearch recent robust winner base {label}", flush=True)
        proposals = _propose_configs(base, runs, count=remaining, scope_policy=scope_policy, lock_to_baseline_scope=False, name_index_offset=name_index_offset + len(out), required_families=required_families, strategy_phase=strategy_phase, allow_expensive=allow_expensive)
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


def _propose_best_path(base: Dict[str, Any], runs: List[Dict[str, Any]], count: int, history: _RunHistory | None = None) -> List[Tuple[str, Dict[str, Any], str]]:
    print("AutoResearch strategy: robust/torch best path first; focused NumPy is fallback only", flush=True)
    strategy = history.strategy_phase(runs) if history else _strategy_phase(runs)
    required_families = strategy["required_families"] or None
    phase = str(strategy["phase"])
    allow_expensive = _strategy_allows_expensive(strategy)
    print(f"AutoResearch strategy phase={phase} plateau={strategy['plateau']} next_action={strategy['next_action']}", flush=True)
    out: list[Tuple[str, Dict[str, Any], str]] = []
    for name, pivot, scope_policy in _pivot_bases():
        remaining = count - len(out)
        if remaining <= 0:
            break
        print(f"Trying AutoResearch best-path base {name} scope={scope_policy}", flush=True)
        out.extend(_propose_configs(pivot, runs, count=remaining, scope_policy=scope_policy, lock_to_baseline_scope=False, name_index_offset=len(out), required_families=required_families, strategy_phase=phase, allow_expensive=allow_expensive))
    if out:
        return out
    print("Curated robust/torch bases are exhausted; trying best recent robust/torch winners", flush=True)
    out = _propose_from_recent_winners(runs, count=count, required_families=required_families, strategy_phase=phase, allow_expensive=allow_expensive, history=history)
    if out:
        return out
    if os.environ.get("AUTORESEARCH_ALLOW_FOCUSED_FALLBACK", "1") != "1":
        return []
    print("Recent robust/torch winner follow-ups are exhausted; falling back to focused NumPy search", flush=True)
    return _propose_configs(base, runs, count=count, required_families=required_families, strategy_phase=phase, allow_expensive=allow_expensive)


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
            "search_strategy": autoresearch.get("search_strategy"),
            "strategy_phase": autoresearch.get("strategy_phase"),
            "scope_policy": autoresearch.get("scope_policy"),
            "cost_tier": autoresearch.get("cost_tier"),
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
    "review_positive_rate",
    "inspect_full_tile_errors",
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
        value = _clamp_param(path, value)
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
        cost_tier = _config_cost_tier(cfg)
        signature = _search_signature(cfg)
        if signature in tested or signature in seen:
            return None
        seen.add(signature)
        autoresearch = cfg.setdefault("autoresearch", {})
        autoresearch["parent_reason"] = reason
        autoresearch["scope_policy"] = str(autoresearch.get("scope_policy") or base.get("dataset", {}).get("research_scope") or "promotion_action")
        autoresearch["intent"] = "promotion_action"
        autoresearch["run_profile"] = "exploration"
        autoresearch["promotion_action_id"] = action_id
        autoresearch["promotable"] = False
        autoresearch["proposal_status"] = "generated"
        autoresearch["changed_path"] = ".".join(path)
        autoresearch["mutation_family"] = _mutation_family(path)
        autoresearch["cost_tier"] = cost_tier
        autoresearch["promotion_required"] = ["seed_repeat_leave_one_out", "full_tile_validation", "promotion_checks_eligible"]
        name = f"auto_{stamp}_promotion_{action_id}_{len(proposals)+1}_{'_'.join(path)}_{_proposal_value_slug(value)}.yaml"
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
        seed = int(_get_nested(base, ("training", "seed"), DEFAULT_SEED))
        for seed_delta in SEED_REPEAT_DELTAS:
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

    proposals.sort(key=lambda item: _cost_tier_rank(item[1].get("autoresearch", {}).get("cost_tier", "expensive")))
    return proposals


def _promotion_or_fallback_proposals(runs: List[Dict[str, Any]], base: Dict[str, Any], ready_payload: dict[str, Any] | None, count: int, history: _RunHistory | None = None) -> tuple[List[Tuple[str, Dict[str, Any], str]], str | None]:
    """Return auto-action proposals, falling back to normal exploration if exhausted."""
    if ready_payload and ready_payload.get("action_id") in _AUTO_ACTIONS:
        action_proposals = _generate_promotion_action_proposals(runs, ready_payload, count=count)
        if action_proposals:
            return action_proposals, str(ready_payload.get("action_id"))
        print(f"Promotion gate action={ready_payload.get('action_id')} proposals exhausted; falling back to normal exploration", flush=True)
    return _propose_best_path(base, runs, count=count, history=history), None


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
            if not command and top_action.get("id") in {"review_positive_rate", "inspect_full_tile_errors"}:
                command = _review_blocker_command(evidence)
                if command:
                    next_action = "Run blocker remediation plan"
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
                "promotion_failure_reasons": [str(item) for item in gate.get("warnings", [])] if isinstance(gate.get("warnings"), list) else [],
                "promotion_actions": actions,
                "candidate_evidence": evidence,
                "proposals": [],
            }
    except Exception as exc:
        LOGGER.exception("Promotion readiness check failed")
        print(f"Promotion readiness check skipped: {exc}", flush=True)
        if os.environ.get("AUTORESEARCH_FAIL_ON_SNAPSHOT_ERROR", "0") == "1":
            return {
                "status": "promotion_readiness_error",
                "next_action": "Promotion readiness check failed; pause exploration and inspect dashboard snapshot.",
                "error": str(exc),
                "proposals": [],
            }
    return None


def _review_blocker_command(evidence: dict[str, Any]) -> str | None:
    """Return a safe local diagnostic command for review-only promotion blockers."""
    candidate_run_id = str(evidence.get("candidate_run_id") or "candidate")
    candidate_artifact = Path(str(evidence.get("candidate_artifact_dir") or ROOT / "experiments" / "runs" / candidate_run_id))
    weak_id = str((evidence.get("loo") or {}).get("worst_fold_id") or "")
    args = [
        DEFAULT_PYTHON,
        "scripts/plan_hard_negative_retrain.py",
        "--max-commands",
        "4",
        "--ratio-threshold",
        "2.0",
        "--base-config",
        _repo_arg(candidate_artifact / "config.json"),
        "--pretty",
    ]
    if weak_id:
        args.extend(["--heldout-segment", weak_id])
    return _shell_command(args)


def _auto_execute_ready_payload_command(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Run a dashboard-provided promotion evidence command when explicitly enabled."""
    if os.environ.get("AUTORESEARCH_AUTO_PROMOTE", "0") != "1":
        return None
    command = payload.get("command")
    if not command:
        return None
    try:
        command_args = shlex.split(str(command))
    except ValueError as exc:
        return {"automation_status": "SKIPPED_INVALID_COMMAND", "automation_error": str(exc)}
    if not command_args:
        return None
    command_text = " ".join(command_args)
    allowed_evidence_command = any(
        token in command_text
        for token in ("scripts/evaluate_leave_one_out.py", "scripts/infer_full_tile.py", "scripts/plan_hard_negative_retrain.py", "scripts/compare_threshold_caps.py")
    )
    if payload.get("safe_to_execute_from_dashboard") is False:
        if not allowed_evidence_command:
            return {"automation_status": "SKIPPED_UNSAFE_COMMAND"}
        print("Promotion evidence command is dashboard-flagged unsafe but matches local evidence allowlist; executing under guard timeout.", flush=True)
    candidate_run_id = str(payload.get("candidate_run_id") or payload.get("action_id") or "promotion_ready")
    timeout = int(os.environ.get("AUTORESEARCH_PROMOTION_TIMEOUT_SECONDS", "3600"))
    summary_json = LOGS / f"{candidate_run_id}_promotion_action.summary.json"
    reasons = [str(item) for item in payload.get("promotion_failure_reasons", [])] if isinstance(payload.get("promotion_failure_reasons"), list) else []
    if payload.get("action_id"):
        _append_promotion_reason(reasons, f"promotion_action_{payload.get('action_id')}")
    return _run_automated_promotion(command_args, candidate_run_id, summary_json, timeout, promotion_failure_reasons=reasons)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(header + body)
    tmp.replace(path)


def _run_experiment_checked(config_path: Path) -> None:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "16")
    env.setdefault("MKL_NUM_THREADS", "16")
    subprocess.run([sys.executable, "run_experiment.py", "--config", str(config_path)], cwd=ROOT, env=env, check=True)


def _acquire_autoresearch_lock(lock: Any) -> None:
    """Acquire the process lock or fail closed when no lock backend is available."""
    if fcntl is not None:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return
    try:
        import msvcrt
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("autoresearch locking requires fcntl or msvcrt; no safe fallback is available") from exc
    msvcrt_any = cast(Any, msvcrt)
    lock.seek(0)
    msvcrt_any.locking(lock.fileno(), msvcrt_any.LK_NBLCK, 1)


def _research_harness() -> Any:
    """Instantiate the active research harness for the autoresearch loop."""
    from harness.vesuvius_harness import VesuviusHarness
    return VesuviusHarness(sys.modules[__name__])


def main() -> int:
    """Run AutoResearch after fail-closed harness initialization, then config pruning."""
    parser = argparse.ArgumentParser(description="Run or plan the local AutoResearch cycle")
    parser.add_argument("--plan", action="store_true", help="Print planned proposals without writing configs or launching experiments")
    parser.add_argument("--json", action="store_true", help="Emit planning output as JSON; only valid with --plan")
    args = parser.parse_args()
    if args.json and not args.plan:
        parser.error("--json requires --plan")

    LOGS.mkdir(parents=True, exist_ok=True)
    CONFIGS.mkdir(parents=True, exist_ok=True)
    try:
        harness = _research_harness()
    except ImportError as exc:
        print(f"Failed to initialize research harness: {exc}", file=sys.stderr, flush=True)
        return 1

    if args.plan:
        profiler = _CycleProfiler()
        history = _RunHistory(profiler)
        runs = history.recent_runs()
        if not runs:
            print(json.dumps({"status": "needs_baseline", "proposals": []}, indent=2 if args.json else None))
            return 0
        ready_payload = harness.promotion_ready_payload()
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
        manual_payload = _promotion_phase_manual_action(runs, history=history)
        if manual_payload:
            if args.json:
                print(json.dumps(manual_payload, indent=2, sort_keys=True))
            else:
                print(f"AutoResearch promotion action required: {manual_payload.get('next_action')}")
                if manual_payload.get("command"):
                    print(f"Command: {manual_payload['command']}")
            return 0
        base = harness.best_base_config(runs)
        proposal_count = int(os.environ.get("AUTORESEARCH_PROPOSALS", "3"))
        if args.json:
            with contextlib.redirect_stdout(sys.stderr):
                with profiler.measure("proposal_generation"):
                    proposals, source_action = _promotion_or_fallback_proposals(runs, base, ready_payload, proposal_count, history=history)
        else:
            with profiler.measure("proposal_generation"):
                proposals, source_action = _promotion_or_fallback_proposals(runs, base, ready_payload, proposal_count, history=history)
        payload = {"status": "planned", "proposal_count": len(proposals), "fallback_from_action": source_action, "proposals": _proposal_plan(proposals)}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            for item in payload["proposals"]:
                print(f"{item['name']}: {item['reason']} [{item.get('strategy_phase')}/{item.get('mutation_family')}]")
            if not proposals:
                print("No novel one-change proposals remain.")
        return 0

    lock_path = Path(os.environ.get("AUTORESEARCH_LOCK_PATH", "/tmp/vesuvius_autoresearch_pytest.lock" if os.environ.get("PYTEST_CURRENT_TEST") else str(LOCK_PATH)))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as lock:
        try:
            _acquire_autoresearch_lock(lock)
        except (BlockingIOError, PermissionError, OSError):
            print(f"{datetime.now(timezone.utc).isoformat()} another autoresearch run is active; exiting safely")
            return 0
        pruned = _prune_stale_configs()
        if pruned:
            print(f"Pruned {pruned} stale generated config(s)", flush=True)
        init_db(DB_PATH)
        profiler = _CycleProfiler()
        history = _RunHistory(profiler)
        runs = history.recent_runs()
        if not runs:
            print("No prior runs found; executing baseline first")
            with profiler.measure("experiment_subprocess"):
                _run_experiment_checked(BASELINE)
            history.invalidate()
            runs = history.recent_runs()
        base = harness.best_base_config(runs)
        proposal_count = int(os.environ.get("AUTORESEARCH_PROPOSALS", "3"))
        print(f"AutoResearch local-only cycle: loaded {len(runs)} prior runs; proposal_count={proposal_count}; no web/LLM calls", flush=True)
        ready_payload = harness.promotion_ready_payload()
        if ready_payload and ready_payload.get("action_id") not in _AUTO_ACTIONS:
            automation_result = _auto_execute_ready_payload_command(ready_payload)
            if automation_result:
                ready_payload.update(automation_result)
            print(f"Promotion gate is ready; pausing exploration. Next action: {ready_payload.get('next_action')}", flush=True)
            if ready_payload.get("command"):
                print(f"Command: {ready_payload['command']}", flush=True)
            if ready_payload.get("automation_status"):
                print(f"Automation status: {ready_payload['automation_status']}", flush=True)
            return 0
        manual_payload = _promotion_phase_manual_action(runs, history=history)
        if manual_payload:
            if os.environ.get("AUTORESEARCH_AUTO_PROMOTE", "0") == "1" and manual_payload.get("command"):
                try:
                    command_args = shlex.split(str(manual_payload["command"]))
                except ValueError as exc:
                    manual_payload["automation_status"] = "SKIPPED_INVALID_COMMAND"
                    manual_payload["automation_error"] = str(exc)
                else:
                    timeout = int(os.environ.get("AUTORESEARCH_PROMOTION_TIMEOUT_SECONDS", "3600"))
                    reasons = [str(item) for item in manual_payload.get("promotion_failure_reasons", [])] if isinstance(manual_payload.get("promotion_failure_reasons"), list) else []
                    summary_json = Path(str(manual_payload.get("summary_json") or LOGS / "manual_promotion_action.summary.json"))
                    manual_payload.update(_run_automated_promotion(command_args, str(manual_payload.get("candidate_run_id") or "manual_promotion"), summary_json, timeout, promotion_failure_reasons=reasons))
            print(f"AutoResearch promotion action required: {manual_payload.get('next_action')}", flush=True)
            if manual_payload.get("command"):
                print(f"Command: {manual_payload['command']}", flush=True)
            if manual_payload.get("automation_status"):
                print(f"Automation status: {manual_payload['automation_status']}", flush=True)
            return 0
        with profiler.measure("proposal_generation"):
            proposals, source_action = _promotion_or_fallback_proposals(runs, base, ready_payload, proposal_count, history=history)
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
            with profiler.measure("config_dump"):
                _dump_config_with_comment(cfg_path, cfg, reason)
            print(f"Running generated experiment {cfg_path.name}: {reason}", flush=True)
            try:
                with profiler.measure("experiment_subprocess"):
                    _run_experiment_checked(cfg_path)
            except subprocess.CalledProcessError:
                try:
                    cfg_path.unlink()
                except FileNotFoundError:
                    pass
                raise
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

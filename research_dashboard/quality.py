from __future__ import annotations

from typing import Any

import numpy as np


_REASON_ACTIONS: dict[str, tuple[str, str, str]] = {
    "positive_rate_ratio_suspicious": ("calibrate_positive_rate", "Tighten positive-rate constraints before promotion", "blocker"),
    "positive_rate_ratio_review": ("review_positive_rate", "Review positive-rate ratio before promotion", "warning"),
    "fixed_threshold_weak": ("calibrate_probability_scale", "Calibrate probability scale; fixed 0.5 threshold is weak", "warning"),
    "weak_ap_lift": ("improve_ranking_signal", "Improve AP/prevalence lift before threshold sweeps", "warning"),
    "weak_full_tile_f1": ("inspect_full_tile_errors", "Inspect full-tile false positives and false negatives", "blocker"),
    "blank_or_no_positive_mask": ("check_inference_wiring", "Check checkpoint/input normalization; decoded mask is blank", "blocker"),
    "flat_probabilities": ("check_probability_scale", "Check inference wiring; probabilities are flat", "blocker"),
    "flooding": ("mine_hard_negatives", "Mine hard negatives and reduce flooding", "blocker"),
    "possible_flooding": ("review_flooding", "Review flooding risk and positive-rate constraint", "warning"),
    "speckle_or_isolated_positives": ("mine_hard_negatives", "Mine hard negatives from speckled false positives", "blocker"),
    "many_isolated_positives": ("mine_hard_negatives", "Mine hard negatives from isolated false positives", "blocker"),
    "fragmented_threshold_mask": ("inspect_decoded_noise", "Inspect fragmented decoded mask and stitching behavior", "blocker"),
    "unstructured_noise": ("inspect_decoded_noise", "Inspect unstructured decoded noise before promotion", "blocker"),
    "high_frequency_noise": ("inspect_decoded_noise", "Inspect high-frequency decoded noise before promotion", "blocker"),
    "invalid_values": ("check_artifact_integrity", "Check decoded artifact numerics", "blocker"),
}


def quality_next_actions(verdict: dict[str, Any] | None, *, target: str | None = None, run_id: str | None = None, segment_id: str | None = None) -> list[dict[str, Any]]:
    if not isinstance(verdict, dict) or verdict.get("verdict") == "pass":
        return []
    reasons = list(verdict.get("reasons") or verdict.get("flags") or [])
    if not reasons:
        reasons = ["quality_review"] if verdict.get("verdict") == "review" else ["quality_failed"]
    actions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for reason in reasons:
        action_id, label, severity = _REASON_ACTIONS.get(str(reason), ("quality_review", "Review full-tile quality evidence before promotion", "warning" if verdict.get("verdict") == "review" else "blocker"))
        key = f"{action_id}:{reason}"
        if key in seen:
            continue
        seen.add(key)
        if verdict.get("verdict") == "fail" and severity == "warning":
            severity = "blocker"
        actions.append({
            "id": action_id,
            "label": label,
            "kind": "quality",
            "source": "quality_verdict",
            "reason": str(reason),
            "severity": severity,
            "target": target,
            "run_id": run_id,
            "segment_id": segment_id,
            "writes_artifacts": False,
            "safe_to_execute_from_dashboard": False,
        })
    return actions


def quality_actions_for_verdict(verdict: str | None) -> list[dict[str, Any]]:
    return quality_next_actions({"verdict": str(verdict or "review"), "reasons": []})


def decoded_output_quality(probs: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    arr = np.asarray(probs)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D probability map, got shape {list(arr.shape)}")

    finite = np.isfinite(arr)
    finite_fraction = float(finite.mean())
    p = np.where(finite, arr, 0.0).astype(np.float32)
    p = np.clip(p, 0.0, 1.0)
    h, w = p.shape
    mask = p >= float(threshold)
    m = mask.astype(np.uint8)
    positive_rate = float(mask.mean())

    q01, q05, q95, q99 = np.quantile(p, [0.01, 0.05, 0.95, 0.99])
    dynamic_range = float(q99 - q01)
    robust_spread = float(q95 - q05)
    std = float(p.std())

    diffs = []
    if w > 1:
        diffs.append(np.abs(np.diff(p, axis=1)).ravel())
    if h > 1:
        diffs.append(np.abs(np.diff(p, axis=0)).ravel())
    mean_abs_neighbor_diff = float(np.mean(np.concatenate(diffs))) if diffs else 0.0
    noise_index = float(mean_abs_neighbor_diff / max(dynamic_range, 1e-6))

    centered = p - float(p.mean())
    var = float(np.mean(centered * centered))
    covs = []
    if w > 1:
        covs.append(float(np.mean(centered[:, :-1] * centered[:, 1:])))
    if h > 1:
        covs.append(float(np.mean(centered[:-1, :] * centered[1:, :])))
    neighbor_autocorr = float(np.mean(covs) / var) if var > 1e-12 and covs else 1.0

    padded = np.pad(m, 1)
    neighbors = np.zeros_like(m, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                neighbors += padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]

    positives = int(m.sum())
    if positives:
        supported_positive_rate = float(((m == 1) & (neighbors >= 2)).sum() / positives)
        isolated_positive_rate = float(((m == 1) & (neighbors == 0)).sum() / positives)
    else:
        supported_positive_rate = 0.0
        isolated_positive_rate = 0.0

    transitions = 0
    edges = 0
    if w > 1:
        transitions += int((m[:, 1:] != m[:, :-1]).sum())
        edges += h * (w - 1)
    if h > 1:
        transitions += int((m[1:, :] != m[:-1, :]).sum())
        edges += (h - 1) * w
    transition_density = float(transitions / edges) if edges else 0.0

    def clip01(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    if positive_rate < 0.001:
        positive_score = 0.0
    elif positive_rate < 0.005:
        positive_score = 0.25
    elif positive_rate <= 0.50:
        positive_score = 1.0
    elif positive_rate >= 0.75:
        positive_score = 0.0
    else:
        positive_score = clip01((0.75 - positive_rate) / 0.25)

    contrast_score = clip01((dynamic_range - 0.05) / 0.25)
    coherence_score = clip01((supported_positive_rate - 0.25) / 0.50) * (1.0 - clip01((transition_density - 0.15) / 0.35))
    smoothness_score = clip01(neighbor_autocorr / 0.50) * (1.0 - clip01((noise_index - 0.25) / 0.50))
    score = float(0.25 * positive_score + 0.25 * contrast_score + 0.30 * coherence_score + 0.20 * smoothness_score)

    flags: list[str] = []
    if finite_fraction < 0.999:
        flags.append("invalid_values")
    if positive_rate < 0.001:
        flags.append("blank_or_no_positive_mask")
    if positive_rate >= 0.75:
        flags.append("flooding")
    elif positive_rate >= 0.60:
        flags.append("possible_flooding")
    if dynamic_range < 0.05 or std < 0.015:
        flags.append("flat_probabilities")
    if positives and supported_positive_rate < 0.25:
        flags.append("speckle_or_isolated_positives")
    if positives and isolated_positive_rate > 0.30:
        flags.append("many_isolated_positives")
    if transition_density > 0.40:
        flags.append("fragmented_threshold_mask")
    if neighbor_autocorr < 0.05 and transition_density > 0.30:
        flags.append("unstructured_noise")
    if noise_index > 0.50 and neighbor_autocorr < 0.20:
        flags.append("high_frequency_noise")

    hard_fail_flags = {"blank_or_no_positive_mask", "flooding", "flat_probabilities", "speckle_or_isolated_positives", "many_isolated_positives", "unstructured_noise", "high_frequency_noise"}
    if any(flag in hard_fail_flags for flag in flags) or score < 0.45:
        verdict = "fail"
    elif score < 0.70 or flags:
        verdict = "review"
    else:
        verdict = "pass"

    return {
        "score": score,
        "verdict": verdict,
        "flags": flags,
        "finite_fraction": finite_fraction,
        "positive_rate": positive_rate,
        "dynamic_range": dynamic_range,
        "robust_spread": robust_spread,
        "std": std,
        "mean_abs_neighbor_diff": mean_abs_neighbor_diff,
        "noise_index": noise_index,
        "neighbor_autocorr": neighbor_autocorr,
        "supported_positive_rate": supported_positive_rate,
        "isolated_positive_rate": isolated_positive_rate,
        "transition_density": transition_density,
    }


def metrics_quality_verdict(metrics: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    val_f1 = _as_float(metrics.get("val_f1", metrics.get("tile_f1")))
    ap = _as_float(metrics.get("average_precision"))
    val_rate = _as_float(metrics.get("val_positive_rate", metrics.get("label_positive_rate")))
    pred_rate = _as_float(metrics.get("pred_positive_rate"))
    fixed_f1 = _as_float(metrics.get("fixed_threshold_f1"))
    ap_lift = _as_float(metrics.get("ap_prevalence_lift"))
    if ap_lift is None and ap is not None and val_rate is not None and val_rate > 1e-12:
        ap_lift = ap / val_rate
    ratio = pred_rate / val_rate if pred_rate is not None and val_rate is not None and val_rate > 1e-12 else None

    score = 0.0
    if val_f1 is not None:
        score += min(0.35, val_f1)
    if ap_lift is not None:
        score += min(0.25, max(0.0, (ap_lift - 1.0) * 0.15))
    if ratio is not None:
        if 0.5 <= ratio <= 2.0:
            score += 0.25
        elif 0.25 <= ratio <= 3.0:
            score += 0.12
            reasons.append("positive_rate_ratio_review")
        else:
            reasons.append("positive_rate_ratio_suspicious")
    if fixed_f1 is not None and val_f1 is not None:
        if fixed_f1 >= 0.5 * val_f1:
            score += 0.15
        else:
            reasons.append("fixed_threshold_weak")

    if ap_lift is not None and ap_lift < 1.25:
        reasons.append("weak_ap_lift")
    if val_f1 is not None and val_f1 < 0.12:
        reasons.append("weak_full_tile_f1")

    if not reasons and score >= 0.70:
        verdict = "pass"
    elif score >= 0.45 and "positive_rate_ratio_suspicious" not in reasons:
        verdict = "review"
    else:
        verdict = "fail"
    return {"score": float(score), "verdict": verdict, "reasons": reasons, "ap_prevalence_lift": ap_lift, "pred_positive_rate_ratio": ratio}


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None

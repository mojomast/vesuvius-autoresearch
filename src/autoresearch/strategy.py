from __future__ import annotations

from ._legacy import legacy

_m = legacy()

PIVOT_CONFIGS = _m.PIVOT_CONFIGS
_strategy_phase = _m._strategy_phase
_ranked_recent_torch_bases = _m._ranked_recent_torch_bases
_propose_from_recent_winners = _m._propose_from_recent_winners
_pivot_bases = _m._pivot_bases
_propose_best_path = _m._propose_best_path
_propose_with_pivots = _m._propose_with_pivots
_promotion_or_fallback_proposals = _m._promotion_or_fallback_proposals

__all__ = ["PIVOT_CONFIGS", "_strategy_phase", "_ranked_recent_torch_bases", "_propose_from_recent_winners", "_pivot_bases", "_propose_best_path", "_propose_with_pivots", "_promotion_or_fallback_proposals"]

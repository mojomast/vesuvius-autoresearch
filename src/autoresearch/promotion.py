from __future__ import annotations

from ._legacy import legacy

_m = legacy()

MetricContract = _m.MetricContract
METRIC_CONTRACT = _m.METRIC_CONTRACT
validate_metric_contract = _m.validate_metric_contract
_promotion_gate = _m._promotion_gate
_run_quality_score = _m._run_quality_score
_promotion_next_action = _m._promotion_next_action
_record_promotion_status = _m._record_promotion_status
_reported_promotion_outputs = _m._reported_promotion_outputs
_validate_reported_promotion_outputs = _m._validate_reported_promotion_outputs
_run_automated_promotion = _m._run_automated_promotion
_manual_promotion_candidate = _m._manual_promotion_candidate
_promotion_phase_manual_action = _m._promotion_phase_manual_action
_promotion_ready_payload = _m._promotion_ready_payload
_auto_execute_ready_payload_command = _m._auto_execute_ready_payload_command
_promotion_ready_message = _m._promotion_ready_message

__all__ = ["MetricContract", "METRIC_CONTRACT", "validate_metric_contract", "_promotion_gate", "_run_quality_score", "_promotion_next_action", "_record_promotion_status", "_reported_promotion_outputs", "_validate_reported_promotion_outputs", "_run_automated_promotion", "_manual_promotion_candidate", "_promotion_phase_manual_action", "_promotion_ready_payload", "_auto_execute_ready_payload_command", "_promotion_ready_message"]

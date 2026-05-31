from __future__ import annotations

from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "overnight_safe_loop.sh"


def test_quality_gate_failure_is_run_local_not_loop_fatal() -> None:
    text = SCRIPT.read_text()

    assert "QUALITY_FAIL " in text
    assert "QUALITY_STOP " not in text
    assert "stop_file.write_text" not in text
    assert "QUALITY_FAIL artifact quality gate failed for $after_run_id; rejecting run-local candidate and continuing research loop" in text
    assert "STOP artifact quality gate failed" not in text


def test_unlimited_runtime_is_supported_but_explicit_stop_marker_remains() -> None:
    text = SCRIPT.read_text()

    assert 'MAX_SECONDS="${OVERNIGHT_MAX_SECONDS:-0}"' in text
    assert 'if [ "$MAX_SECONDS" -gt 0 ] && [ $((now - started)) -ge "$MAX_SECONDS" ]; then' in text
    assert 'if [ -f "$STOP" ]; then' in text
    assert 'log "STOP marker exists: $STOP"' in text


def test_stale_and_evidence_limits_continue_with_guarded_override() -> None:
    text = SCRIPT.read_text()

    assert "EVIDENCE_LIMIT reached; next cycle will allow exploration past promotion action" in text
    assert "STALE_LIMIT reached; enabling one exploration override cycle instead of stopping" in text
    assert text.count("export SCROLL_RESEARCH_ALLOW_PROMOTION_OVERRIDE=1") >= 2
    assert text.count("export AUTORESEARCH_PAUSE_WHEN_PROMOTION_READY=0") >= 2
    assert text.count("export AUTORESEARCH_CONTINUE_AFTER_PROMOTION_ACTION=1") >= 2

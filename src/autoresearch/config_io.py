from __future__ import annotations

from ._legacy import legacy

_m = legacy()

ROOT = _m.ROOT
CONFIGS = _m.CONFIGS
LOGS = _m.LOGS
LOCK_PATH = _m.LOCK_PATH
BASELINE = _m.BASELINE
SCOPE_KEYS = _m.SCOPE_KEYS
SEARCH_PATHS = _m.SEARCH_PATHS
SIGNATURE_DEFAULTS = _m.SIGNATURE_DEFAULTS
BALANCED_CALIBRATION_PATH = _m.BALANCED_CALIBRATION_PATH
BALANCED_CALIBRATION_FIELDS = _m.BALANCED_CALIBRATION_FIELDS
PARAM_BOUNDS = _m.PARAM_BOUNDS

_set_nested = _m._set_nested
_get_nested = _m._get_nested
_candidate_is_noop = _m._candidate_is_noop
_apply_candidate = _m._apply_candidate
_dump_config_with_comment = _m._dump_config_with_comment

__all__ = ["ROOT", "CONFIGS", "LOGS", "LOCK_PATH", "BASELINE", "SCOPE_KEYS", "SEARCH_PATHS", "SIGNATURE_DEFAULTS", "BALANCED_CALIBRATION_PATH", "BALANCED_CALIBRATION_FIELDS", "PARAM_BOUNDS", "_set_nested", "_get_nested", "_candidate_is_noop", "_apply_candidate", "_dump_config_with_comment"]

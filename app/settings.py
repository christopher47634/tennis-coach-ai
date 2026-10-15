"""Runtime configuration: built-in defaults + optional TOML overrides.

The defaults below reproduce the thresholds that used to be hard-coded in
pipeline.py.  ``config/default.toml`` mirrors them (a test keeps both in
sync); point ``TENNIS_CONFIG`` at another TOML file to override any subset.
"""

import copy
import os
import tomllib

DEFAULTS = {
    "scoring": {
        "severity_std_multiplier": 2.0,
        "good_below": 0.30,
        "moderate_below": 0.60,
        "empty_phase_score": 50,
        "empty_phase_severity": 0.5,
    },
    "phases": {
        "weights": [0.10, 0.20, 0.25, 0.30, 0.15],
    },
    "detection": {
        "min_detection_rate": 0.3,
        "min_valid_frames": 15,
        "smooth_seconds": 0.15,
        "smooth_min_window": 3,
        "min_prominence": 0.6,
        "rest_fraction": 0.85,
        "accel_fallback_fraction": 0.55,
        "loading_fallback_fraction": 0.4,
        "knee_min_drop": 5.0,
        "knee_loading_fraction": 0.3,
        "contact_half_min": 2,
        "contact_half_fraction": 0.1,
        "min_phase_frames": 2,
        "low_confidence_below": 0.5,
    },
    "limits": {
        "max_upload_mb": 200,
        "max_duration_seconds": 120,
        "max_dimension": 4096,
        "max_concurrent_analyses": 2,
        "queue_timeout_seconds": 30,
        "output_ttl_seconds": 3600,
    },
}

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "default.toml"
)


def _merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | None = None) -> dict:
    """Return DEFAULTS overlaid with the TOML file at ``path``.

    ``path`` falls back to $TENNIS_CONFIG; if neither is set the built-in
    defaults are returned unchanged.
    """
    cfg = copy.deepcopy(DEFAULTS)
    path = path or os.environ.get("TENNIS_CONFIG")
    if path:
        with open(path, "rb") as f:
            _merge(cfg, tomllib.load(f))
    if len(cfg["phases"]["weights"]) != 5:
        raise ValueError("phases.weights must contain exactly 5 values")
    return cfg


CONFIG = load_config()

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
        "min_frames_for_phases": 10,
        "smooth_min_window": 5,
        "smooth_divisor": 20,
        "peak_search_min": 15,
        "peak_search_divisor": 30,
        "knee_missing_default": 160.0,
        "knee_min_drop": 5.0,
        "knee_loading_fraction": 0.3,
        "loading_default_divisor": 4,
        "contact_decel_fraction": 0.35,
        "contact_half_width_min": 3,
        "contact_half_width_divisor": 5,
        "followthrough_min_fraction": 0.05,
        "min_phase_width_floor": 8,
        "min_phase_width_divisor": 20,
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

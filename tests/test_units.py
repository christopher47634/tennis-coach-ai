"""Unit tests for the pure scoring helpers and configuration."""

import copy
import os
import tomllib

import numpy as np
import pytest

import pipeline
import settings
from helpers import load_fixture

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES_PATH = os.path.join(ROOT, "data", "expert_baselines.json")


def _lm(x, y, z=0.0):
    return {"x": x, "y": y, "z": z}


def test_angle_between_right_angle_and_straight():
    assert pipeline.angle_between(_lm(1, 0), _lm(0, 0), _lm(0, 1)) == pytest.approx(90.0, abs=1e-3)
    assert pipeline.angle_between(_lm(-1, 0), _lm(0, 0), _lm(1, 0)) == pytest.approx(180.0, abs=0.1)


def test_score_angles_rating_buckets():
    bl = {"j": {"mean": 100, "std": 10}}  # severity = deviation / 20
    assert pipeline.score_angles({"j": 100}, bl)["j"]["rating"] == "good"
    assert pipeline.score_angles({"j": 107}, bl)["j"]["rating"] == "moderate"  # 0.35
    assert pipeline.score_angles({"j": 115}, bl)["j"]["rating"] == "needs_work"  # 0.75
    assert pipeline.score_angles({"j": 400}, bl)["j"]["severity"] == 1.0


def test_load_baselines_file_and_fallback(tmp_path):
    data = pipeline.load_baselines(BASELINES_PATH)
    assert set(pipeline.JOINT_NAMES) <= set(data)
    assert pipeline.load_baselines(str(tmp_path / "missing.json")) is pipeline.DEFAULT_BASELINES


def test_too_few_frames_are_split_evenly():
    phases = pipeline._detect_phases(np.linspace(0.5, 0.2, 8), [None] * 8, 8)
    assert phases == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]


def test_low_detection_rate_fails():
    frames = load_fixture("serve_base")
    sparse = [f if i % 10 == 0 else [] for i, f in enumerate(frames)]  # 10% detected
    res = pipeline.analyze_landmark_sequence(sparse, pipeline.load_baselines(BASELINES_PATH))
    assert res["success"] is False


def test_empty_input_fails():
    assert pipeline.analyze_landmark_sequence([], pipeline.DEFAULT_BASELINES)["success"] is False


def test_reference_skeleton_shape():
    sk = pipeline._generate_reference_skeleton(pipeline.DEFAULT_BASELINES)
    assert sk["total_frames"] == len(sk["frames"]) == 120


# ── configuration ──

def test_default_toml_matches_builtin_defaults():
    with open(settings.DEFAULT_CONFIG_PATH, "rb") as f:
        assert tomllib.load(f) == settings.DEFAULTS


def test_phase_defs_follow_config_weights():
    assert [p["weight"] for p in pipeline.PHASE_DEFS] == settings.DEFAULTS["phases"]["weights"]
    assert sum(settings.DEFAULTS["phases"]["weights"]) == pytest.approx(1.0)


def test_load_config_partial_override(tmp_path):
    f = tmp_path / "my.toml"
    f.write_text("[scoring]\ngood_below = 0.5\n", encoding="utf-8")
    cfg = settings.load_config(str(f))
    assert cfg["scoring"]["good_below"] == 0.5
    assert cfg["scoring"]["moderate_below"] == settings.DEFAULTS["scoring"]["moderate_below"]
    assert settings.DEFAULTS["scoring"]["good_below"] == 0.30  # defaults not mutated


def test_load_config_env_var(tmp_path, monkeypatch):
    f = tmp_path / "env.toml"
    f.write_text("[detection]\nmin_detection_rate = 0.9\n", encoding="utf-8")
    monkeypatch.setenv("TENNIS_CONFIG", str(f))
    assert settings.load_config()["detection"]["min_detection_rate"] == 0.9


def test_load_config_rejects_bad_weights(tmp_path):
    f = tmp_path / "bad.toml"
    f.write_text("[phases]\nweights = [1.0]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        settings.load_config(str(f))


def test_config_changes_rating():
    cfg = copy.deepcopy(settings.DEFAULTS)
    cfg["scoring"]["good_below"] = 0.9
    bl = {"j": {"mean": 100, "std": 10}}
    assert pipeline.score_angles({"j": 107}, bl, cfg)["j"]["rating"] == "good"

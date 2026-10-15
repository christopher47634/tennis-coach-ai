"""Unit tests for the pure scoring helpers and configuration."""

import copy
import os
import tomllib

import numpy as np
import pytest

import pipeline
import settings
from angles import compute_frame_angles
from helpers import FIXTURE_SIZE, load_fixture
from phases import PhaseDetectionError, detect_phases
from reference import generate_reference_skeleton

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


def test_too_few_frames_fail_explicitly():
    with pytest.raises(PhaseDetectionError):
        detect_phases(np.linspace(0.5, -0.5, 8), [np.nan] * 8, 30.0)


def test_low_detection_rate_fails():
    frames = load_fixture("serve_base")
    sparse = [f if i % 10 == 0 else [] for i, f in enumerate(frames)]  # 10% detected
    res = pipeline.analyze_landmark_sequence(
        sparse, pipeline.load_baselines(BASELINES_PATH), frame_size=FIXTURE_SIZE)
    assert res["success"] is False


def test_empty_input_fails():
    assert pipeline.analyze_landmark_sequence([], pipeline.DEFAULT_BASELINES)["success"] is False


def test_reference_skeleton_angles_come_from_the_pose():
    sk = generate_reference_skeleton()
    assert sk["total_frames"] == len(sk["frames"]) == 120
    f = sk["frames"][0]
    assert f["angles"] == {k: round(v, 1) for k, v in compute_frame_angles(f["landmarks"]).items()}


def test_phase_baselines_cover_every_phase_and_joint():
    pb = pipeline.load_phase_baselines(BASELINES_PATH)
    for pdef in pipeline.PHASE_DEFS:
        assert set(pb[pdef["name"]]) == set(pipeline.JOINT_NAMES), pdef["name"]


def test_baselines_for_phase_falls_back_per_joint():
    base = {"a": {"mean": 1, "std": 1}, "b": {"mean": 2, "std": 1}}
    merged = pipeline.baselines_for_phase("p", base, {"p": {"b": {"mean": 9, "std": 1}}})
    assert merged["a"]["mean"] == 1 and merged["b"]["mean"] == 9


def test_commentary_distinguishes_knee_too_straight_from_too_deep():
    scores = {"right_knee_flexion": {"value": 170, "expert_mean": 135, "severity": 1.0, "rating": "needs_work"}}
    straight = pipeline._generate_phase_commentary("蓄力期", scores, {})
    scores["right_knee_flexion"]["value"] = 90
    deep = pipeline._generate_phase_commentary("蓄力期", scores, {})
    assert "不足" in straight and "过深" in deep


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


# ── ffmpeg fallback ──

def test_transcode_without_ffmpeg_logs_and_keeps_file(tmp_path, monkeypatch, caplog):
    src = tmp_path / "o.mp4"
    src.write_bytes(b"mp4v")
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: None)
    assert pipeline._transcode_h264(str(src)) is False
    assert src.read_bytes() == b"mp4v"
    assert "ffmpeg not found" in caplog.text


def test_transcode_failure_logs_and_removes_temp(tmp_path, monkeypatch, caplog):
    src = tmp_path / "o.mp4"
    src.write_bytes(b"mp4v")

    class Proc:
        returncode = 1
        stderr = b"boom"

    def fake_run(cmd, **kw):
        (tmp_path / "o_h264.mp4").write_bytes(b"partial")
        return Proc()
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    assert pipeline._transcode_h264(str(src)) is False
    assert sorted(p.name for p in tmp_path.iterdir()) == ["o.mp4"]
    assert "boom" in caplog.text

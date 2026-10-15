"""Angle geometry: pixel-space correctness (report section E, test 1)."""

import math

import pytest

from angles import (
    L_HIP, L_SHOULDER, R_HIP, R_SHOULDER, angle_between, compute_frame_angles,
    mirror_landmarks, wrist_height_rel,
)
from helpers import body


def _lm(x, y):
    return {"x": x, "y": y, "z": 0.0}


def test_pixel_right_angle_in_1920x1080_is_90_degrees():
    """Normalised coords of a pixel-space right angle must still give 90 degrees."""
    w, h = 1920, 1080
    elbow = _lm(960 / w, 540 / h)
    shoulder = _lm((960 + 100) / w, (540 + 100) / h)   # +(100, 100) px
    wrist = _lm((960 + 100) / w, (540 - 100) / h)      # +(100, -100) px
    assert angle_between(shoulder, elbow, wrist, (w, h)) == pytest.approx(90.0, abs=1e-6)
    # ... which the old normalised-coordinate maths got wrong (not 90):
    assert abs(angle_between(shoulder, elbow, wrist) - 90.0) > 5


def test_135_degree_corner_not_distorted_by_aspect_ratio():
    w, h = 1920, 1080
    b = _lm(0.5, 0.5)
    a = _lm(0.5 + 200 / w, 0.5)                                            # +x px
    c = _lm(0.5 + 200 * math.cos(math.radians(135)) / w, 0.5 + 200 * math.sin(math.radians(135)) / h)
    assert angle_between(a, b, c, (w, h)) == pytest.approx(135.0, abs=1e-6)


def test_trunk_tilt_follows_shoulder_line():
    level = compute_frame_angles(body(0.4), (1000, 1000))
    assert level['trunk_lateral_tilt'] == pytest.approx(0.0, abs=1e-6)

    lm = body(0.4)
    lm[L_SHOULDER] = _lm(0.45, 0.30 - 0.0577)    # left shoulder up: 30 degrees over a 0.1 span
    lm[R_SHOULDER] = _lm(0.55, 0.30)
    tilted = compute_frame_angles(lm, (1000, 1000))
    assert tilted['trunk_lateral_tilt'] == pytest.approx(30.0, abs=0.1)
    assert tilted['trunk_lateral_tilt'] != pytest.approx(level['trunk_lateral_tilt'], abs=1)


def test_trunk_tilt_is_not_the_old_constant_180():
    for dy in (0.0, 0.02, -0.03, 0.08):
        lm = body(0.4)
        lm[L_SHOULDER] = _lm(0.45, 0.30 + dy)
        assert compute_frame_angles(lm, (1000, 1000))['trunk_lateral_tilt'] < 90


def test_hip_shoulder_separation_is_angle_between_lines():
    lm = body(0.4)
    assert compute_frame_angles(lm, (1000, 1000))['hip_shoulder_separation'] == pytest.approx(0.0, abs=1e-6)
    lm[L_SHOULDER] = _lm(0.45, 0.30 - 0.0577)             # shoulders 30 deg, hips still level
    assert compute_frame_angles(lm, (1000, 1000))['hip_shoulder_separation'] == pytest.approx(30.0, abs=0.1)
    lm[L_HIP] = _lm(0.46, 0.55 - 0.0577 * 0.8)            # hips follow: separation shrinks to 0
    lm[R_HIP] = _lm(0.54, 0.55)
    sep = compute_frame_angles(lm, (1000, 1000))['hip_shoulder_separation']
    assert sep < 30.0


def test_degenerate_points_do_not_crash():
    lm = [_lm(0.5, 0.5)] * 33
    angles = compute_frame_angles(lm)       # all joints coincide: angles undefined, simply omitted
    assert angles == {} or all(v == v for v in angles.values())
    assert wrist_height_rel(lm, 16) != wrist_height_rel(lm, 16)   # NaN


def test_wrist_height_is_relative_to_shoulders_and_torso():
    assert wrist_height_rel(body(0.55), 16, (1000, 1000)) == pytest.approx(1.0)
    assert wrist_height_rel(body(0.05), 16, (1000, 1000)) == pytest.approx(-1.0)
    # zoom / pan invariance: scale the whole body and shift it
    shifted = [{"x": p["x"] * 0.5 + 0.1, "y": p["y"] * 0.5 + 0.3, "z": 0.0} for p in body(0.05)]
    assert wrist_height_rel(shifted, 16, (1000, 1000)) == pytest.approx(-1.0)


def test_mirror_swaps_left_and_right():
    lm = body(0.1)
    m = mirror_landmarks(lm)
    assert m[16] == lm[15] and m[15] == lm[16] and m[R_SHOULDER] == lm[L_SHOULDER]
    assert m[R_HIP] == lm[L_HIP]

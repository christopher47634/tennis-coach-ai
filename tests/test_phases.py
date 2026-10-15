"""Phase detection on synthetic wrist trajectories (report section E, test 2)."""

import numpy as np
import pytest

import pipeline
from helpers import body, serve_curve, slow_motion
from phases import PhaseDetectionError, detect_phases, interpolate_nan, smooth

SIZE = (1000, 1000)
BASELINES = pipeline.DEFAULT_BASELINES


def _frames(wrist_y, missing=None, tail_empty=0.0, left_hand=False):
    n = len(wrist_y)
    last = int(n * (1 - tail_empty))
    out = []
    for i, y in enumerate(wrist_y):
        if i >= last or (missing is not None and i in missing):
            out.append([])
        else:
            out.append(body(float(y)))
    return out


def _check_ordered(bounds, n):
    assert len(bounds) == 5
    for (s, e) in bounds:
        assert 0 <= s < e <= n
    for (_, prev_end), (start, _) in zip(bounds, bounds[1:]):
        assert start == prev_end            # contiguous, no overlap, no gap


def _analyse(y, fps=30.0, **kw):
    res = pipeline.analyze_landmark_sequence(_frames(y, **kw), BASELINES, frame_size=SIZE, fps=fps)
    assert res["success"], res.get("error")
    return res


def test_five_monotone_phases_normal_speed():
    y = serve_curve(300, 80, 150, 200)
    res = _analyse(y)
    _check_ordered(res["phase_boundaries"], 300)
    assert abs(res["contact_frame"] - 150) <= 6
    # contact window brackets the peak
    contact_s, contact_e = res["phase_boundaries"][3]
    assert contact_s <= 150 < contact_e
    assert 60 <= res["phase_boundaries"][0][0] <= 110          # starts near the wrist leaving rest, not frame 0
    assert res["confidence"] >= 0.7


def test_slow_motion_scales_boundaries_and_keeps_order():
    y = serve_curve(300, 80, 150, 200)
    fast = _analyse(y)
    slow = _analyse(slow_motion(y, 4))
    _check_ordered(slow["phase_boundaries"], 1200)
    assert abs(slow["contact_frame"] - 4 * 150) <= 24
    for (fs, fe), (ss, se) in zip(fast["phase_boundaries"], slow["phase_boundaries"]):
        assert ss == pytest.approx(4 * fs, abs=40)
        assert se == pytest.approx(4 * fe, abs=40)


def test_empty_tail_30_percent_is_not_scored_as_follow_through():
    y = serve_curve(300, 60, 110, 150)           # whole serve inside the first 70%
    res = _analyse(y, tail_empty=0.3)
    _check_ordered(res["phase_boundaries"], 300)
    assert res["phase_boundaries"][-1][1] <= 210    # nothing past the last frame with a person
    assert abs(res["contact_frame"] - 110) <= 6


def test_ten_percent_missing_frames_do_not_fake_a_wrist_drop():
    y = serve_curve(300, 80, 150, 200)
    rng = np.random.default_rng(0)
    missing = set(rng.choice(300, 30, replace=False).tolist())
    clean = _analyse(y)
    gappy = _analyse(y, missing=missing)
    _check_ordered(gappy["phase_boundaries"], 300)
    assert abs(gappy["contact_frame"] - clean["contact_frame"]) <= 5
    for (cs, ce), (gs, ge) in zip(clean["phase_boundaries"], gappy["phase_boundaries"]):
        assert abs(cs - gs) <= 8 and abs(ce - ge) <= 8


def test_all_conditions_together():
    y = slow_motion(serve_curve(300, 60, 110, 150), 4)           # 1200 frames
    rng = np.random.default_rng(1)
    missing = set(rng.choice(int(1200 * 0.7), 120, replace=False).tolist())
    res = _analyse(y, missing=missing, tail_empty=0.3)
    _check_ordered(res["phase_boundaries"], 1200)
    assert abs(res["contact_frame"] - 440) <= 30


def test_detection_works_without_known_frame_count():
    """Denominator is the frames actually read, so a 0/odd container frame count cannot break it."""
    y = serve_curve(120, 30, 60, 90)
    res = _analyse(y)
    assert res["total_frames"] == 120


def test_flat_signal_fails_explicitly():
    y = np.full(200, 0.55)
    res = pipeline.analyze_landmark_sequence(_frames(y), BASELINES, frame_size=SIZE)
    assert res["success"] is False
    assert "发球" in res["error"] or "挥拍" in res["error"]


def test_too_few_valid_frames_raise():
    rel = np.full(100, np.nan)
    rel[:5] = [1.0, 0.5, 0.0, -0.5, -1.0]
    with pytest.raises(PhaseDetectionError):
        detect_phases(rel, np.full(100, np.nan), 30.0)


def test_left_handed_player_is_mirrored():
    y = serve_curve(300, 80, 150, 200)
    frames = []
    for lm in _frames(y):
        if lm:
            lm = list(lm)
            lm[15], lm[16] = lm[16], lm[15]            # the swinging wrist is now the left one
        frames.append(lm)
    res = pipeline.analyze_landmark_sequence(frames, BASELINES, frame_size=SIZE)
    assert res["success"] and res["handedness"] == "left"
    assert abs(res["contact_frame"] - 150) <= 6


def test_smooth_does_not_pull_edges_toward_zero():
    out = smooth(np.full(200, 0.5), 15)
    assert out == pytest.approx(np.full(200, 0.5))
    ramp = smooth(np.linspace(1, 2, 50), 9)
    assert ramp[0] > 0.9 and ramp[-1] < 2.1


def test_smooth_window_larger_than_signal():
    assert len(smooth(np.ones(4), 15)) == 4


def test_interpolate_nan_fills_gaps_and_holds_edges():
    a = interpolate_nan([np.nan, 1.0, np.nan, np.nan, 4.0, np.nan])
    assert a.tolist() == [1.0, 1.0, 2.0, 3.0, 4.0, 4.0]
    assert np.isnan(interpolate_nan([np.nan, np.nan])).all()


def test_start_knee_not_biased_by_smoothing_edges():
    """Constant knee angle + constant wrist: no spurious 'loading' from edge shrinkage."""
    y = serve_curve(300, 80, 150, 200)
    res = _analyse(y)
    s, e = res["phase_boundaries"][0][0], res["phase_boundaries"][1][0]
    # without a real knee bend the loading start falls back to a fixed fraction, not frame 'start+3'
    assert e - s >= 2

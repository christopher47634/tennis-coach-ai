"""Golden-score regression: synthetic keypoint fixtures -> expected scores."""

import copy
import json
import os

import pytest

import pipeline
from helpers import FIXTURE_DIR, FIXTURE_FPS, FIXTURE_SIZE, load_fixture
from settings import DEFAULTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINES_PATH = os.path.join(ROOT, "data", "expert_baselines.json")
BASELINES = pipeline.load_baselines(BASELINES_PATH)
PHASE_BASELINES = pipeline.load_phase_baselines(BASELINES_PATH)


def analyze(name, cfg=None):
    return pipeline.analyze_landmark_sequence(
        load_fixture(name), BASELINES, cfg, frame_size=FIXTURE_SIZE, fps=FIXTURE_FPS,
        phase_baselines=PHASE_BASELINES)


with open(os.path.join(FIXTURE_DIR, "expected_scores.json"), encoding="utf-8") as f:
    EXPECTED = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

SCORE_TOL = 1
ANGLE_TOL = 0.5
BOUNDARY_TOL = 2


@pytest.fixture(scope="module", params=sorted(EXPECTED))
def case(request):
    name = request.param
    result = analyze(name)
    return name, result, EXPECTED[name]


def test_success_and_five_phases(case):
    _, result, _ = case
    assert result["success"] is True
    assert [p["name"] for p in result["phases"]] == [d["name"] for d in pipeline.PHASE_DEFS]


def test_overall_score(case):
    _, result, expected = case
    assert result["overall_score"] == pytest.approx(expected["overall_score"], abs=SCORE_TOL)


def test_phase_scores(case):
    _, result, expected = case
    for got, want in zip(result["phases"], expected["phases"]):
        assert got["score"] == pytest.approx(want["score"], abs=SCORE_TOL), got["name"]


def test_phase_angles(case):
    _, result, expected = case
    for got, want in zip(result["phases"], expected["phases"]):
        assert set(got["angles"]) == set(want["angles"]), got["name"]
        for joint, value in want["angles"].items():
            assert got["angles"][joint] == pytest.approx(value, abs=ANGLE_TOL), (got["name"], joint)


def test_phase_boundaries(case):
    _, result, expected = case
    for got, want in zip(result["phase_boundaries"], expected["phase_boundaries"]):
        assert got[0] == pytest.approx(want[0], abs=BOUNDARY_TOL)
        assert got[1] == pytest.approx(want[1], abs=BOUNDARY_TOL)


def test_phases_are_contiguous_and_ordered(case):
    _, result, _ = case
    bounds = result["phase_boundaries"]
    assert bounds[0][0] >= 0 and bounds[-1][1] <= result["total_frames"]
    for (_, prev_end), (start, end) in zip(bounds, bounds[1:]):
        assert start == prev_end
        assert end > start


def test_scoring_drift_is_detected(case):
    """Sanity check on the test itself: a changed threshold must move scores."""
    name, _, expected = case
    cfg = copy.deepcopy(DEFAULTS)
    cfg["scoring"]["severity_std_multiplier"] = 3.0
    drifted = analyze(name, cfg)
    assert abs(drifted["overall_score"] - expected["overall_score"]) > SCORE_TOL

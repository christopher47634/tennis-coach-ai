import json
import os

import numpy as np

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
FIXTURE_SIZE = (1000, 1000)   # the stored keypoints are normalised for a square frame
FIXTURE_FPS = 30.0


def load_fixture(name: str) -> list:
    """Load a sparse keypoint fixture as a list of 33-landmark frames."""
    with open(os.path.join(FIXTURE_DIR, f"{name}.json"), encoding="utf-8") as f:
        data = json.load(f)
    frames = []
    for sparse in data["frames"]:
        lms = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(33)]
        for idx, (x, y, z) in sparse.items():
            lms[int(idx)] = {"x": x, "y": y, "z": z}
        frames.append(lms)
    return frames


def body(wrist_y: float, left_wrist_y: float = 0.52) -> list:
    """A 33-landmark upright body in a square frame with the right wrist at ``wrist_y``.

    Shoulders y=0.30, hips y=0.55 (torso = 0.25 of the frame), so the wrist is
    at +1.0 torso lengths (hip level) when wrist_y=0.55 and -1.0 at 0.05.
    """
    lm = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(33)]

    def put(i, x, y):
        lm[i] = {"x": x, "y": y, "z": 0.0}
    put(11, 0.45, 0.30)
    put(12, 0.55, 0.30)           # shoulders (L, R)
    put(23, 0.46, 0.55)
    put(24, 0.54, 0.55)           # hips
    put(25, 0.46, 0.72)
    put(26, 0.54, 0.72)           # knees
    put(27, 0.46, 0.90)
    put(28, 0.54, 0.90)           # ankles
    put(15, 0.42, left_wrist_y)
    put(13, 0.43, (0.30 + left_wrist_y) / 2)
    put(16, 0.58, wrist_y)
    put(14, 0.57, (0.30 + wrist_y) / 2)
    put(20, 0.59, wrist_y - 0.02)
    return lm


def serve_curve(n: int = 300, rest_end: int = 80, peak: int = 150, back_to_rest: int = 200,
                rest_y: float = 0.55, peak_y: float = 0.05) -> np.ndarray:
    """Wrist y per frame: still -> up to the contact peak -> back down -> still."""
    y = np.full(n, rest_y)
    up = np.arange(rest_end, peak)
    y[rest_end:peak] = rest_y + (peak_y - rest_y) * (1 - np.cos(np.pi * (up - rest_end) / (peak - rest_end))) / 2
    down = np.arange(peak, back_to_rest)
    y[peak:back_to_rest] = peak_y + (rest_y - peak_y) * (1 - np.cos(np.pi * (down - peak) / (back_to_rest - peak))) / 2
    return y


def slow_motion(y: np.ndarray, factor: int) -> np.ndarray:
    """Stretch a curve in time by ``factor`` (same fps, so the motion is slower)."""
    return np.interp(np.arange(len(y) * factor) / factor, np.arange(len(y)), y)

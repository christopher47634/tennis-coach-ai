"""Joint-angle geometry.

All angles are computed in *pixel* space: MediaPipe landmarks are normalised
to the frame width and height separately, so angles taken on the raw
normalised coordinates are distorted by the aspect ratio (a true 135 deg
corner measures about 119 deg in a 1920x1080 frame).  Pass the frame size so
x is scaled by the width and y by the height before any angle is taken.

Only the 2-D image-plane projection is used.  MediaPipe's z estimate is far
noisier than x/y and mixing it in made the angles unstable.
"""

import math

import numpy as np

# MediaPipe landmark indices
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_INDEX, R_INDEX = 19, 20
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# Left/right pairs, used to mirror a left-handed player onto the right side.
_MIRROR_PAIRS = [(11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),
                 (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]

JOINT_NAMES = [
    "right_elbow_flexion", "left_elbow_flexion",
    "right_shoulder_abduction", "left_shoulder_abduction",
    "right_knee_flexion", "left_knee_flexion",
    "trunk_lateral_tilt", "hip_shoulder_separation", "wrist_extension",
]


def _lm_coords(p):
    """Extract (x, y, z) from a landmark, supporting both dicts and objects."""
    if isinstance(p, dict):
        return p['x'], p['y'], p.get('z', 0.0)
    return p.x, p.y, getattr(p, 'z', 0.0)


def _px(p, size):
    """Landmark -> (x_px, y_px) float array."""
    x, y, _ = _lm_coords(p)
    w, h = size
    return np.array([x * w, y * h], dtype=float)


def angle_between(p1, p2, p3, size=(1, 1)):
    """Angle in degrees at p2 formed by p1-p2-p3, measured in pixels.

    ``size`` is (width, height) of the frame the normalised landmarks refer to.
    """
    a, b, c = _px(p1, size), _px(p2, size), _px(p3, size)
    v1, v2 = a - b, c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-9:
        return float('nan')
    cos_angle = float(np.dot(v1, v2) / denom)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def line_angle_deg(p_from, p_to, size=(1, 1)) -> float:
    """Direction of the line p_from->p_to in degrees, image plane, pixel space."""
    d = _px(p_to, size) - _px(p_from, size)
    return math.degrees(math.atan2(d[1], d[0]))


def _fold_axis_angle(deg: float) -> float:
    """Fold the angle between two undirected lines into [0, 90]."""
    deg = abs(deg) % 180.0
    return 180.0 - deg if deg > 90.0 else deg


def shoulder_tilt(lm, size=(1, 1)) -> float:
    """Tilt of the shoulder line from horizontal, 0 (level) .. 90 degrees."""
    return _fold_axis_angle(line_angle_deg(lm[R_SHOULDER], lm[L_SHOULDER], size))


def hip_shoulder_separation(lm, size=(1, 1)) -> float:
    """Angle between the shoulder line and the hip line, 0 .. 90 degrees."""
    shoulder = line_angle_deg(lm[R_SHOULDER], lm[L_SHOULDER], size)
    hip = line_angle_deg(lm[R_HIP], lm[L_HIP], size)
    return _fold_axis_angle(shoulder - hip)


def mirror_landmarks(lm):
    """Swap left and right landmarks (so a left-hander is analysed as right)."""
    out = list(lm)
    for a, b in _MIRROR_PAIRS:
        if a < len(out) and b < len(out):
            out[a], out[b] = out[b], out[a]
    return out


def compute_frame_angles(lm, size=(1, 1)):
    """Compute the 9 serve-relevant joint angles for one frame (degrees)."""
    angles = {}
    angles['right_elbow_flexion'] = angle_between(lm[R_SHOULDER], lm[R_ELBOW], lm[R_WRIST], size)
    angles['left_elbow_flexion'] = angle_between(lm[L_SHOULDER], lm[L_ELBOW], lm[L_WRIST], size)
    angles['right_shoulder_abduction'] = angle_between(lm[R_ELBOW], lm[R_SHOULDER], lm[R_HIP], size)
    angles['left_shoulder_abduction'] = angle_between(lm[L_ELBOW], lm[L_SHOULDER], lm[L_HIP], size)
    angles['right_knee_flexion'] = angle_between(lm[R_HIP], lm[R_KNEE], lm[R_ANKLE], size)
    angles['left_knee_flexion'] = angle_between(lm[L_HIP], lm[L_KNEE], lm[L_ANKLE], size)
    angles['trunk_lateral_tilt'] = shoulder_tilt(lm, size)
    angles['hip_shoulder_separation'] = hip_shoulder_separation(lm, size)
    index = lm[R_INDEX] if len(lm) > R_INDEX else lm[R_WRIST]
    angles['wrist_extension'] = angle_between(lm[R_ELBOW], lm[R_WRIST], index, size)
    return {k: v for k, v in angles.items() if not math.isnan(v)}


def torso_length_px(lm, size=(1, 1)) -> float:
    """Distance from the shoulder midpoint to the hip midpoint, in pixels."""
    sh = (_px(lm[L_SHOULDER], size) + _px(lm[R_SHOULDER], size)) / 2
    hp = (_px(lm[L_HIP], size) + _px(lm[R_HIP], size)) / 2
    return float(np.linalg.norm(hp - sh))


def wrist_height_rel(lm, wrist_idx, size=(1, 1)) -> float:
    """Wrist height relative to the shoulders, in torso lengths.

    Image y grows downward: 0 = wrist at shoulder level, negative = above the
    shoulders, +1 = about hip level.  Dividing by the torso length makes the
    signal independent of camera zoom/distance, and subtracting the shoulder
    makes it independent of camera pan/tilt.  NaN when the torso degenerates.
    """
    torso = torso_length_px(lm, size)
    if torso < 1e-6:
        return float('nan')
    shoulder_y = (_px(lm[L_SHOULDER], size)[1] + _px(lm[R_SHOULDER], size)[1]) / 2
    return float((_px(lm[wrist_idx], size)[1] - shoulder_y) / torso)

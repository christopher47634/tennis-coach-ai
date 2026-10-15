"""Pose extraction and phased serve analysis pipeline.

Adapted from ~/tennis-poc/extract_pose.py and ~/tennis-poc/analyze_angles.py.
Uses MediaPipe 0.10.14 solutions API (NOT the Tasks API).

v2: Implements 5-phase serve analysis engine:
  1. 准备期 (Preparation)
  2. 蓄力期 (Loading/Trophy)
  3. 加速期 (Acceleration)
  4. 击球期 (Contact)
  5. 随挥期 (Follow-through)
"""

import os
import json
import uuid
import numpy as np

from settings import CONFIG

# cv2 / mediapipe are imported lazily inside the video functions so that the
# pure scoring/phase code (and its tests) only needs numpy.

# MediaPipe landmark indices
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
R_INDEX = 20

JOINT_NAMES = [
    "right_elbow_flexion", "left_elbow_flexion",
    "right_shoulder_abduction", "left_shoulder_abduction",
    "right_knee_flexion", "left_knee_flexion",
    "trunk_lateral_tilt", "hip_shoulder_separation", "wrist_extension",
]

# Default expert baselines (overridden by data/expert_baselines.json if present)
DEFAULT_BASELINES = {
    "right_elbow_flexion":      {"mean": 100, "std": 15},
    "left_elbow_flexion":       {"mean": 90,  "std": 15},
    "right_shoulder_abduction": {"mean": 95,  "std": 12},
    "left_shoulder_abduction":  {"mean": 85,  "std": 12},
    "right_knee_flexion":       {"mean": 140, "std": 15},
    "left_knee_flexion":        {"mean": 145, "std": 15},
    "trunk_lateral_tilt":       {"mean": 170, "std": 8},
    "hip_shoulder_separation":  {"mean": 160, "std": 10},
    "wrist_extension":          {"mean": 160, "std": 12},
}

# ── Phase definitions ──────────────────────────────────────────
_PHASE_WEIGHTS = CONFIG["phases"]["weights"]
PHASE_DEFS = [
    {"name": "准备期", "name_en": "Preparation", "weight": _PHASE_WEIGHTS[0]},
    {"name": "蓄力期", "name_en": "Loading/Trophy", "weight": _PHASE_WEIGHTS[1]},
    {"name": "加速期", "name_en": "Acceleration", "weight": _PHASE_WEIGHTS[2]},
    {"name": "击球期", "name_en": "Contact", "weight": _PHASE_WEIGHTS[3]},
    {"name": "随挥期", "name_en": "Follow-through", "weight": _PHASE_WEIGHTS[4]},
]

# Phase-specific key angles and their Chinese names
PHASE_ANGLE_LABELS = {
    "准备期": {
        "right_knee_flexion": "右膝弯曲",
        "left_knee_flexion": "左膝弯曲",
        "right_shoulder_abduction": "右肩外展",
        "left_shoulder_abduction": "左肩外展",
    },
    "蓄力期": {
        "right_knee_flexion": "右膝弯曲",
        "left_knee_flexion": "左膝弯曲",
        "right_shoulder_abduction": "右肩外展",
        "right_elbow_flexion": "右肘弯曲",
        "wrist_extension": "手腕伸展",
    },
    "加速期": {
        "right_shoulder_abduction": "右肩外展",
        "right_elbow_flexion": "右肘弯曲",
        "trunk_lateral_tilt": "躯干侧倾",
        "hip_shoulder_separation": "髋肩分离",
    },
    "击球期": {
        "right_shoulder_abduction": "右肩外展",
        "right_elbow_flexion": "右肘弯曲",
        "wrist_extension": "手腕伸展",
        "trunk_lateral_tilt": "躯干侧倾",
        "hip_shoulder_separation": "髋肩分离",
    },
    "随挥期": {
        "right_elbow_flexion": "右肘弯曲",
        "right_shoulder_abduction": "右肩外展",
        "trunk_lateral_tilt": "躯干侧倾",
        "left_knee_flexion": "左膝弯曲",
    },
}


def load_baselines(baselines_path: str | None = None) -> dict:
    """Load expert baselines from JSON file, falling back to defaults."""
    if baselines_path and os.path.exists(baselines_path):
        with open(baselines_path) as f:
            data = json.load(f)
        if "serve" in data and isinstance(data["serve"], dict):
            return data["serve"]
        return {k: v for k, v in data.items() if isinstance(v, dict) and "mean" in v}
    return DEFAULT_BASELINES


def _lm_coords(p):
    """Extract (x, y, z) from a landmark, supporting both dicts and objects."""
    if isinstance(p, dict):
        return p['x'], p['y'], p['z']
    return p.x, p.y, p.z


def angle_between(p1, p2, p3):
    """Calculate angle at p2 formed by p1-p2-p3 (in degrees)."""
    x1, y1, z1 = _lm_coords(p1)
    x2, y2, z2 = _lm_coords(p2)
    x3, y3, z3 = _lm_coords(p3)
    v1 = np.array([x1 - x2, y1 - y2, z1 - z2])
    v2 = np.array([x3 - x2, y3 - y2, z3 - z2])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))


def mid_point(p1, p2):
    x1, y1, z1 = _lm_coords(p1)
    x2, y2, z2 = _lm_coords(p2)
    return {'x': (x1 + x2) / 2, 'y': (y1 + y2) / 2, 'z': (z1 + z2) / 2}


def compute_frame_angles(lm):
    """Compute 9 serve-relevant joint angles for one frame."""
    angles = {}
    angles['right_elbow_flexion'] = angle_between(lm[R_SHOULDER], lm[R_ELBOW], lm[R_WRIST])
    angles['left_elbow_flexion'] = angle_between(lm[L_SHOULDER], lm[L_ELBOW], lm[L_WRIST])
    angles['right_shoulder_abduction'] = angle_between(lm[R_ELBOW], lm[R_SHOULDER], lm[R_HIP])
    angles['left_shoulder_abduction'] = angle_between(lm[L_ELBOW], lm[L_SHOULDER], lm[L_HIP])
    angles['right_knee_flexion'] = angle_between(lm[R_HIP], lm[R_KNEE], lm[R_ANKLE])
    angles['left_knee_flexion'] = angle_between(lm[L_HIP], lm[L_KNEE], lm[L_ANKLE])
    mid_spine = mid_point(lm[L_SHOULDER], lm[R_SHOULDER])
    angles['trunk_lateral_tilt'] = angle_between(lm[L_SHOULDER], mid_spine, lm[R_SHOULDER])
    angles['hip_shoulder_separation'] = angle_between(lm[L_HIP], lm[R_HIP], lm[R_SHOULDER])
    angles['wrist_extension'] = angle_between(
        lm[R_ELBOW], lm[R_WRIST], lm[R_INDEX] if len(lm) > R_INDEX else lm[R_WRIST]
    )
    return angles


def rate_severity(severity: float, cfg: dict | None = None) -> str:
    """Map a 0-1 severity to good / moderate / needs_work."""
    sc = (cfg or CONFIG)["scoring"]
    if severity < sc["good_below"]:
        return 'good'
    if severity < sc["moderate_below"]:
        return 'moderate'
    return 'needs_work'


def score_angles(angles: dict, baselines: dict, cfg: dict | None = None) -> dict:
    """Score each angle against expert baselines."""
    mult = (cfg or CONFIG)["scoring"]["severity_std_multiplier"]
    scores = {}
    for name, val in angles.items():
        if name in baselines:
            expert = baselines[name]
            deviation = abs(val - expert['mean'])
            severity = min(deviation / (mult * expert['std']), 1.0)
            rating = rate_severity(severity, cfg)
            scores[name] = {
                'value': round(val, 1),
                'expert_mean': expert['mean'],
                'deviation': round(deviation, 1),
                'severity': round(severity, 2),
                'rating': rating,
            }
    return scores


def _color_for_rating(rating: str) -> tuple:
    """Return BGR color for a rating."""
    if rating == 'good':
        return (0, 255, 0)       # green
    elif rating == 'moderate':
        return (0, 255, 255)     # yellow
    else:
        return (0, 0, 255)       # red


# ── Reference skeleton generation ─────────────────────────────

# 5 key poses: 33 landmarks (MediaPipe normalized coords) for
# 站立准备, 膝盖弯曲蓄力, 手臂上举, 击球伸展, 随挥收拍
_KEY_POSES = [
    # Pose 0: 站立准备 (standing ready)
    [
        {'x': 0.50, 'y': 0.10, 'z': 0.0},   # 0  nose
        {'x': 0.48, 'y': 0.09, 'z': 0.0},   # 1  left_eye_inner
        {'x': 0.48, 'y': 0.09, 'z': 0.0},   # 2  left_eye
        {'x': 0.48, 'y': 0.09, 'z': 0.0},   # 3  left_eye_outer
        {'x': 0.52, 'y': 0.09, 'z': 0.0},   # 4  right_eye_inner
        {'x': 0.52, 'y': 0.09, 'z': 0.0},   # 5  right_eye
        {'x': 0.52, 'y': 0.09, 'z': 0.0},   # 6  right_eye_outer
        {'x': 0.47, 'y': 0.10, 'z': 0.0},   # 7  left_ear
        {'x': 0.53, 'y': 0.10, 'z': 0.0},   # 8  right_ear
        {'x': 0.49, 'y': 0.10, 'z': 0.0},   # 9  mouth_left
        {'x': 0.51, 'y': 0.10, 'z': 0.0},   # 10 mouth_right
        {'x': 0.44, 'y': 0.18, 'z': 0.0},   # 11 left_shoulder
        {'x': 0.56, 'y': 0.18, 'z': 0.0},   # 12 right_shoulder
        {'x': 0.40, 'y': 0.30, 'z': 0.0},   # 13 left_elbow
        {'x': 0.60, 'y': 0.30, 'z': 0.0},   # 14 right_elbow
        {'x': 0.38, 'y': 0.42, 'z': 0.0},   # 15 left_wrist
        {'x': 0.62, 'y': 0.42, 'z': 0.0},   # 16 right_wrist
        {'x': 0.37, 'y': 0.43, 'z': 0.0},   # 17 left_pinky
        {'x': 0.63, 'y': 0.43, 'z': 0.0},   # 18 right_pinky
        {'x': 0.37, 'y': 0.43, 'z': 0.0},   # 19 left_index
        {'x': 0.63, 'y': 0.43, 'z': 0.0},   # 20 right_index
        {'x': 0.37, 'y': 0.43, 'z': 0.0},   # 21 left_thumb
        {'x': 0.63, 'y': 0.43, 'z': 0.0},   # 22 right_thumb
        {'x': 0.46, 'y': 0.40, 'z': 0.0},   # 23 left_hip
        {'x': 0.54, 'y': 0.40, 'z': 0.0},   # 24 right_hip
        {'x': 0.44, 'y': 0.58, 'z': 0.0},   # 25 left_knee
        {'x': 0.56, 'y': 0.58, 'z': 0.0},   # 26 right_knee
        {'x': 0.43, 'y': 0.76, 'z': 0.0},   # 27 left_ankle
        {'x': 0.57, 'y': 0.76, 'z': 0.0},   # 28 right_ankle
        {'x': 0.43, 'y': 0.77, 'z': 0.0},   # 29 left_heel
        {'x': 0.57, 'y': 0.77, 'z': 0.0},   # 30 right_heel
        {'x': 0.42, 'y': 0.78, 'z': 0.0},   # 31 left_foot_index
        {'x': 0.58, 'y': 0.78, 'z': 0.0},   # 32 right_foot_index
    ],
    # Pose 1: 膝盖弯曲蓄力 (knee bend loading)
    [
        {'x': 0.50, 'y': 0.12, 'z': 0.0},
        {'x': 0.48, 'y': 0.11, 'z': 0.0},
        {'x': 0.48, 'y': 0.11, 'z': 0.0},
        {'x': 0.48, 'y': 0.11, 'z': 0.0},
        {'x': 0.52, 'y': 0.11, 'z': 0.0},
        {'x': 0.52, 'y': 0.11, 'z': 0.0},
        {'x': 0.52, 'y': 0.11, 'z': 0.0},
        {'x': 0.47, 'y': 0.12, 'z': 0.0},
        {'x': 0.53, 'y': 0.12, 'z': 0.0},
        {'x': 0.49, 'y': 0.12, 'z': 0.0},
        {'x': 0.51, 'y': 0.12, 'z': 0.0},
        {'x': 0.44, 'y': 0.20, 'z': 0.0},
        {'x': 0.56, 'y': 0.20, 'z': 0.0},
        {'x': 0.40, 'y': 0.28, 'z': 0.0},
        {'x': 0.60, 'y': 0.26, 'z': 0.0},
        {'x': 0.38, 'y': 0.38, 'z': 0.0},
        {'x': 0.62, 'y': 0.32, 'z': 0.0},
        {'x': 0.37, 'y': 0.39, 'z': 0.0},
        {'x': 0.63, 'y': 0.33, 'z': 0.0},
        {'x': 0.37, 'y': 0.39, 'z': 0.0},
        {'x': 0.63, 'y': 0.33, 'z': 0.0},
        {'x': 0.37, 'y': 0.39, 'z': 0.0},
        {'x': 0.63, 'y': 0.33, 'z': 0.0},
        {'x': 0.46, 'y': 0.38, 'z': 0.0},
        {'x': 0.54, 'y': 0.38, 'z': 0.0},
        {'x': 0.42, 'y': 0.52, 'z': 0.0},
        {'x': 0.58, 'y': 0.52, 'z': 0.0},
        {'x': 0.43, 'y': 0.70, 'z': 0.0},
        {'x': 0.57, 'y': 0.70, 'z': 0.0},
        {'x': 0.43, 'y': 0.71, 'z': 0.0},
        {'x': 0.57, 'y': 0.71, 'z': 0.0},
        {'x': 0.42, 'y': 0.72, 'z': 0.0},
        {'x': 0.58, 'y': 0.72, 'z': 0.0},
    ],
    # Pose 2: 手臂上举 (arm raised trophy)
    [
        {'x': 0.50, 'y': 0.11, 'z': 0.0},
        {'x': 0.48, 'y': 0.10, 'z': 0.0},
        {'x': 0.48, 'y': 0.10, 'z': 0.0},
        {'x': 0.48, 'y': 0.10, 'z': 0.0},
        {'x': 0.52, 'y': 0.10, 'z': 0.0},
        {'x': 0.52, 'y': 0.10, 'z': 0.0},
        {'x': 0.52, 'y': 0.10, 'z': 0.0},
        {'x': 0.47, 'y': 0.11, 'z': 0.0},
        {'x': 0.53, 'y': 0.11, 'z': 0.0},
        {'x': 0.49, 'y': 0.11, 'z': 0.0},
        {'x': 0.51, 'y': 0.11, 'z': 0.0},
        {'x': 0.44, 'y': 0.19, 'z': 0.0},
        {'x': 0.56, 'y': 0.19, 'z': 0.0},
        {'x': 0.38, 'y': 0.30, 'z': 0.0},
        {'x': 0.62, 'y': 0.12, 'z': 0.0},
        {'x': 0.36, 'y': 0.40, 'z': 0.0},
        {'x': 0.60, 'y': 0.05, 'z': 0.0},
        {'x': 0.35, 'y': 0.41, 'z': 0.0},
        {'x': 0.61, 'y': 0.04, 'z': 0.0},
        {'x': 0.35, 'y': 0.41, 'z': 0.0},
        {'x': 0.61, 'y': 0.04, 'z': 0.0},
        {'x': 0.35, 'y': 0.41, 'z': 0.0},
        {'x': 0.61, 'y': 0.04, 'z': 0.0},
        {'x': 0.46, 'y': 0.39, 'z': 0.0},
        {'x': 0.54, 'y': 0.39, 'z': 0.0},
        {'x': 0.43, 'y': 0.55, 'z': 0.0},
        {'x': 0.57, 'y': 0.55, 'z': 0.0},
        {'x': 0.43, 'y': 0.73, 'z': 0.0},
        {'x': 0.57, 'y': 0.73, 'z': 0.0},
        {'x': 0.43, 'y': 0.74, 'z': 0.0},
        {'x': 0.57, 'y': 0.74, 'z': 0.0},
        {'x': 0.42, 'y': 0.75, 'z': 0.0},
        {'x': 0.58, 'y': 0.75, 'z': 0.0},
    ],
    # Pose 3: 击球伸展 (contact — full extension)
    [
        {'x': 0.50, 'y': 0.10, 'z': 0.0},
        {'x': 0.48, 'y': 0.09, 'z': 0.0},
        {'x': 0.48, 'y': 0.09, 'z': 0.0},
        {'x': 0.48, 'y': 0.09, 'z': 0.0},
        {'x': 0.52, 'y': 0.09, 'z': 0.0},
        {'x': 0.52, 'y': 0.09, 'z': 0.0},
        {'x': 0.52, 'y': 0.09, 'z': 0.0},
        {'x': 0.47, 'y': 0.10, 'z': 0.0},
        {'x': 0.53, 'y': 0.10, 'z': 0.0},
        {'x': 0.49, 'y': 0.10, 'z': 0.0},
        {'x': 0.51, 'y': 0.10, 'z': 0.0},
        {'x': 0.44, 'y': 0.17, 'z': 0.0},
        {'x': 0.56, 'y': 0.17, 'z': 0.0},
        {'x': 0.38, 'y': 0.28, 'z': 0.0},
        {'x': 0.64, 'y': 0.08, 'z': 0.0},
        {'x': 0.36, 'y': 0.38, 'z': 0.0},
        {'x': 0.62, 'y': 0.02, 'z': 0.0},
        {'x': 0.35, 'y': 0.39, 'z': 0.0},
        {'x': 0.63, 'y': 0.01, 'z': 0.0},
        {'x': 0.35, 'y': 0.39, 'z': 0.0},
        {'x': 0.63, 'y': 0.01, 'z': 0.0},
        {'x': 0.35, 'y': 0.39, 'z': 0.0},
        {'x': 0.63, 'y': 0.01, 'z': 0.0},
        {'x': 0.46, 'y': 0.38, 'z': 0.0},
        {'x': 0.54, 'y': 0.38, 'z': 0.0},
        {'x': 0.44, 'y': 0.56, 'z': 0.0},
        {'x': 0.56, 'y': 0.56, 'z': 0.0},
        {'x': 0.43, 'y': 0.74, 'z': 0.0},
        {'x': 0.57, 'y': 0.74, 'z': 0.0},
        {'x': 0.43, 'y': 0.75, 'z': 0.0},
        {'x': 0.57, 'y': 0.75, 'z': 0.0},
        {'x': 0.42, 'y': 0.76, 'z': 0.0},
        {'x': 0.58, 'y': 0.76, 'z': 0.0},
    ],
    # Pose 4: 随挥收拍 (follow-through)
    [
        {'x': 0.50, 'y': 0.10, 'z': 0.0},
        {'x': 0.48, 'y': 0.09, 'z': 0.0},
        {'x': 0.48, 'y': 0.09, 'z': 0.0},
        {'x': 0.48, 'y': 0.09, 'z': 0.0},
        {'x': 0.52, 'y': 0.09, 'z': 0.0},
        {'x': 0.52, 'y': 0.09, 'z': 0.0},
        {'x': 0.52, 'y': 0.09, 'z': 0.0},
        {'x': 0.47, 'y': 0.10, 'z': 0.0},
        {'x': 0.53, 'y': 0.10, 'z': 0.0},
        {'x': 0.49, 'y': 0.10, 'z': 0.0},
        {'x': 0.51, 'y': 0.10, 'z': 0.0},
        {'x': 0.44, 'y': 0.18, 'z': 0.0},
        {'x': 0.56, 'y': 0.18, 'z': 0.0},
        {'x': 0.38, 'y': 0.30, 'z': 0.0},
        {'x': 0.58, 'y': 0.14, 'z': 0.0},
        {'x': 0.36, 'y': 0.40, 'z': 0.0},
        {'x': 0.50, 'y': 0.28, 'z': 0.0},
        {'x': 0.35, 'y': 0.41, 'z': 0.0},
        {'x': 0.49, 'y': 0.29, 'z': 0.0},
        {'x': 0.35, 'y': 0.41, 'z': 0.0},
        {'x': 0.49, 'y': 0.29, 'z': 0.0},
        {'x': 0.35, 'y': 0.41, 'z': 0.0},
        {'x': 0.49, 'y': 0.29, 'z': 0.0},
        {'x': 0.46, 'y': 0.38, 'z': 0.0},
        {'x': 0.54, 'y': 0.38, 'z': 0.0},
        {'x': 0.44, 'y': 0.56, 'z': 0.0},
        {'x': 0.56, 'y': 0.56, 'z': 0.0},
        {'x': 0.43, 'y': 0.74, 'z': 0.0},
        {'x': 0.57, 'y': 0.74, 'z': 0.0},
        {'x': 0.43, 'y': 0.75, 'z': 0.0},
        {'x': 0.57, 'y': 0.75, 'z': 0.0},
        {'x': 0.42, 'y': 0.76, 'z': 0.0},
        {'x': 0.58, 'y': 0.76, 'z': 0.0},
    ],
]


def _lerp_pose(pose_a: list, pose_b: list, t: float) -> list:
    """Linearly interpolate two 33-landmark poses. t in [0, 1]."""
    result = []
    for a, b in zip(pose_a, pose_b):
        result.append({
            'x': a['x'] + (b['x'] - a['x']) * t,
            'y': a['y'] + (b['y'] - a['y']) * t,
            'z': a['z'] + (b['z'] - a['z']) * t,
        })
    return result


def _generate_reference_skeleton(baselines: dict) -> dict:
    """Generate a 120-frame ideal serve skeleton sequence.

    5 phases × 24 frames each.  Sinusoidal interpolation between
    key poses so the motion looks smooth.  Joint angles in each
    frame are computed from the generated landmarks AND overridden
    to match expert baseline mean values where possible.

    Returns dict with keys: frames, fps, total_frames, description.
    """
    total_frames = 120
    fps = 30
    frames_per_phase = total_frames // 5  # 24
    phase_names = [p['name'] for p in PHASE_DEFS]
    num_key_poses = len(_KEY_POSES)

    frames = []
    for i in range(total_frames):
        phase_idx = min(i // frames_per_phase, 4)
        # Position within the current phase transition [0, 1)
        t_in_phase = (i % frames_per_phase) / frames_per_phase

        # Key pose A = phase_idx, Key pose B = phase_idx + 1 (clamped)
        pose_a_idx = phase_idx
        pose_b_idx = min(phase_idx + 1, num_key_poses - 1)

        if pose_a_idx == pose_b_idx:
            t = 0.0
        else:
            # Sinusoidal easing: smooth acceleration/deceleration
            t = (1.0 - np.cos(np.pi * t_in_phase)) / 2.0

        lm = _lerp_pose(_KEY_POSES[pose_a_idx], _KEY_POSES[pose_b_idx], float(t))

        # Compute actual angles from the generated landmarks
        angles = compute_frame_angles(lm)
        # Override angles with expert baseline means for accuracy
        for joint_name, bl in baselines.items():
            if joint_name in angles:
                angles[joint_name] = float(bl['mean'])

        frames.append({
            'landmarks': lm,
            'phase': phase_names[phase_idx],
            'angles': {k: round(v, 1) for k, v in angles.items()},
        })

    return {
        'description': '基于专家基线数据生成的理想发球骨架序列',
        'frames': frames,
        'fps': fps,
        'total_frames': total_frames,
    }


# ── Angle comparison ──────────────────────────────────────────

_RATING_COLORS_HEX = {
    'good': '#00ff88',
    'moderate': '#ffaa00',
    'needs_work': '#ff4444',
}


def _build_angle_comparison(user_angles: dict, baselines: dict, cfg: dict | None = None) -> dict:
    """Compare user angles against expert baselines.

    Returns dict keyed by joint name, each containing:
        user_value, expert_value, deviation, rating, color

    Rating thresholds (deviation as fraction of 2*std, see config/default.toml):
        good:       < 0.30  (green)
        moderate:   0.30 – 0.60  (yellow)
        needs_work: > 0.60  (red)
    """
    mult = (cfg or CONFIG)["scoring"]["severity_std_multiplier"]
    comparison = {}
    for joint_name, bl in baselines.items():
        user_val = user_angles.get(joint_name)
        if user_val is None:
            continue
        expert_val = bl['mean']
        deviation = abs(user_val - expert_val)
        severity = min(deviation / (mult * bl['std']), 1.0) if bl.get('std') else 0.0
        rating = rate_severity(severity, cfg)

        comparison[joint_name] = {
            'user_value': round(float(user_val), 1),
            'expert_value': round(float(expert_val), 1),
            'deviation': round(float(deviation), 1),
            'rating': rating,
            'color': _RATING_COLORS_HEX[rating],
        }

    return comparison


# Mapping from our joint names to the MediaPipe landmarks that define them
JOINT_LANDMARKS = {
    'right_elbow_flexion':      [R_SHOULDER, R_ELBOW, R_WRIST],
    'left_elbow_flexion':       [L_SHOULDER, L_ELBOW, L_WRIST],
    'right_shoulder_abduction': [R_ELBOW, R_SHOULDER, R_HIP],
    'left_shoulder_abduction':  [L_ELBOW, L_SHOULDER, L_HIP],
    'right_knee_flexion':       [R_HIP, R_KNEE, R_ANKLE],
    'left_knee_flexion':        [L_HIP, L_KNEE, L_ANKLE],
    'wrist_extension':          [R_ELBOW, R_WRIST, R_INDEX],
}

# Full skeleton connections for drawing (pairs of landmark indices)
SKELETON_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    (11, 23), (12, 24), (23, 24),                       # torso
    (23, 25), (25, 27), (24, 26), (26, 28),             # legs
    (0, 11), (0, 12),                                     # neck
]

# ── Phase colors for overlay (BGR) ─────────────────────────────
PHASE_COLORS_BGR = [
    (200, 200, 200),  # 准备期: light gray
    (0, 200, 255),    # 蓄力期: orange
    (0, 0, 255),      # 加速期: red
    (0, 255, 0),      # 击球期: bright green
    (255, 200, 0),    # 随挥期: cyan/light blue
]


# ── Phase boundary detection ───────────────────────────────────

def _smooth(arr, window=5):
    """Simple moving average smoothing."""
    if len(arr) < window:
        return arr.copy()
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')


def _detect_phases(wrist_y: np.ndarray, knee_angles: list, total_frames: int,
                   cfg: dict | None = None) -> list:
    """Detect 5 serve phases using right wrist y-coordinate + knee angles.

    In MediaPipe coordinates, y=0 is top of frame, y=1 is bottom.
    The serve motion has these characteristics:
      - Wrist y decreases (rises) during ball toss and arm cocking
      - Wrist y is at minimum (highest point) at the trophy position
      - Wrist y increases during the downward swing to contact
      - Knee angle decreases (bends more) during loading/leg drive

    Since wrist y changes can be subtle in some videos, we use a hybrid
    approach: find the serve window from wrist trajectory, then use
    biomechanical landmarks (knee bend, wrist peak) to split it into phases.

    Returns list of (start_frame, end_frame) tuples for each phase.
    """
    dc = (cfg or CONFIG)["detection"]
    n = len(wrist_y)
    if n < dc["min_frames_for_phases"]:
        chunk = max(1, n // 5)
        return [(i * chunk, min((i + 1) * chunk, n)) for i in range(5)]

    # Smooth the wrist y signal
    wy = _smooth(wrist_y, window=max(dc["smooth_min_window"], n // dc["smooth_divisor"]))

    # ── Step 1: Find the dominant serve action ──
    # The video may contain multiple serves. Find the one with the deepest
    # wrist rise (largest drop in y value from preceding local max to local min).
    search_win = max(dc["peak_search_min"], n // dc["peak_search_divisor"])

    # Find candidate peaks: local minima of y (wrist at highest point)
    candidate_peaks = []
    for i in range(search_win, n - search_win):
        local_min = np.min(wy[max(0, i - search_win):i + search_win + 1])
        if abs(wy[i] - local_min) < 1e-6:
            candidate_peaks.append(i)

    # Deduplicate nearby peaks
    if candidate_peaks:
        merged = [candidate_peaks[0]]
        for p in candidate_peaks[1:]:
            if p - merged[-1] > search_win:
                merged.append(p)
        candidate_peaks = merged
    if not candidate_peaks:
        candidate_peaks = [int(np.argmin(wy))]

    # Pick the peak with the largest y-drop from its preceding maximum
    best_peak = candidate_peaks[0]
    best_drop = 0.0
    for peak_cand in candidate_peaks:
        lookback = max(0, peak_cand - search_win * 3)
        pre_max = np.max(wy[lookback:peak_cand]) if peak_cand > lookback else wy[0]
        drop = pre_max - wy[peak_cand]
        if drop > best_drop:
            best_drop = drop
            best_peak = peak_cand

    peak_idx = best_peak
    peak_y = wy[peak_idx]

    # ── Step 2: Find the serve window boundaries ──
    # serve_start: where the significant wrist descent begins before the peak
    pre_lookback = 0  # start from beginning of video
    y_before = wy[pre_lookback]
    y_range_before = y_before - peak_y

    if y_range_before > 0.02:
        # Find where the wrist starts descending toward the peak
        threshold_start = y_before - y_range_before * 0.12
        serve_start = pre_lookback
        for i in range(peak_idx, pre_lookback - 1, -1):
            if wy[i] > threshold_start:
                serve_start = max(0, i)
                break
    else:
        serve_start = max(0, peak_idx - search_win * 3)

    # serve_end: where the wrist settles after the peak (post-contact)
    post_lookforward = n - 1  # extend to end of video
    y_after = wy[post_lookforward]
    y_range_after = y_after - peak_y

    if y_range_after > 0.02:
        threshold_end = peak_y + y_range_after * 0.80
        serve_end = post_lookforward
        for i in range(peak_idx, post_lookforward + 1):
            if wy[i] > threshold_end:
                serve_end = min(n, i + 3)
                break
    else:
        serve_end = min(n, peak_idx + search_win * 3)

    # Ensure minimum window size (at least 50% of video or 100 frames)
    # Also expand to cover full video for better timeline coverage
    serve_start = 0  # start from beginning of video
    serve_end = n    # end at end of video
    serve_duration = serve_end - serve_start

    # ── Step 3: Find biomechanical landmarks within the serve window ──

    # A. Loading start: where knee bending accelerates (knee angle drops)
    loading_start = serve_start + serve_duration // dc["loading_default_divisor"]  # default: 25% into serve
    if knee_angles:
        ka_window = []
        for i in range(serve_start, min(peak_idx + 1, len(knee_angles))):
            ka_window.append((i, knee_angles[i] if knee_angles[i] is not None else dc["knee_missing_default"]))
        if len(ka_window) > 5:
            ka_vals = np.array([k[1] for k in ka_window])
            ka_smooth = _smooth(ka_vals, window=max(3, len(ka_vals) // 8))
            # Find where knee angle drops below the starting value by a meaningful amount
            start_knee = ka_smooth[0]
            min_knee_val = np.min(ka_smooth)
            knee_drop = start_knee - min_knee_val
            if knee_drop > dc["knee_min_drop"]:  # meaningful bend
                threshold_knee = start_knee - knee_drop * dc["knee_loading_fraction"]
                for i in range(len(ka_smooth)):
                    if ka_smooth[i] < threshold_knee:
                        loading_start = serve_start + i
                        break
                # Also find the deepest knee bend as the end of loading
                deepest_knee_idx = int(np.argmin(ka_smooth))
                # Peak of serve motion (trophy position) aligns with deep knee bend
                # Use the deeper of: knee-bend peak or wrist peak
                if deepest_knee_idx > 3:
                    peak_idx = max(peak_idx, serve_start + deepest_knee_idx)

    # Ensure loading_start is well before peak
    loading_start = max(serve_start + 3, min(loading_start, peak_idx - 8))

    # B. Contact point: after peak, the wrist starts descending then stabilizes
    # Look for the inflection point in wrist y after the peak
    contact_search_end = min(serve_end, peak_idx + serve_duration // 2)
    post_peak_wy = wy[peak_idx:contact_search_end]
    if len(post_peak_wy) > 5:
        deriv = np.gradient(post_peak_wy)
        # Find where the y-increase (arm coming down) decelerates
        max_increase_idx = int(np.argmax(deriv))
        # Contact is where the rate of descent starts to slow
        contact_offset = max_increase_idx
        if max_increase_idx + 1 < len(deriv):
            threshold_deriv = deriv[max_increase_idx] * dc["contact_decel_fraction"]
            for j in range(max_increase_idx, len(deriv)):
                if deriv[j] < threshold_deriv:
                    contact_offset = j
                    break
        contact_idx = peak_idx + contact_offset
    else:
        contact_idx = peak_idx + max(3, (serve_end - peak_idx) // 3)

    # ── Step 4: Construct 5 phases with reasonable proportions ──
    # Phase 1: Preparation = [serve_start, loading_start]
    # Phase 2: Loading = [loading_start, peak_idx]
    # Phase 3: Acceleration = [peak_idx, mid-point between peak and contact]
    # Phase 4: Contact = [around contact_idx, narrow window]
    # Phase 5: Follow-through = [after contact, serve_end]

    accel_duration = contact_idx - peak_idx
    contact_half_width = max(dc["contact_half_width_min"], accel_duration // dc["contact_half_width_divisor"])

    accel_end = max(peak_idx + 3, contact_idx - contact_half_width)
    contact_end = min(contact_idx + contact_half_width, serve_end - 5)
    if contact_end <= accel_end:
        contact_end = accel_end + max(3, accel_duration // 4)

    # If follow-through is too short (< 5% of serve), extend it
    if serve_end - contact_end < serve_duration * dc["followthrough_min_fraction"]:
        contact_end = max(accel_end + 3, serve_end - max(5, serve_duration // 8))

    phases = [
        (serve_start, loading_start),
        (loading_start, peak_idx),
        (peak_idx, accel_end),
        (accel_end, contact_end),
        (contact_end, serve_end),
    ]

    # Ensure no phase has zero width and phases don't overlap
    for i in range(len(phases)):
        s, e = phases[i]
        if e <= s:
            e = s + 3
        phases[i] = (s, e)
        if i > 0 and phases[i][0] < phases[i - 1][1]:
            phases[i] = (phases[i - 1][1], max(phases[i - 1][1] + 3, phases[i][1]))

    # Enforce minimum phase width (at least 5% of serve duration, min 8 frames)
    min_phase_width = max(dc["min_phase_width_floor"], serve_duration // dc["min_phase_width_divisor"])
    for i in range(len(phases)):
        s, e = phases[i]
        if e - s < min_phase_width:
            # Try to expand by stealing from neighbors
            deficit = min_phase_width - (e - s)
            # Steal from next phase if possible
            if i < len(phases) - 1:
                next_s, next_e = phases[i + 1]
                steal = min(deficit, (next_e - next_s) // 2)
                phases[i] = (s, e + steal)
                phases[i + 1] = (next_s + steal, next_e)
            # Still too small? steal from previous
            if phases[i][1] - phases[i][0] < min_phase_width and i > 0:
                prev_s, prev_e = phases[i - 1]
                steal = min(min_phase_width - (phases[i][1] - phases[i][0]), (prev_e - prev_s) // 2)
                phases[i - 1] = (prev_s, prev_e - steal)
                phases[i] = (phases[i][0] - steal, phases[i][1])

    return phases


# ── Phase commentary generation (Chinese) ──────────────────────

def _generate_phase_commentary(phase_name: str, phase_scores: dict, baselines: dict) -> str:
    """Generate Chinese biomechanical commentary for a phase."""
    if not phase_scores:
        return f"{phase_name}阶段未检测到足够姿态数据。"

    comments = []

    # Get key angles for this phase
    key_angles = PHASE_ANGLE_LABELS.get(phase_name, {})
    good_angles = []
    moderate_angles = []
    bad_angles = []

    for angle_name, cn_label in key_angles.items():
        if angle_name in phase_scores:
            s = phase_scores[angle_name]
            if s['rating'] == 'good':
                good_angles.append(cn_label)
            elif s['rating'] == 'moderate':
                moderate_angles.append((cn_label, s['value'], s['expert_mean']))
            else:
                bad_angles.append((cn_label, s['value'], s['expert_mean']))

    if phase_name == "准备期":
        if good_angles:
            comments.append(f"准备姿势良好，{good_angles[0]}接近职业水平。")
        if bad_angles:
            cn, val, ref = bad_angles[0]
            comments.append(f"{cn}偏差较大({val:.0f}° vs 参考{ref:.0f}°)，建议调整站姿。")
        if moderate_angles:
            cn, val, ref = moderate_angles[0]
            comments.append(f"{cn}略有偏差({val:.0f}°)，可通过热身拉伸改善。")
        if not comments:
            comments.append("准备阶段基本到位，保持身体放松。")

    elif phase_name == "蓄力期":
        if good_angles:
            comments.append(f"蓄力充分，{good_angles[0]}到位，为发力奠定基础。")
        if bad_angles:
            cn, val, ref = bad_angles[0]
            if "膝" in cn:
                comments.append(f"{cn}不足({val:.0f}° vs 参考{ref:.0f}°)，膝关节弯曲不够会影响蹬地发力。")
            else:
                comments.append(f"{cn}需要改进({val:.0f}° vs 参考{ref:.0f}°)。")
        if not comments:
            comments.append("蓄力阶段动能储备良好，注意保持膝盖弯曲。")

    elif phase_name == "加速期":
        if good_angles:
            comments.append(f"加速阶段发力链流畅，{good_angles[0]}表现优秀。")
        if bad_angles:
            cn, val, ref = bad_angles[0]
            if "髋" in cn or "躯干" in cn:
                comments.append(f"{cn}不足({val:.0f}°)，影响力量从下肢向上传递。")
            else:
                comments.append(f"{cn}偏离理想值({val:.0f}° vs 参考{ref:.0f}°)，影响加速效率。")
        if not comments:
            comments.append("加速阶段力量传导良好，髋肩分离到位。")

    elif phase_name == "击球期":
        if good_angles:
            comments.append(f"击球瞬间{good_angles[0]}到位，击球质量有保障。")
        if bad_angles:
            cn, val, ref = bad_angles[0]
            comments.append(f"击球时{cn}不理想({val:.0f}° vs 参考{ref:.0f}°)，会导致击球不稳定。")
        if len(good_angles) >= 2:
            comments.append("多个关键角度在职业水平范围内，击球技术扎实。")
        if not comments:
            comments.append("击球阶段基本稳定，注意手腕固定。")

    elif phase_name == "随挥期":
        if good_angles:
            comments.append(f"随挥完整，{good_angles[0]}自然流畅。")
        if bad_angles:
            cn, val, ref = bad_angles[0]
            comments.append(f"随挥时{cn}不够自然({val:.0f}° vs 参考{ref:.0f}°)，建议放松完成收拍动作。")
        if not comments:
            comments.append("随挥阶段完成度良好，注意保持身体平衡。")

    if not comments:
        overall_sev = np.mean([s['severity'] for s in phase_scores.values()])
        if overall_sev < 0.3:
            comments.append(f"{phase_name}阶段技术动作接近职业水平。")
        elif overall_sev < 0.6:
            comments.append(f"{phase_name}阶段有改进空间，注意关键角度的控制。")
        else:
            comments.append(f"{phase_name}阶段需要重点练习，建议参考职业选手动作进行模仿训练。")

    return "".join(comments)


# ── Pure scoring pipeline (no video / model dependency) ───────

_NO_PERSON_ERROR = "Could not detect person in video. Please upload a clear video of a tennis player."


def analyze_landmark_sequence(
    frame_landmarks: list,
    baselines: dict,
    total_frames: int | None = None,
    cfg: dict | None = None,
) -> dict:
    """Phase detection + scoring on an already-extracted landmark sequence.

    ``frame_landmarks`` holds one entry per video frame: a list of 33
    landmarks (dicts with x/y/z, or objects with .x/.y/.z), or an empty
    list/None when no person was detected in that frame.  This is everything
    that happens after MediaPipe, so it can be tested without a model.
    """
    cfg = cfg or CONFIG
    if total_frames is None:
        total_frames = len(frame_landmarks)

    all_frame_angles = []
    wrist_y_values = []
    knee_angle_values = []
    detected = 0

    for frame_idx, lm_list in enumerate(frame_landmarks):
        if lm_list:
            detected += 1
            wy = _lm_coords(lm_list[R_WRIST])[1] if len(lm_list) > R_WRIST else 1.0
            wrist_y_values.append(wy)
            try:
                angles = compute_frame_angles(lm_list)
                all_frame_angles.append({'frame': frame_idx, 'angles': angles})
                knee_angle_values.append(angles.get('right_knee_flexion'))
            except Exception:
                knee_angle_values.append(None)
        else:
            wrist_y_values.append(1.0)
            knee_angle_values.append(None)

    # Check detection rate
    if total_frames == 0 or detected / total_frames < cfg["detection"]["min_detection_rate"]:
        return {"success": False, "error": _NO_PERSON_ERROR}

    if not all_frame_angles:
        return {"success": False, "error": _NO_PERSON_ERROR}

    # ── Detect phases ──
    wrist_arr = np.array(wrist_y_values)
    phase_boundaries = _detect_phases(wrist_arr, knee_angle_values, total_frames, cfg)

    # ── Compute per-phase scores ──
    sc = cfg["scoring"]
    phases_result = []

    for pdef, (start_f, end_f) in zip(PHASE_DEFS, phase_boundaries):
        # Gather angles for frames in this phase
        phase_angles_list = [
            fa['angles'] for fa in all_frame_angles
            if start_f <= fa['frame'] < end_f
        ]

        if not phase_angles_list:
            # No data for this phase
            phases_result.append({
                "name": pdef["name"],
                "name_en": pdef["name_en"],
                "start_frame": start_f,
                "end_frame": end_f,
                "score": sc["empty_phase_score"],
                "angles": {},
                "commentary": f"{pdef['name']}阶段未检测到足够姿态数据。",
            })
            continue

        # Average angles across phase
        avg_angles = {}
        for key in JOINT_NAMES:
            vals = [a[key] for a in phase_angles_list if key in a]
            if vals:
                avg_angles[key] = round(float(np.mean(vals)), 1)

        # Score against baselines
        phase_scores = score_angles(avg_angles, baselines, cfg)

        # Phase score = 1 - mean severity, scaled to 0-100
        severities = [s['severity'] for s in phase_scores.values()]
        phase_severity = float(np.mean(severities)) if severities else sc["empty_phase_severity"]
        phase_score = round((1 - phase_severity) * 100)

        # Generate commentary
        commentary = _generate_phase_commentary(pdef["name"], phase_scores, baselines)

        phases_result.append({
            "name": pdef["name"],
            "name_en": pdef["name_en"],
            "start_frame": start_f,
            "end_frame": end_f,
            "score": phase_score,
            "angles": avg_angles,
            "commentary": commentary,
        })

    # ── Overall score = weighted average ──
    weighted_sum = 0.0
    weight_total = 0.0
    for pdef, phase_r in zip(PHASE_DEFS, phases_result):
        weighted_sum += pdef["weight"] * phase_r["score"]
        weight_total += pdef["weight"]
    overall_score = round(weighted_sum / weight_total) if weight_total > 0 else sc["empty_phase_score"]

    # ── Build joint_scores for backward compatibility (use contact phase) ──
    contact_phase = phases_result[3] if len(phases_result) > 3 else phases_result[-1]
    joint_scores = score_angles(contact_phase.get("angles", {}), baselines, cfg)
    angle_comparison = _build_angle_comparison(contact_phase.get("angles", {}), baselines, cfg)

    return {
        "success": True,
        "overall_score": overall_score,
        "phases": phases_result,
        "phase_boundaries": phase_boundaries,
        "joint_scores": joint_scores,
        "detected_frames": detected,
        "total_frames": total_frames,
        "angle_comparison": angle_comparison,
    }


# ── Main pipeline ──────────────────────────────────────────────

def extract_and_analyze(
    video_path: str,
    baselines: dict,
    output_dir: str,
) -> dict:
    """
    Full pipeline: extract pose from video, detect serve phases,
    compute per-phase angles and scores, generate overlay video.

    Returns dict matching TASK_SPEC format:
        success, overall_score, phases[], overlay_video_url, total_frames
    """
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"success": False, "error": "Could not open video file."}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare overlay video
    os.makedirs(output_dir, exist_ok=True)
    uid = uuid.uuid4().hex[:12]
    overlay_filename = f"overlay_{uid}.mp4"
    overlay_path = os.path.join(output_dir, overlay_filename)

    all_frame_landmarks = []
    all_frames_bgr = []

    # ── Single MediaPipe pass ──
    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                lm_list = results.pose_landmarks.landmark
                all_frame_landmarks.append([{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in lm_list])
            else:
                all_frame_landmarks.append([])
            all_frames_bgr.append(frame.copy())

    cap.release()

    # ── Phase detection + scoring (pure computation) ──
    analysis = analyze_landmark_sequence(all_frame_landmarks, baselines, total_frames)
    if not analysis["success"]:
        return analysis
    phases_result = analysis.pop("phases")
    phase_boundaries = analysis.pop("phase_boundaries")

    # ── Generate reference skeleton ──
    reference_skeleton = _generate_reference_skeleton(baselines)
    # ── Generate overlay with phase coloring ──
    _redraw_overlay_phases(
        all_frame_landmarks, phases_result, phase_boundaries,
        overlay_path, fps, w, h, all_frames_bgr,
    )

    # Re-encode to H.264 for browser compatibility
    h264_path = overlay_path.replace('.mp4', '_h264.mp4')
    import subprocess
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', overlay_path,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            h264_path
        ], capture_output=True, timeout=120)
        if os.path.exists(h264_path):
            os.replace(h264_path, overlay_path)
    except Exception:
        pass  # keep original if ffmpeg fails

    return {
        "success": True,
        "overall_score": analysis["overall_score"],
        "phases": phases_result,
        "joint_scores": analysis["joint_scores"],  # backward compat
        "overlay_video_path": overlay_path,
        "overlay_filename": overlay_filename,
        "detected_frames": analysis["detected_frames"],
        "total_frames": total_frames,
        "reference_skeleton": reference_skeleton,
        "angle_comparison": analysis["angle_comparison"],
    }


def _redraw_overlay_phases(
    all_frame_landmarks: list,
    phases_result: list,
    phase_boundaries: list,
    overlay_path: str,
    fps: float,
    w: int,
    h: int,
    all_frames_bgr: list | None = None,
):
    """Draw skeleton overlay with phase-colored indicators."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(overlay_path, fourcc, fps, (w, h))

    # Build frame -> phase index mapping
    frame_to_phase = {}
    for pi, (start_f, end_f) in enumerate(phase_boundaries):
        for f in range(start_f, end_f):
            frame_to_phase[f] = pi

    for i, frame_lms in enumerate(all_frame_landmarks):
        if all_frames_bgr and i < len(all_frames_bgr):
            frame = all_frames_bgr[i].copy()
        else:
            frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Determine phase color
        pi = frame_to_phase.get(i, 0)
        phase_color = PHASE_COLORS_BGR[pi % len(PHASE_COLORS_BGR)]

        if frame_lms:
            # Draw connections with phase color tint
            for (i1, j1) in SKELETON_CONNECTIONS:
                if i1 < len(frame_lms) and j1 < len(frame_lms):
                    pt1 = (int(frame_lms[i1]['x'] * w), int(frame_lms[i1]['y'] * h))
                    pt2 = (int(frame_lms[j1]['x'] * w), int(frame_lms[j1]['y'] * h))
                    cv2.line(frame, pt1, pt2, (255, 255, 0), 2)

            # Draw joint dots
            for idx in range(len(frame_lms)):
                cx, cy = int(frame_lms[idx]['x'] * w), int(frame_lms[idx]['y'] * h)
                cv2.circle(frame, (cx, cy), 4, phase_color, -1)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 0), 1)

        # Draw phase label at top-left
        phase_name = PHASE_DEFS[pi]["name"] if pi < len(PHASE_DEFS) else ""
        cv2.putText(frame, phase_name, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, phase_color, 2)

        # Phase timeline bar at bottom
        bar_y = h - 20
        bar_h = 12
        total = len(all_frame_landmarks)
        for pj, (sf, ef) in enumerate(phase_boundaries):
            x1 = int(sf / total * w)
            x2 = int(ef / total * w)
            cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_h),
                          PHASE_COLORS_BGR[pj % len(PHASE_COLORS_BGR)], -1)
        # Current position marker
        cur_x = int(i / total * w) if total > 0 else 0
        cv2.line(frame, (cur_x, bar_y - 2), (cur_x, bar_y + bar_h + 2),
                 (255, 255, 255), 2)

        out.write(frame)

    out.release()

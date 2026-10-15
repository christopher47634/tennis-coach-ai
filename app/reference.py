"""Reference ("ideal serve") skeleton animation shown next to the user's result.

The animation is interpolated between five hand-authored key poses.  The
joint angles attached to each frame are computed from the interpolated
landmarks themselves (they are NOT overwritten with baseline means), so the
labels always agree with the pose that is drawn.
"""

import numpy as np

from angles import compute_frame_angles

PHASE_NAMES = ["准备期", "蓄力期", "加速期", "击球期", "随挥期"]

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


def generate_reference_skeleton() -> dict:
    """Generate a 120-frame ideal serve skeleton sequence.

    5 phases x 24 frames, sinusoidal easing between consecutive key poses.
    Returns dict with keys: frames, fps, total_frames, description.
    """
    total_frames = 120
    fps = 30
    frames_per_phase = total_frames // 5
    num_key_poses = len(_KEY_POSES)

    frames = []
    for i in range(total_frames):
        phase_idx = min(i // frames_per_phase, 4)
        t_in_phase = (i % frames_per_phase) / frames_per_phase
        pose_a_idx = phase_idx
        pose_b_idx = min(phase_idx + 1, num_key_poses - 1)
        if pose_a_idx == pose_b_idx:
            t = 0.0
        else:
            t = (1.0 - np.cos(np.pi * t_in_phase)) / 2.0

        lm = _lerp_pose(_KEY_POSES[pose_a_idx], _KEY_POSES[pose_b_idx], float(t))
        angles = compute_frame_angles(lm)
        frames.append({
            'landmarks': lm,
            'phase': PHASE_NAMES[phase_idx],
            'angles': {k: round(v, 1) for k, v in angles.items()},
        })

    return {
        'description': '示意性的理想发球骨架动画（关键姿势插值，角度由骨架本身计算）',
        'frames': frames,
        'fps': fps,
        'total_frames': total_frames,
    }

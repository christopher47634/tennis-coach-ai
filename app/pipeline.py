"""Serve analysis pipeline: pose extraction -> phase detection -> scoring -> overlay.

Module layout:
  angles.py     pixel-space joint-angle geometry
  phases.py     serve phase detection (contact point = highest wrist)
  reference.py  the ideal-serve reference skeleton animation
  pipeline.py   scoring, commentary, video I/O (this file)

The five phases are 准备期 / 蓄力期 / 加速期 / 击球期 / 随挥期.
"""

import json
import logging
import math
import os
import shutil
import subprocess
import uuid

import numpy as np

from angles import (  # noqa: F401  (re-exported for callers/tests)
    JOINT_NAMES, L_WRIST, R_WRIST, angle_between, compute_frame_angles, mirror_landmarks,
    wrist_height_rel, _lm_coords,
)
from phases import PhaseDetectionError, detect_phases
from reference import generate_reference_skeleton
from settings import CONFIG

# cv2 / mediapipe are imported lazily inside the video functions so that the
# pure scoring/phase code (and its tests) only needs numpy.

log = logging.getLogger("tennis.pipeline")

# Fallback single-pose baselines (used when data/expert_baselines.json is missing).
# trunk_lateral_tilt = shoulder-line tilt from horizontal; hip_shoulder_separation
# = angle between shoulder line and hip line (both in degrees, 0..90).
DEFAULT_BASELINES = {
    "right_elbow_flexion":      {"mean": 100, "std": 15},
    "left_elbow_flexion":       {"mean": 90,  "std": 15},
    "right_shoulder_abduction": {"mean": 95,  "std": 12},
    "left_shoulder_abduction":  {"mean": 85,  "std": 12},
    "right_knee_flexion":       {"mean": 140, "std": 15},
    "left_knee_flexion":        {"mean": 145, "std": 15},
    "trunk_lateral_tilt":       {"mean": 25,  "std": 10},
    "hip_shoulder_separation":  {"mean": 20,  "std": 12},
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


def _read_baseline_file(path):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def load_baselines(baselines_path: str | None = None) -> dict:
    """Load the single-pose expert baselines (joint -> {mean, std}), or defaults."""
    data = _read_baseline_file(baselines_path)
    if data is None:
        return DEFAULT_BASELINES
    if "serve" in data and isinstance(data["serve"], dict):
        return data["serve"]
    return {k: v for k, v in data.items() if isinstance(v, dict) and "mean" in v}


def load_phase_baselines(baselines_path: str | None = None) -> dict:
    """Load per-phase baselines (phase name -> joint -> {mean, std}); {} if absent."""
    data = _read_baseline_file(baselines_path) or {}
    phases = data.get("serve_phases")
    return phases if isinstance(phases, dict) else {}


def baselines_for_phase(phase_name: str, baselines: dict, phase_baselines: dict | None) -> dict:
    """Per-phase baselines, falling back joint-by-joint to the single-pose ones."""
    merged = dict(baselines)
    merged.update((phase_baselines or {}).get(phase_name, {}))
    return merged


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


# ── Phase commentary generation (Chinese) ──────────────────────

def _side(val: float, ref: float) -> str:
    """Direction word for a deviation from the reference angle."""
    return "偏大" if val > ref else "偏小"


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
            if "膝" in cn and val > ref:
                comments.append(f"{cn}不足({val:.0f}° vs 参考{ref:.0f}°)，膝关节弯曲不够会影响蹬地发力。")
            elif "膝" in cn:
                comments.append(f"{cn}过深({val:.0f}° vs 参考{ref:.0f}°)，下蹲过低会拖慢蹬伸衔接。")
            else:
                comments.append(f"{cn}{_side(val, ref)}需要改进({val:.0f}° vs 参考{ref:.0f}°)。")
        if not comments:
            comments.append("蓄力阶段动能储备良好，注意保持膝盖弯曲。")

    elif phase_name == "加速期":
        if good_angles:
            comments.append(f"加速阶段发力链流畅，{good_angles[0]}表现优秀。")
        if bad_angles:
            cn, val, ref = bad_angles[0]
            if "髋" in cn or "躯干" in cn:
                comments.append(f"{cn}{_side(val, ref)}({val:.0f}° vs 参考{ref:.0f}°)，影响力量从下肢向上传递。")
            else:
                comments.append(f"{cn}{_side(val, ref)}，偏离理想值({val:.0f}° vs 参考{ref:.0f}°)，影响加速效率。")
        if not comments:
            comments.append("加速阶段力量传导良好，髋肩分离到位。")

    elif phase_name == "击球期":
        if good_angles:
            comments.append(f"击球瞬间{good_angles[0]}到位，击球质量有保障。")
        if bad_angles:
            cn, val, ref = bad_angles[0]
            comments.append(f"击球时{cn}{_side(val, ref)}({val:.0f}° vs 参考{ref:.0f}°)，会导致击球不稳定。")
        if len(good_angles) >= 2:
            comments.append("多个关键角度在职业水平范围内，击球技术扎实。")
        if not comments:
            comments.append("击球阶段基本稳定，注意手腕固定。")

    elif phase_name == "随挥期":
        if good_angles:
            comments.append(f"随挥完整，{good_angles[0]}自然流畅。")
        if bad_angles:
            cn, val, ref = bad_angles[0]
            comments.append(f"随挥时{cn}{_side(val, ref)}，不够自然({val:.0f}° vs 参考{ref:.0f}°)，建议放松完成收拍动作。")
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
_LOW_CONFIDENCE_WARNING = "阶段划分置信度较低，评分仅供参考；建议上传机位固定、包含完整发球的视频。"


def _dominant_side_is_left(rel_right: np.ndarray, rel_left: np.ndarray) -> bool:
    """The hitting arm is the one whose wrist gets highest (smallest y)."""
    def top(a):
        v = a[~np.isnan(a)]
        return float(np.percentile(v, 2)) if len(v) else float('inf')
    return top(rel_left) < top(rel_right)


def analyze_landmark_sequence(
    frame_landmarks: list,
    baselines: dict,
    cfg: dict | None = None,
    frame_size: tuple = (1, 1),
    fps: float = 30.0,
    phase_baselines: dict | None = None,
) -> dict:
    """Phase detection + scoring on an already-extracted landmark sequence.

    ``frame_landmarks`` holds one entry per frame actually read from the
    video: a list of 33 landmarks (dicts with x/y/z, or objects with
    .x/.y/.z), or an empty list/None where no person was detected.
    ``frame_size`` is (width, height) in pixels, needed for correct angles.
    This is everything that happens after MediaPipe, so it can be tested
    without a model.
    """
    cfg = cfg or CONFIG
    phase_baselines = phase_baselines or {}
    total = len(frame_landmarks)
    if total == 0:
        return {"success": False, "error": _NO_PERSON_ERROR}

    usable = [lm if (lm and len(lm) > 28) else None for lm in frame_landmarks]
    detected = sum(1 for lm in usable if lm is not None)
    if detected / total < cfg["detection"]["min_detection_rate"]:
        return {"success": False, "error": _NO_PERSON_ERROR}

    # Hitting side: mirror a left-hander so every "right_*" joint is the hitting arm.
    rel_r = np.array([wrist_height_rel(lm, R_WRIST, frame_size) if lm else np.nan for lm in usable])
    rel_l = np.array([wrist_height_rel(lm, L_WRIST, frame_size) if lm else np.nan for lm in usable])
    left_handed = _dominant_side_is_left(rel_r, rel_l)
    if left_handed:
        usable = [mirror_landmarks(lm) if lm else None for lm in usable]
        rel_r = rel_l

    frame_angles = [compute_frame_angles(lm, frame_size) if lm else None for lm in usable]
    knee = np.array([a.get('right_knee_flexion', np.nan) if a else np.nan for a in frame_angles])

    try:
        det = detect_phases(rel_r, knee, fps, cfg)
    except PhaseDetectionError as e:
        return {"success": False, "error": str(e)}
    phase_boundaries = det["boundaries"]

    sc = cfg["scoring"]
    phases_result = []
    for pdef, (start_f, end_f) in zip(PHASE_DEFS, phase_boundaries):
        bl = baselines_for_phase(pdef["name"], baselines, phase_baselines)
        in_phase = [a for a in frame_angles[start_f:end_f] if a]
        base = {
            "name": pdef["name"], "name_en": pdef["name_en"],
            "start_frame": start_f, "end_frame": end_f,
        }
        if not in_phase:
            phases_result.append({
                **base, "score": sc["empty_phase_score"], "angles": {},
                "commentary": f"{pdef['name']}阶段未检测到足够姿态数据。",
            })
            continue

        avg_angles = {}
        for key in JOINT_NAMES:
            vals = [a[key] for a in in_phase if key in a]
            if vals:
                avg_angles[key] = round(float(np.mean(vals)), 1)

        phase_scores = score_angles(avg_angles, bl, cfg)
        severities = [s['severity'] for s in phase_scores.values()]
        phase_severity = float(np.mean(severities)) if severities else sc["empty_phase_severity"]
        phases_result.append({
            **base,
            "score": round((1 - phase_severity) * 100),
            "angles": avg_angles,
            "commentary": _generate_phase_commentary(pdef["name"], phase_scores, bl),
        })

    weighted_sum = sum(p["weight"] * r["score"] for p, r in zip(PHASE_DEFS, phases_result))
    weight_total = sum(p["weight"] for p in PHASE_DEFS)
    overall_score = round(weighted_sum / weight_total) if weight_total > 0 else sc["empty_phase_score"]

    # joint_scores / angle_comparison describe the contact phase.
    contact_phase = phases_result[3]
    contact_bl = baselines_for_phase(PHASE_DEFS[3]["name"], baselines, phase_baselines)
    joint_scores = score_angles(contact_phase.get("angles", {}), contact_bl, cfg)
    angle_comparison = _build_angle_comparison(contact_phase.get("angles", {}), contact_bl, cfg)

    warnings = list(det["notes"])
    if det["confidence"] < cfg["detection"]["low_confidence_below"]:
        warnings.append(_LOW_CONFIDENCE_WARNING)

    return {
        "success": True,
        "overall_score": overall_score,
        "phases": phases_result,
        "phase_boundaries": phase_boundaries,
        "joint_scores": joint_scores,
        "detected_frames": detected,
        "total_frames": total,
        "angle_comparison": angle_comparison,
        "confidence": det["confidence"],
        "contact_frame": det["contact_frame"],
        "handedness": "left" if left_handed else "right",
        "warnings": warnings,
    }


# ── Video I/O ──────────────────────────────────────────────────

def _transcode_h264(src: str) -> bool:
    """Re-encode the OpenCV mp4v output to H.264 so browsers can play it.

    Returns False (and logs why) on any failure; the temporary file is always
    removed.  On failure ``src`` is left as the unplayable-in-browsers mp4v.
    """
    exe = shutil.which('ffmpeg')
    if not exe:
        log.error("ffmpeg not found; overlay stays mp4v (most browsers cannot play it)")
        return False
    tmp = src[:-4] + '_h264.mp4'
    try:
        proc = subprocess.run(
            [exe, '-y', '-i', src, '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
             '-pix_fmt', 'yuv420p', '-movflags', '+faststart', tmp],
            capture_output=True, timeout=180,
        )
        if proc.returncode != 0:
            tail = proc.stderr.decode('utf-8', 'replace')[-500:]
            log.error("ffmpeg exited with %s: %s", proc.returncode, tail)
            return False
        os.replace(tmp, src)
        return True
    except subprocess.TimeoutExpired:
        log.error("ffmpeg timed out re-encoding %s", os.path.basename(src))
        return False
    except OSError as e:
        log.error("ffmpeg could not be run: %s", e)
        return False
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def extract_and_analyze(
    video_path: str,
    baselines: dict,
    output_dir: str,
    phase_baselines: dict | None = None,
    cfg: dict | None = None,
) -> dict:
    """Full pipeline: pose extraction, phase detection, scoring, overlay video.

    Memory stays flat: pass 1 keeps only the (tiny) landmark lists, pass 2
    re-reads the video and streams overlay frames straight to disk.
    """
    import cv2

    cfg = cfg or CONFIG
    lim = cfg["limits"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"success": False, "error": "Could not open video file."}

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not (1.0 <= fps <= 480.0) or math.isnan(fps):
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        cap.release()
        return {"success": False, "error": "Could not read video dimensions."}
    if max(w, h) > lim["max_dimension"]:
        cap.release()
        return {"success": False, "error": f"Video resolution too large (max {lim['max_dimension']}px)."}
    max_frames = int(lim["max_duration_seconds"] * fps)

    import mediapipe as mp

    landmarks_per_frame = []
    too_long = False
    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if len(landmarks_per_frame) >= max_frames:
                too_long = True
                break
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                landmarks_per_frame.append(
                    [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in results.pose_landmarks.landmark])
            else:
                landmarks_per_frame.append([])
    cap.release()

    if too_long:
        return {"success": False, "error": f"Video too long (max {lim['max_duration_seconds']} seconds)."}

    analysis = analyze_landmark_sequence(
        landmarks_per_frame, baselines, cfg, frame_size=(w, h), fps=fps, phase_baselines=phase_baselines)
    if not analysis["success"]:
        return analysis

    os.makedirs(output_dir, exist_ok=True)
    overlay_filename = f"overlay_{uuid.uuid4().hex}.mp4"
    overlay_path = os.path.join(output_dir, overlay_filename)
    try:
        _write_overlay(video_path, landmarks_per_frame, analysis["phase_boundaries"], overlay_path, fps, w, h)
        playable = _transcode_h264(overlay_path)
    except BaseException:
        if os.path.exists(overlay_path):
            os.unlink(overlay_path)
        raise
    warnings = analysis["warnings"]
    if not playable:
        warnings = warnings + ["叠加视频转码失败，浏览器可能无法播放；评分结果不受影响。"]

    return {
        "success": True,
        "overall_score": analysis["overall_score"],
        "phases": analysis["phases"],
        "joint_scores": analysis["joint_scores"],
        "overlay_video_path": overlay_path,
        "overlay_filename": overlay_filename,
        "overlay_playable": playable,
        "detected_frames": analysis["detected_frames"],
        "total_frames": analysis["total_frames"],
        "fps": fps,
        "reference_skeleton": generate_reference_skeleton(),
        "angle_comparison": analysis["angle_comparison"],
        "confidence": analysis["confidence"],
        "contact_frame": analysis["contact_frame"],
        "handedness": analysis["handedness"],
        "warnings": warnings,
    }


def _write_overlay(
    video_path: str,
    all_frame_landmarks: list,
    phase_boundaries: list,
    overlay_path: str,
    fps: float,
    w: int,
    h: int,
):
    """Stream the source video to ``overlay_path`` with skeleton + phase markers."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter(overlay_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError("could not open video writer")

    frame_to_phase = {}
    for pi, (start_f, end_f) in enumerate(phase_boundaries):
        for f in range(start_f, end_f):
            frame_to_phase[f] = pi

    total = len(all_frame_landmarks)
    try:
        for i, frame_lms in enumerate(all_frame_landmarks):
            ret, frame = cap.read()
            if not ret:
                break
            pi = frame_to_phase.get(i)
            phase_color = PHASE_COLORS_BGR[pi % len(PHASE_COLORS_BGR)] if pi is not None else (160, 160, 160)

            if frame_lms:
                for (i1, j1) in SKELETON_CONNECTIONS:
                    if i1 < len(frame_lms) and j1 < len(frame_lms):
                        pt1 = (int(frame_lms[i1]['x'] * w), int(frame_lms[i1]['y'] * h))
                        pt2 = (int(frame_lms[j1]['x'] * w), int(frame_lms[j1]['y'] * h))
                        cv2.line(frame, pt1, pt2, (255, 255, 0), 2)
                for lm in frame_lms:
                    cx, cy = int(lm['x'] * w), int(lm['y'] * h)
                    cv2.circle(frame, (cx, cy), 4, phase_color, -1)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 0), 1)

            if pi is not None:
                cv2.putText(frame, PHASE_DEFS[pi]["name"], (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, phase_color, 2)

            bar_y, bar_h = h - 20, 12
            for pj, (sf, ef) in enumerate(phase_boundaries):
                cv2.rectangle(frame, (int(sf / total * w), bar_y), (int(ef / total * w), bar_y + bar_h),
                              PHASE_COLORS_BGR[pj % len(PHASE_COLORS_BGR)], -1)
            cur_x = int(i / total * w)
            cv2.line(frame, (cur_x, bar_y - 2), (cur_x, bar_y + bar_h + 2), (255, 255, 255), 2)
            out.write(frame)
    finally:
        out.release()
        cap.release()

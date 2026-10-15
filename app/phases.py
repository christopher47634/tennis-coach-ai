"""Serve phase detection.

The signal is the hitting-wrist height relative to the shoulders, in torso
lengths (see ``angles.wrist_height_rel``): 0 at shoulder level, negative above
the shoulders, about +1 at hip level.  The *lowest* value of that curve (wrist
highest) is the **contact point**, not the trophy position.  Everything else is
found by walking backwards/forwards from it:

    准备期 [serve_start, loading_start)  wrist still near its resting height
    蓄力期 [loading_start, accel_start)  knee bend + toss + trophy, up to the racket drop
    加速期 [accel_start, contact-half)   racket drop -> swing up to the ball
    击球期 [contact-half, contact+half]  a narrow window around the contact frame
    随挥期 (contact+half, serve_end)     until the wrist returns to rest

The five phases cover only the serve itself ([serve_start, serve_end)), not the
whole clip, so empty lead-in / tail footage is not scored.  Missing frames are
NaN (never a fake value) and are interpolated for tracing; the result carries a
0..1 confidence, and an unusable clip raises ``PhaseDetectionError`` instead of
being force-fitted into five segments.
"""

import numpy as np

from settings import CONFIG


class PhaseDetectionError(Exception):
    """Raised when no credible serve can be located in the signal."""


def interpolate_nan(arr) -> np.ndarray:
    """Linear interpolation over NaN gaps; leading/trailing gaps hold the edge value."""
    a = np.asarray(arr, dtype=float).copy()
    valid = ~np.isnan(a)
    if not valid.any() or valid.all():
        return a
    idx = np.arange(len(a))
    a[~valid] = np.interp(idx[~valid], idx[valid], a[valid])
    return a


def smooth(arr, window: int) -> np.ndarray:
    """Moving average that is not pulled toward 0 at the edges.

    The kernel is renormalised by how many samples actually fall inside the
    window at each position, so a constant signal stays constant.
    """
    a = np.asarray(arr, dtype=float)
    n = len(a)
    window = int(max(1, min(window, n)))
    if window <= 1 or n == 0:
        return a.copy()
    kernel = np.ones(window)
    num = np.convolve(a, kernel, mode='same')
    den = np.convolve(np.ones(n), kernel, mode='same')
    return num / den


def _enforce_order(bounds: list, min_width: int, lo: int, hi: int) -> list:
    """Make bounds strictly increasing with at least ``min_width`` between them."""
    b = list(bounds)
    b[0] = max(b[0], lo)
    b[-1] = min(b[-1], hi)
    for i in range(len(b) - 2, 0, -1):          # right -> left
        b[i] = min(b[i], b[i + 1] - min_width)
    for i in range(1, len(b) - 1):              # left -> right
        b[i] = max(b[i], b[i - 1] + min_width)
    return b


def detect_phases(wrist_rel, knee_angles, fps: float = 30.0, cfg: dict | None = None) -> dict:
    """Locate the five serve phases.

    ``wrist_rel`` and ``knee_angles`` hold one value per video frame, NaN where
    the player was not detected.  Returns::

        {"boundaries": [(start, end) x 5],   # frame indices, end exclusive
         "contact_frame": int, "serve_start": int, "serve_end": int,
         "confidence": float, "notes": [str]}

    Raises ``PhaseDetectionError`` when there is no usable serve.
    """
    dc = (cfg or CONFIG)["detection"]
    fps = fps if fps and fps > 0 else 30.0
    rel_raw = np.asarray(wrist_rel, dtype=float)
    n = len(rel_raw)
    valid = ~np.isnan(rel_raw)
    n_valid = int(valid.sum())
    if n_valid < dc["min_valid_frames"]:
        raise PhaseDetectionError("可用的姿态帧太少，无法定位发球动作。")

    first_valid = int(np.argmax(valid))
    last_valid = int(n - 1 - np.argmax(valid[::-1]))
    span = last_valid - first_valid + 1
    mw = dc["min_phase_frames"]
    notes = []

    rel = smooth(interpolate_nan(rel_raw), max(dc["smooth_min_window"], int(round(fps * dc["smooth_seconds"]))))

    # ── contact = highest wrist (minimum y) ──
    seg = rel[first_valid:last_valid + 1]
    contact = first_valid + int(np.argmin(seg))
    contact_level = float(rel[contact])

    edge = max(3, span // 10)
    rest_before = float(np.median(rel[first_valid:first_valid + edge]))
    rest_after = float(np.median(rel[last_valid - edge + 1:last_valid + 1]))
    prominence = max(rest_before, rest_after) - contact_level
    if prominence < dc["min_prominence"]:
        raise PhaseDetectionError("没有检测到明显的发球挥拍（手腕上举幅度不足），请上传包含完整发球动作的视频。")

    # ── walk back from contact to where the wrist left its resting height ──
    start_clipped = True
    serve_start = first_valid
    rise = rest_before - contact_level
    if rise >= dc["min_prominence"]:
        level = contact_level + dc["rest_fraction"] * rise
        for i in range(contact, first_valid - 1, -1):
            if rel[i] >= level:
                serve_start = i
                start_clipped = False
                break
    else:
        notes.append("视频开头手腕已处于高位，起势可能被截断。")

    # ── walk forward to where it comes back to rest ──
    end_clipped = True
    serve_end = last_valid + 1
    fall = rest_after - contact_level
    if fall >= dc["min_prominence"]:
        level = contact_level + dc["rest_fraction"] * fall
        for i in range(contact, last_valid + 1):
            if rel[i] >= level:
                serve_end = min(last_valid + 1, i + 1)
                end_clipped = False
                break
    else:
        notes.append("视频结尾手腕仍处于高位，随挥可能被截断。")

    rise_len = max(contact - serve_start, 1)

    # ── racket drop (accel start): lowest wrist point between trophy and contact ──
    lo = serve_start + int(0.3 * rise_len)
    hi = contact - max(1, int(0.1 * rise_len))
    accel_start = None
    if hi - lo >= 2:
        window = rel[lo:hi + 1]
        j = int(np.argmax(window))
        if 0 < j < len(window) - 1 and window[j] - min(window[0], window[-1]) > 0.02:
            accel_start = lo + j
    if accel_start is None:
        accel_start = serve_start + int(round(dc["accel_fallback_fraction"] * rise_len))
        notes.append("没有检出明显的引拍下落，加速期起点为估计值。")

    # ── knee bend -> loading start ──
    loading_start = serve_start + int(round(dc["loading_fallback_fraction"] * max(accel_start - serve_start, 1)))
    knee = np.asarray(knee_angles, dtype=float) if knee_angles is not None else np.full(n, np.nan)
    if len(knee) == n and (~np.isnan(knee[serve_start:accel_start + 1])).sum() > 5:
        k = smooth(interpolate_nan(knee), max(3, int(round(fps * 0.1))))[serve_start:accel_start + 1]
        drop = float(k[0] - np.min(k))
        if drop > dc["knee_min_drop"]:
            thr = k[0] - drop * dc["knee_loading_fraction"]
            below = np.nonzero(k < thr)[0]
            if len(below):
                loading_start = serve_start + int(below[0])

    half = max(dc["contact_half_min"], int(round(dc["contact_half_fraction"] * rise_len)))
    contact_lo = contact - half
    contact_hi = contact + half + 1

    # Not enough footage after contact: pull the contact window back, don't invent frames.
    bounds = [serve_start, loading_start, accel_start, contact_lo, contact_hi, serve_end]
    if serve_end - serve_start < 5 * mw:
        raise PhaseDetectionError("检测到的发球片段过短，无法划分五个阶段。")
    bounds = _enforce_order(bounds, mw, serve_start, serve_end)
    boundaries = [(int(bounds[i]), int(bounds[i + 1])) for i in range(5)]

    # ── confidence ──
    det_rate = n_valid / n
    c_det = min(1.0, det_rate / 0.8)
    c_prom = min(1.0, prominence / 1.0)
    confidence = (0.35 * c_prom + 0.25 * c_det
                  + 0.2 * (0.5 if start_clipped else 1.0) + 0.2 * (0.5 if end_clipped else 1.0))
    if contact - first_valid < 3 or last_valid - contact < 3:
        confidence *= 0.6
        notes.append("击球点贴近视频首尾，结果可信度较低。")

    return {
        "boundaries": boundaries,
        "contact_frame": int(contact),
        "serve_start": int(serve_start),
        "serve_end": int(serve_end),
        "confidence": round(float(min(1.0, confidence)), 2),
        "notes": notes,
    }

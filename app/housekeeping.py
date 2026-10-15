"""Output-directory housekeeping: overlay videos contain user footage, so they expire."""

import logging
import os
import re
import time

log = logging.getLogger("tennis.housekeeping")

# overlay_<32 hex>.mp4, plus the transient ffmpeg file overlay_<hex>_h264.mp4
OUTPUT_NAME_RE = re.compile(r"^overlay_[0-9a-f]{32}(_h264)?\.mp4$")


def cleanup_outputs(output_dir: str, ttl_seconds: float, now: float | None = None) -> int:
    """Delete generated overlay files older than ``ttl_seconds``; returns the count."""
    now = time.time() if now is None else now
    removed = 0
    try:
        names = os.listdir(output_dir)
    except OSError:
        return 0
    for name in names:
        if not OUTPUT_NAME_RE.match(name):
            continue
        path = os.path.join(output_dir, name)
        try:
            if now - os.path.getmtime(path) > ttl_seconds:
                os.unlink(path)
                removed += 1
        except OSError as e:
            log.warning("could not remove %s: %s", name, e)
    return removed

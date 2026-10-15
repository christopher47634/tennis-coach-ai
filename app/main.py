"""FastAPI backend for tennis video analysis.

Endpoints:
  GET  /                    -> serves static frontend files
  GET  /api/health          -> liveness probe
  POST /api/analyze         -> upload video, returns analysis JSON
  GET  /outputs/{filename}  -> serves generated overlay videos (expire after a TTL)

All error responses use FastAPI's ``{"detail": "..."}`` shape.
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Add parent to path so imports work regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from housekeeping import OUTPUT_NAME_RE, cleanup_outputs
from pipeline import extract_and_analyze, load_baselines, load_phase_baselines
from settings import CONFIG

log = logging.getLogger("tennis.api")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.environ.get("TENNIS_OUTPUT_DIR") or os.path.join(DATA_DIR, "outputs")
BASELINES_PATH = os.path.join(DATA_DIR, "expert_baselines.json")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

LIMITS = CONFIG["limits"]
MAX_UPLOAD_BYTES = int(LIMITS["max_upload_mb"] * 1024 * 1024)
ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
CHUNK = 1024 * 1024
MULTIPART_OVERHEAD = 64 * 1024

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

BASELINES = load_baselines(BASELINES_PATH)
PHASE_BASELINES = load_phase_baselines(BASELINES_PATH)

# Bounds concurrent analyses (each is CPU/RAM heavy).
_analysis_slots = asyncio.Semaphore(int(LIMITS["max_concurrent_analyses"]))


class _UploadTooLarge(Exception):
    pass


def _too_large_body() -> bytes:
    return json.dumps({"detail": f"Video too large. Max {LIMITS['max_upload_mb']}MB."}).encode()


class UploadSizeLimitMiddleware:
    """Reject oversized uploads while the body is still arriving.

    Checks Content-Length up front and counts bytes as they stream in (for
    chunked requests that carry no length), so an over-limit body is cut off
    instead of being buffered in full before a size check.  The framework may
    turn the interrupted read into its own 400, so once the limit has been hit
    whatever response the app produces is replaced with a 413.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/api/analyze" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return

        limit = MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD
        declared = dict(scope["headers"]).get(b"content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            await self._respond(send)
            return

        received = 0
        exceeded = False
        responded = False

        async def counting_receive():
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True
                    raise _UploadTooLarge()
            return message

        async def replacing_send(message):
            nonlocal responded
            if not exceeded:
                await send(message)
            elif not responded:
                responded = True
                await self._respond(send)

        try:
            await self.app(scope, counting_receive, replacing_send)
        except _UploadTooLarge:
            pass
        if exceeded and not responded:
            await self._respond(send)

    @staticmethod
    async def _respond(send):
        body = _too_large_body()
        await send({
            "type": "http.response.start", "status": 413,
            "headers": [(b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()), (b"connection", b"close")],
        })
        await send({"type": "http.response.body", "body": body})


@asynccontextmanager
async def lifespan(_app):
    cleanup_outputs(OUTPUT_DIR, LIMITS["output_ttl_seconds"])
    yield


app = FastAPI(
    title="Tennis Coach AI",
    description="Analyze tennis serve videos with pose estimation and rule-based feedback.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(UploadSizeLimitMiddleware)


# ── Health check ──────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok"}


def _safe_suffix(filename: str) -> str | None:
    """Whitelisted lowercase extension of the client filename, else None.

    Only the last path component counts (the client may send a Windows or a
    POSIX path), and the result is never used to build a path, only to pick a
    value out of a fixed set.
    """
    base = filename.replace("\\", "/").rsplit("/", 1)[-1]
    _, dot, ext = base.rpartition(".")
    suffix = f".{ext.lower()}" if dot else ""
    return suffix if suffix in ALLOWED_SUFFIXES else None


# ── Video analysis endpoint ──────────────────────────────────
@app.post("/api/analyze")
async def analyze_video(video: UploadFile = File(...)):
    """Accept a video upload, run pose analysis, return scores + overlay URL."""
    if not video.filename:
        raise HTTPException(status_code=400, detail="No video file provided.")
    suffix = _safe_suffix(video.filename)
    if suffix is None:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Allowed: " + ", ".join(sorted(ALLOWED_SUFFIXES)) + ".",
        )

    # Stream the upload to disk (MediaPipe/OpenCV need a path), enforcing the limit while writing.
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="upload_")
    try:
        size = 0
        with os.fdopen(fd, "wb") as tmp:
            while True:
                chunk = await video.read(CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413, detail=f"Video too large. Max {LIMITS['max_upload_mb']}MB.")
                tmp.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="Empty video file.")

        try:
            await asyncio.wait_for(_analysis_slots.acquire(), timeout=LIMITS["queue_timeout_seconds"])
        except asyncio.TimeoutError:
            raise HTTPException(status_code=503, detail="Server is busy analysing other videos, try again shortly.")
        try:
            cleanup_outputs(OUTPUT_DIR, LIMITS["output_ttl_seconds"])
            result = await run_in_threadpool(
                extract_and_analyze,
                video_path=tmp_path,
                baselines=BASELINES,
                output_dir=OUTPUT_DIR,
                phase_baselines=PHASE_BASELINES,
            )
        finally:
            _analysis_slots.release()

        if not result["success"]:
            raise HTTPException(status_code=422, detail=result["error"])

        return {
            "overall_score": result["overall_score"],
            "phases": result.get("phases", []),
            "total_frames": result.get("total_frames", 0),
            "fps": result.get("fps", 30),
            "overlay_video_url": f"/outputs/{result['overlay_filename']}",
            "overlay_playable": result.get("overlay_playable", True),
            "reference_skeleton": result.get("reference_skeleton", {}),
            "angle_comparison": result.get("angle_comparison", {}),
            "confidence": result.get("confidence"),
            "handedness": result.get("handedness"),
            "warnings": result.get("warnings", []),
        }
    except HTTPException:
        raise
    except Exception:
        log.exception("analysis failed")
        raise HTTPException(status_code=500, detail="Analysis failed. Please try another video.")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Overlay video serving ────────────────────────────────────
@app.get("/outputs/{filename}")
async def serve_output(filename: str):
    """Serve generated overlay videos from the outputs directory."""
    if not OUTPUT_NAME_RE.match(filename) or filename.endswith("_h264.mp4"):
        raise HTTPException(status_code=404, detail="File not found.")
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(filepath, media_type="video/mp4")


# ── Static files (frontend) ─────────────────────────────────
# Mount LAST so API routes take priority
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

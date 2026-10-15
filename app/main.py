"""FastAPI backend for tennis video analysis.

Endpoints:
  GET  /                    → serves static frontend files
  POST /api/analyze         → upload video, returns analysis JSON
  GET  /outputs/{filename}  → serves generated overlay videos
"""

import os
import sys
import tempfile
import traceback

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Add parent to path so imports work regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import load_baselines, extract_and_analyze
from coach import generate_advice  # noqa: F401

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
BASELINES_PATH = os.path.join(DATA_DIR, "expert_baselines.json")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Load baselines once at startup
BASELINES = load_baselines(BASELINES_PATH)

app = FastAPI(
    title="Tennis Coach AI",
    description="Analyze tennis serve videos with pose estimation and get coaching advice.",
    version="1.0.0",
)


# ── Health check ──────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ── Video analysis endpoint ──────────────────────────────────
@app.post("/api/analyze")
async def analyze_video(video: UploadFile = File(...)):
    """Accept a video file upload, run pose analysis, return scores + coaching advice."""
    # Validate file
    if not video.filename:
        raise HTTPException(status_code=400, detail="No video file provided.")

    # Check file size (max 200MB)
    MAX_SIZE = 200 * 1024 * 1024  # 200MB
    content = await video.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="Video too large. Max 200MB.")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty video file.")

    # Save upload to a temp file (MediaPipe needs a file path, not a stream)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=_suffix(video.filename))
    try:
        tmp.write(content)
        tmp.close()

        # Run the analysis pipeline
        result = extract_and_analyze(
            video_path=tmp.name,
            baselines=BASELINES,
            output_dir=OUTPUT_DIR,
        )

        if not result["success"]:
            return JSONResponse(
                status_code=422,
                content={"error": result["error"]},
            )

        return {
            "overall_score": result["overall_score"],
            "phases": result.get("phases", []),
            "total_frames": result.get("total_frames", 0),
            "overlay_video_url": f"/outputs/{result['overlay_filename']}",
            "reference_skeleton": result.get("reference_skeleton", {}),
            "angle_comparison": result.get("angle_comparison", {}),
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ── Overlay video serving ────────────────────────────────────
@app.get("/outputs/{filename}")
async def serve_output(filename: str):
    """Serve generated overlay videos from data/outputs/."""
    # Sanitize filename to prevent path traversal
    safe = os.path.basename(filename)
    filepath = os.path.join(OUTPUT_DIR, safe)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(filepath, media_type="video/mp4")


# ── Static files (frontend) ─────────────────────────────────
# Mount LAST so API routes take priority
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def _suffix(filename: str) -> str:
    """Extract file extension."""
    if '.' in filename:
        return '.' + filename.rsplit('.', 1)[-1].lower()
    return '.mp4'

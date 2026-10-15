FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgl1 + libglib2.0-0: runtime libs for the full OpenCV build that mediapipe pulls in;
# ffmpeg: re-encodes the overlay to H.264 for browsers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# model_complexity=2 downloads its weights on first use, which would fail for a
# non-root user (site-packages is read-only to them): fetch it at build time.
RUN python -c "import mediapipe as mp; mp.solutions.pose.Pose(model_complexity=2).close()"

COPY app/ ./app/
COPY data/expert_baselines.json ./data/expert_baselines.json
COPY config/ ./config/

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p data/outputs \
    && chown -R appuser:appuser data
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).status == 200 else 1)"

# One worker on purpose: each analysis is memory-heavy and concurrency is bounded
# inside the app (limits.max_concurrent_analyses).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

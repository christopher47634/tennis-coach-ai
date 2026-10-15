"""HTTP-level tests for /api/analyze (report section E, test 3).

The pose pipeline is replaced by a stub (MediaPipe is not needed to test the
upload handling), except for one test that uses the real OpenCV when present.
"""

import os
import time

import pytest
from fastapi.testclient import TestClient

import housekeeping
import main

client = TestClient(main.app)
MB = 1024 * 1024


def _ok_result(**kw):
    base = {
        "success": True, "overall_score": 71, "phases": [], "total_frames": 90, "fps": 30.0,
        "overlay_filename": "overlay_" + "a" * 32 + ".mp4", "overlay_playable": True,
        "reference_skeleton": {}, "angle_comparison": {}, "confidence": 0.9, "handedness": "right",
        "warnings": [],
    }
    base.update(kw)
    return base


@pytest.fixture
def stub(monkeypatch):
    """Replace the pipeline; records the temp file path it was handed."""
    seen = {}

    def fake(video_path, **kwargs):
        seen["path"] = video_path
        seen["existed"] = os.path.exists(video_path)
        seen["size"] = os.path.getsize(video_path)
        seen["suffix"] = os.path.splitext(video_path)[1]
        return seen.get("result") or _ok_result()

    monkeypatch.setattr(main, "extract_and_analyze", fake)
    return seen


def _post(name="serve.mp4", data=b"\x00\x01video-bytes", ctype="video/mp4"):
    return client.post("/api/analyze", files={"video": (name, data, ctype)})


def test_empty_file_is_400(stub):
    r = _post(data=b"")
    assert r.status_code == 400
    assert "path" not in stub                      # never reached the pipeline


def test_201mb_is_413_and_not_analysed(stub):
    r = _post(data=b"\x00" * (201 * MB))
    assert r.status_code == 413
    assert "detail" in r.json()
    assert "path" not in stub


def test_exactly_at_limit_is_accepted(stub):
    r = _post(data=b"\x00" * (200 * MB))
    assert r.status_code == 200
    assert stub["size"] == 200 * MB


def test_non_video_bytes_give_422_with_detail(monkeypatch):
    """Garbage bytes: the pipeline reports 'cannot open', the API turns it into 422."""
    monkeypatch.setattr(
        main, "extract_and_analyze", lambda **kw: {"success": False, "error": "Could not open video file."})
    r = _post(data=b"this is definitely not a video" * 100)
    assert r.status_code == 422
    assert r.json() == {"detail": "Could not open video file."}


def test_non_video_bytes_422_with_real_opencv():
    pytest.importorskip("cv2")
    r = _post(data=b"this is definitely not a video" * 100)
    assert r.status_code == 422


@pytest.mark.parametrize("name", ["a./x", "a.", ".", "..", "x/..", "a/b/c", "noext", "evil.exe", "v.mp4/",
                                  "a\\..\\b", "x" * 400])
def test_odd_filenames_never_500(stub, name):
    r = _post(name=name)
    assert r.status_code in (200, 400), (name, r.status_code, r.text)


def test_extension_whitelist(stub):
    assert _post(name="clip.MOV").status_code == 200
    assert stub["suffix"] == ".mov"
    assert _post(name="C:\\Users\\me\\clip.mp4").status_code == 200
    r = _post(name="script.py")
    assert r.status_code == 400 and "Unsupported" in r.json()["detail"]


def test_temp_upload_is_removed_after_success_and_failure(stub, monkeypatch):
    assert _post().status_code == 200
    assert stub["existed"] and not os.path.exists(stub["path"])

    def boom(**kw):
        stub["path"] = kw["video_path"]
        raise RuntimeError("secret C:\\internal\\path")
    monkeypatch.setattr(main, "extract_and_analyze", boom)
    r = _post()
    assert r.status_code == 500
    assert not os.path.exists(stub["path"])


def test_500_does_not_echo_exception_text(monkeypatch):
    def boom(**kw):
        raise RuntimeError("secret C:\\internal\\path token=abc123")
    monkeypatch.setattr(main, "extract_and_analyze", boom)
    r = _post()
    assert r.status_code == 500
    assert "secret" not in r.text and "abc123" not in r.text


def test_success_response_shape(stub):
    r = _post()
    body = r.json()
    assert r.status_code == 200
    assert body["overall_score"] == 71
    assert body["overlay_video_url"] == "/outputs/overlay_" + "a" * 32 + ".mp4"
    assert body["confidence"] == 0.9 and body["warnings"] == []


def test_analysis_runs_off_the_event_loop(monkeypatch):
    """Health stays responsive while an analysis is running in the thread pool."""
    import threading
    started, release = threading.Event(), threading.Event()

    def slow(**kw):
        started.set()
        release.wait(5)
        return _ok_result()
    monkeypatch.setattr(main, "extract_and_analyze", slow)

    results = {}
    t = threading.Thread(target=lambda: results.update(r=_post()))
    t.start()
    assert started.wait(5)
    t0 = time.time()
    assert client.get("/api/health").status_code == 200
    assert time.time() - t0 < 2
    release.set()
    t.join(10)
    assert results["r"].status_code == 200


def test_busy_server_returns_503(stub, monkeypatch):
    import asyncio
    monkeypatch.setattr(main, "_analysis_slots", asyncio.Semaphore(0))
    monkeypatch.setitem(main.LIMITS, "queue_timeout_seconds", 0.05)
    r = _post()
    assert r.status_code == 503
    assert "detail" in r.json()


def test_chunked_upload_is_cut_off_while_streaming(stub, monkeypatch):
    """No Content-Length: the middleware counts bytes as they arrive and stops at the limit."""
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1 * MB)

    def gen():
        yield (b"--b\r\nContent-Disposition: form-data; name=\"video\"; filename=\"v.mp4\"\r\n"
               b"Content-Type: video/mp4\r\n\r\n")
        for _ in range(40):                      # 40 MB offered, limit is 1 MB (+64 KB slack)
            yield b"\x00" * MB
        yield b"\r\n--b--\r\n"

    r = client.post("/api/analyze", content=gen(), headers={"content-type": "multipart/form-data; boundary=b"})
    assert r.status_code == 413
    assert "path" not in stub


# ── outputs ──

def test_outputs_rejects_traversal_and_unknown_names(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "OUTPUT_DIR", str(tmp_path))
    (tmp_path / "secret.txt").write_text("x")
    name = "overlay_" + "b" * 32 + ".mp4"
    (tmp_path / name).write_bytes(b"mp4")
    assert client.get(f"/outputs/{name}").status_code == 200
    assert client.get("/outputs/secret.txt").status_code == 404
    assert client.get("/outputs/..%2Fsecret.txt").status_code == 404
    assert client.get("/outputs/overlay_" + "c" * 32 + ".mp4").status_code == 404


def test_cleanup_outputs_removes_only_expired_overlays(tmp_path):
    old = tmp_path / ("overlay_" + "a" * 32 + ".mp4")
    fresh = tmp_path / ("overlay_" + "b" * 32 + ".mp4")
    stale_tmp = tmp_path / ("overlay_" + "c" * 32 + "_h264.mp4")
    other = tmp_path / "keep.txt"
    for f in (old, fresh, stale_tmp, other):
        f.write_bytes(b"x")
    now = time.time()
    for f in (old, stale_tmp, other):
        os.utime(f, (now - 7200, now - 7200))
    assert housekeeping.cleanup_outputs(str(tmp_path), 3600, now) == 2
    assert not old.exists() and not stale_tmp.exists()
    assert fresh.exists() and other.exists()


def test_frontend_reads_detail_field():
    js = open(os.path.join(main.STATIC_DIR, "app.js"), encoding="utf-8").read()
    assert "data.detail" in js

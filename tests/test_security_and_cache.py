"""Unit tests for security rate-limit eviction and output cache headers."""
import main as api_main
import security
from fastapi.testclient import TestClient


def test_rate_limit_evicts_oldest_under_active_flood(monkeypatch):
    security.reset_rate_limit()
    monkeypatch.setattr(security, "_MAX_TRACKED_IPS", 3)
    monkeypatch.setenv("RATE_LIMIT_MAX", "5")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "600")

    class _Client:
        def __init__(self, host):
            self.host = host

    class _Req:
        def __init__(self, host):
            self.client = _Client(host)

    for i in range(6):
        security.rate_limit(_Req(f"10.0.0.{i}"))

    assert len(security._hits) <= 3
    security.reset_rate_limit()


def test_serve_output_sets_private_no_store_cache_header(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "OUTPUT_DIR", str(tmp_path))
    fname = "overlay_1234567890abcdef1234567890abcdef.mp4"
    (tmp_path / fname).write_bytes(b"fake-mp4")
    client = TestClient(api_main.app)
    resp = client.get(f"/outputs/{fname}")
    assert resp.status_code == 200
    assert resp.headers.get("cache-control") == "private, no-store"

"""Optional bearer-token auth and a per-IP sliding-window rate limit for /api/analyze.

Both are FastAPI dependencies and read their settings from the environment on
every request, so they can be changed (and tested) without re-importing.

  API_TOKEN                  if set, requests need ``Authorization: Bearer <token>``
  RATE_LIMIT_MAX             requests allowed per window per client IP (default 5; 0 disables)
  RATE_LIMIT_WINDOW_SECONDS  window length in seconds (default 600)

The client IP is the direct peer address.  Behind a reverse proxy, start uvicorn
with ``--proxy-headers --forwarded-allow-ips=<proxy ip>`` so it reflects
X-Forwarded-For; the header is never trusted here directly.
"""

import hmac
import logging
import math
import os
import threading
import time
from collections import deque

from fastapi import HTTPException, Request

log = logging.getLogger("tennis.security")

DEFAULT_RATE_MAX = 5
DEFAULT_RATE_WINDOW = 600.0
_MAX_TRACKED_IPS = 10000

_hits: dict[str, deque] = {}
_lock = threading.Lock()


def _env_number(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value >= 0 else default


def reset_rate_limit() -> None:
    with _lock:
        _hits.clear()


def log_startup_config() -> None:
    if os.environ.get("API_TOKEN"):
        log.info("API_TOKEN is set: /api/analyze requires a Bearer token.")
    else:
        log.warning("API_TOKEN is not set: /api/analyze is open to anyone who can reach it.")
    limit = int(_env_number("RATE_LIMIT_MAX", DEFAULT_RATE_MAX))
    if limit:
        window = _env_number("RATE_LIMIT_WINDOW_SECONDS", DEFAULT_RATE_WINDOW)
        log.info("Rate limit: %d analyses per client IP per %.0fs.", limit, window)
    else:
        log.warning("Rate limit disabled (RATE_LIMIT_MAX=0).")


def require_token(request: Request) -> None:
    token = os.environ.get("API_TOKEN")
    if not token:
        return
    scheme, _, supplied = request.headers.get("authorization", "").partition(" ")
    ok = scheme.lower() == "bearer" and hmac.compare_digest(
        supplied.strip().encode(), token.encode())
    if not ok:
        raise HTTPException(
            status_code=401, detail="Missing or invalid API token.",
            headers={"WWW-Authenticate": "Bearer"})


def rate_limit(request: Request) -> None:
    limit = int(_env_number("RATE_LIMIT_MAX", DEFAULT_RATE_MAX))
    if limit <= 0:
        return
    window = _env_number("RATE_LIMIT_WINDOW_SECONDS", DEFAULT_RATE_WINDOW)
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _lock:
        if len(_hits) > _MAX_TRACKED_IPS:      # drop idle entries so the table cannot grow forever
            for key in [k for k, q in _hits.items() if not q or now - q[-1] >= window]:
                del _hits[key]
        q = _hits.setdefault(ip, deque())
        while q and now - q[0] >= window:
            q.popleft()
        if len(q) >= limit:
            retry = max(1, math.ceil(window - (now - q[0])))
            raise HTTPException(
                status_code=429, detail="Too many analysis requests, try again later.",
                headers={"Retry-After": str(retry)})
        q.append(now)

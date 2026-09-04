import os
import threading
import time
import uuid
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

API_KEY = os.environ.get("LEXRAG_API_KEY", "").strip()

OPEN_PATH_PREFIXES = ("/health", "/ui", "/marketing", "/openapi.json", "/docs", "/redoc")

RATE_LIMITS = [
    ("/api/chat", 60, 60.0),
    ("/api/ingest", 30, 60.0),
    ("/api/", 300, 60.0),
]

_hits: dict = defaultdict(deque)
_hits_lock = threading.Lock()


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _limit_for(path: str):
    for prefix, max_hits, window in RATE_LIMITS:
        if path.startswith(prefix):
            return max_hits, window
    return None


def check_rate_limit(request: Request) -> bool:
    rule = _limit_for(request.url.path)
    if not rule:
        return True
    max_hits, window = rule
    now = time.monotonic()
    group = "/" + "/".join(request.url.path.strip("/").split("/")[:2])
    key = (_client_ip(request), group)
    with _hits_lock:
        q = _hits[key]
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= max_hits:
            return False
        q.append(now)
    return True


def is_authorized(request: Request) -> bool:
    if not API_KEY:
        return True
    if request.headers.get("x-api-key", "") == API_KEY:
        return True
    if request.query_params.get("api_key", "") == API_KEY:
        return True
    return False


class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        path = request.url.path

        if request.method == "OPTIONS" or path == "/" or path.startswith(OPEN_PATH_PREFIXES):
            resp = await call_next(request)
            _secure_headers(resp, request_id)
            return resp

        if path.startswith("/api/"):
            if not check_rate_limit(request):
                resp = JSONResponse({"detail": "Rate limit exceeded. Slow down."}, status_code=429)
                _secure_headers(resp, request_id)
                return resp
            if not is_authorized(request):
                resp = JSONResponse({"detail": "Missing or invalid API key."}, status_code=401)
                _secure_headers(resp, request_id)
                return resp

        resp = await call_next(request)
        _secure_headers(resp, request_id)
        return resp


def _secure_headers(resp, request_id: str):
    resp.headers["X-Request-ID"] = request_id
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "same-origin"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; base-uri 'self'"
    )

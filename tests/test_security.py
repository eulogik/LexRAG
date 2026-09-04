"""Tests for API-key auth and rate limiting (no heavy deps)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from starlette.requests import Request

import api.security as sec
from api.security import SecurityMiddleware, check_rate_limit, is_authorized


def _req(path="/api/chat", ip="10.9.9.1", headers=None):
    scope = {
        "type": "http", "http_version": "1.1", "method": "GET", "scheme": "http",
        "path": path, "query_string": b"", "headers": headers or [],
        "server": ("test", 80), "client": (ip, 1234),
    }
    return Request(scope)


async def _ok_response(request):
    from starlette.responses import JSONResponse
    return JSONResponse({"ok": True})


def test_open_mode_allows_without_key(monkeypatch):
    monkeypatch.setattr(sec, "API_KEY", "")
    assert is_authorized(_req()) is True


def test_auth_rejects_missing_key(monkeypatch):
    import asyncio
    monkeypatch.setattr(sec, "API_KEY", "secret")
    resp = asyncio.run(SecurityMiddleware(None).dispatch(_req(ip="10.9.9.2"), _ok_response))
    assert resp.status_code == 401


def test_auth_accepts_header_key(monkeypatch):
    import asyncio
    monkeypatch.setattr(sec, "API_KEY", "secret")
    req = _req(ip="10.9.9.3", headers=[(b"x-api-key", b"secret")])
    resp = asyncio.run(SecurityMiddleware(None).dispatch(req, _ok_response))
    assert resp.status_code == 200
    assert resp.headers["X-Request-ID"]


def test_auth_accepts_query_key(monkeypatch):
    monkeypatch.setattr(sec, "API_KEY", "secret")
    scope_req = _req(ip="10.9.9.4")
    scope_req.scope["query_string"] = b"api_key=secret"
    assert is_authorized(scope_req) is True


def test_open_paths_skip_auth(monkeypatch):
    import asyncio
    monkeypatch.setattr(sec, "API_KEY", "secret")
    resp = asyncio.run(SecurityMiddleware(None).dispatch(_req(path="/health", ip="10.9.9.5"), _ok_response))
    assert resp.status_code == 200


def test_rate_limit_trips_after_60_chat_hits():
    ip = "10.9.9.6"
    for _ in range(60):
        assert check_rate_limit(_req("/api/chat", ip)) is True
    assert check_rate_limit(_req("/api/chat", ip)) is False


def test_rate_limit_is_per_client():
    for _ in range(60):
        assert check_rate_limit(_req("/api/chat", "10.9.9.7")) is True
    assert check_rate_limit(_req("/api/chat", "10.9.9.8")) is True

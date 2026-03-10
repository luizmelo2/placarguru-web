import os

import pytest

import utils
from utils import fetch_release_file


class _Resp:
    def __init__(self, content: bytes, status_ok: bool = True):
        self.content = content
        self._ok = status_ok
        self.headers = {"ETag": "abc", "Last-Modified": "Mon, 01 Jan 2024 00:00:00 GMT"}

    def raise_for_status(self):
        if not self._ok:
            raise RuntimeError("http error")


def test_fetch_release_file_rejects_non_allowlisted_host(monkeypatch):
    fetch_release_file.clear()
    monkeypatch.setenv("PG_RELEASE_ALLOWED_HOSTS", "example.com")

    with pytest.raises(ValueError):
        fetch_release_file("https://github.com/owner/repo/file.xlsx")


def test_fetch_release_file_retries_and_succeeds(monkeypatch):
    fetch_release_file.clear()
    monkeypatch.setenv("PG_RELEASE_ALLOWED_HOSTS", "github.com")
    monkeypatch.setenv("PG_RELEASE_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("PG_RELEASE_BACKOFF_S", "0")

    calls = {"n": 0}

    def fake_get(url, timeout, verify):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("temporary")
        return _Resp(b"ok")

    monkeypatch.setattr(utils.requests, "get", fake_get)

    content, etag, last_mod = fetch_release_file("https://github.com/owner/repo/file.xlsx")

    assert calls["n"] == 3
    assert content == b"ok"
    assert etag == "abc"
    assert "2024" in last_mod


def test_fetch_release_file_checksum_validation(monkeypatch):
    fetch_release_file.clear()
    monkeypatch.setenv("PG_RELEASE_ALLOWED_HOSTS", "github.com")
    monkeypatch.setenv("PG_RELEASE_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("PG_RELEASE_SHA256", "deadbeef")

    monkeypatch.setattr(utils.requests, "get", lambda *args, **kwargs: _Resp(b"content"))

    with pytest.raises(RuntimeError):
        fetch_release_file("https://github.com/owner/repo/file.xlsx")


def test_fetch_release_file_uses_tls_verify_by_default(monkeypatch):
    fetch_release_file.clear()
    monkeypatch.setenv("PG_RELEASE_ALLOWED_HOSTS", "github.com")
    monkeypatch.delenv("PG_RELEASE_VERIFY_SSL", raising=False)

    seen = {"verify": None}

    def fake_get(url, timeout, verify):
        seen["verify"] = verify
        return _Resp(b"ok")

    monkeypatch.setattr(utils.requests, "get", fake_get)

    fetch_release_file("https://github.com/owner/repo/file.xlsx")
    assert seen["verify"] is True


def test_fetch_release_file_can_disable_tls_verify_via_env(monkeypatch):
    fetch_release_file.clear()
    monkeypatch.setenv("PG_RELEASE_ALLOWED_HOSTS", "github.com")
    monkeypatch.setenv("PG_RELEASE_VERIFY_SSL", "0")

    seen = {"verify": None}

    def fake_get(url, timeout, verify):
        seen["verify"] = verify
        return _Resp(b"ok")

    monkeypatch.setattr(utils.requests, "get", fake_get)

    fetch_release_file("https://github.com/owner/repo/file.xlsx")
    assert seen["verify"] is False

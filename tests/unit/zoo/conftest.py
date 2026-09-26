"""Shared fixtures for ``dfwb.zoo`` tests: an isolated cache/state, and a local HTTP server for
the weight manager's download tests (no real network; sockets are always closed)."""

from __future__ import annotations

import dataclasses
import http.server
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest


@dataclasses.dataclass
class Isolated:
    cache: Path
    state: Path


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Isolated:
    """A private cache root and licence-acknowledgement store for every zoo test."""
    paths = Isolated(cache=tmp_path / "cache", state=tmp_path / "state")
    monkeypatch.setenv("DFWB_CACHE_ROOT", str(paths.cache))
    monkeypatch.setenv("DFWB_STATE_DIR", str(paths.state))
    return paths


class _Handler(http.server.BaseHTTPRequestHandler):
    routes: dict[str, bytes] = {}
    requests: list[str] = []

    def log_message(self, *args: object) -> None:  # keep test output quiet
        pass

    def do_GET(self) -> None:
        type(self).requests.append(self.path)
        body = type(self).routes.get(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@dataclasses.dataclass
class Server:
    url: str
    routes: dict[str, bytes]
    requests: list[str]


@pytest.fixture
def server() -> Iterator[Server]:
    routes: dict[str, bytes] = {}
    requests: list[str] = []
    handler = type("Handler", (_Handler,), {"routes": routes, "requests": requests})
    httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield Server(f"http://127.0.0.1:{httpd.server_port}", routes, requests)
    finally:
        httpd.shutdown()
        thread.join()
        httpd.server_close()

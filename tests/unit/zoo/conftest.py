"""Shared fixtures for ``dfwb.zoo`` tests: an isolated cache/state, and a local HTTP server for
the weight manager's download tests (no real network; sockets are always closed)."""

from __future__ import annotations

import dataclasses
import http.server
import threading
from collections.abc import Iterator
from pathlib import Path

import platformdirs
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


def _snapshot(root: Path) -> frozenset[tuple[str, int, int]]:
    """``(relative path, size, mtime_ns)`` for every file under ``root``, or an empty set when
    ``root`` does not exist -- comparable before/after so any write, delete or edit shows up,
    including the directory coming into existence at all."""
    if not root.is_dir():
        return frozenset()
    return frozenset(
        (str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    )


@pytest.fixture(autouse=True)
def _never_touch_the_real_cache_or_state_dir() -> Iterator[None]:
    """A safety net independent of ``isolated`` above: every zoo test is expected to redirect
    ``DFWB_CACHE_ROOT``/``DFWB_STATE_DIR`` into ``tmp_path``, but if one ever does not (a bug, or
    a stray ``monkeypatch.undo()`` reverting that redirection early), this catches it rather than
    letting the test silently read from or write to the machine's real, shared directories --
    which the controller (not this suite) is responsible for cleaning up, so this only asserts
    and never deletes anything itself."""
    real_cache = Path(platformdirs.user_cache_dir("dfwb"))
    real_state = Path(platformdirs.user_state_dir("dfwb"))
    before = (_snapshot(real_cache), _snapshot(real_state))
    yield
    after = (_snapshot(real_cache), _snapshot(real_state))
    assert after == before, (
        f"a zoo test wrote to the real {real_cache} or {real_state} instead of a redirected "
        "DFWB_CACHE_ROOT/DFWB_STATE_DIR"
    )


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

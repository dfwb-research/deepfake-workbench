import hashlib
import http.server
import socket
import threading
from collections.abc import Iterator

import pytest

from dfwb.core.errors import ContractError, InstallationError
from dfwb.core.fetch import fetch

CONTENT = b"pretend these are model weights\n" * 100
SHA256 = hashlib.sha256(CONTENT).hexdigest()


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:  # keep test output quiet
        pass

    def do_GET(self) -> None:
        if self.path == "/model.bin":
            self.send_response(200)
            self.send_header("Content-Length", str(len(CONTENT)))
            self.end_headers()
            self.wfile.write(CONTENT)
        elif self.path == "/truncated.bin":
            # Claim the full length but send only part of it, then close: nothing at the socket
            # layer complains, so what actually catches this is the sha256 not matching.
            self.send_response(200)
            self.send_header("Content-Length", str(len(CONTENT)))
            self.end_headers()
            self.wfile.write(CONTENT[: len(CONTENT) // 2])
            self.close_connection = True
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture
def server() -> Iterator[str]:
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        thread.join()


def test_fetch_downloads_and_verifies_the_right_sha(server, tmp_path):
    dest = tmp_path / "weights" / "model.bin"
    result = fetch(f"{server}/model.bin", SHA256, dest)
    assert result == dest
    assert dest.read_bytes() == CONTENT


def test_fetch_removes_the_file_and_raises_on_sha_mismatch(server, tmp_path):
    dest = tmp_path / "model.bin"
    wrong = "0" * 64
    with pytest.raises(ContractError, match="sha256"):
        fetch(f"{server}/model.bin", wrong, dest)
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []


def test_fetch_is_offline_aware(tmp_path, monkeypatch, server):
    monkeypatch.setenv("DFWB_OFFLINE", "1")
    dest = tmp_path / "model.bin"
    with pytest.raises(InstallationError, match="DFWB_OFFLINE"):
        fetch(f"{server}/model.bin", SHA256, dest)
    assert not dest.exists()


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", "on", "ON"])
def test_offline_env_truthy_spellings_are_all_offline(tmp_path, monkeypatch, server, value):
    monkeypatch.setenv("DFWB_OFFLINE", value)
    with pytest.raises(InstallationError, match="DFWB_OFFLINE"):
        fetch(f"{server}/model.bin", SHA256, tmp_path / "model.bin")


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_offline_env_other_values_mean_online(tmp_path, monkeypatch, server, value):
    monkeypatch.setenv("DFWB_OFFLINE", value)
    dest = tmp_path / "model.bin"
    result = fetch(f"{server}/model.bin", SHA256, dest)
    assert result == dest
    assert dest.read_bytes() == CONTENT


def test_fetch_leaves_no_partial_file_after_an_interrupted_transfer(server, tmp_path):
    dest = tmp_path / "model.bin"
    with pytest.raises(ContractError):
        fetch(f"{server}/truncated.bin", SHA256, dest)
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []


def test_fetch_raises_installation_error_on_http_error(server, tmp_path):
    dest = tmp_path / "model.bin"
    with pytest.raises(InstallationError, match="404"):
        fetch(f"{server}/nope.bin", SHA256, dest)
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []


def test_fetch_raises_installation_error_on_connection_refused(tmp_path):
    dest = tmp_path / "model.bin"
    # Nothing listens here: an ephemeral port on loopback that was never bound.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with pytest.raises(InstallationError):
        fetch(f"http://127.0.0.1:{port}/model.bin", SHA256, dest)
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []

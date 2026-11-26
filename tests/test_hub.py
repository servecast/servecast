"""The LLM's config.json from the Hugging Face hub: the user's token goes to the hub and nowhere else.

A local server stands in for the hub. The hub answers config.json with a redirect within its own host (a 307 to
/api/resolve-cache/..., observed on 2026-11-26), and urllib copies every header of a request to the location it is
redirected to, whatever the host.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from servecast.core import hub

TOKEN = "test-token-not-a-real-one"
CONFIG = {"model_type": "llama", "num_hidden_layers": 2}


class Hub:
    """A local server: records each request's path and Authorization header, and answers from `routes`."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, str | None]] = []
        self.routes: dict[str, tuple[int, dict[str, str], bytes]] = {}
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                server.seen.append((self.path.split("?")[0], self.headers.get("Authorization")))
                status, headers, body = server.routes.get(self.path.split("?")[0], (404, {}, b""))
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:  # the test's output stays quiet
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def servers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[tuple[Hub, Hub]]:
    """The hub at 127.0.0.1 and another host at localhost (another host name, so a redirect to it leaves the hub)."""
    first, other = Hub(), Hub()
    monkeypatch.setattr(hub, "URL", f"http://127.0.0.1:{first.port}/{{repo}}/resolve/{{revision}}/config.json")
    monkeypatch.setenv("SERVECAST_CACHE_DIR", str(tmp_path / "cache"))
    yield first, other
    first.close()
    other.close()


PATH = "/org/gated/resolve/main/config.json"
OK = (200, {"Content-Type": "application/json"}, json.dumps(CONFIG).encode())


def test_a_gated_models_config_is_fetched_with_the_token_and_cached_without_it(
        servers: tuple[Hub, Hub], monkeypatch: pytest.MonkeyPatch) -> None:
    first, _ = servers
    first.routes[PATH] = OK
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    assert hub.fetch_config("org/gated") == CONFIG
    assert first.seen == [(PATH, f"Bearer {TOKEN}")]
    cached = hub.cached_path("org/gated").read_text(encoding="utf-8")
    assert json.loads(cached) == CONFIG and TOKEN not in cached


def test_a_redirect_within_the_hub_keeps_the_token(servers: tuple[Hub, Hub], monkeypatch: pytest.MonkeyPatch) -> None:
    first, _ = servers
    first.routes[PATH] = (307, {"Location": "/api/resolve-cache/org/gated/config.json"}, b"")
    first.routes["/api/resolve-cache/org/gated/config.json"] = OK
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    assert hub.fetch_config("org/gated") == CONFIG
    assert [auth for _, auth in first.seen] == [f"Bearer {TOKEN}", f"Bearer {TOKEN}"]


def test_a_redirect_to_another_host_does_not_carry_the_token(servers: tuple[Hub, Hub],
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    first, other = servers
    first.routes[PATH] = (302, {"Location": f"http://localhost:{other.port}/elsewhere/config.json"}, b"")
    other.routes["/elsewhere/config.json"] = OK
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    assert hub.fetch_config("org/gated") == CONFIG
    assert first.seen == [(PATH, f"Bearer {TOKEN}")] and other.seen == [("/elsewhere/config.json", None)]


def test_a_gated_model_without_a_token_is_an_input_error_naming_hf_token(servers: tuple[Hub, Hub],
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    first, _ = servers
    first.routes[PATH] = (401, {}, b"")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(ValueError, match="HF_TOKEN") as refused:
        hub.fetch_config("org/gated")
    assert first.seen == [(PATH, None)] and TOKEN not in str(refused.value)

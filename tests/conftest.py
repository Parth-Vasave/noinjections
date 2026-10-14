"""Shared pytest fixtures."""

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from email.message import Message
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
WEB_FIXTURES = FIXTURES / "web"


@dataclass(frozen=True)
class Route:
    """A canned response. A ``list`` body is sent chunk by chunk with no Content-Length, after
    sleeping ``delay`` seconds before each chunk; a ``bytes`` body is sent at once.
    """

    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes | list[bytes] = b""
    delay: float = 0.0


@dataclass(frozen=True)
class RecordedRequest:
    path: str
    headers: Message


class LocalServer:
    """A local HTTP server serving ``tests/fixtures/web/``, plus routes a test registers."""

    def __init__(self, httpd: ThreadingHTTPServer) -> None:
        self.httpd = httpd
        host, port = httpd.server_address[:2]
        self.base_url = f"http://{host}:{port}"
        self.routes: dict[str, Route] = {}
        self.requests: list[RecordedRequest] = []

    def url(self, path: str = "/") -> str:
        return self.base_url + path

    def route(
        self,
        path: str,
        status: int = 200,
        *,
        body: bytes | list[bytes] = b"",
        headers: dict[str, str] | None = None,
        delay: float = 0.0,
    ) -> str:
        """Serve a canned response at ``path`` (which wins over a fixture file); returns its URL."""
        self.routes[path] = Route(status, headers or {}, body, delay)
        return self.url(path)

    def redirect(self, path: str, location: str, status: int = 302) -> str:
        return self.route(path, status, headers={"Location": location})


class _Handler(SimpleHTTPRequestHandler):
    server: "_Server"

    def do_GET(self) -> None:
        state = self.server.state
        state.requests.append(RecordedRequest(self.path, self.headers))
        route = state.routes.get(urlsplit(self.path).path)
        if route is None:
            super().do_GET()
            return
        self.send_response(route.status)
        for name, value in route.headers.items():
            self.send_header(name, value)
        if isinstance(route.body, bytes):
            self.send_header("Content-Length", str(len(route.body)))
            self.end_headers()
            self.wfile.write(route.body)
            return
        self.end_headers()  # HTTP/1.0: the body ends when the connection closes
        for chunk in route.body:
            time.sleep(route.delay)
            self.wfile.write(chunk)
            self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:
        pass


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    state: LocalServer

    def handle_error(self, request: object, client_address: object) -> None:
        pass  # clients that time out on purpose leave broken pipes behind


@pytest.fixture(scope="session")
def _local_httpd() -> Iterator[LocalServer]:
    handler = partial(_Handler, directory=str(WEB_FIXTURES))
    httpd = _Server(("127.0.0.1", 0), handler)
    state = LocalServer(httpd)
    httpd.state = state
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield state
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def local_server(_local_httpd: LocalServer) -> Iterator[LocalServer]:
    """A local HTTP server serving ``tests/fixtures/web/``. Routes and requests reset per test."""
    yield _local_httpd
    _local_httpd.routes.clear()
    _local_httpd.requests.clear()

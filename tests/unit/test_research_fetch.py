import importlib.util
import socket
import ssl
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpcore
import httpx
import pytest


@pytest.fixture
def fetch_boundary(monkeypatch):
    network = SimpleNamespace(connections=[], tls=[], requests=[], status=200, headers={}, body=b"Document")
    clients = []
    client_class = httpx.Client
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)

    def make_client(*args, **kwargs):
        client = client_class(*args, **kwargs)
        clients.append(client)
        return client

    class Stream:
        def __init__(self):
            self.pending = b""
            self.remaining = b""

        def write(self, buffer, timeout=None):
            self.pending += buffer
            if b"\r\n\r\n" not in self.pending:
                return
            network.requests.append(self.pending)
            if self.pending.startswith(b"CONNECT "):
                self.remaining = b"HTTP/1.1 200 Connection established\r\n\r\n"
            else:
                headers = {"Content-Length": str(len(network.body)), "Connection": "close", **network.headers}
                self.remaining = (
                    f"HTTP/1.1 {network.status} Response\r\n".encode()
                    + b"".join(f"{name}: {value}\r\n".encode() for name, value in headers.items())
                    + b"\r\n"
                    + network.body
                )
            self.pending = b""

        def read(self, max_bytes, timeout=None):
            data, self.remaining = self.remaining[:max_bytes], self.remaining[max_bytes:]
            return data

        def start_tls(self, ssl_context, server_hostname=None, timeout=None):
            network.tls.append((server_hostname, ssl_context.check_hostname, ssl_context.verify_mode))
            return self

        def get_extra_info(self, info):
            return None

        def close(self):
            pass

    def connect_tcp(self, host, port, **kwargs):
        network.connections.append((host, port))
        return Stream()

    monkeypatch.setattr(httpcore.SyncBackend, "connect_tcp", connect_tcp)
    monkeypatch.setattr(httpx, "Client", make_client)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))])
    converter = ModuleType("markdownify")
    converter.markdownify = lambda text: f"markdown:{text}"
    monkeypatch.setitem(sys.modules, "markdownify", converter)

    def load_fetch():
        path = Path(__file__).parents[1] / "langchain" / "research-swarm" / "tools" / "src" / "agent" / "utils.py"
        spec = importlib.util.spec_from_file_location("research_fetch", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.fetch_doc

    yield load_fetch, network
    for client in clients:
        client.close()


@pytest.mark.parametrize(
    "addresses",
    [
        ["127.0.0.1"],
        ["169.254.169.254"],
        ["10.0.0.1"],
        ["::1"],
        ["fec0::1"],
        ["8.8.8.8", "192.168.1.1"],
    ],
)
def test_fetch_rejects_every_nonpublic_dns_destination(fetch_boundary, monkeypatch, addresses):
    load_fetch, network = fetch_boundary
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *args, **kwargs: [(0, socket.SOCK_STREAM, 6, "", (address, 443)) for address in addresses]
    )

    result = load_fetch()("https://docs.example/document")

    assert result.startswith("Encountered an HTTP error:")
    assert network.connections == []


@pytest.mark.parametrize(
    "url, address, port, host, tls",
    [
        ("https://docs.example:8443/document?part=1", "8.8.8.8", 8443, "docs.example:8443", "docs.example"),
        ("https://docs.example/document", "2606:4700:4700::1111", 443, "docs.example", "docs.example"),
        ("http://docs.example/document", "8.8.8.8", 80, "docs.example", None),
    ],
)
def test_fetch_pins_connection_but_preserves_document_host_and_tls(fetch_boundary, monkeypatch, url, address, port, host, tls):
    load_fetch, network = fetch_boundary
    resolutions = []

    def resolve(host, port, **kwargs):
        resolutions.append(host)
        resolved = address if len(resolutions) == 1 else "127.0.0.1"
        return [(0, socket.SOCK_STREAM, 6, "", (resolved, port))]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)

    assert load_fetch()(url) == "markdown:Document"
    assert resolutions == ["docs.example"]
    assert network.connections == [(address, port)]
    assert f"Host: {host}\r\n".encode() in network.requests[0]
    assert network.tls == ([(tls, True, ssl.CERT_REQUIRED)] if tls else [])


def test_fetch_does_not_follow_redirects_or_environment_proxies(fetch_boundary, monkeypatch):
    load_fetch, network = fetch_boundary
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8123")
    network.status = 302
    network.headers = {"Location": "http://169.254.169.254/document"}

    result = load_fetch()("https://docs.example/document")

    assert result.startswith("Encountered an HTTP error:")
    assert network.connections == [("8.8.8.8", 443)]
    assert len(network.requests) == 1


def test_fetch_tries_remaining_public_addresses_after_connection_failure(fetch_boundary, monkeypatch):
    load_fetch, network = fetch_boundary
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *args, **kwargs: [(0, socket.SOCK_STREAM, 6, "", (address, 443)) for address in ["8.8.8.8", "1.1.1.1"]]
    )
    connect = httpcore.SyncBackend.connect_tcp

    def connect_with_failure(self, host, port, **kwargs):
        if host == "8.8.8.8":
            network.connections.append((host, port))
            raise httpcore.ConnectError("unreachable")
        return connect(self, host, port, **kwargs)

    monkeypatch.setattr(httpcore.SyncBackend, "connect_tcp", connect_with_failure)

    assert load_fetch()("https://docs.example/document") == "markdown:Document"
    assert network.connections == [("8.8.8.8", 443), ("1.1.1.1", 443)]


@pytest.mark.parametrize("url", ["file:///etc/hosts", "https://user:secret@docs.example/document", "https://docs.example:invalid"])
def test_fetch_rejects_unsafe_or_invalid_urls(fetch_boundary, url):
    load_fetch, network = fetch_boundary

    assert load_fetch()(url).startswith("Encountered an HTTP error:")
    assert network.connections == []

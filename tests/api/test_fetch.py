import socket
import ssl
import threading
from types import SimpleNamespace

import pytest
import urllib3

from api import fetch
from tests.api.conftest import img_bytes

PUBLIC = "93.184.216.34"


def dns(monkeypatch, table):
    def gai(host, port, **kw):
        ip = table.get(host, host)
        fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(fam, socket.SOCK_STREAM, 6, "", (ip, port))]
    monkeypatch.setattr(fetch.socket, "getaddrinfo", gai)


class FakeResp:
    def __init__(self, status=200, headers=None, body=b"", chunks=None, delay=None):
        self.status, self.headers = status, headers or {}
        self.chunks = chunks if chunks is not None else [body]
        self.delay = delay

    def stream(self, n, decode_content=False):
        for c in self.chunks:
            if self.delay:
                self.delay()
            yield c

    def release_conn(self):
        pass

    def close(self):
        pass


def serve(monkeypatch, responses, seen=None):
    """Replace the HTTP layer; responses is a list (one per hop) or a callable."""
    it = iter(responses)

    def urlopen(self, method, target, **kw):
        if seen is not None:
            seen.append((self.host, self.conn_kw["server_hostname"], kw["headers"], target))
        r = next(it)
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(urllib3.HTTPSConnectionPool, "urlopen", urlopen)


def link(c, url):
    return c.post("/v1/check", data={"url": url})


@pytest.mark.parametrize("url", [
    "https://127.0.0.1/a.png", "https://10.0.0.1/a.png", "https://169.254.169.254/latest/meta-data",
    "https://[::1]/a.png", "https://[::ffff:10.0.0.1]/a.png", "https://100.64.0.1/a.png",
    "https://224.0.0.1/a.png", "https://0.0.0.0/a.png",
    "https://internal.example/a.png",  # hostname resolving to a private IP
    "http://example.com/a.png", "https://user:pw@example.com/a.png", "ftp://example.com/a.png",
])
def test_blocked(build, monkeypatch, url):
    dns(monkeypatch, {"internal.example": "192.168.1.5"})
    serve(monkeypatch, [])  # any connection attempt would raise StopIteration -> 502, not 400
    r = link(build(), url)
    assert r.status_code == 400 and r.json() == {"detail": "That link can't be used."}


def test_any_blocked_address_blocks(build, monkeypatch):
    monkeypatch.setattr(fetch.socket, "getaddrinfo", lambda h, p, **k: [
        (socket.AF_INET, 1, 6, "", (PUBLIC, p)), (socket.AF_INET, 1, 6, "", ("10.0.0.1", p))])
    assert link(build(), "https://mixed.example/a.png").status_code == 400


def test_redirect_to_private_blocked(build, monkeypatch):
    dns(monkeypatch, {"cdn.example": PUBLIC, "evil.example": "10.0.0.1"})
    serve(monkeypatch, [FakeResp(302, {"Location": "https://evil.example/x.png"})])
    assert link(build(), "https://cdn.example/a.png").status_code == 400


def test_redirect_to_http_blocked(build, monkeypatch):
    dns(monkeypatch, {"cdn.example": PUBLIC})
    serve(monkeypatch, [FakeResp(301, {"Location": "http://cdn.example/x.png"})])
    assert link(build(), "https://cdn.example/a.png").status_code == 400


def test_redirect_limit(build, monkeypatch):
    dns(monkeypatch, {"cdn.example": PUBLIC})
    hop = FakeResp(302, {"Location": "/next.png"})
    serve(monkeypatch, [hop] * 4)
    assert link(build(), "https://cdn.example/a.png").status_code == 502
    serve(monkeypatch, [hop] * 3 + [FakeResp(200, body=img_bytes(255))])
    assert link(build(), "https://cdn.example/a.png").status_code == 200


def test_size_cap(build, monkeypatch):
    dns(monkeypatch, {"cdn.example": PUBLIC})
    big = b"\0" * (fetch.MAX_BYTES // 2 + 1)
    serve(monkeypatch, [FakeResp(chunks=[big, big])])
    assert link(build(), "https://cdn.example/a.png").status_code == 413
    serve(monkeypatch, [FakeResp(headers={"Content-Length": str(fetch.MAX_BYTES + 1)}, chunks=[])])
    assert link(build(), "https://cdn.example/a.png").status_code == 413


def test_timeouts(build, monkeypatch):
    dns(monkeypatch, {"cdn.example": PUBLIC})
    serve(monkeypatch, [urllib3.exceptions.ConnectTimeoutError(None, "t")])
    assert link(build(), "https://cdn.example/a.png").status_code == 504
    # slow trickle past the 10 s total
    t = iter(range(0, 100, 11))
    monkeypatch.setattr(fetch, "_clock", lambda: next(t))
    serve(monkeypatch, [FakeResp(chunks=[b"x"])])
    assert link(build(), "https://cdn.example/a.png").status_code == 504


@pytest.mark.parametrize("status", [401, 403, 404, 410])
def test_expired(build, monkeypatch, status):
    dns(monkeypatch, {"cdn.example": PUBLIC})
    serve(monkeypatch, [FakeResp(status)])
    r = link(build(), "https://cdn.example/a.png")
    assert r.status_code == 400 and "expired or not be public" in r.json()["detail"]


def test_other_failures_502(build, monkeypatch):
    dns(monkeypatch, {"cdn.example": PUBLIC})
    serve(monkeypatch, [FakeResp(500), OSError("reset")])
    c = build()
    assert link(c, "https://cdn.example/a.png").status_code == 502
    assert link(c, "https://cdn.example/a.png").status_code == 502


def test_exactly_one_input(build):
    c = build()
    assert c.post("/v1/check").status_code == 400
    both = c.post("/v1/check", data={"url": "https://x.example/a.png"}, files={"image": ("a.png", img_bytes(), "image/png")})
    assert both.status_code == 400


def test_happy_path_and_logging(build, monkeypatch, caplog):
    dns(monkeypatch, {"cdn.example": PUBLIC})
    seen = []
    serve(monkeypatch, [FakeResp(302, {"Location": "/b.png?sig=SECRET"}), FakeResp(200, body=img_bytes(255))], seen)
    with caplog.at_level("INFO", logger="sixfingers"):
        r = link(build(), "https://cdn.example/a.png?token=SECRET")
    assert r.status_code == 200 and r.json()["verdict"] == "likely_ai"
    assert all(s[0] == PUBLIC and s[1] == "cdn.example" and s[2]["Host"] == "cdn.example" for s in seen)
    assert seen[0][2]["User-Agent"] == "SixFingers/1.0"
    ev = [x for x in caplog.records if getattr(x, "event", None) == "check"][0]
    assert ev.input == "link" and ev.host == "cdn.example"
    assert "SECRET" not in repr(ev.__dict__)


def test_link_counts_toward_rate_limit(build, monkeypatch):
    dns(monkeypatch, {"cdn.example": PUBLIC})
    serve(monkeypatch, [FakeResp(200, body=img_bytes(255))])
    c = build(rate_limit=1)
    assert link(c, "https://cdn.example/a.png").status_code == 200
    assert link(c, "https://cdn.example/a.png").status_code == 429


def test_socket_connects_to_vetted_ip(build, monkeypatch):
    calls = []
    answers = iter([PUBLIC, "10.0.0.1"])  # a second lookup would be the rebinding answer

    def gai(host, port, **kw):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (next(answers), port))]
    monkeypatch.setattr(fetch.socket, "getaddrinfo", gai)

    def cc(address, *a, **k):
        calls.append(address)
        raise OSError("stop")
    monkeypatch.setattr(urllib3.util.connection, "create_connection", cc)
    assert link(build(), "https://rebind.example/a.png").status_code == 502
    assert calls == [(PUBLIC, 443)]

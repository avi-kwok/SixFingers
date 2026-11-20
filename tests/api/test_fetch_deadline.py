"""A real local TLS server that trickles bytes must not hold a fetch past its total deadline."""
import datetime
import functools
import socket
import ssl
import threading
import time

import pytest
import urllib3
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from api import fetch
from api.validate import UploadError


@pytest.fixture(scope="module")
def cert(tmp_path_factory):
    """Self-signed certificate for localhost; returns (cert path, key path)."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)
    crt = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
           .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
           .not_valid_after(now + datetime.timedelta(days=1))
           .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)
           .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
           .sign(key, hashes.SHA256()))
    d = tmp_path_factory.mktemp("tls")
    (d / "c.pem").write_bytes(crt.public_bytes(serialization.Encoding.PEM))
    (d / "k.pem").write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                serialization.NoEncryption()))
    return str(d / "c.pem"), str(d / "k.pem")


def trickle(conn, data, gap):
    for i in range(len(data)):
        conn.sendall(data[i:i + 1])
        time.sleep(gap)


def handle(raw, ctx, mode):
    try:
        with ctx.wrap_socket(raw, server_side=True) as conn:
            req = b""
            while b"\r\n\r\n" not in req:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                req += chunk
            if mode == "redirect" and req.startswith(b"GET /a "):
                conn.sendall(b"HTTP/1.1 302 Found\r\nLocation: /b\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            elif mode == "ok":
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\nimg")
            elif mode == "body":
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nConnection: close\r\n\r\n")
                trickle(conn, b"x" * 1000, 0.3)
            else:  # headers, or the second redirect hop
                trickle(conn, b"HTTP/1.1 200 OK\r\n" + b"X-Pad: y\r\n" * 100, 0.3)
    except OSError:
        pass  # the client hung up, which is the point


@pytest.fixture
def server(cert, monkeypatch):
    """Start a trickling server; returns a function mode -> https url."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(*cert)
    lsock = socket.create_server(("127.0.0.1", 0))
    port = lsock.getsockname()[1]
    state = {}

    def accept():
        while True:
            try:
                raw, _ = lsock.accept()
            except OSError:
                return
            threading.Thread(target=handle, args=(raw, ctx, state["mode"]), daemon=True).start()
    threading.Thread(target=accept, daemon=True).start()

    # Loopback is blocked by the real guard, so vet to 127.0.0.1 here and trust the test certificate.
    def vet(url):
        u = urllib3.util.parse_url(url)
        return "localhost", port, u.path or "/", "127.0.0.1"
    monkeypatch.setattr(fetch, "_vet", vet)
    monkeypatch.setattr(urllib3, "HTTPSConnectionPool", functools.partial(urllib3.HTTPSConnectionPool, ca_certs=cert[0]))
    monkeypatch.setattr(fetch, "TOTAL_TIMEOUT", 2.0)

    def start(mode):
        state["mode"] = mode
        return f"https://localhost:{port}/a"
    yield start
    lsock.close()


@pytest.mark.parametrize("mode", ["headers", "body", "redirect"])
def test_trickling_server_hits_total_deadline(server, mode):
    url = server(mode)
    t0 = time.monotonic()
    with pytest.raises(UploadError) as e:
        fetch.fetch_image(url)
    assert e.value.status == 504
    assert time.monotonic() - t0 < fetch.TOTAL_TIMEOUT + 2


def test_fast_server_still_works(server):
    assert fetch.fetch_image(server("ok")) == b"img"

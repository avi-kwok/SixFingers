"""Fetch an image from an https link without letting the server reach internal addresses."""
import functools
import ipaddress
import socket
import threading
import time
from urllib.parse import urljoin, urlsplit

import urllib3

from api.validate import MAX_BYTES, UploadError

CONNECT_TIMEOUT, TOTAL_TIMEOUT, MAX_REDIRECTS = 5.0, 10.0, 3
BLOCKED = UploadError(400, "That link can't be used.")
EXPIRED = UploadError(400, "Couldn't fetch that image. The link may have expired or not be public.")
HEADERS = {"User-Agent": "SixFingers/1.0", "Accept-Encoding": "identity", "Accept": "image/*"}

_clock = time.monotonic


class _Conn(urllib3.connection.HTTPSConnection):
    """HTTPS connection that hands its raw socket to the deadline watchdog."""

    def __init__(self, *a, watch, **kw):
        super().__init__(*a, **kw)
        self._watch = watch

    def _new_conn(self):
        sock = super()._new_conn()
        self._watch(sock.dup())  # TLS detaches the original socket object; a dup stays usable and is never reused
        return sock


def _vet(url: str) -> tuple[str, int, str, str]:
    """Return (host, port, request target, vetted ip) or raise BLOCKED."""
    try:
        u = urlsplit(url.strip())
        host, port = u.hostname, u.port or 443
    except ValueError:
        raise BLOCKED from None
    if u.scheme != "https" or not host or u.username is not None or u.password is not None:
        raise BLOCKED
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise UploadError(502, "Couldn't fetch that image.") from None
    ips = []
    for info in infos:
        a = ipaddress.ip_address(info[4][0].split("%")[0])
        a = getattr(a, "ipv4_mapped", None) or a
        if not a.is_global or a.is_multicast:  # is_global alone lets IPv4 multicast through
            raise BLOCKED
        ips.append(str(info[4][0]))
    if not ips:
        raise BLOCKED
    target = (u.path or "/") + (f"?{u.query}" if u.query else "")
    return host, port, target, ips[0]


def fetch_image(url: str) -> bytes:
    # Socket timeouts are per recv, so a server sending 1 byte every few seconds never trips them.
    # A watchdog shuts the socket down at the deadline, which fails any blocked connect, handshake or read.
    socks, expired = [], threading.Event()

    def kill(sock):
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def watch(sock):
        socks.append(sock)
        if expired.is_set():  # the watchdog fired while this socket was connecting
            kill(sock)

    def on_deadline():
        expired.set()
        for s in list(socks):
            kill(s)

    timer = threading.Timer(TOTAL_TIMEOUT, on_deadline)
    timer.daemon = True
    timer.start()
    try:
        return _fetch(url, _clock() + TOTAL_TIMEOUT, watch, expired)
    finally:
        timer.cancel()
        timer.join()
        for s in socks:
            s.close()


def _fetch(url, deadline, watch, expired) -> bytes:
    for _ in range(MAX_REDIRECTS + 1):
        host, port, target, ip = _vet(url)
        # Connect to the vetted IP; SNI, certificate check and Host stay on the hostname.
        left = deadline - _clock()
        if left <= 0:
            raise UploadError(504, "The link took too long to respond.")
        pool = urllib3.HTTPSConnectionPool(ip, port, server_hostname=host, assert_hostname=host,
                                           retries=False, timeout=urllib3.Timeout(connect=min(CONNECT_TIMEOUT, left), read=left))
        pool.ConnectionCls = functools.partial(_Conn, watch=watch)
        try:
            resp = pool.urlopen("GET", target, headers={**HEADERS, "Host": host if port == 443 else f"{host}:{port}"},
                                redirect=False, preload_content=False, decode_content=False)
            try:
                if resp.status in (301, 302, 303, 307, 308) and resp.headers.get("Location"):
                    url = urljoin(url, resp.headers["Location"])
                    continue
                if resp.status in (401, 403, 404, 410):
                    raise EXPIRED
                if resp.status == 429:
                    raise UploadError(502, "That site is limiting requests right now. Try again later or upload the file.")
                if resp.status != 200:
                    raise UploadError(502, "Couldn't fetch that image.")
                if int(resp.headers.get("Content-Length") or 0) > MAX_BYTES:
                    raise UploadError(413, "Image is larger than 10 MB.")
                buf = bytearray()
                for chunk in resp.stream(65536, decode_content=False):
                    buf += chunk
                    if len(buf) > MAX_BYTES:
                        raise UploadError(413, "Image is larger than 10 MB.")
                    if _clock() > deadline:
                        raise TimeoutError
                if expired.is_set():  # a body cut off at the deadline can look like a normal end
                    raise TimeoutError
                return bytes(buf)
            finally:
                resp.release_conn()
                resp.close()
        except UploadError:
            raise
        except urllib3.exceptions.NewConnectionError:  # subclasses ConnectTimeoutError in urllib3 2.x
            raise UploadError(502, "Couldn't fetch that image.") from None
        except (TimeoutError, urllib3.exceptions.TimeoutError):
            raise UploadError(504, "The link took too long to respond.") from None
        except Exception:
            if expired.is_set():
                raise UploadError(504, "The link took too long to respond.") from None
            raise UploadError(502, "Couldn't fetch that image.") from None
        finally:
            pool.close()
    raise UploadError(502, "Couldn't fetch that image.")  # too many redirects

"""Cloudflare Turnstile check that guards the paid judge. Off unless SIXFINGERS_TURNSTILE_SECRET is set."""
import logging
import os

import httpx

log = logging.getLogger("sixfingers")
SITEVERIFY = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
HEADER = "cf-turnstile-response"
TIMEOUT_S = 3.0


def enabled() -> bool:
    return bool(os.environ.get("SIXFINGERS_TURNSTILE_SECRET"))


def _hostname_ok(host) -> bool:
    from urllib.parse import urlparse
    from api.main import PROD_ORIGIN  # lazy: main imports this module
    default = urlparse(PROD_ORIGIN).hostname
    allowed = [h.strip() for h in os.environ.get("SIXFINGERS_TURNSTILE_HOSTNAMES", default).split(",") if h.strip()]
    if os.environ.get("SIXFINGERS_DEV") == "1":
        allowed += ["localhost", "example.com"]  # example.com is what Cloudflare's test keys return
    return isinstance(host, str) and any(host == a or host.endswith("." + a) for a in allowed)


def verify(token: str | None) -> bool:
    """True when Cloudflare accepts the token for an allowed hostname. A missing token, a rejection or any failure is False.
    The visitor's IP is deliberately not sent (remoteip is optional)."""
    if not token or len(token) > 2048:
        return False
    try:
        r = httpx.post(SITEVERIFY, data={"secret": os.environ["SIXFINGERS_TURNSTILE_SECRET"], "response": token},
                       timeout=TIMEOUT_S)
        j = r.json()
        return j.get("success") is True and _hostname_ok(j.get("hostname"))
    except Exception as e:
        log.warning("turnstile siteverify failed: %s", type(e).__name__)
        return False

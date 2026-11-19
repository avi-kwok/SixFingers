"""sixfingers doctor: quick health check. Stdlib only. Never prints secret values."""
import json
import os
import shutil
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOMAIN = os.environ.get("SIXFINGERS_DOMAIN", "api.sixfingers.avikwok.dev")
DATA = Path(os.environ.get("SIXFINGERS_DATA_DIR", REPO / "data"))
MODELS = Path(os.environ.get("SIXFINGERS_MODELS_DIR", REPO / "models"))
# The API persists its daily judge counter here (JSON); WARN if absent.
BUDGET_FILE = Path(os.environ.get("SIXFINGERS_BUDGET_FILE", DATA / "judge_budget.json"))
results = []


def check(name, status, detail=""):
    results.append(status)
    print(f"[{status:4}] {name}" + (f": {detail}" if detail else ""))


def port_open(host, port):
    try:
        socket.create_connection((host, port), timeout=3).close()
        return True
    except OSError:
        return False


def env_value(name):
    if os.environ.get(name):
        return os.environ[name]
    f = REPO / ".env"
    try:
        for l in f.read_text().splitlines():
            if l.startswith(name + "="):
                v = l.split("=", 1)[1].strip().strip("'\"")
                if v:
                    return v
    except OSError:
        pass
    return ""


def env_has_key(name):
    return bool(env_value(name))


def get_json(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:  # /v1/health answers 503 with a JSON body while loading or with no model
        return json.load(e)


def main():
    for port, name in ((8000, "api (local)"), (80, "http"), (443, "https")):
        ok = port_open("127.0.0.1", port)
        check(f"port {port} {name}", "OK" if ok else "FAIL", "listening" if ok else "nothing listening")

    try:
        with socket.create_connection((DOMAIN, 443), timeout=5) as s, ssl.create_default_context().wrap_socket(s, server_hostname=DOMAIN) as t:
            days = (ssl.cert_time_to_seconds(t.getpeercert()["notAfter"]) - time.time()) / 86400
        check("tls cert", "OK" if days > 14 else "WARN", f"{DOMAIN} valid for {days:.0f} more days")
    except Exception as e:  # noqa: BLE001
        check("tls cert", "FAIL", f"{DOMAIN}: {type(e).__name__} (DNS record or Caddy not ready?)")

    cur = MODELS / "current"
    check("models/current", "OK" if cur.exists() else "FAIL", os.readlink(cur) if cur.is_symlink() else "missing symlink")
    try:
        health = get_json("http://127.0.0.1:8000/v1/health")
        served, status = health.get("model_version"), health.get("status")
        want = os.readlink(cur) if cur.is_symlink() else "?"
        if status == "no_model":
            check("api model loaded", "FAIL", f"no model (models/current -> {want})")
        elif status == "loading":
            check("api model loaded", "WARN", f"still loading backbones for {served}; re-run in a minute")
        else:
            check("api model loaded", "OK" if served == want else "WARN", f"serving {served} (models/current -> {want})")
    except Exception as e:  # noqa: BLE001
        check("api model loaded", "FAIL", type(e).__name__)

    if (env_value("SIXFINGERS_JUDGE_BACKEND") or "api") == "command":
        check("SIXFINGERS_JUDGE_CMD", "OK" if env_has_key("SIXFINGERS_JUDGE_CMD") else "FAIL", "present" if env_has_key("SIXFINGERS_JUDGE_CMD") else "missing (judge backend is command)")
    else:
        check("ANTHROPIC_API_KEY", "OK" if env_has_key("ANTHROPIC_API_KEY") else "FAIL", "present" if env_has_key("ANTHROPIC_API_KEY") else "missing (judge falls back to detector only)")
    proxy = env_value("SIXFINGERS_TRUST_PROXY") == "1"
    check("SIXFINGERS_TRUST_PROXY", "OK" if proxy else "FAIL", "1" if proxy else "not 1 (uvicorn already trusts X-Forwarded-For from 127.0.0.1; 1 makes the API's own client-IP logic explicit)")
    check("DISCORD_WEBHOOK_URL", "OK" if env_has_key("DISCORD_WEBHOOK_URL") else "WARN", "present" if env_has_key("DISCORD_WEBHOOK_URL") else "missing (loop reports skip Discord)")

    try:
        b = json.loads(BUDGET_FILE.read_text())
        check("judge budget", "OK", json.dumps(b))
    except Exception:  # noqa: BLE001
        check("judge budget", "WARN", f"no counter file at {BUDGET_FILE}")

    loop_on = Path("/etc/cron.d/sixfingers-loop").exists()
    if not loop_on:
        check("nightly loop", "OK", "off (opt-in, see RUNBOOK)")
    reports = sorted((DATA / "loop_reports").glob("*.json")) if (DATA / "loop_reports").is_dir() else []
    if reports:
        last = json.loads(reports[-1].read_text())
        check("last nightly loop", "OK" if last.get("outcome") != "error" else "WARN", f"{last['date']} {last['outcome']}")
    elif loop_on:
        check("last nightly loop", "WARN", "no report yet")

    try:
        tz = subprocess.run(["timedatectl", "show", "-p", "Timezone", "--value"], capture_output=True, text=True, timeout=5).stdout.strip()
        check("timezone", "OK" if tz in ("UTC", "Etc/UTC") else "WARN", tz if tz in ("UTC", "Etc/UTC") else f"{tz or 'unknown'} (the loop cron time 0 9 * * * assumes UTC)")
    except (OSError, subprocess.TimeoutExpired):
        pass  # no timedatectl (not systemd): skip

    du = shutil.disk_usage(REPO)
    free = du.free / 2**30
    check("disk", "OK" if free > 10 else "WARN", f"{free:.1f} GiB free")
    return 1 if "FAIL" in results else 0


if __name__ == "__main__":
    sys.exit(main())

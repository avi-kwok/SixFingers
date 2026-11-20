"""Turnstile gate: only Cloudflare's published TEST secret keys appear here; siteverify is mocked, no network."""
import httpx
import pytest

from tests.api.conftest import FakeClient, JUDGE_JSON, img_bytes, post
from tests.api.test_limits_launch import _count
from tests.api.test_audio_api import audio_on, client, post_audio, tone  # noqa: F401
from tests.api.video_clips import requires_ffmpeg
from tests.api.test_video_api import clips, post_video  # noqa: F401  (clips is a fixture)

TEST_SECRET = "1x0000000000000000000000000000000AA"  # Cloudflare's published always-passes test secret
HDR = {"cf-turnstile-response": "XXXX.DUMMY.TOKEN.XXXX"}  # Cloudflare's published dummy token


@pytest.fixture
def siteverify(monkeypatch):
    calls = []

    def fake(url, data=None, timeout=None):
        calls.append((url, data, timeout))
        out = fake.out
        if isinstance(out, Exception):
            raise out
        return httpx.Response(200, json=out)
    fake.out = {"success": True, "hostname": "sixfingers.avikwok.dev"}
    monkeypatch.setattr("api.turnstile.httpx.post", fake)
    fake.calls = calls
    return fake


def _post(c, headers=None):
    return c.post("/v1/check", files={"image": ("a.png", img_bytes(), "image/png")}, headers=headers or {})


def test_off_is_unchanged(build, siteverify):
    fake = FakeClient(JUDGE_JSON)
    j = _post(build(fake)).json()
    assert j["source"] == "judge" and not siteverify.calls


def test_valid_token_calls_judge(build, siteverify, monkeypatch):
    monkeypatch.setenv("SIXFINGERS_TURNSTILE_SECRET", TEST_SECRET)
    fake = FakeClient(JUDGE_JSON)
    j = _post(build(fake), HDR).json()
    assert j["source"] == "judge" and len(fake.calls) == 1
    url, data, timeout = siteverify.calls[0]
    assert url.endswith("/turnstile/v0/siteverify") and "remoteip" not in data and timeout <= 3
    assert data == {"secret": TEST_SECRET, "response": HDR["cf-turnstile-response"]}


@pytest.mark.parametrize("case", ["missing", "invalid", "timeout", "hostname"])
def test_bad_token_is_detector_only_and_free(build, siteverify, monkeypatch, tmp_path, case):
    monkeypatch.setenv("SIXFINGERS_TURNSTILE_SECRET", TEST_SECRET)
    monkeypatch.setenv("SIXFINGERS_JUDGE_PER_IP_DAY", "1")
    if case == "invalid":
        siteverify.out = {"success": False, "error-codes": ["invalid-input-response"]}
    if case == "timeout":
        siteverify.out = httpx.TimeoutException("slow")
    if case == "hostname":
        siteverify.out = {"success": True, "hostname": "evil.com"}
    fake = FakeClient(JUDGE_JSON)
    c = build(fake)
    r = _post(c, None if case == "missing" else HDR)
    j = r.json()
    assert r.status_code == 200 and j["source"] == "detector_only" and j["judge_unavailable_reason"] == "verification"
    assert not fake.calls and _count(tmp_path) == 0
    siteverify.out = {"success": True, "hostname": "sixfingers.avikwok.dev"}
    assert _post(c, HDR).json()["source"] == "judge"  # the per-IP quota (1) was untouched by the failed check


def test_video_gated(build, siteverify, monkeypatch, clips, tmp_path):  # noqa: F811
    monkeypatch.setenv("SIXFINGERS_TURNSTILE_SECRET", TEST_SECRET)
    data = clips["mp4"].read_bytes()
    fake = FakeClient(JUDGE_JSON)
    c = build(fake)
    j = post_video(c, data).json()
    assert j["source"] == "detector_only" and j["judge_unavailable_reason"] == "verification"
    assert not fake.calls and _count(tmp_path) == 0
    j = c.post("/v1/check-video", files={"video": ("a.mp4", data, "video/mp4")}, headers=HDR).json()
    assert j["judge_unavailable_reason"] != "verification"


@pytest.mark.parametrize("host,ok", [("sixfingers.avikwok.dev", True), ("www.sixfingers.avikwok.dev", True),
                                      ("notsixfingers.avikwok.dev", False), ("example.com", False), (None, False)])
def test_hostname_default(build, siteverify, monkeypatch, host, ok):
    monkeypatch.setenv("SIXFINGERS_TURNSTILE_SECRET", TEST_SECRET)
    siteverify.out = {"success": True, "hostname": host}
    assert (_post(build(FakeClient(JUDGE_JSON)), HDR).json()["source"] == "judge") is ok


def test_hostname_env_and_dev(build, siteverify, monkeypatch):
    monkeypatch.setenv("SIXFINGERS_TURNSTILE_SECRET", TEST_SECRET)
    monkeypatch.setenv("SIXFINGERS_TURNSTILE_HOSTNAMES", "a.test, b.test")
    siteverify.out = {"success": True, "hostname": "x.b.test"}
    assert _post(build(FakeClient(JUDGE_JSON)), HDR).json()["source"] == "judge"
    siteverify.out = {"success": True, "hostname": "example.com"}
    assert _post(build(FakeClient(JUDGE_JSON)), HDR).json()["source"] == "detector_only"
    monkeypatch.setenv("SIXFINGERS_DEV", "1")
    assert _post(build(FakeClient(JUDGE_JSON)), HDR).json()["source"] == "judge"


@requires_ffmpeg
def test_audio_not_gated(client, siteverify, monkeypatch, tmp_path):
    monkeypatch.setenv("SIXFINGERS_TURNSTILE_SECRET", TEST_SECRET)
    p = tone(tmp_path / "a.wav")
    r = post_audio(client, p.read_bytes())  # no token
    assert r.status_code == 200 and not siteverify.calls

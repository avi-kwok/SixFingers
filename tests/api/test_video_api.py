import base64
import io
import json
import tempfile
import threading

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.judge import MODEL
from api.limits import RateLimiter
from api.main import create_app
from ml.common.api_models import VideoJudgeOutput
from ml.common.judge_backend import ApiBackend
from tests.api.conftest import FakeClient, StubEmbedder, command_script, img_bytes
from tests.api.video_clips import make_clip, playlist_text, requires_ffmpeg

pytestmark = requires_ffmpeg

VIDEO_JSON = {"verdict": "likely_ai", "confidence": 0.8, "explanation": "Shapes melt between frames.",
              "cues": [{"frame": 3, "region": "clock", "cue": "digits warp"},
                       {"frame": 99, "region": "sky", "cue": "bad frame number"}],
              "overrode_detector": False}


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips")
    return {"mp4": make_clip(d / "c5.mp4", seconds=5),
            "webm": make_clip(d / "c2.webm", seconds=2, kind="webm"),
            "mov": make_clip(d / "c2.mov", seconds=2)}


def post_video(c, data, name="a.mp4", ctype="video/mp4"):
    return c.post("/v1/check-video", files={"video": (name, data, ctype)})


def test_video_success(build, clips):
    fake = FakeClient(json.dumps(VIDEO_JSON))
    r = post_video(build(fake), clips["mp4"].read_bytes())
    assert r.status_code == 200
    j = r.json()
    assert j["source"] == "judge"
    frames = j["detector"]["frames"]
    assert len(frames) == 16 and [f["index"] for f in frames] == list(range(1, 17))
    times = [f["time_s"] for f in frames]
    assert all(a < b for a, b in zip(times, times[1:]))
    top4 = sorted((f["score"] for f in frames), reverse=True)[:4]
    assert j["detector"]["score"] == pytest.approx(sum(top4) / 4)
    assert len(j["judge"]["cues"]) == 1  # frame 99 does not exist and is dropped
    assert j["judge"]["cues"][0]["time_s"] == frames[2]["time_s"]
    assert j["audio"] == {"analysed": False, "note": "Audio is not analysed yet."}
    assert set(j["timings_ms"]) == {"validate", "frames", "detector", "judge", "total"}
    assert j["model_version"] == "v1"
    assert j["duration_s"] == pytest.approx(5.0, abs=0.1)
    assert j["judge"]["overrode_detector"] == (j["verdict"] != j["detector"]["band"])
    assert j["judge"]["model"] == MODEL

    content = fake.calls[0]["messages"][0]["content"]
    assert [b["type"] for b in content] == ["image", "text"]
    raw = base64.b64decode(content[0]["source"]["data"])
    with Image.open(io.BytesIO(raw)) as grid:
        assert grid.format == "JPEG" and max(grid.size) <= 1568
    text = (fake.calls[0]["system"] + content[1]["text"]).lower()
    for word in ("detector", "score", "probe", "t_ai", "t_real", "backbones", "clip_vitl14"):
        assert word not in text


def test_video_without_judge_is_detector_only(build, clips):
    j = post_video(build(client=None), clips["mp4"].read_bytes()).json()
    assert j["source"] == "detector_only" and j["judge"] is None
    assert j["judge_unavailable_reason"] == "disabled"
    assert j["verdict"] == j["detector"]["band"]


def test_video_refusal_is_422_and_logged(build, clips, caplog):
    with caplog.at_level("INFO", logger="sixfingers"):
        r = post_video(build(FakeClient("", stop_reason="refusal")), clips["mp4"].read_bytes())
    assert r.status_code == 422 and r.json() == {"detail": "We can't analyse this video."}
    ev = [x for x in caplog.records if getattr(x, "event", None) == "refusal"]
    assert len(ev) == 1 and not hasattr(ev[0], "verdict")


def test_video_costs_three_checks(build, clips):
    c = build(rate_limit=5)
    assert post_video(c, clips["mp4"].read_bytes()).status_code == 200
    r = post_video(c, clips["mp4"].read_bytes())
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0
    assert [c.post("/v1/check", files={"image": ("a.png", img_bytes(), "image/png")}).status_code
            for _ in range(3)] == [200, 200, 429]


def test_video_budget_exhausted(build, clips):
    fake = FakeClient(json.dumps(VIDEO_JSON))
    c = build(fake, budget_limit=1)
    data = clips["mp4"].read_bytes()
    assert post_video(c, data).json()["source"] == "judge"
    j = post_video(c, data).json()
    assert j["source"] == "detector_only" and j["judge_unavailable_reason"] == "budget"
    assert len(fake.calls) == 1


def test_video_bad_inputs(build, clips):
    c = build(rate_limit=100)
    assert post_video(c, img_bytes(), "a.png", "image/png").status_code == 415
    assert post_video(c, playlist_text(9).encode(), "a.m3u8", "application/vnd.apple.mpegurl").status_code == 415
    assert post_video(c, b"\x00\x00\x00\x18ftypisom" + b"junk" * 20).status_code == 400
    assert c.post("/v1/check-video").status_code == 400
    long = make_clip(clips["mp4"].parent / "long.mp4", seconds=61, size="160x120", rate=1)
    assert post_video(c, long.read_bytes()).status_code == 413


def test_video_size_cap(build, clips, monkeypatch):
    monkeypatch.setattr("api.video.MAX_BYTES", 1000)
    assert post_video(build(), clips["mp4"].read_bytes()).status_code == 413


def test_large_video_spools_past_image_cap(build, clips):
    noise = make_clip(clips["mp4"].parent / "noise.mp4", seconds=4, size="320x240", kind="noise")
    data = noise.read_bytes()
    assert len(data) > 10 * 1024 * 1024
    assert post_video(build(), data).status_code == 200


def test_webm_and_mov_accepted(build, clips):
    c = build()
    assert post_video(c, clips["webm"].read_bytes(), "a.webm", "video/webm").status_code == 200
    assert post_video(c, clips["mov"].read_bytes(), "a.mov", "video/quicktime").status_code == 200


def test_temp_files_removed(build, clips, tmp_path, monkeypatch):
    d = tmp_path / "t"
    d.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(d))
    c = build()
    assert post_video(c, clips["mp4"].read_bytes()).status_code == 200
    assert list(d.iterdir()) == []
    assert post_video(c, b"\x00\x00\x00\x18ftypisom" + b"junk" * 20).status_code == 400
    assert list(d.iterdir()) == []


def test_video_judge_runs_concurrently(clips, models_dir, tmp_path, monkeypatch):
    started = threading.Event()
    seen = []

    class WaitingEmbedder(StubEmbedder):
        def embed(self, backbone, image):
            seen.append(started.wait(2))  # only true if the judge call began while frames are scored
            return super().embed(backbone, image)

    def ask(self, **kw):
        started.set()
        return VideoJudgeOutput(**VIDEO_JSON), (0, 0)
    monkeypatch.setattr(ApiBackend, "ask", ask)
    app = create_app(embedder=WaitingEmbedder(), anthropic_client=FakeClient(), models_dir=models_dir,
                     budget_path=tmp_path / "b.json")
    assert post_video(TestClient(app), clips["mp4"].read_bytes()).status_code == 200
    assert seen and all(seen)


def test_video_command_backend(build, clips, tmp_path, monkeypatch):
    saved = tmp_path / "request.json"
    body = (f"import sys, json\n"
            f"open({str(saved)!r}, 'w').write(sys.stdin.read())\n"
            f"print(json.dumps({VIDEO_JSON!r}))\n")
    monkeypatch.setenv("SIXFINGERS_JUDGE_BACKEND", "command")
    monkeypatch.setenv("SIXFINGERS_JUDGE_CMD", command_script(tmp_path, body))
    j = post_video(build(), clips["mp4"].read_bytes()).json()
    assert j["source"] == "judge"
    req = json.loads(saved.read_text())
    assert set(req) == {"model", "system", "evidence", "image", "schema", "max_tokens"}
    assert req["image"]["media_type"] == "image/jpeg"


def test_limiter_cost():
    lim = RateLimiter(5, salt="x")
    assert lim.allow("k", 3) == 0
    assert lim.allow("k", 3) > 0
    assert lim.allow("k") == 0
    assert lim.allow("k") == 0
    assert lim.allow("k") > 0
    assert RateLimiter(2, salt="x").allow("k", 3) == 3600


def test_video_busy_returns_503(clips, models_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("SIXFINGERS_MAX_CONCURRENT", "1")
    monkeypatch.setattr("api.main.BUSY_WAIT_S", 0.2)
    entered, release = threading.Event(), threading.Event()

    class Slow(StubEmbedder):
        def embed(self, backbone, image):
            entered.set()
            release.wait(10)
            return super().embed(backbone, image)
    app = create_app(Slow(), None, models_dir=models_dir, budget_path=tmp_path / "budget.json",
                     rate_limit=100, usage_path=tmp_path / "usage.sqlite")
    data = clips["webm"].read_bytes()
    first = []
    t = threading.Thread(target=lambda: first.append(post_video(TestClient(app), data, "a.webm", "video/webm").status_code))
    t.start()
    assert entered.wait(10)
    r = post_video(TestClient(app), data, "a.webm", "video/webm")
    release.set()
    t.join(15)
    assert r.status_code == 503 and "busy" in r.json()["detail"]
    assert first == [200]


def test_video_no_model_never_calls_judge(clips, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    fake = FakeClient(json.dumps(VIDEO_JSON))
    app = create_app(StubEmbedder(), fake, models_dir=empty, budget_path=tmp_path / "budget.json",
                     usage_path=tmp_path / "usage.sqlite")
    assert post_video(TestClient(app), clips["webm"].read_bytes(), "a.webm", "video/webm").status_code == 503
    b = tmp_path / "budget.json"
    assert fake.calls == [] and (not b.exists() or json.loads(b.read_text())["count"] == 0)


def test_video_per_ip_judge_quota(build, clips, monkeypatch):
    monkeypatch.setenv("SIXFINGERS_JUDGE_PER_IP_DAY", "1")
    fake = FakeClient(json.dumps(VIDEO_JSON))
    c = build(fake)
    data = clips["webm"].read_bytes()
    assert post_video(c, data, "a.webm", "video/webm").json()["source"] == "judge"
    j = post_video(c, data, "a.webm", "video/webm").json()
    assert j["source"] == "detector_only" and j["judge_unavailable_reason"] == "budget"
    assert len(fake.calls) == 1

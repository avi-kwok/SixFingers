"""Video and audio checks feed the usage counter and feedback; video gets an overall verdict."""
import sqlite3

import pytest
from fastapi.testclient import TestClient

from api.main import create_app, overall_verdict
from tests.api.conftest import StubEmbedder
from tests.api.test_audio_api import FakeAudioEmbedder, audio_on, make_audio_model, tone, video_with_audio  # noqa: F401
from tests.api.video_clips import make_clip, requires_ffmpeg

pytestmark = requires_ffmpeg


@pytest.fixture
def client(tmp_path, models_dir, audio_on):  # noqa: F811
    make_audio_model(models_dir)
    app = create_app(embedder=StubEmbedder(), anthropic_client=None, models_dir=models_dir,
                     budget_path=tmp_path / "budget.json", audio_embedder=FakeAudioEmbedder(),
                     usage_path=tmp_path / "usage.sqlite")
    return TestClient(app)


def _rows(tmp_path, sql):
    with sqlite3.connect(tmp_path / "usage.sqlite") as con:
        return con.execute(sql).fetchall()


def _fb(c, check_id, **kw):
    return c.post("/v1/feedback", json={"check_id": check_id, "answer": "right", **kw})


def test_audio_check_is_recorded_and_takes_feedback(client, tmp_path):
    r = client.post("/v1/check-audio", files={"audio": ("a.wav", tone(tmp_path / "a.wav").read_bytes(), "audio/wav")})
    assert r.status_code == 200
    cid = r.json()["check_id"]
    assert cid
    assert _rows(tmp_path, "SELECT kind, verdict, n FROM daily_counts") == [("audio", "likely_ai", 1)]
    assert _fb(client, cid, actual="ai").status_code == 200
    assert _fb(client, cid).status_code == 409
    assert _rows(tmp_path, "SELECT kind, answer FROM feedback") == [("audio", "right")]
    assert client.get("/v1/scorecard").status_code in (200, 404)  # usage block is added to the card when one exists


def test_video_check_is_recorded_and_takes_feedback(client, tmp_path):
    clip = video_with_audio(tmp_path / "v.mp4")
    r = client.post("/v1/check-video", files={"video": ("v.mp4", clip.read_bytes(), "video/mp4")})
    assert r.status_code == 200
    j = r.json()
    cid = j["check_id"]
    assert cid
    # the counter stores the overall verdict
    assert _rows(tmp_path, "SELECT kind, verdict, n FROM daily_counts") == [("video", j["overall_verdict"], 1)]
    assert _fb(client, cid, actual="real").status_code == 200
    assert _rows(tmp_path, "SELECT kind, actual FROM feedback") == [("video", "real")]


def test_usage_summary_counts_all_kinds(client, tmp_path):
    client.post("/v1/check-audio", files={"audio": ("a.wav", tone(tmp_path / "a.wav").read_bytes(), "audio/wav")})
    clip = make_clip(tmp_path / "v.mp4", seconds=2)
    client.post("/v1/check-video", files={"video": ("v.mp4", clip.read_bytes(), "video/mp4")})
    assert client.app.state.usage.summary()["checks_total"] == 2


def test_video_overall_matches_rule_with_voice(client, tmp_path):
    clip = video_with_audio(tmp_path / "v.mp4")
    j = client.post("/v1/check-video", files={"video": ("v.mp4", clip.read_bytes(), "video/mp4")}).json()
    assert j["audio"]["analysed"] is True
    assert (j["overall_verdict"], j["overall_flagged_by"]) == overall_verdict(j["verdict"], j["audio"]["band"])


def test_video_without_voice_overall_is_picture(build, tmp_path, monkeypatch):
    monkeypatch.delenv("SIXFINGERS_ENABLE_AUDIO", raising=False)
    clip = make_clip(tmp_path / "v.mp4", seconds=2)
    j = build().post("/v1/check-video", files={"video": ("v.mp4", clip.read_bytes(), "video/mp4")}).json()
    assert j["overall_verdict"] == j["verdict"]
    assert j["overall_flagged_by"] == (["picture"] if j["verdict"] == "likely_ai" else [])
    assert j["check_id"]


@pytest.mark.parametrize("picture,voice,expected,flagged", [
    ("likely_ai", "likely_real", "likely_ai", ["picture"]),
    ("likely_real", "likely_ai", "likely_ai", ["voice"]),
    ("likely_ai", "likely_ai", "likely_ai", ["picture", "voice"]),
    ("likely_real", "likely_real", "likely_real", []),
    ("likely_real", None, "likely_real", []),
    ("uncertain", "likely_real", "uncertain", []),
    ("uncertain", None, "uncertain", []),
    ("likely_ai", None, "likely_ai", ["picture"]),
])
def test_overall_rule(picture, voice, expected, flagged):
    assert overall_verdict(picture, voice) == (expected, flagged)

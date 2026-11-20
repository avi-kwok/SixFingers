"""Audio checks: off by default (no route, video audio note unchanged); on with SIXFINGERS_ENABLE_AUDIO=1.

A fake embedder stands in for XLS-R, so no test needs the GPU or the model download.
"""
import json
import subprocess
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import audio as aud
from api.main import create_app
from ml.common.schemas import ProbeArtifact, save_probe
from tests.api.conftest import StubEmbedder
from tests.api.video_clips import FFMPEG, make_clip, requires_ffmpeg

pytestmark = requires_ffmpeg

LAYER, D = 6, 4
DEFAULT_AUDIO = {"analysed": False, "note": "Audio is not analysed yet."}


class FakeAudioEmbedder:
    """First feature = 20 * (std - 0.05): loud clips look fake, quiet ones real."""
    def __init__(self):
        self.calls = 0

    def embed(self, batch):
        self.calls += 1
        v = 20 * (np.asarray(batch).std(axis=1) - 0.05)
        return {LAYER: np.stack([v, *[np.zeros_like(v)] * (D - 1)], axis=1).astype(np.float32)}


def make_audio_model(models_dir, name="a3"):
    d = models_dir / "audio" / name
    d.mkdir(parents=True)
    save_probe(ProbeArtifact(backbone=f"xlsr300m_L{LAYER}", embed_dim=D, mean=[0.0] * D, std=[1.0] * D,
                             weights=[1.0, 0, 0, 0], bias=0.0), d / "probe.json")
    (d / "meta.json").write_text(json.dumps({"version": name, "layer": LAYER, "threshold_p_fake": 0.5}))


def tone(path, seconds=2.0, volume=4.0, codec=()):  # lavfi sine peaks at 1/8, so 4.0 peaks at 0.5
    subprocess.run([FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", f"sine=frequency=440:duration={seconds}", "-af", f"volume={volume}", *codec, str(path)],
                   check=True)
    return path


def video_with_audio(path, seconds=3):
    subprocess.run([FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=320x240:rate=25",
                    "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True)
    return path


@pytest.fixture
def audio_on(monkeypatch):
    monkeypatch.setenv("SIXFINGERS_ENABLE_AUDIO", "1")


@pytest.fixture
def audio_off(monkeypatch):
    monkeypatch.delenv("SIXFINGERS_ENABLE_AUDIO", raising=False)


@pytest.fixture
def client(tmp_path, models_dir, audio_on):
    make_audio_model(models_dir)
    emb = FakeAudioEmbedder()
    app = create_app(embedder=StubEmbedder(), anthropic_client=None, models_dir=models_dir,
                     budget_path=tmp_path / "budget.json", audio_embedder=emb)
    c = TestClient(app)
    c.embedder = emb
    return c


def post_audio(c, data, name="a.wav", ctype="audio/wav"):
    return c.post("/v1/check-audio", files={"audio": (name, data, ctype)})


def post_video(c, data):
    return c.post("/v1/check-video", files={"video": ("a.mp4", data, "video/mp4")})


# --- flag off ---

def test_flag_off_has_no_route_and_video_audio_unchanged(build, tmp_path, audio_off):
    c = build()
    assert post_audio(c, tone(tmp_path / "t.wav").read_bytes()).status_code == 404
    j = post_video(c, video_with_audio(tmp_path / "v.mp4").read_bytes()).json()
    assert j["audio"] == DEFAULT_AUDIO
    assert "audio are not analysed" in j["disclaimer"]
    assert "audio_enabled" not in c.get("/v1/health").json()


# --- flag on ---

def test_loud_and_quiet_clips(client, tmp_path):
    loud = post_audio(client, tone(tmp_path / "loud.wav", volume=4.0).read_bytes())
    assert loud.status_code == 200
    j = loud.json()
    assert j["verdict"] == j["audio"]["band"] == "likely_ai" and j["source"] == "detector_only"
    a = j["audio"]
    assert a["analysed"] is True and a["p_fake"] >= a["threshold"] == 0.5 and a["model_version"] == "a3"
    assert "first 4 seconds" in a["note"] and "cannot hear audio" in a["note"]
    assert j["duration_s"] == pytest.approx(2.0, abs=0.05)
    assert set(j["timings_ms"]) == {"validate", "detector", "total"}
    quiet = post_audio(client, tone(tmp_path / "quiet.wav", volume=0.05).read_bytes()).json()
    assert quiet["verdict"] == "likely_real" and quiet["audio"]["p_fake"] < 0.5
    assert client.get("/v1/health").json()["audio_enabled"] is True


@pytest.mark.parametrize("ext,codec", [("mp3", ["-c:a", "libmp3lame"]), ("flac", []), ("ogg", ["-c:a", "libopus"]),
                                       ("m4a", ["-c:a", "aac"]), ("webm", ["-c:a", "libopus"])])
def test_formats(client, tmp_path, ext, codec):
    data = tone(tmp_path / f"t.{ext}", codec=codec).read_bytes()
    assert aud.sniff_audio(data[:16]) is not None
    r = post_audio(client, data, name=f"t.{ext}", ctype="application/octet-stream")
    assert r.status_code == 200, r.text
    assert r.json()["verdict"] == "likely_ai"


def test_sniff_mp3_frame_sync_but_not_adts():
    assert aud.sniff_audio(b"\xff\xfb\x90\x64" + b"\x00" * 12) == ("mp3", "mp3")  # MPEG-1 layer III, no ID3 tag
    assert aud.sniff_audio(b"\xff\xf1\x50\x80" + b"\x00" * 12) is None  # ADTS AAC uses layer bits 00


def test_caps_and_bad_input(client, tmp_path):
    assert post_audio(client, b"").status_code == 400
    assert client.post("/v1/check-audio").status_code == 400
    r = post_audio(client, b"RIFF\x00\x00\x00\x00WAVEjunkjunk")
    assert r.status_code == 400 and r.json()["detail"] == "Could not read this audio."
    r = post_audio(client, b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, name="a.png", ctype="image/png")
    assert r.status_code == 415
    r = post_audio(client, tone(tmp_path / "long.wav", seconds=62, volume=4.0).read_bytes())
    assert r.status_code == 413 and "60 seconds" in r.json()["detail"]
    r = post_audio(client, b"RIFF" + b"\x00" * (aud.MAX_BYTES + 200_000))
    assert r.status_code == 413 and "25 MB" in r.json()["detail"]


def test_upload_over_cap_without_length_header(tmp_path):
    with open(tmp_path / "big", "wb") as f:
        f.write(b"\x00" * (aud.MAX_BYTES + 1))
    with open(tmp_path / "big", "rb") as src, pytest.raises(aud.UploadError) as e:
        aud.load(src, tmp_path)
    assert e.value.status == 413 and e.value.detail == "Audio is larger than 25 MB."


def test_video_with_and_without_audio_track(client, tmp_path):
    j = post_video(client, video_with_audio(tmp_path / "v.mp4").read_bytes()).json()
    assert j["audio"]["analysed"] is True and j["audio"]["band"] == "likely_ai"
    assert "separate prototype" in j["disclaimer"]
    j = post_video(client, make_clip(tmp_path / "silent.mp4", seconds=2).read_bytes()).json()
    assert j["audio"] == {"analysed": False, "note": "This video has no audio track."}


def test_missing_audio_model_is_503_not_a_crash(tmp_path, models_dir, audio_on):
    c = TestClient(create_app(embedder=StubEmbedder(), anthropic_client=None, models_dir=models_dir,
                              budget_path=tmp_path / "budget.json"))
    assert post_audio(c, tone(tmp_path / "t.wav").read_bytes()).status_code == 503
    assert c.get("/v1/health").json()["audio_enabled"] is False


def test_xlsr_loads_once_on_first_use(tmp_path, monkeypatch):
    make_audio_model(tmp_path)
    made = []

    def fake_xlsr(layers):
        made.append(layers)
        return FakeAudioEmbedder()

    monkeypatch.setattr(aud.features, "XLSR", fake_xlsr)
    scorer = aud.AudioScorer(tmp_path)
    assert made == []  # nothing loaded at start-up
    x = np.full(16_000, 0.3, dtype=np.float32)
    threads = [threading.Thread(target=scorer.score, args=(x,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [sorted(m) for m in made] == [[LAYER]]


# --- speech gate: no voice verdict unless the scored 4 s sounds like speech ---

def add_gate(models_dir, threshold=0.0, name="a3"):
    """Fake gate on the same fake feature: loud clips count as speech, quiet ones do not."""
    probe = ProbeArtifact(backbone=f"xlsr300m_L{LAYER}", embed_dim=D, mean=[0.0] * D, std=[1.0] * D,
                          weights=[1.0, 0, 0, 0], bias=0.0)
    (models_dir / "audio" / name / "gate.json").write_text(
        json.dumps({"probe": probe.model_dump(), "layer": LAYER, "threshold_logit": threshold}))


@pytest.fixture
def gated(tmp_path, models_dir, audio_on):
    make_audio_model(models_dir)
    add_gate(models_dir)
    return TestClient(create_app(embedder=StubEmbedder(), anthropic_client=None, models_dir=models_dir,
                                 budget_path=tmp_path / "budget.json", audio_embedder=FakeAudioEmbedder()))


def test_gate_passes_speech_like_and_blocks_other_audio(gated, tmp_path):
    ok = post_audio(gated, tone(tmp_path / "loud.wav", volume=4.0).read_bytes())
    assert ok.status_code == 200 and ok.json()["audio"]["analysed"] is True
    no = post_audio(gated, tone(tmp_path / "quiet.wav", volume=0.05).read_bytes())
    assert no.status_code == 422 and no.json()["detail"] == aud.NO_SPEECH


def test_gate_off_scores_everything_as_before(client, tmp_path):  # no gate.json in the model folder
    r = post_audio(client, tone(tmp_path / "quiet.wav", volume=0.05).read_bytes())
    assert r.status_code == 200 and r.json()["verdict"] == "likely_real"


def test_video_without_speech_gets_a_note_and_the_picture_decides(gated, tmp_path):
    j = post_video(gated, video_with_audio(tmp_path / "v.mp4").read_bytes()).json()  # a sine is loud: counts as speech here
    assert j["audio"]["analysed"] is True
    quiet = tmp_path / "q.mp4"
    subprocess.run([FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=25",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-af", "volume=0.05",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(quiet)], check=True)
    q = post_video(gated, quiet.read_bytes()).json()
    assert q["audio"] == {"analysed": False, "note": aud.NO_SPEECH}
    assert q["overall_verdict"] == q["verdict"] and "voice" not in q["overall_flagged_by"]


def test_gate_decision_uses_the_threshold(tmp_path):
    make_audio_model(tmp_path)
    add_gate(tmp_path, threshold=100.0)  # nothing reaches it
    s = aud.AudioScorer(tmp_path, FakeAudioEmbedder())
    assert s.score(np.full(16_000, 0.3, dtype=np.float32)) == aud.AudioInfo(note=aud.NO_SPEECH)

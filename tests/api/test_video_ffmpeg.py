import io
import socket
import subprocess

import pytest
from PIL import Image

from api import video
from api.validate import UploadError
from api.video import Clip, binaries, extract_frames, frame_grid, load, probe, sniff_video, top_k_mean
from tests.api.video_clips import FFMPEG, make_clip, playlist_text, requires_ffmpeg

pytestmark = requires_ffmpeg


def load_path(path, work):
    work.mkdir(exist_ok=True)
    with open(path, "rb") as f:
        return load(f, work)


def png_bytes():
    b = io.BytesIO()
    Image.new("RGB", (4, 4)).save(b, "PNG")
    return b.getvalue()


def jpeg_bytes():
    b = io.BytesIO()
    Image.new("RGB", (4, 4)).save(b, "JPEG")
    return b.getvalue()


def test_sniff_video(tmp_path):
    mp4 = make_clip(tmp_path / "a.mp4", seconds=1)
    assert sniff_video(mp4.read_bytes()[:16]) == ("mp4", "mov")
    mov = make_clip(tmp_path / "a.mov", seconds=1, kind="mov")
    assert sniff_video(mov.read_bytes()[:16]) == ("mov", "mov")
    webm = make_clip(tmp_path / "a.webm", seconds=1, kind="webm")
    assert sniff_video(webm.read_bytes()[:16]) == ("webm", "matroska")
    assert sniff_video(png_bytes()) is None
    assert sniff_video(jpeg_bytes()) is None
    assert sniff_video(b"#EXTM3U\n#EXT-X-TARGETDURATION:1\n") is None
    assert sniff_video(b"") is None


def test_load_and_sample_mp4(tmp_path):
    clip = load_path(make_clip(tmp_path / "c.mp4", seconds=5), tmp_path / "work")
    assert abs(clip.duration - 5.0) < 0.1
    frames = extract_frames(clip, tmp_path / "work")
    assert len(frames) == 16
    times = [t for t, _ in frames]
    assert times == sorted(set(times))
    assert all(0 < t < 5 for t in times)
    for _, im in frames:
        assert im.mode == "RGB" and im.size == (320, 240)


@pytest.mark.parametrize("kind", ["webm", "mov"])
def test_short_webm_and_mov_sample(tmp_path, kind):
    clip = load_path(make_clip(tmp_path / f"b.{kind}", seconds=2, kind=kind), tmp_path / "work")
    assert len(extract_frames(clip, tmp_path / "work")) >= 8


def test_very_short_clip_drops_duplicate_frames(tmp_path):
    clip = load_path(make_clip(tmp_path / "s.mp4", seconds=0.2, rate=25), tmp_path / "work")
    frames = extract_frames(clip, tmp_path / "work")
    assert 0 < len(frames) < 16
    times = [t for t, _ in frames]
    assert times == sorted(set(times))
    assert all(a[1].tobytes() != b[1].tobytes() for a, b in zip(frames, frames[1:]))


def test_over_60_seconds_is_413(tmp_path):
    p = make_clip(tmp_path / "long.mp4", seconds=61, size="160x120", rate=1)
    with pytest.raises(UploadError) as e:
        load_path(p, tmp_path / "work")
    assert e.value.status == 413


def test_load_errors(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(UploadError) as e:
        load(io.BytesIO(png_bytes()), work)
    assert e.value.status == 415
    with pytest.raises(UploadError) as e:
        load(io.BytesIO(b"\x00\x00\x00\x18ftypisom" + b"junk" * 500), work)
    assert e.value.status == 400
    with pytest.raises(UploadError) as e:
        load(io.BytesIO(b""), work)
    assert e.value.status == 400
    monkeypatch.setattr("api.video.MAX_BYTES", 1000)
    with pytest.raises(UploadError) as e:
        load_path(make_clip(tmp_path / "big.mp4", seconds=5), work)
    assert e.value.status == 413


@pytest.mark.parametrize("name, text, demuxer", [
    ("hls.m3u8", None, "hls"),
    ("concat.txt", "ffconcat version 1.0\nfile 'http://127.0.0.1:{port}/x.mp4'\n", "concat"),
])
def test_playlist_is_never_fetched(tmp_path, name, text, demuxer):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    srv.settimeout(1.0)
    try:
        port = srv.getsockname()[1]
        pl = tmp_path / name
        pl.write_text(playlist_text(port) if text is None else text.format(port=port))
        assert sniff_video(pl.read_bytes()[:16]) is None
        with pytest.raises(UploadError):
            probe(pl, demuxer)
        with pytest.raises(UploadError):
            extract_frames(Clip(pl, "mp4", demuxer, 4.0, 320, 240), tmp_path)
        with pytest.raises(socket.timeout):
            srv.accept()  # ffmpeg must never have connected
    finally:
        srv.close()


def _frames(n, size):
    return [(i * 0.5, Image.new("RGB", size, (i * 15 % 255, 80, 160))) for i in range(n)]


def _grid_size(jpg):
    assert jpg[:2] == b"\xff\xd8"
    return Image.open(io.BytesIO(jpg)).size


def test_frame_grid_sizes():
    w, h = _grid_size(frame_grid(_frames(16, (320, 240))))
    assert max(w, h) <= 1568
    w, h = _grid_size(frame_grid(_frames(16, (1080, 1920))))
    assert w <= 1568 and h <= 1568
    assert _grid_size(frame_grid(_frames(3, (320, 240))))[0] > 0


def test_top_k_mean():
    assert top_k_mean([0.1, 0.9, 0.8, 0.7, 0.6, 0.2]) == pytest.approx(0.75)
    assert top_k_mean([0.5, 0.7]) == pytest.approx(0.6)


def test_binaries_missing_is_503(monkeypatch):
    monkeypatch.delenv("SIXFINGERS_FFMPEG", raising=False)
    monkeypatch.delenv("SIXFINGERS_FFPROBE", raising=False)
    monkeypatch.setattr("api.video.shutil.which", lambda name: None)
    with pytest.raises(UploadError) as e:
        binaries()
    assert e.value.status == 503


def test_binaries_env_path_must_exist(monkeypatch, tmp_path):
    monkeypatch.setenv("SIXFINGERS_FFMPEG", str(tmp_path / "missing"))
    with pytest.raises(UploadError) as e:
        binaries()
    assert e.value.status == 503


def test_probe_timeout_is_400(tmp_path, monkeypatch):
    p = tmp_path / "x.mp4"
    p.write_bytes(b"x")

    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="ffprobe", timeout=30)

    monkeypatch.setattr(video.subprocess, "run", boom)
    with pytest.raises(UploadError) as e:
        probe(p, "mov")
    assert e.value.status == 400


def test_deadline_mid_seek_keeps_frames(tmp_path, monkeypatch):
    clip = load_path(make_clip(tmp_path / "d.mp4", seconds=5), tmp_path / "work")
    real_run = subprocess.run
    calls = []

    def run(*a, **k):
        calls.append(1)
        if len(calls) > 3:
            raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1)
        return real_run(*a, **k)

    monkeypatch.setattr(video.subprocess, "run", run)
    frames = extract_frames(clip, tmp_path / "work")
    assert len(frames) == 3


def test_deadline_on_first_seek_is_400(tmp_path, monkeypatch):
    clip = load_path(make_clip(tmp_path / "e.mp4", seconds=5), tmp_path / "work")

    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1)

    monkeypatch.setattr(video.subprocess, "run", boom)
    with pytest.raises(UploadError) as e:
        extract_frames(clip, tmp_path / "work")
    assert e.value.status == 400


def _cut_clip(path, seconds=8, size="320x240", rate=25):
    half = seconds / 2
    subprocess.run([FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", f"testsrc=s={size}:d={half}:r={rate}",
                    "-f", "lavfi", "-i", f"testsrc2=s={size}:d={half}:r={rate},negate",
                    "-filter_complex", "[0][1]concat=n=2:v=1:a=0,format=yuv420p",
                    "-c:v", "libx264", str(path)], check=True)
    return path


def test_scene_cut_adds_a_frame(tmp_path):
    # 40 s clip with a cut at 20.0 s: slices are 2.5 s, so the nearest even frames (18.75, 21.25) are far away
    clip = load_path(_cut_clip(tmp_path / "cut.mp4", seconds=40, size="160x120", rate=5), tmp_path / "work")
    frames = extract_frames(clip, tmp_path / "work")
    times = [t for t, _ in frames]
    assert len(frames) == 17 and times == sorted(times)
    assert any(abs(t - 20.0) < 0.3 for t in times if t not in video.sample_times(clip.duration))


def test_scene_cut_near_even_frame_is_skipped(tmp_path):
    clip = load_path(_cut_clip(tmp_path / "cut2.mp4", seconds=8), tmp_path / "work")  # cut at 4.0, evens at 3.75/4.25
    assert len(extract_frames(clip, tmp_path / "work")) == 16


def test_scene_pass_failure_falls_back_to_even_frames(tmp_path, monkeypatch):
    clip = load_path(_cut_clip(tmp_path / "cut3.mp4", seconds=6), tmp_path / "work")
    monkeypatch.setattr(video, "scene_cuts", lambda *a, **k: [])
    assert len(extract_frames(clip, tmp_path / "work")) == 16


def test_frame_grid_fits_twenty_frames():
    w, h = _grid_size(frame_grid(_frames(20, (1080, 1920))))
    assert w <= 1568 and h <= 1568

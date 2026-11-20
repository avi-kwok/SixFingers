"""The batch frame sampler must return the same frames as the API's sampler."""
import subprocess

from api import video as vid
from ml.video import extract
from tests.api.video_clips import FFMPEG, make_clip, requires_ffmpeg

pytestmark = requires_ffmpeg


def test_fast_frames_equal_api_frames(tmp_path):
    src = make_clip(tmp_path / "c.mp4", seconds=3)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    with open(src, "rb") as f:
        clip = vid.load(f, tmp_path / "a")
    api = vid.extract_frames(clip, tmp_path / "a")
    fast = extract.fast_frames(clip, tmp_path / "b")
    assert len(api) == len(fast) == vid.N_FRAMES
    assert all(a.tobytes() == b.tobytes() for (_, a), (_, b) in zip(api, fast))


def test_fast_frames_equal_api_frames_with_a_scene_cut(tmp_path):
    # 18 s clip with a hard cut at 9.0 s: 0.5625 s from the nearest even frame, so the API adds a cut frame
    src = tmp_path / "cut.mp4"
    subprocess.run([FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-filter_complex",
                    "testsrc=d=9:s=320x240:r=25[a];testsrc2=d=9:s=320x240:r=25[b];[a][b]concat=n=2:v=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    with open(src, "rb") as f:
        clip = vid.load(f, tmp_path / "a")
    api = vid.extract_frames(clip, tmp_path / "a")
    fast = extract.fast_frames(clip, tmp_path / "b")
    assert len(api) == len(fast) == vid.N_FRAMES + 1
    assert all(abs(a - b) <= 1 / 25 for (a, _), (b, _) in zip(api, fast))  # fast reports the decoded frame time
    assert all(a.tobytes() == b.tobytes() for (_, a), (_, b) in zip(api, fast))

"""Synthetic ffmpeg clips shared by the video tests."""
import shutil
import subprocess
from pathlib import Path

import pytest

FFMPEG = shutil.which("ffmpeg")
requires_ffmpeg = pytest.mark.skipif(FFMPEG is None or shutil.which("ffprobe") is None,
                                     reason="ffmpeg not installed")


def make_clip(path: Path, seconds: float = 5, size: str = "320x240", rate: int = 25, kind: str = "mp4") -> Path:
    if kind == "noise":
        src = f"nullsrc=s={size}:d={seconds}:r={rate},geq=random(1)*255:128:128"
        codec = ["-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv420p"]
    elif kind == "webm":
        src = f"testsrc=duration={seconds}:size={size}:rate={rate}"
        codec = ["-c:v", "libvpx-vp9", "-b:v", "200k"]
    else:  # mp4 or mov; the extension selects the muxer
        src = f"testsrc=duration={seconds}:size={size}:rate={rate}"
        codec = ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    subprocess.run([FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", src, *codec, str(path)], check=True)
    return path


def playlist_text(port: int) -> str:
    return (f"#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\n"
            f"http://127.0.0.1:{port}/seg.ts\n#EXT-X-ENDLIST\n")

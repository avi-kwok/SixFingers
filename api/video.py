"""Video input: byte sniffing, size and length checks, and frame sampling with ffmpeg.

ffmpeg only reads a local file the caller wrote into its own temporary folder.
"""
import io
import json
import logging
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from math import ceil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from api.validate import MAX_PIXELS, UploadError

log = logging.getLogger("sixfingers")

MAX_BYTES = 100 * 1024 * 1024
MAX_SECONDS = 60
N_FRAMES = 16
MAX_EXTRA = 4  # scene-cut frames added on top of the even ones
MAX_FRAMES = 20
CUT_GAP_S = 0.5  # a cut this close to an existing frame adds nothing
SCENE_THRESHOLD = 0.4
TILE = 392
GRID_EDGE = 1568
TIMEOUT_S = 30
SAFE = ["-protocol_whitelist", "file,pipe"]
SCALE = "scale='min(1280,iw)':'min(1280,ih)':force_original_aspect_ratio=decrease"
UNREADABLE = "Could not read this video."
GRID_COLS, GRID_PER_GRID = 4, 8
HEADER_H, LABEL_H, GAP = 28, 24, 8


@dataclass
class Clip:
    path: Path
    format: str  # mp4|mov|webm
    demuxer: str  # mov|matroska
    duration: float
    width: int
    height: int
    has_audio: bool = False


def sniff_video(head: bytes) -> tuple[str, str] | None:
    """Return (format, demuxer) from the container magic bytes, or None."""
    if head[4:8] == b"ftyp":
        return ("mov" if head[8:12] == b"qt  " else "mp4"), "mov"
    if head[4:8] in (b"moov", b"mdat", b"wide", b"free"):
        return "mov", "mov"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "webm", "matroska"
    return None


def save_upload(src, dst: Path, max_bytes: int | None = None, too_big: str = "Video is larger than 100 MB.") -> bytes:
    """Copy the upload to dst in 1 MiB chunks and return its first 16 bytes."""
    limit = MAX_BYTES if max_bytes is None else max_bytes
    total, head = 0, b""
    with open(dst, "wb") as out:
        while chunk := src.read(1 << 20):
            total += len(chunk)
            if total > limit:
                raise UploadError(413, too_big)
            if len(head) < 16:
                head += chunk[:16 - len(head)]
            out.write(chunk)
    if total == 0:
        raise UploadError(400, "Empty upload.")
    return head


def binaries() -> tuple[str, str]:
    ffmpeg = os.environ.get("SIXFINGERS_FFMPEG") or shutil.which("ffmpeg")
    ffprobe = os.environ.get("SIXFINGERS_FFPROBE") or shutil.which("ffprobe")
    if not (ffmpeg and ffprobe and os.path.isfile(ffmpeg) and os.path.isfile(ffprobe)):
        raise UploadError(503, "Video checks are not available right now.")
    return ffmpeg, ffprobe


def _run(argv: list[str], timeout: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        raise UploadError(400, UNREADABLE) from None


def probe(path: Path, demuxer: str) -> tuple[float, int, int, bool]:
    _, ffprobe = binaries()
    argv = [ffprobe, "-hide_banner", "-v", "error", *SAFE, "-f", demuxer,
            "-show_entries", "format=duration:stream=codec_type,width,height,duration",
            "-of", "json", f"file:{path}"]
    p = _run(argv, TIMEOUT_S)
    if p.returncode != 0:
        log.info("ffprobe exited with %s", p.returncode)
        raise UploadError(400, UNREADABLE)
    try:
        info = json.loads(p.stdout)
        video = next(s for s in info["streams"] if s.get("codec_type") == "video")
        duration = float(info["format"].get("duration") or video.get("duration"))
        w, h = int(video["width"]), int(video["height"])
        has_audio = any(s.get("codec_type") == "audio" for s in info["streams"])
    except (StopIteration, KeyError, TypeError, ValueError, AttributeError):
        raise UploadError(400, UNREADABLE) from None
    if not math.isfinite(duration) or duration <= 0 or w <= 0 or h <= 0:
        raise UploadError(400, UNREADABLE)
    return duration, w, h, has_audio


def load(src, tmp: Path) -> Clip:
    path = tmp / "input"
    head = save_upload(src, path)
    kind = sniff_video(head)
    if kind is None:
        raise UploadError(415, "Only MP4, WebM and MOV videos are supported.")
    duration, w, h, has_audio = probe(path, kind[1])
    if duration > MAX_SECONDS + 0.5:  # 0.5 s grace for container rounding
        raise UploadError(413, "Video is longer than 60 seconds.")
    if w * h > MAX_PIXELS:
        raise UploadError(413, "Video frames are too large (limit 40 megapixels).")
    return Clip(path, kind[0], kind[1], duration, w, h, has_audio)


def sample_times(duration: float, n: int = N_FRAMES) -> list[float]:
    """Centres of n equal segments: avoids black first frames and end-of-file seeks."""
    return [round((k + 0.5) * duration / n, 3) for k in range(n)]


def _grab(clip: Clip, ffmpeg: str, t: float, dst: Path, timeout: float) -> Image.Image | None:
    # The demuxer is forced from the sniffed bytes and only file/pipe protocols are allowed,
    # so a crafted file can never be read as a playlist (HLS, concat) or make ffmpeg fetch a URL.
    argv = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "2", *SAFE,
            "-f", clip.demuxer, "-ss", f"{t:.3f}", "-i", f"file:{clip.path}",
            "-map", "0:v:0", "-an", "-sn", "-dn", "-frames:v", "1", "-vf", SCALE, "-y", str(dst)]
    p = _run(argv, timeout)
    if p.returncode != 0 or not dst.is_file():
        return None
    with Image.open(dst) as im:
        return im.convert("RGB")


def scene_cuts(clip: Clip, ffmpeg: str, timeout: float) -> list[float]:
    """Timestamps of hard cuts from one decoding pass; [] if the pass fails or times out."""
    argv = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "info", "-threads", "2", *SAFE,
            "-f", clip.demuxer, "-i", f"file:{clip.path}", "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vf", f"scale=320:-2,select='gt(scene,{SCENE_THRESHOLD})',showinfo", "-f", "null", "-"]
    try:
        p = _run(argv, timeout)
    except UploadError:
        return []
    if p.returncode != 0:
        return []
    cuts = []
    for m in re.finditer(rb"pts_time:([0-9.]+)", p.stderr):
        t = float(m.group(1))
        if 0 < t < clip.duration:
            cuts.append(round(t, 3))
    return cuts


def extract_frames(clip: Clip, tmp: Path) -> list[tuple[float, Image.Image]]:
    ffmpeg, _ = binaries()
    deadline = time.monotonic() + TIMEOUT_S
    frames: list[tuple[float, Image.Image]] = []
    for k, t in enumerate(sample_times(clip.duration)):
        left = deadline - time.monotonic()
        if left <= 0:
            break
        try:
            frame = _grab(clip, ffmpeg, t, tmp / f"frame_{k:02d}.png", min(TIMEOUT_S, left))
        except UploadError:
            break  # deadline ran out mid-seek: keep the frames already taken
        if frame is None:
            continue
        if frames and frame.tobytes() == frames[-1][1].tobytes():
            continue  # short clips seek to the same frame twice
        frames.append((t, frame))
    if not frames:
        raise UploadError(400, UNREADABLE)
    left = deadline - time.monotonic()
    if left > 0:
        added = 0
        # shortcut: first cuts in time order win, not the strongest; fine for clips under 60 s
        for t in scene_cuts(clip, ffmpeg, left):
            if added >= MAX_EXTRA or len(frames) >= MAX_FRAMES:
                break
            if any(abs(t - ft) < CUT_GAP_S for ft, _ in frames):
                continue
            left = deadline - time.monotonic()
            if left <= 0:
                break
            try:
                frame = _grab(clip, ffmpeg, t, tmp / f"cut_{added:02d}.png", min(TIMEOUT_S, left))
            except UploadError:
                break
            if frame is not None:
                frames.append((t, frame))
                added += 1
        frames.sort(key=lambda f: f[0])
    return frames


def frame_grid(frames: list[tuple[float, Image.Image]]) -> bytes:
    """One JPEG with the frames as grids of 8 (labelled #index and timestamp), at most GRID_EDGE px a side."""
    font = ImageFont.load_default(size=18)
    tiles = []
    for t, im in frames:
        tile = im.copy()
        tile.thumbnail((TILE, TILE))
        tiles.append((t, tile))
    cell_w = max(im.width for _, im in tiles)
    tile_h = max(im.height for _, im in tiles)
    cell_h = tile_h + LABEL_H
    groups = [tiles[i:i + GRID_PER_GRID] for i in range(0, len(tiles), GRID_PER_GRID)]
    heights = [HEADER_H + ceil(len(g) / GRID_COLS) * cell_h for g in groups]
    canvas = Image.new("RGB", (GRID_COLS * cell_w, sum(heights) + GAP * (len(groups) - 1)), "white")
    draw = ImageDraw.Draw(canvas)
    y, idx = 0, 0
    for g_i, group in enumerate(groups):
        first = g_i * GRID_PER_GRID + 1
        draw.text((6, y + 4), f"Frames {first}-{first + len(group) - 1}", fill="black", font=font)
        y += HEADER_H
        for j, (t, im) in enumerate(group):
            idx += 1
            x0, y0 = (j % GRID_COLS) * cell_w, y + (j // GRID_COLS) * cell_h
            canvas.paste(im, (x0 + (cell_w - im.width) // 2, y0 + (tile_h - im.height) // 2))
            draw.rectangle([x0, y0 + tile_h, x0 + cell_w - 1, y0 + cell_h - 1], fill="black")
            draw.text((x0 + 4, y0 + tile_h + 3), f"#{idx}  {t:.1f} s", fill="white", font=font)
        y += ceil(len(group) / GRID_COLS) * cell_h + GAP
    canvas.thumbnail((GRID_EDGE, GRID_EDGE))
    out = io.BytesIO()
    canvas.save(out, "JPEG", quality=90)
    return out.getvalue()


def top_k_mean(scores: list[float], k: int = 4) -> float:
    top = sorted(scores, reverse=True)[:k]
    return sum(top) / len(top)

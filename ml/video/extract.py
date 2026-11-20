"""Frames and embeddings for every downloaded AIGVDBench video, sampled exactly like the API.

Two steps, so the slow ffmpeg work does not hold the GPU:

    python -m ml.video.extract frames            # CPU: ffmpeg frames -> normalized JPEGs, one .npz per video
    python -m ml.video.extract embed             # GPU: CLIP ViT-L/14 + SigLIP (fp16 on MPS) per frame

Videos are opened with api.video.load (the same 100 MB / 60 s / 40 MP limits). fast_frames picks the
same frames as api.video.extract_frames (16 even frames plus up to 4 scene cuts) in fewer decoding
passes and without the 30 s budget. Each frame then gets the training normalizer (512 px centre crop,
JPEG q95), as FeatureEmbedder does in the API. Shards hold 100 videos in a fixed random order.
A video the API would refuse gets a .err file with the reason instead of an .npz.
"""
from __future__ import annotations

import argparse
import io
import os
import random
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from api import video as vid
from api.detector import VIDEO_BACKBONES
from api.validate import UploadError
from ml.common.paths import features_dir
from ml.data.normalize import encode, normalize_image
from ml.video import data

SHARD = 100


def frames_path(row: dict) -> Path:
    return features_dir() / "video" / "frames" / (row["path"] + ".npz")


def emb_dir() -> Path:
    return features_dir() / "video" / "emb"


def pack(jpgs: list[bytes], times: list[float]) -> dict:
    """Variable-length JPEG bytes -> flat uint8 array + offsets (no pickle needed to load)."""
    off = np.cumsum([0] + [len(j) for j in jpgs]).astype(np.int64)
    return {"jpg": np.frombuffer(b"".join(jpgs), dtype=np.uint8), "off": off, "t": np.asarray(times, np.float32)}


def unpack(z) -> list[Image.Image]:
    jpg, off = z["jpg"], z["off"]
    return [Image.open(io.BytesIO(jpg[off[i]:off[i + 1]].tobytes())).convert("RGB") for i in range(len(off) - 1)]


def pick_times(duration: float, cuts: list[float]) -> list[float]:
    """The API's timestamps: 16 segment centres, then the first cuts not within CUT_GAP_S of a frame."""
    times = vid.sample_times(duration)
    extra = []
    for t in cuts:
        if len(extra) >= vid.MAX_EXTRA or len(times) + len(extra) >= vid.MAX_FRAMES:
            break
        if all(abs(t - ft) >= vid.CUT_GAP_S for ft in times + extra):
            extra.append(t)
    return sorted(times + extra)


def select_expr(times: list[float]) -> str:
    """ffmpeg select: keep the first frame at or after each time, as an accurate `-ss t` seek does."""
    one = "if(isnan(prev_t),gte(t,{0}),gte(t,{0})*lt(prev_t,{0}))"
    return "gt(" + "+".join(one.format(f"{t:.3f}") for t in times) + ",0)"


HW = ["-hwaccel", "videotoolbox"]  # Mac media engine; ffmpeg falls back to software decoding where it cannot


def decode_args(clip: vid.Clip) -> tuple[list[str], str]:
    """(input flags, filter prefix). Hardware decoding only for plain yuv420p video: its frames come back as
    nv12, and converting them to yuv420p first gives the same pixels as software decoding (checked with md5)."""
    _, ffprobe = vid.binaries()
    p = vid._run([ffprobe, "-v", "error", *vid.SAFE, "-f", clip.demuxer, "-select_streams", "v:0",
                  "-show_entries", "stream=pix_fmt", "-of", "csv=p=0", f"file:{clip.path}"], 60)
    if p.returncode == 0 and p.stdout.strip() == b"yuv420p":
        return HW, "format=yuv420p,"
    return [], ""


def scene_cuts(clip: vid.Clip, ffmpeg: str) -> list[float]:
    """api.video.scene_cuts with hardware decoding and no time budget.
    Videos shorter than 16 s skip it: every moment is then within 0.5 s of an even frame, so no cut is ever added."""
    if clip.duration / vid.N_FRAMES < 2 * vid.CUT_GAP_S:
        return []
    hw, fmt = decode_args(clip)
    argv = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "info", *hw, "-threads", "2", *vid.SAFE,
            "-f", clip.demuxer, "-i", f"file:{clip.path}", "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vf", f"{fmt}scale=320:-2,select='gt(scene,{vid.SCENE_THRESHOLD})',showinfo", "-f", "null", "-"]
    p = vid._run(argv, 600)
    if p.returncode != 0:
        return []
    ts = (round(float(m.group(1)), 3) for m in re.finditer(rb"pts_time:([0-9.]+)", p.stderr))
    return [t for t in ts if 0 < t < clip.duration]


def fast_frames(clip: vid.Clip, tmp: Path) -> list[tuple[float, Image.Image]]:
    """Same frames as api.video.extract_frames, from at most 2 decoding passes instead of up to 21 seeks.

    The API seeks once per frame, which decodes from the previous keyframe every time; that is fine
    for one upload but too slow for thousands of videos. Here pass 1 finds scene cuts (as the API
    does) and pass 2 decodes the video once and keeps the frames at the chosen times.
    Frames are written as BMP (lossless, like the API's PNG, but faster to write).
    Identical consecutive frames are dropped, as in the API. There is no 30 s budget here.
    """
    ffmpeg, _ = vid.binaries()
    times = pick_times(clip.duration, scene_cuts(clip, ffmpeg))
    hw, fmt = decode_args(clip)
    argv = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "info", *hw, "-threads", "2", *vid.SAFE,
            "-f", clip.demuxer, "-i", f"file:{clip.path}", "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vf", f"select='{select_expr(times)}',{fmt}{vid.SCALE},showinfo", "-fps_mode", "passthrough",
            "-y", str(tmp / "f_%03d.bmp")]
    p = vid._run(argv, 600)
    files = sorted(tmp.glob("f_*.bmp"))
    if p.returncode != 0 or not files:
        raise UploadError(400, vid.UNREADABLE)
    got = [float(m.group(1)) for m in re.finditer(rb"pts_time:([0-9.]+)", p.stderr)]
    frames: list[tuple[float, Image.Image]] = []
    for i, f in enumerate(files):
        with Image.open(f) as im:
            im = im.convert("RGB")
        if frames and im.tobytes() == frames[-1][1].tobytes():
            continue
        frames.append((got[i] if i < len(got) else float("nan"), im))
    return frames


def frames_one(row: dict) -> str:
    dst = frames_path(row)
    err = dst.with_suffix(".err")
    if dst.exists() or err.exists():
        return "cached"
    src = data.root() / row["path"]
    if not src.is_file():
        return "missing"  # not downloaded (yet); retried on the next run
    try:
        with tempfile.TemporaryDirectory(prefix="sf-vframes-") as tmp, open(src, "rb") as f:
            clip = vid.load(f, Path(tmp))
            frames = fast_frames(clip, Path(tmp))
            jpgs = [encode(normalize_image(im)) for _, im in frames]
    except UploadError as e:
        dst.parent.mkdir(parents=True, exist_ok=True)
        err.write_text(f"{e.status}: {e.detail}\n")
        return "refused"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmpf = dst.with_name(dst.stem + ".tmp.npz")
    np.savez(tmpf, **pack(jpgs, [t for t, _ in frames]))
    tmpf.replace(dst)
    return "ok"


def run_frames(rows: list[dict], workers: int) -> None:
    counts: dict[str, int] = {}
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:  # ffmpeg runs in subprocesses, so threads are enough
        for i, s in enumerate(ex.map(frames_one, rows), 1):
            counts[s] = counts.get(s, 0) + 1
            if i % 200 == 0 or i == len(rows):
                print(f"{i}/{len(rows)} {counts} {i / (time.time() - t0):.1f} videos/s", flush=True)


def ordered(rows: list[dict], split: str) -> list[dict]:
    """A split's rows in a fixed random order, so every finished prefix is a balanced random subset."""
    rows = sorted((r for r in rows if r["split"] == split), key=lambda r: r["path"])
    random.Random(0).shuffle(rows)
    return rows


def shard_rows(rows: list[dict], split: str) -> list[list[dict]]:
    rows = ordered(rows, split)
    return [rows[i:i + SHARD] for i in range(0, len(rows), SHARD)]


def run_embed(rows: list[dict], splits: list[str], batch: int, shards: int | None = None, device: str | None = None,
              first: int = 0) -> None:
    """device None: MPS when available (fp16), as the API on a Mac. "cpu": fp32, as the API on a CPU server."""
    from ml.features.backbones import load_backbone
    # the API's video setup (FeatureEmbedder(half=True)); frames were normalized in the frames step
    models = {b: load_backbone(b, device=device, half=True) for b in VIDEO_BACKBONES}
    dev = next(iter(models.values())).device
    for split in splits:
        for k, shard in list(enumerate(shard_rows(rows, split)))[first:shards]:
            out = emb_dir() / f"{split}_{k:03d}.npz"
            if out.exists():  # another worker may have written it; checked again before saving
                continue
            pending = [r for r in shard if not frames_path(r).exists() and not frames_path(r).with_suffix(".err").exists()]
            if pending:
                print(f"{out.name}: {len(pending)} videos still have no frames; next split", flush=True)
                break
            t0 = time.time()
            keep, nf, times, feats = [], [], [], {b: [] for b in models}
            for r in shard:
                p = frames_path(r)
                if not p.exists():
                    continue
                with np.load(p) as z:
                    ims, t = unpack(z), z["t"]
                for b, m in models.items():
                    feats[b].append(np.concatenate([m.embed(ims[i:i + batch]) for i in range(0, len(ims), batch)]).astype(np.float16))
                keep.append(r)
                nf.append(len(ims))
                times.append(t)
            res = {"path": np.array([r["path"] for r in keep]), "gen": np.array([r["gen"] for r in keep]),
                   "y": np.array([r["label"] for r in keep], np.int8), "nf": np.array(nf, np.int32),
                   "t": np.concatenate(times), "device": np.array([dev] * len(keep))}
            res.update({b: np.concatenate(v) for b, v in feats.items()})
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():  # a parallel worker finished this shard first; keep its file
                print(f"{out.name}: written by another worker, kept", flush=True)
                continue
            tmp = out.with_name(f"{out.stem}.{os.getpid()}.tmp.npz")  # one temp file per worker, so writers never mix
            np.savez(tmp, **res)
            tmp.replace(out)
            print(f"{out.name}: {len(keep)} videos, {int(sum(nf))} frames in {time.time() - t0:.0f} s", flush=True)


def load_split(split: str) -> dict:
    """All shards of a split -> {"path", "gen", "y", "nf", "t", <backbone>: [sum(nf), D]}."""
    parts = []
    for p in sorted(emb_dir().glob(f"{split}_[0-9][0-9][0-9].npz")):
        with np.load(p) as z:
            part = {k: z[k] for k in z.files}
        part.setdefault("device", np.array(["mps"] * len(part["y"])))  # the first shards were all made on MPS
        parts.append(part)
    if not parts:
        raise FileNotFoundError(f"no embeddings for {split} in {emb_dir()}")
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["frames", "embed"])
    ap.add_argument("--splits", nargs="+", default=["val", "test", "train"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="frames: first N videos per split, in the fixed random order")
    ap.add_argument("--first", type=int, default=0, help="embed: skip the first N shards (a second worker starts later)")
    ap.add_argument("--device", default=None, help="embed: cpu to embed without the GPU (fp32)")
    ap.add_argument("--shards", type=int, default=None, help=f"embed: first N shards ({SHARD} videos each) per split")
    a = ap.parse_args(argv)
    rows = data.read_manifest()
    rows = [r for s in a.splits for r in ordered(rows, s)[:a.limit]]
    if a.step == "frames":
        run_frames(rows, a.workers)
    else:
        run_embed(data.read_manifest(), a.splits, a.batch, a.shards, a.device, a.first)  # shards are cut from the full split order


if __name__ == "__main__":
    main()

"""Pick and download a balanced AIGVDBench subset, split by generator.

AIGVDBench (Hugging Face AIGVDBench/AIGVDBench, card licence CC BY 4.0; its real videos come from
OpenVid-1M, which is for research and non-commercial use) ships one large zip per generator.
We never download a whole zip: the zip index is read with HTTP range requests, and then only
the chosen videos are fetched.

    python -m ml.video.data --dry-run   # print counts and sizes, download nothing
    python -m ml.video.data             # download into <raw>/video/aigvdbench and write manifest.csv

Splits (by generator, so a test generator is never seen in training):
  train: open generators minus VAL_FAMILIES, with real videos from the dataset's train ids
  val:   open generators of VAL_FAMILIES, with real videos from the val ids
  test:  closed models (Sora, Kling, Gen-2, Gen-3), with real videos from the test ids
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import struct
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ml.common.paths import raw_dir

REPO = "datasets/AIGVDBench/AIGVDBench"
OPEN = ["I2V/EasyAnimate", "I2V/LTX", "I2V/Pyramid-Flow", "I2V/SEINE", "I2V/SVD", "I2V/VideoCrafter",
        "T2V/AccVideo", "T2V/AnimateDiff", "T2V/Cogvideox1.5", "T2V/EasyAnimate", "T2V/HunyuanVideo",
        "T2V/IPOC", "T2V/LTX", "T2V/Open-Sora", "T2V/Pyramid-Flow", "T2V/RepVideo", "T2V/VideoCrafter",
        "T2V/Wan2.1", "V2V/Cogvideox1.5", "V2V/LTX"]
VAL_FAMILIES = {"HunyuanVideo", "Pyramid-Flow", "SEINE"}  # held out from training, used to pick C and threshold
CLOSED = ["Sora", "kling", "Gen2", "Gen3"]  # test only
COUNTS = {"train_fake": 3000, "train_real": 3000, "val_fake_per_gen": 150, "val_real": 600,
          "test_fake_per_gen": 500, "test_real": 2000}
SEED = 0


def root() -> Path:
    return raw_dir() / "video" / "aigvdbench"


def family(gen: str) -> str:
    return gen.split("/")[-1]


def split_of(gen: str) -> str:
    if gen == "real":
        raise ValueError("real videos take the split of their id")
    if gen in CLOSED:
        return "test"
    return "val" if family(gen) in VAL_FAMILIES else "train"


def zip_path(gen: str) -> str:
    if gen == "real":
        return "AIGVDBench/Real/Real.zip"
    return f"AIGVDBench/ClosedSource/{gen}.zip" if gen in CLOSED else f"AIGVDBench/OpenSource/{gen}.zip"


def select(members: dict[str, list], ids: dict[str, list[str]], counts: dict = COUNTS, seed: int = SEED) -> list[dict]:
    """members: zip path -> [(name, size, ...)]; ids: split -> video ids. Returns rows to download.

    Open generators share the dataset's id lists, so no clip id is in two splits. The ids are clip ids:
    a few source videos have clips in more than one split.
    """
    rng = random.Random(seed)
    rows = []

    def pick(gen, split, n, allowed=None):
        mem = [m for m in members[zip_path(gen)] if not m[0].endswith("/")]
        if allowed is not None:
            mem = [m for m in mem if m[0].rsplit("/", 1)[-1] in allowed]
        mem.sort()
        for name, size, *_ in rng.sample(mem, min(n, len(mem))):
            rows.append({"gen": gen, "split": split, "label": int(gen != "real"), "member": name, "bytes": size,
                         "zip": zip_path(gen), "path": f"{gen}/{name.rsplit('/', 1)[-1]}"})

    train_gens = [g for g in OPEN if split_of(g) == "train"]
    per = -(-counts["train_fake"] // len(train_gens))
    sets = {s: set(v) for s, v in ids.items()}
    for g in train_gens:
        pick(g, "train", per, sets["train"])
    for g in OPEN:
        if split_of(g) == "val":
            pick(g, "val", counts["val_fake_per_gen"], sets["val"])
    for g in CLOSED:
        pick(g, "test", counts["test_fake_per_gen"])
    for s, n in (("train", counts["train_real"]), ("val", counts["val_real"]), ("test", counts["test_real"])):
        pick("real", s, n, sets[s])
    return rows


class _Fetcher:
    """One ranged HTTP GET per video, using the zip index (offsets) we already read.

    The Hugging Face resolve URL redirects to a signed CDN URL that expires after about an hour,
    so it is re-resolved every 20 minutes and after any HTTP error.
    """

    def __init__(self, index: dict[str, dict[str, zipfile.ZipInfo]]):
        import requests
        self.index, self.http, self.urls, self.lock = index, requests.Session(), {}, threading.Lock()

    def url(self, zpath: str, fresh: bool = False) -> str:
        with self.lock:
            u, t = self.urls.get(zpath, (None, 0.0))
            if fresh or u is None or time.time() - t > 1200:
                r = self.http.head(f"https://huggingface.co/{REPO}/resolve/main/{zpath}", allow_redirects=False, timeout=30)
                u = r.headers.get("location") if r.is_redirect else r.url
                self.urls[zpath] = (u, time.time())
            return u

    def get(self, row: dict, dst: Path) -> None:
        if dst.is_file() and dst.stat().st_size == row["bytes"]:
            return
        info = self.index[row["zip"]][row["member"]]
        start = info.header_offset
        end = start + 30 + 2 * len(info.orig_filename.encode()) + 1024 + info.compress_size  # slack for the local extra field
        for attempt in range(4):
            try:
                r = self.http.get(self.url(row["zip"], fresh=attempt > 0), headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                r.raise_for_status()
                data = unpack_local(r.content, info)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        dst.parent.mkdir(parents=True, exist_ok=True)
        part = dst.with_name(dst.name + ".part")
        part.write_bytes(data)
        part.replace(dst)


def unpack_local(buf: bytes, info: zipfile.ZipInfo) -> bytes:
    """Bytes starting at a zip local file header -> the member's uncompressed data (CRC checked)."""
    if buf[:4] != b"PK\x03\x04":
        raise OSError("not a local file header")
    n, e = struct.unpack("<HH", buf[26:30])
    raw = buf[30 + n + e: 30 + n + e + info.compress_size]
    if len(raw) != info.compress_size:
        raise OSError("short read")
    if info.compress_type == zipfile.ZIP_DEFLATED:
        raw = zlib.decompress(raw, -15)
    elif info.compress_type != zipfile.ZIP_STORED:
        raise OSError(f"unsupported compression {info.compress_type}")
    if len(raw) != info.file_size or zlib.crc32(raw) != info.CRC:
        raise OSError("size or CRC mismatch")
    return raw


def list_members() -> dict[str, list[zipfile.ZipInfo]]:
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    gens = OPEN + CLOSED + ["real"]

    def one(g):
        with fs.open(f"{REPO}/{zip_path(g)}", "rb", block_size=1 << 20) as f:
            return zip_path(g), zipfile.ZipFile(f).infolist()
    with ThreadPoolExecutor(8) as ex:
        return dict(ex.map(one, gens))


def split_ids() -> dict[str, list[str]]:
    from huggingface_hub import hf_hub_download
    out = {}
    for s in ("train", "val", "test"):
        p = hf_hub_download("AIGVDBench/AIGVDBench", f"Split/{s}.jsonl", repo_type="dataset")
        out[s] = [json.loads(line)["Video_id"] for line in open(p) if line.strip()]
    return out


def manifest_path() -> Path:
    return root() / "manifest.csv"


def read_manifest() -> list[dict]:
    with open(manifest_path(), newline="") as f:
        return [dict(r, label=int(r["label"]), bytes=int(r["bytes"])) for r in csv.DictReader(f)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    members = list_members()
    rows = select({z: [(i.filename, i.file_size) for i in infos] for z, infos in members.items()}, split_ids())
    by = {}
    for r in rows:
        k = (r["split"], "fake" if r["label"] else "real")
        n, b = by.get(k, (0, 0))
        by[k] = (n + 1, b + r["bytes"])
    for k, (n, b) in sorted(by.items()):
        print(f"{k[0]:5s} {k[1]}: {n} videos, {b / 1e9:.1f} GB")
    print(f"total: {len(rows)} videos, {sum(r['bytes'] for r in rows) / 1e9:.1f} GB", flush=True)
    if a.dry_run:
        return
    root().mkdir(parents=True, exist_ok=True)
    keys = ["path", "gen", "split", "label", "bytes", "zip", "member"]
    with open(manifest_path(), "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows({k: r[k] for k in keys} for r in rows)
    fetch = _Fetcher({z: {i.filename: i for i in infos} for z, infos in members.items()})
    order = sorted(rows, key=lambda r: ("val", "test", "train").index(r["split"]))  # val/test first: the baseline needs them
    def get(r):
        try:
            fetch.get(r, root() / r["path"])
        except Exception as e:  # one bad member must not stop the rest; later steps skip missing files
            print(f"FAILED {r['path']}: {e!r}", flush=True)
    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for _ in ex.map(get, order):
            done += 1
            if done % 200 == 0 or done == len(order):
                print(f"{done}/{len(order)} downloaded", flush=True)


if __name__ == "__main__":
    main()

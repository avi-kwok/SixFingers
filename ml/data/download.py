"""Downloaders, one per source type. All are resumable: files already on disk are skipped and
already-indexed items count toward --limit. Raw files go to data/raw/<name>/, plus an index.jsonl."""
import argparse
import csv
import hashlib
import io
import json
import random
import shutil
import subprocess
import zipfile
from pathlib import Path

import requests

from ml.data.sources import append_index, enabled_sources, holdout_generator, read_index, source_dir
from ml.data.splits import REAL_TEST_GENERATOR, _bucket

COCO_ANN = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
OI_CSV = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"
OI_IMG = "https://open-images-dataset.s3.amazonaws.com/validation/{}.jpg"
RAISE_CSV = "http://loki.disi.unitn.it/RAISE/getFile.php?p=1k"
RAPIDATA = "Rapidata/Flux_SD3_MJ_Dalle_Human_Alignment_Dataset"
HOLDOUT_PROMPT_PCT = 30  # share of prompts (by hash) whose images go to the held-out generator


def prompt_is_holdout(group: str) -> bool:
    # salted so the partition is independent of the split hash (unsalted, training prompts only got buckets >= 30)
    return _bucket("holdout:" + group) < HOLDOUT_PROMPT_PCT


def fetch(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    return dest


def _done(src: dict) -> set[str]:
    return {r["file"] for r in read_index(src["name"])}


def _row(src: dict, file: str, group: str, **extra) -> dict:
    return {"file": file, "source": src["name"], "generator": src["generator"],
            "side": src["side"], "license": src["license"], "group": group, **extra}


def coco_captions(root: Path) -> dict[int, str]:
    zpath = fetch(COCO_ANN, root / "_annotations_trainval2017.zip")
    with zipfile.ZipFile(zpath) as z:
        ann = json.loads(z.read("annotations/captions_val2017.json"))
    caps: dict[int, str] = {}
    for a in sorted(ann["annotations"], key=lambda a: a["id"]):
        caps.setdefault(a["image_id"], a["caption"].strip())
    return caps


def coco(src: dict, limit: int) -> None:
    root = source_dir(src["name"])
    caps = coco_captions(root)
    ids = sorted(caps)
    random.Random(0).shuffle(ids)  # seeded sample so reruns and --limit growth are stable
    done = _done(src)
    for i in ids[:limit]:
        name = f"{i:012d}.jpg"
        if name in done:
            continue
        fetch(f"http://images.cocodataset.org/val2017/{name}", root / name)
        append_index(src["name"], _row(src, name, f"coco:{i}", caption=caps[i]))


def openimages(src: dict, limit: int) -> None:
    root = source_dir(src["name"])
    path = fetch(OI_CSV, root / "_validation-images.csv")
    rows = sorted(csv.DictReader(path.open(encoding="utf-8")), key=lambda r: r["ImageID"])
    random.Random(0).shuffle(rows)
    done = _done(src)
    for r in rows[:limit]:
        name = f"{r['ImageID']}.jpg"
        if name in done:
            continue
        fetch(OI_IMG.format(r["ImageID"]), root / name)
        append_index(src["name"], _row(src, name, f"oi:{r['ImageID']}", license_url=r["License"]))


def raise_(src: dict, limit: int) -> None:
    """NEF -> full-quality JPEG via macOS `sips`; the normalizer then treats it like any other image."""
    if not shutil.which("sips"):  # shortcut: macOS-only decoder; RAISE is test-only so Linux (the loop VM) skips it
        print("[raise] skipped: no `sips` on this platform", flush=True)
        return
    root = source_dir(src["name"])
    zpath = fetch(RAISE_CSV, root / "_RAISE_1k.csv.zip")
    with zipfile.ZipFile(zpath) as z:
        rows = list(csv.DictReader(io.TextIOWrapper(z.open("RAISE_1k.csv"), encoding="utf-8-sig")))
    rows.sort(key=lambda r: r["File"])
    random.Random(0).shuffle(rows)
    done = _done(src)
    for r in rows[: min(limit, src.get("max_limit", limit))]:
        name = f"{r['File']}.jpg"
        if name in done:
            continue
        nef = fetch(r["NEF"], root / "_nef" / f"{r['File']}.NEF")
        subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "100", str(nef),
                        "--out", str(root / name)], check=True, capture_output=True)
        nef.unlink()  # 10 MB each; the JPEG is what we keep
        append_index(src["name"], _row(src, name, f"raise:{r['File']}", device=r.get("Device")))


def rapidata(src: dict, limit: int) -> None:
    """Pairs of (image1, image2) from two generators per shard; each unique image path is kept once."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    root = source_dir(src["name"])
    done = _done(src)
    n = len(done)
    holdout = src["generator"] == holdout_generator()
    # prompts are partitioned by hash so the held-out generator never shares a prompt with training generators
    # shards are listed per source in sources.yaml because generators are spread unevenly over the 26 files
    shards = src.get("shards") or sorted(f for f in HfApi().list_repo_files(RAPIDATA, repo_type="dataset") if f.endswith(".parquet"))
    for shard in shards:
        if n >= limit:
            return
        local = hf_hub_download(RAPIDATA, shard, repo_type="dataset")
        pf = pq.ParquetFile(local)
        t = pf.read().to_pylist()
        for row in t:
            for k in ("1", "2"):
                if row[f"model{k}"] != src["model_name"]:
                    continue
                key = row[f"image{k}_path"]
                name = key.replace("/", "_")
                if name in done or n >= limit:
                    continue
                gid = hashlib.sha1(row["prompt"].encode()).hexdigest()[:12]
                group = f"rapidata:{gid}"
                if prompt_is_holdout(group) != holdout:  # not this source's partition: never written to disk
                    continue
                img = row[f"image{k}"]["bytes"]
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_bytes(img)
                append_index(src["name"], _row(src, name, group, caption=row["prompt"]))
                done.add(name)
                n += 1


DOWNLOADERS = {"coco": coco, "openimages": openimages, "raise": raise_, "rapidata": rapidata}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--limit", type=int, default=200, help="images per source")
    grp.add_argument("--grow", type=int, help="add N images to each training source (report_test is never grown)")
    ap.add_argument("--source", help="one source name; default all enabled downloadable sources")
    a = ap.parse_args()
    for s in enabled_sources():
        fn = DOWNLOADERS.get(s["downloader"])
        if fn is None or (a.source and s["name"] != a.source):
            continue
        if a.grow is None:
            limit = a.limit
            print(f"[{s['name']}] downloading up to {limit}", flush=True)
        else:
            if s["generator"] in (holdout_generator(), REAL_TEST_GENERATOR):
                continue
            limit = len(read_index(s["name"])) + a.grow
            print(f"[{s['name']}] grow by {a.grow} (limit {limit})", flush=True)
        fn(s, limit)
        print(f"[{s['name']}] {len(read_index(s['name']))} indexed", flush=True)


if __name__ == "__main__":
    main()

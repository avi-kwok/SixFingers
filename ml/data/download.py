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
import time
import zipfile
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import requests
from PIL import Image, ImageOps

from ml.data.sources import append_index, enabled_sources, holdout_generator, read_index, source_dir
from ml.data.splits import REAL_TEST_GENERATOR, _bucket

COCO_ANN = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
OI_CSV = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"
OI_IMG = "https://open-images-dataset.s3.amazonaws.com/validation/{}.jpg"
OI_ORIG_LICENSE = "https://creativecommons.org/licenses/by/2.0/"  # CC BY 2.0 as written in the CSV License column
OI_ORIG_LONG_EDGE = 1600
OI_ORIG_QUALITY = 90
OI_ORIG_WORKERS = 2  # Flickr answers HTTP 429 after a burst of about 37 requests from 4 workers, so stay small
OI_ORIG_PAUSE_S = (3.0, 8.0)  # random wait before each request, so requests never arrive in bursts; tests set it to (0, 0)
OI_ORIG_TIMEOUT_S = (10, 120)  # connect, read: originals are up to ~16 MB
OI_ORIG_RETRIES = 5  # after a 429, a 5xx or a network error
OI_ORIG_BACKOFF_S = 15.0  # wait before the first retry, doubled for each next one; tests set it to 0
OI_ORIG_MAX_ERRORS = OI_ORIG_WORKERS  # one full round of in-flight images failing every retry means a rate-limit block, so stop and send fewer requests to a blocked server
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
    taken = {Path(r["file"]).stem for r in read_index("openimages_original")}  # same seeded order: a kept original is never re-fetched as a web copy
    for r in [r for r in rows if r["ImageID"] not in taken][:limit]:
        name = f"{r['ImageID']}.jpg"
        if name in done:
            continue
        fetch(OI_IMG.format(r["ImageID"]), root / name)
        append_index(src["name"], _row(src, name, f"oi:{r['ImageID']}", license_url=r["License"]))


def _oiorig_fetch(url: str) -> tuple[str, bytes | None, str]:
    """Download one Flickr original. Returns ("ok", JPEG bytes at 1600 px long edge, camera make) or
    (reason, None, ""): "dead" (4xx other than 429, or not an image), "no_exif" (no camera make),
    "error" (every retry failed)."""
    for attempt in range(OI_ORIG_RETRIES + 1):
        time.sleep(random.uniform(*OI_ORIG_PAUSE_S))
        try:
            resp = requests.get(url, timeout=OI_ORIG_TIMEOUT_S)
            code = resp.status_code
        except requests.RequestException:
            code = None
        if code == 200:
            break
        if code is not None and code != 429 and code < 500:
            return "dead", None, ""
        if attempt < OI_ORIG_RETRIES:
            time.sleep(OI_ORIG_BACKOFF_S * 2 ** attempt)
    else:
        return "error", None, ""
    try:
        im = Image.open(io.BytesIO(resp.content))
        make = str(im.getexif().get(271) or "").strip(" \x00")
        if not make:  # no camera make: often a scan, a screenshot or an export
            return "no_exif", None, ""
        im = ImageOps.exif_transpose(im).convert("RGB")  # the saved copy has no EXIF, so apply the rotation now
        im.thumbnail((OI_ORIG_LONG_EDGE, OI_ORIG_LONG_EDGE), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=OI_ORIG_QUALITY)
        return "ok", buf.getvalue(), make
    except (OSError, ValueError, Image.DecompressionBombError):
        return "dead", None, ""


def oiorig(src: dict, limit: int) -> None:
    """Open Images originals: the Flickr original of each CC BY 2.0 validation image that the openimages source has
    not taken, kept only when its EXIF names a camera make, stored at 1600 px on the long edge as JPEG q90.
    Workers only download and resize; this thread writes every file and index line, so writes need no lock.
    Dead links and photos without a camera make go to _rejected.txt and are not fetched again."""
    root = source_dir(src["name"])
    root.mkdir(parents=True, exist_ok=True)
    csv_path = fetch(OI_CSV, source_dir("openimages") / "_validation-images.csv")
    rows = sorted(csv.DictReader(csv_path.open(encoding="utf-8")), key=lambda r: r["ImageID"])
    random.Random(0).shuffle(rows)  # seeded order so reruns and --limit/--grow are stable
    rejected = root / "_rejected.txt"
    skip = {Path(r["file"]).stem for name in ("openimages", src["name"]) for r in read_index(name)}
    if rejected.exists():
        skip |= set(rejected.read_text().split())
    todo = iter([r for r in rows if r["License"] == OI_ORIG_LICENSE and r["ImageID"] not in skip])
    n, counts, errors_in_row, running = len(read_index(src["name"])), Counter(), 0, {}
    with ThreadPoolExecutor(OI_ORIG_WORKERS) as ex, rejected.open("a") as rej:
        while n < limit:
            while len(running) < OI_ORIG_WORKERS and (r := next(todo, None)) is not None:
                running[ex.submit(_oiorig_fetch, r["OriginalURL"])] = r
            if not running:
                break
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in finished:
                r = running.pop(fut)
                status, jpeg, make = fut.result()
                counts[status] += 1
                errors_in_row = errors_in_row + 1 if status == "error" else 0
                if status in ("dead", "no_exif"):
                    rej.write(r["ImageID"] + "\n")
                    rej.flush()
                elif status == "ok" and n < limit:  # results past the limit are dropped, so no overshoot
                    name = f"{r['ImageID']}.jpg"
                    (root / name).write_bytes(jpeg)
                    append_index(src["name"], _row(src, name, f"oiorig:{r['ImageID']}", license_url=r["License"],
                                                   author=r["Author"], author_url=r["AuthorProfileURL"],
                                                   landing_url=r["OriginalLandingURL"], camera_make=make))
                    n += 1
                    if n % 100 == 0:
                        print(f"[{src['name']}] kept {n}; this run so far: {dict(counts)}", flush=True)
            if errors_in_row >= OI_ORIG_MAX_ERRORS:
                print(f"[{src['name']}] stopped: {errors_in_row} images in a row failed every retry "
                      "(likely rate limited); rerun later, the download resumes", flush=True)
                break
        for fut in running:  # limit reached: drop work that has not started
            fut.cancel()
    print(f"[{src['name']}] kept {n}; this run: {dict(counts)}", flush=True)


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
        nef.unlink()  # 10 MB each; only the JPEG is kept
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


DOWNLOADERS = {"coco": coco, "openimages": openimages, "oiorig": oiorig, "raise": raise_, "rapidata": rapidata}


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

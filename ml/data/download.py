"""Downloaders, one per source type. All are resumable: files already on disk are skipped and
already-indexed items count toward --limit. Raw files go to data/raw/<name>/, plus an index.jsonl."""
import argparse
import csv
import hashlib
import html
import io
import json
import random
import re
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
# Wikimedia Commons rules: https://wikitech.wikimedia.org/wiki/Robot_policy and the Wikimedia User-Agent policy
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
COMMONS_HEADERS = {"User-Agent": "SixFingers/0.1 (https://github.com/avi-kwok/SixFingers)"}
COMMONS_WORKERS = 2  # the robot policy allows at most 2 file downloads at once
COMMONS_MAX_MBPS = 25.0  # the robot policy's limit on total download speed
COMMONS_PAUSE_S = (2.0, 5.0)  # random wait before each file download; tests set it to (0, 0)
COMMONS_API_PAUSE_S = 1.0  # wait after each API request (5x after a slow one); tests set it to 0
COMMONS_TIMEOUT_S = (10, 120)  # connect, read: originals can be 30 MB
COMMONS_RETRIES = 5  # after a 429, a 5xx, a maxlag error or a network error
COMMONS_BACKOFF_S = 15.0  # wait before the first retry when the server sends no Retry-After, doubled each time
COMMONS_MAX_ERRORS = COMMONS_WORKERS  # files in a row that failed every retry before the run stops
COMMONS_MIN_FILES = 20  # camera categories with fewer files are skipped
COMMONS_BATCH = 10  # files checked per category per API request
COMMONS_PER_CATEGORY = 3  # files kept per camera category, so many cameras and phones are covered
COMMONS_MIN_SIDE = 1000
COMMONS_TAKEN_BEFORE = "2022:01:01"  # EXIF date format; AI photo generators were rare before 2022
COMMONS_LICENSE = re.compile(r"(cc0|public domain|pd\b|cc by(-sa)? \d)", re.I)  # matched at the start of LicenseShortName
COMMONS_AI = re.compile(r"\bai[- ]generated|\bai art\b|artificial intelligence|generative|midjourney|stable diffusion|dall[-·]e|firefly", re.I)
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


def _commons_get(url: str, params: dict | None = None):
    """GET with the Commons User-Agent. Retries after a 429, a 5xx, a maxlag error or a network error, waiting
    the server's Retry-After seconds when it sends them. Returns the response (200 or another 4xx), or None
    when every retry failed."""
    for attempt in range(COMMONS_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=COMMONS_HEADERS, timeout=COMMONS_TIMEOUT_S)
        except requests.RequestException:
            resp = None
        if resp is not None and resp.status_code != 429 and resp.status_code < 500 \
                and resp.headers.get("MediaWiki-API-Error") != "maxlag":
            return resp
        if attempt < COMMONS_RETRIES:
            after = resp.headers.get("Retry-After", "") if resp is not None else ""
            time.sleep(float(after) if after.isdigit() else COMMONS_BACKOFF_S * 2 ** attempt)
    return None


def _commons_api(params: dict) -> dict:
    """One Action API query. Only the main thread calls it, so requests go one at a time. Afterwards it waits
    COMMONS_API_PAUSE_S, or 5 times that when the answer took over 1 s (robot policy)."""
    start = time.monotonic()
    resp = _commons_get(COMMONS_API, {"action": "query", "format": "json", "formatversion": 2, "maxlag": 5, **params})
    time.sleep(COMMONS_API_PAUSE_S * (5 if time.monotonic() - start > 1 else 1))
    data = resp.json() if resp is not None and resp.status_code == 200 else None
    if data is None or "error" in data:
        raise RuntimeError(f"Commons API failed: {data['error'] if data else getattr(resp, 'status_code', 'no answer')}")
    return data


def _commons_query(params: dict):
    """Yields every batch of an API query, following 'continue' until the answer is complete."""
    cont: dict = {}
    while True:
        data = _commons_api({**params, **cont})
        yield data
        if "continue" not in data:
            return
        cont = data["continue"]


def _commons_categories(root: Path) -> list[str]:
    """Camera categories ("Taken with <camera>") with at least COMMONS_MIN_FILES files. Cached in
    _categories.txt so reruns walk the same list in the same order."""
    path = root / "_categories.txt"
    if not path.exists():
        params = {"list": "allcategories", "acprefix": "Taken with ", "acmin": COMMONS_MIN_FILES, "aclimit": 500,
                  "acprop": "size"}
        cats = sorted(c["category"] for d in _commons_query(params) for c in d["query"]["allcategories"]
                      if c["files"] >= COMMONS_MIN_FILES and not COMMONS_AI.search(c["category"]))
        path.write_text("\n".join(cats) + "\n")
    return path.read_text().splitlines()


def _commons_candidates(cat: str, skip: set[str]) -> list[dict]:
    """Up to COMMONS_BATCH unseen files of one category, in a seeded random order from its first 500 files, with
    their image info, EXIF, licence and categories."""
    members = _commons_api({"list": "categorymembers", "cmtitle": "Category:" + cat, "cmtype": "file",
                            "cmlimit": 500, "cmprop": "ids"})["query"]["categorymembers"]
    ids = sorted(str(m["pageid"]) for m in members)
    random.Random(cat).shuffle(ids)
    ids = [i for i in ids if i not in skip][:COMMONS_BATCH]
    if not ids:
        return []
    params = {"pageids": "|".join(ids), "prop": "imageinfo|categories", "cllimit": "max",
              "iiprop": "url|size|mime|metadata|extmetadata", "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist"}
    pages: dict[int, dict] = {}
    for d in _commons_query(params):
        for p in d["query"]["pages"]:
            q = pages.setdefault(p["pageid"], {**p, "categories": []})
            q["categories"] += p.get("categories", [])  # a long category list can arrive over several batches
            if "imageinfo" in p:
                q["imageinfo"] = p["imageinfo"]
    return [pages[int(i)] for i in ids if int(i) in pages]


def _commons_info(page: dict) -> tuple[dict, dict, dict]:
    """(image info, EXIF tags by name, extmetadata values by name) of one file page."""
    info = (page.get("imageinfo") or [{}])[0]
    exif = {m["name"]: str(m["value"]).strip(" \x00") for m in info.get("metadata") or []}
    ext = {k: str(v.get("value", "")).strip() for k, v in (info.get("extmetadata") or {}).items()}
    return info, exif, ext


def _commons_reason(page: dict) -> str | None:
    """Why a file is skipped before it is downloaded, or None when it passes every filter."""
    info, exif, ext = _commons_info(page)
    taken = exif.get("DateTimeOriginal", "")
    if not info:
        return "missing"
    if any(COMMONS_AI.search(c["title"]) for c in page["categories"]):
        return "ai_category"
    if info.get("mime") != "image/jpeg":
        return "not_jpeg"
    if min(info.get("width", 0), info.get("height", 0)) < COMMONS_MIN_SIDE:
        return "small"
    if not COMMONS_LICENSE.match(ext.get("LicenseShortName", "")):
        return "licence"
    if not (exif.get("Make") and exif.get("Model")):
        return "no_camera"
    if not re.match(r"\d{4}:\d\d:\d\d", taken) or not "1990" <= taken < COMMONS_TAKEN_BEFORE:  # also drops placeholder dates like 0000:00:00
        return "date"
    return None


def _commons_fetch(url: str) -> tuple[str, bytes | None]:
    """Download one original after a random pause, keeping both workers together under COMMONS_MAX_MBPS.
    Returns ("ok", JPEG bytes at 1600 px long edge), ("dead", None) for a 4xx or a file that is not an image,
    or ("error", None) when every retry failed."""
    time.sleep(random.uniform(*COMMONS_PAUSE_S))
    start = time.monotonic()
    resp = _commons_get(url)
    if resp is None:
        return "error", None
    if resp.status_code != 200:
        return "dead", None
    min_s = len(resp.content) * 8 / (COMMONS_MAX_MBPS * 1e6 / COMMONS_WORKERS)
    time.sleep(max(0.0, min_s - (time.monotonic() - start)))
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(resp.content))).convert("RGB")
        im.thumbnail((OI_ORIG_LONG_EDGE, OI_ORIG_LONG_EDGE), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=OI_ORIG_QUALITY)
        return "ok", buf.getvalue()
    except (OSError, ValueError, Image.DecompressionBombError):
        return "dead", None


def commons(src: dict, limit: int) -> None:
    """Wikimedia Commons camera photos: files from the "Taken with <camera>" categories, walked in a seeded random
    order, at most COMMONS_PER_CATEGORY per category. A file is kept only when it is a JPEG with a short side of
    at least 1000 px, has a CC0, public domain, CC BY or CC BY-SA licence, a camera make and model in EXIF, an EXIF
    date taken before 2022, and no category that mentions AI generation. All of this is checked through the API,
    so rejected files are never downloaded. Kept files are stored at 1600 px on the long edge as JPEG q90.
    This thread makes every API request and writes every file and index line; workers only download and resize.
    Rejected and dead files go to _rejected.txt (page id and reason) and are not checked again."""
    root = source_dir(src["name"])
    root.mkdir(parents=True, exist_ok=True)
    rejected = root / "_rejected.txt"
    index = read_index(src["name"])
    skip = {Path(r["file"]).stem for r in index}
    if rejected.exists():
        skip |= {line.split()[0] for line in rejected.read_text().splitlines() if line.strip()}
    per_cat = Counter(r["category"] for r in index)
    n, counts, errors_in_row = len(index), Counter(), 0
    try:
        cats = _commons_categories(root)
    except RuntimeError as e:
        print(f"[{src['name']}] stopped: {e}; rerun later", flush=True)
        return
    random.Random(0).shuffle(cats)
    with ThreadPoolExecutor(COMMONS_WORKERS) as ex, rejected.open("a") as rej:
        for cat in cats:
            if n >= limit or errors_in_row >= COMMONS_MAX_ERRORS:
                break
            if per_cat[cat] >= COMMONS_PER_CATEGORY:
                continue
            try:
                pages = _commons_candidates(cat, skip)
            except RuntimeError as e:
                print(f"[{src['name']}] stopped: {e}; rerun later, the download resumes", flush=True)
                break
            todo = []
            for p in pages:
                if reason := _commons_reason(p):
                    counts[reason] += 1
                    rej.write(f"{p['pageid']} {reason}\n")
                    skip.add(str(p["pageid"]))
                else:
                    todo.append(p)
            rej.flush()
            todo, running = iter(todo), {}
            while n < limit and per_cat[cat] < COMMONS_PER_CATEGORY:
                while len(running) < min(COMMONS_WORKERS, COMMONS_PER_CATEGORY - per_cat[cat], limit - n) \
                        and (p := next(todo, None)) is not None:
                    running[ex.submit(_commons_fetch, p["imageinfo"][0]["url"])] = p
                if not running:
                    break
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for fut in finished:
                    p = running.pop(fut)
                    status, jpeg = fut.result()
                    counts[status] += 1
                    errors_in_row = errors_in_row + 1 if status == "error" else 0
                    if status == "dead":
                        rej.write(f"{p['pageid']} dead\n")
                        rej.flush()
                    elif status == "ok" and n < limit and per_cat[cat] < COMMONS_PER_CATEGORY:
                        info, exif, ext = _commons_info(p)
                        name = f"{p['pageid']}.jpg"
                        (root / name).write_bytes(jpeg)
                        author = html.unescape(re.sub(r"<[^>]+>", "", ext.get("Artist", ""))).strip()
                        append_index(src["name"], _row(
                            src, name, f"commons:{p['pageid']}", title=p["title"], author=author,
                            license_name=ext["LicenseShortName"], license_url=ext.get("LicenseUrl") or None,
                            landing_url=info["descriptionurl"], camera_make=exif["Make"], camera_model=exif["Model"],
                            date_taken=exif["DateTimeOriginal"], category=cat))
                        n += 1
                        per_cat[cat] += 1
                        if n % 100 == 0:
                            print(f"[{src['name']}] kept {n}; this run so far: {dict(counts)}", flush=True)
                if errors_in_row >= COMMONS_MAX_ERRORS:
                    print(f"[{src['name']}] stopped: {errors_in_row} files in a row failed every retry "
                          "(likely rate limited); rerun later, the download resumes", flush=True)
                    break
            for fut in running:  # limit reached or stopped: drop work that has not started
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


DOWNLOADERS = {"coco": coco, "openimages": openimages, "oiorig": oiorig, "commons": commons, "raise": raise_,
               "rapidata": rapidata}


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

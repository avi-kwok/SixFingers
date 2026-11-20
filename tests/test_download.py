import csv
import io
import random
import sys
import threading
import time
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from ml.data import download, generate_local
from ml.data.sources import append_index, enabled_sources, holdout_generator, read_index, source_dir
from ml.data.splits import assign_splits


def test_partition_share_and_stable():
    groups = [f"rapidata:{i:04d}" for i in range(2000)]
    share = sum(download.prompt_is_holdout(g) for g in groups) / len(groups)
    assert 0.25 < share < 0.35
    assert [download.prompt_is_holdout(g) for g in groups] == [download.prompt_is_holdout(g) for g in groups]


def test_training_prompts_keep_70_15_15_split():
    from ml.data.splits import assign_splits
    import pandas as pd
    groups = [g for g in (f"rapidata:{i:05d}" for i in range(20000)) if not download.prompt_is_holdout(g)]
    split = assign_splits(pd.DataFrame({"generator": "dalle3", "group": groups}), holdout="flux")
    share = split.value_counts(normalize=True)
    assert abs(share["train"] - 0.70) < 0.02 and abs(share["val"] - 0.15) < 0.02 and abs(share["gate"] - 0.15) < 0.02


def _jpeg(v: int) -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (16, 16), (v, v, v)).save(b, "JPEG")
    return b.getvalue()


def test_rapidata_partitions_prompts_between_holdout_and_training(tmp_path, monkeypatch):
    rows = {"prompt": [], "model1": [], "model2": [], "image1": [], "image2": [],
            "image1_path": [], "image2_path": []}
    for i in range(40):
        rows["prompt"].append(f"a photo number {i}")
        rows["model1"].append("flux")
        rows["model2"].append("dalle-3")
        rows["image1"].append({"bytes": _jpeg(i)})
        rows["image2"].append({"bytes": _jpeg(255 - i)})
        rows["image1_path"].append(f"flux/{i}_a.jpg")
        rows["image2_path"].append(f"dalle/{i}_b.jpg")
    pq_path = tmp_path / "shard.parquet"
    pq.write_table(pa.table(rows), pq_path)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(pq_path))
    monkeypatch.setenv("SIXFINGERS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(download, "holdout_generator", lambda *a, **k: "flux")

    def src(name, generator, model_name):
        return {"name": name, "side": "ai", "generator": generator, "model_name": model_name,
                "license": "test", "shards": ["x.parquet"]}

    download.rapidata(src("rapidata_flux", "flux", "flux"), 1000)
    download.rapidata(src("rapidata_dalle3", "dalle3", "dalle-3"), 1000)

    flux = read_index("rapidata_flux")
    dalle = read_index("rapidata_dalle3")
    assert flux and dalle
    assert not {r["group"] for r in flux} & {r["group"] for r in dalle}
    assert all(download.prompt_is_holdout(r["group"]) for r in flux)
    for name, rows_ in (("rapidata_flux", flux), ("rapidata_dalle3", dalle)):
        on_disk = {p.name for p in source_dir(name).iterdir()} - {"index.jsonl"}
        assert on_disk == {r["file"] for r in rows_}


def test_grow_skips_report_test_sources_and_uses_index_plus_n(tmp_path, monkeypatch):
    monkeypatch.setenv("SIXFINGERS_DATA_DIR", str(tmp_path))
    append_index("normal", {"file": "a.jpg", "group": "g"})
    append_index("normal", {"file": "b.jpg", "group": "g"})
    append_index("normal", {"file": "c.jpg", "group": "g"})
    sources = [
        {"name": "holdout", "downloader": "rapidata", "generator": "flux"},
        {"name": "raise", "downloader": "raise", "generator": "real_raise"},
        {"name": "normal", "downloader": "coco", "generator": "real_coco"},
    ]
    calls = []
    monkeypatch.setattr(download, "enabled_sources", lambda *a, **k: sources)
    monkeypatch.setattr(download, "holdout_generator", lambda *a, **k: "flux")
    monkeypatch.setattr(download, "DOWNLOADERS", {
        name: (lambda s, limit, name=name: calls.append((s["name"], limit)))
        for name in ("rapidata", "raise", "coco")})
    monkeypatch.setattr(sys, "argv", ["x", "--grow", "5"])
    download.main()
    assert calls == [("normal", 8)]


def test_generate_local_grow_adds_n_images(tmp_path, monkeypatch):
    monkeypatch.setenv("SIXFINGERS_DATA_DIR", str(tmp_path))
    for i in range(6):
        append_index("coco", {"file": f"{i:012d}.jpg", "caption": f"caption {i}", "group": f"coco:{i}"})
    for i in range(2):
        append_index("sdxl_turbo", {"file": f"{i:012d}.png", "group": f"coco:{i}"})

    def fake_pipe(caption, generator=None, **kw):
        return SimpleNamespace(images=[Image.new("RGB", (64, 64), (128, 128, 128))])

    monkeypatch.setattr(generate_local, "load_pipe", lambda cfg: (fake_pipe, "cpu"))
    monkeypatch.setattr(sys, "argv", ["x", "--grow", "3"])
    generate_local.main()
    assert len({r["file"] for r in read_index("sdxl_turbo")}) == 5


OI_CSV_HEADER = ["ImageID", "Subset", "OriginalURL", "OriginalLandingURL", "License", "AuthorProfileURL", "Author",
                 "Title", "OriginalSize", "OriginalMD5", "Thumbnail300KURL", "Rotation"]
OI_SRC = {"name": "openimages_original", "side": "real", "generator": "real_oi_original", "license": "test"}


def _exif_jpeg(size, make=None) -> bytes:
    b = io.BytesIO()
    im = Image.new("RGB", size, (90, 120, 150))
    if make:
        exif = Image.Exif()
        exif[271] = make  # Make tag
        im.save(b, "JPEG", exif=exif.tobytes())
    else:
        im.save(b, "JPEG")
    return b.getvalue()


def _oi_row(image_id, license_url=download.OI_ORIG_LICENSE):
    return {"ImageID": image_id, "Subset": "validation", "OriginalURL": f"https://flickr.test/{image_id}.jpg",
            "OriginalLandingURL": f"https://flickr.test/photos/{image_id}", "License": license_url,
            "AuthorProfileURL": f"https://flickr.test/people/{image_id}", "Author": f"Author {image_id}",
            "Title": "", "OriginalSize": "", "OriginalMD5": "", "Thumbnail300KURL": "", "Rotation": ""}


def _write_oi_csv(rows):
    path = source_dir("openimages") / "_validation-images.csv"  # fetch() skips the download when this exists
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OI_CSV_HEADER)
        w.writeheader()
        w.writerows(rows)


class _Resp:
    def __init__(self, status_code, content=b""):
        self.status_code = status_code
        self.content = content


def _mock_http(routes, delay=0.0):
    """routes: url -> [(status, content), ...]; the last entry repeats. Unknown urls get 404.
    Returns the list of requested urls and a dict with the peak number of concurrent requests."""
    seen, state, lock = [], {"now": 0, "peak": 0}, threading.Lock()

    def get(url, timeout=None, **_):
        with lock:
            seen.append(url)
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
            seq = routes.get(url, [(404, b"")])
            status, content = seq.pop(0) if len(seq) > 1 else seq[0]
        try:
            time.sleep(delay)
        finally:
            with lock:
                state["now"] -= 1
        return _Resp(status, content)

    return get, seen, state


def test_oiorig_keeps_cc_by_camera_photos_and_records_attribution(tmp_path, monkeypatch):
    monkeypatch.setenv("SIXFINGERS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(download, "OI_ORIG_RETRIES", 2)
    monkeypatch.setattr(download, "OI_ORIG_BACKOFF_S", 0)
    monkeypatch.setattr(download, "OI_ORIG_PAUSE_S", (0, 0))
    sa_license = "https://creativecommons.org/licenses/by-sa/2.0/"
    _write_oi_csv([
        _oi_row("CAM"),
        _oi_row("NOEXIF"),
        _oi_row("SA", license_url=sa_license),
        _oi_row("TAKEN"),
        _oi_row("GONE"),
        _oi_row("FLAKY"),
        _oi_row("BUSY"),
    ])
    append_index("openimages", {"file": "TAKEN.jpg", "group": "oi:TAKEN"})
    busy = "https://flickr.test/BUSY.jpg"
    routes = {
        "https://flickr.test/CAM.jpg": [(200, _exif_jpeg((2400, 1200), "Canon"))],
        "https://flickr.test/NOEXIF.jpg": [(200, _exif_jpeg((200, 200)))],
        "https://flickr.test/SA.jpg": [(200, _exif_jpeg((200, 200), "Canon"))],
        "https://flickr.test/TAKEN.jpg": [(200, _exif_jpeg((200, 200), "Canon"))],
        "https://flickr.test/GONE.jpg": [(404, b"")],
        "https://flickr.test/FLAKY.jpg": [(503, b""), (200, _exif_jpeg((800, 600), "NIKON"))],
        busy: [(429, b"")],
    }
    get, seen, _ = _mock_http(routes)
    monkeypatch.setattr(download.requests, "get", get)

    download.oiorig(OI_SRC, 10)

    root = source_dir(OI_SRC["name"])
    idx = {r["file"]: r for r in read_index(OI_SRC["name"])}
    assert set(idx) == {"CAM.jpg", "FLAKY.jpg"}
    cam = idx["CAM.jpg"]
    assert cam["group"] == "oiorig:CAM"
    assert cam["license_url"] == download.OI_ORIG_LICENSE
    assert cam["author"] == "Author CAM"
    assert cam["author_url"] == "https://flickr.test/people/CAM"
    assert cam["landing_url"] == "https://flickr.test/photos/CAM"
    assert cam["camera_make"] == "Canon"
    with Image.open(root / "CAM.jpg") as im:
        assert im.size == (1600, 800)
    with Image.open(root / "FLAKY.jpg") as im:
        assert im.size == (800, 600)
    assert "https://flickr.test/SA.jpg" not in seen
    assert "https://flickr.test/TAKEN.jpg" not in seen
    assert set((root / "_rejected.txt").read_text().split()) == {"NOEXIF", "GONE"}
    assert seen.count(busy) == 3  # one try plus OI_ORIG_RETRIES=2
    assert "BUSY" not in (root / "_rejected.txt").read_text().split()

    seen.clear()
    download.oiorig(OI_SRC, 10)
    assert seen == [busy] * 3  # only the image that was never settled is requested again


def test_oiorig_runs_in_parallel_and_stops_at_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("SIXFINGERS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(download, "OI_ORIG_BACKOFF_S", 0)
    monkeypatch.setattr(download, "OI_ORIG_PAUSE_S", (0, 0))
    ids = [f"P{i:02d}" for i in range(12)]
    _write_oi_csv([_oi_row(i) for i in ids])
    jpeg = _exif_jpeg((64, 48), "Canon")
    get, seen, state = _mock_http({f"https://flickr.test/{i}.jpg": [(200, jpeg)] for i in ids}, delay=0.05)
    monkeypatch.setattr(download.requests, "get", get)

    download.oiorig(OI_SRC, 3)

    root = source_dir(OI_SRC["name"])
    assert len(read_index(OI_SRC["name"])) == 3
    assert len(list(root.glob("*.jpg"))) == 3
    assert 2 <= state["peak"] <= download.OI_ORIG_WORKERS
    assert len(seen) <= 3 + download.OI_ORIG_WORKERS

    download.oiorig(OI_SRC, 5)
    files = {r["file"] for r in read_index(OI_SRC["name"])}
    assert len(files) == 5
    assert len(read_index(OI_SRC["name"])) == 5


def test_oiorig_stops_when_every_request_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("SIXFINGERS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(download, "OI_ORIG_RETRIES", 0)
    monkeypatch.setattr(download, "OI_ORIG_BACKOFF_S", 0)
    monkeypatch.setattr(download, "OI_ORIG_PAUSE_S", (0, 0))
    monkeypatch.setattr(download, "OI_ORIG_MAX_ERRORS", 3)
    ids = [f"R{i:02d}" for i in range(10)]
    _write_oi_csv([_oi_row(i) for i in ids])
    get, seen, _ = _mock_http({f"https://flickr.test/{i}.jpg": [(429, b"")] for i in ids})
    monkeypatch.setattr(download.requests, "get", get)

    download.oiorig(OI_SRC, 10)  # a normal return, so the other sources still run

    assert read_index(OI_SRC["name"]) == []
    rejected = source_dir(OI_SRC["name"]) / "_rejected.txt"
    assert not rejected.exists() or rejected.read_text() == ""
    assert len(seen) <= download.OI_ORIG_MAX_ERRORS + download.OI_ORIG_WORKERS


def test_openimages_skips_photos_held_as_originals(tmp_path, monkeypatch):
    monkeypatch.setenv("SIXFINGERS_DATA_DIR", str(tmp_path))
    ids = [f"O{i:02d}" for i in range(10)]
    _write_oi_csv([_oi_row(i) for i in ids])
    order = sorted(ids)
    random.Random(0).shuffle(order)  # the order openimages uses; the held id is the first one it would fetch
    held = order[0]
    append_index(OI_SRC["name"], {"file": f"{held}.jpg", "source": OI_SRC["name"], "generator": OI_SRC["generator"],
                                  "side": "real", "license": "test", "group": f"oiorig:{held}"})

    def fake_fetch(url, dest):
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x")
        return dest

    monkeypatch.setattr(download, "fetch", fake_fetch)
    download.openimages({"name": "openimages", "side": "real", "generator": "real_openimages", "license": "test"}, 3)

    files = [r["file"] for r in read_index("openimages")]
    assert f"{held}.jpg" not in files
    assert len(files) == 3


def test_oiorig_source_is_enabled_and_never_report_test():
    src = next(s for s in enabled_sources() if s["name"] == "openimages_original")
    assert src["side"] == "real"
    assert src["generator"] == "real_oi_original"
    assert download.DOWNLOADERS[src["downloader"]] is download.oiorig
    groups = [f"oiorig:{i}" for i in range(500)]
    split = assign_splits(pd.DataFrame({"generator": "real_oi_original", "group": groups}), holdout_generator())
    assert set(split) == {"train", "val", "gate"}

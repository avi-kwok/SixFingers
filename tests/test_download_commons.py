"""The wikimedia_commons downloader, with every HTTP request mocked (no network)."""
import io
import json
import threading
import time
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

from ml.data import download
from ml.data.sources import enabled_sources, holdout_generator, read_index, source_dir
from ml.data.splits import assign_splits

SRC = {"name": "wikimedia_commons", "side": "real", "generator": "real_commons", "license": "test"}
UPLOAD = "https://upload.test/{}.jpg"


def _jpeg(size=(64, 48)) -> bytes:
    b = io.BytesIO()
    Image.new("RGB", size, (90, 120, 150)).save(b, "JPEG")
    return b.getvalue()


def _page(pid, *, mime="image/jpeg", size=(3000, 2000), lic="CC BY-SA 4.0", lic_url="https://cc.test/by-sa/4.0",
          make="Canon", model="Canon EOS 5D", taken="2015:06:13 14:22:01", cats=("Category:Birds",)):
    meta = [{"name": k, "value": v} for k, v in (("Make", make), ("Model", model), ("DateTimeOriginal", taken)) if v]
    ext = {"LicenseShortName": {"value": lic}, "Artist": {"value": f'<a href="//commons.test/User:A{pid}">Author &amp; {pid}</a>'}}
    if lic_url:
        ext["LicenseUrl"] = {"value": lic_url}
    return {"pageid": pid, "title": f"File:Photo {pid}.jpg", "categories": [{"ns": 14, "title": c} for c in cats],
            "imageinfo": [{"url": UPLOAD.format(pid), "descriptionurl": f"https://commons.test/wiki/File:Photo_{pid}.jpg",
                           "mime": mime, "width": size[0], "height": size[1], "metadata": meta, "extmetadata": ext}]}


class _Resp:
    def __init__(self, status_code=200, content=b"", headers=None, data=None):
        self.status_code, self.content, self.headers, self._data = status_code, content, headers or {}, data

    def json(self):
        return self._data


class FakeCommons:
    """Serves the API from `categories` (name -> list of pages) and files from `files` (pageid -> [responses];
    the last one repeats). Records every request."""

    def __init__(self, categories, files=None, delay=0.0):
        self.categories, self.files, self.delay = categories, files or {}, delay
        self.pages = {p["pageid"]: p for ps in categories.values() for p in ps}
        self.api, self.downloads, self.headers = [], [], []
        self.lock, self.now, self.peak = threading.Lock(), 0, 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.headers.append(headers)
        if params is not None:
            self.api.append(params)
            if params.get("list") == "allcategories":
                return _Resp(data={"query": {"allcategories": [
                    {"category": c, "files": len(ps)} for c, ps in self.categories.items()]}})
            if params.get("list") == "categorymembers":
                ps = self.categories[params["cmtitle"].removeprefix("Category:")]
                return _Resp(data={"query": {"categorymembers": [{"pageid": p["pageid"]} for p in ps]}})
            ids = [int(i) for i in params["pageids"].split("|")]
            return _Resp(data={"query": {"pages": [json.loads(json.dumps(self.pages[i])) for i in ids]}})
        pid = int(url.rsplit("/", 1)[1].removesuffix(".jpg"))
        with self.lock:
            self.downloads.append(pid)
            self.now += 1
            self.peak = max(self.peak, self.now)
            seq = self.files.get(pid, [_Resp(content=_jpeg())])
            resp = seq.pop(0) if len(seq) > 1 else seq[0]
        try:
            time.sleep(self.delay)
        finally:
            with self.lock:
                self.now -= 1
        return resp


@pytest.fixture
def fast(tmp_path, monkeypatch):
    monkeypatch.setenv("SIXFINGERS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(download, "COMMONS_PAUSE_S", (0, 0))
    monkeypatch.setattr(download, "COMMONS_API_PAUSE_S", 0)
    monkeypatch.setattr(download, "COMMONS_BACKOFF_S", 0)
    monkeypatch.setattr(download, "COMMONS_MIN_FILES", 1)

    def install(fake):
        monkeypatch.setattr(download.requests, "get", fake.get)
        return fake
    return install


def _rejected() -> dict[str, str]:
    path = source_dir(SRC["name"]) / "_rejected.txt"
    return dict(line.split() for line in path.read_text().splitlines()) if path.exists() else {}


def test_filters_licences_and_attribution(fast, monkeypatch):
    monkeypatch.setattr(download, "COMMONS_PER_CATEGORY", 20)
    monkeypatch.setattr(download, "COMMONS_BATCH", 20)
    pages = [
        _page(1),                                            # CC BY-SA: kept
        _page(2, lic="CC BY 2.0", lic_url="https://cc.test/by/2.0"),
        _page(3, lic="CC0", lic_url=None),
        _page(4, lic="Public domain", lic_url=None),
        _page(10, size=(3000, 900)),                         # short side under 1000 px
        _page(11, mime="image/png"),
        _page(12, lic="CC BY-NC 2.0"),
        _page(13, lic="GFDL"),
        _page(14, lic="CC BY-ND 4.0"),
        _page(15, model=None),                               # make without model
        _page(16, make=None),
        _page(17, taken="2023:05:01 10:00:00"),             # taken after the AI cut-off
        _page(18, taken="2022:01:01 00:00:00"),
        _page(19, taken=None),
        _page(20, cats=("Category:Birds", "Category:AI-generated images")),
        _page(21, cats=("Category:Images made with Midjourney",)),
        _page(22, cats=("Category:Stable Diffusion images",)),
        _page(23, cats=("Category:DALL-E 3",)),
        _page(30),                                           # passes the filters but the file is gone
        {"pageid": 31, "missing": True},
    ]
    big = io.BytesIO()
    Image.new("RGB", (2400, 1200), (10, 20, 30)).save(big, "JPEG")
    fake = fast(FakeCommons({"Taken with Canon EOS 5D": pages, "Taken with Stable Diffusion": [_page(40)]},
                            files={1: [_Resp(content=big.getvalue())], 30: [_Resp(404)]}))

    download.commons(SRC, 50)

    idx = {r["file"]: r for r in read_index(SRC["name"])}
    assert set(idx) == {"1.jpg", "2.jpg", "3.jpg", "4.jpg"}
    assert sorted(fake.downloads) == [1, 2, 3, 4, 30]  # rejected files are never downloaded
    one = idx["1.jpg"]
    assert one == {**one, "source": "wikimedia_commons", "generator": "real_commons", "side": "real",
                   "group": "commons:1", "title": "File:Photo 1.jpg", "author": "Author & 1",
                   "license_name": "CC BY-SA 4.0", "license_url": "https://cc.test/by-sa/4.0",
                   "landing_url": "https://commons.test/wiki/File:Photo_1.jpg", "camera_make": "Canon",
                   "camera_model": "Canon EOS 5D", "date_taken": "2015:06:13 14:22:01",
                   "category": "Taken with Canon EOS 5D"}
    assert idx["3.jpg"]["license_url"] is None and idx["3.jpg"]["license_name"] == "CC0"
    with Image.open(source_dir(SRC["name"]) / "1.jpg") as im:
        assert im.size == (1600, 800) and im.format == "JPEG"
    assert _rejected() == {"10": "small", "11": "not_jpeg", "12": "licence", "13": "licence", "14": "licence",
                           "15": "no_camera", "16": "no_camera", "17": "date", "18": "date", "19": "date",
                           "20": "ai_category", "21": "ai_category", "22": "ai_category", "23": "ai_category",
                           "30": "dead", "31": "missing"}
    assert not any(p.get("cmtitle") == "Category:Taken with Stable Diffusion" for p in fake.api)
    assert all(h == download.COMMONS_HEADERS for h in fake.headers)
    assert all(p["maxlag"] == 5 for p in fake.api)


def test_resume_skips_kept_and_rejected_files(fast):
    cats = {"Taken with Nikon D90": [_page(i) for i in range(1, 6)] + [_page(9, lic="GFDL")]}
    fake = fast(FakeCommons(cats))
    download.commons(SRC, 2)
    assert len(read_index(SRC["name"])) == 2
    first = list(fake.downloads)

    fake.downloads.clear()
    fake.api.clear()
    download.commons(SRC, 3)  # the category list is cached; one more file is fetched
    assert len(read_index(SRC["name"])) == 3
    assert len(fake.downloads) == 1 and fake.downloads[0] not in first
    assert not any(p.get("list") == "allcategories" for p in fake.api)
    assert all("9" not in p.get("pageids", "").split("|") for p in fake.api)  # rejected: never checked again

    fake.downloads.clear()
    fake.api.clear()
    download.commons(SRC, 3)  # the category already holds its 3 files: no request at all
    assert fake.downloads == [] and fake.api == []
    assert len({r["file"] for r in read_index(SRC["name"])}) == 3


def test_caps_each_category_and_downloads_two_at_a_time(fast, monkeypatch):
    monkeypatch.setattr(download, "COMMONS_PER_CATEGORY", 2)
    cats = {f"Taken with Phone {c}": [_page(c * 100 + i) for i in range(6)] for c in range(1, 4)}
    fake = fast(FakeCommons(cats, delay=0.05))

    download.commons(SRC, 50)

    rows = read_index(SRC["name"])
    assert len(rows) == 6
    assert {r["category"] for r in rows} == set(cats)
    assert all(sum(r["category"] == c for r in rows) == 2 for c in cats)
    assert fake.peak == download.COMMONS_WORKERS
    assert len(fake.downloads) == 6


def test_honours_retry_after_and_maxlag(fast, monkeypatch):
    sleeps = []
    monkeypatch.setattr(download, "time", SimpleNamespace(sleep=sleeps.append, monotonic=time.monotonic))
    fake = fast(FakeCommons({"Taken with Canon": [_page(1)]}, files={1: [_Resp(429, headers={"Retry-After": "3"}),
                                                                         _Resp(content=_jpeg())]}))
    real_get, lagged = fake.get, []

    def get(url, params=None, **kw):
        if params is not None and params.get("list") == "allcategories" and not lagged:
            lagged.append(1)
            return _Resp(headers={"MediaWiki-API-Error": "maxlag", "Retry-After": "7"},
                         data={"error": {"code": "maxlag"}})
        return real_get(url, params=params, **kw)

    monkeypatch.setattr(download.requests, "get", get)
    download.commons(SRC, 1)

    assert len(read_index(SRC["name"])) == 1
    assert 7.0 in sleeps and 3.0 in sleeps
    assert sum(p.get("list") == "allcategories" for p in fake.api) == 1  # the retried request


def test_stops_when_every_download_fails(fast, monkeypatch):
    monkeypatch.setattr(download, "COMMONS_RETRIES", 0)
    cats = {f"Taken with Cam {c}": [_page(c * 10 + i) for i in range(3)] for c in range(1, 5)}
    fake = fast(FakeCommons(cats, files={c * 10 + i: [_Resp(503)] for c in range(1, 5) for i in range(3)}))

    download.commons(SRC, 10)  # a normal return, so the other sources still run

    assert read_index(SRC["name"]) == []
    assert _rejected() == {}
    assert len(fake.downloads) <= download.COMMONS_MAX_ERRORS + download.COMMONS_WORKERS


def test_api_failure_stops_cleanly(fast, monkeypatch):
    monkeypatch.setattr(download, "COMMONS_RETRIES", 1)
    monkeypatch.setattr(download.requests, "get", lambda *a, **k: _Resp(503))
    download.commons(SRC, 5)
    assert read_index(SRC["name"]) == []


def test_source_is_enabled_and_never_report_test():
    src = next(s for s in enabled_sources() if s["name"] == "wikimedia_commons")
    assert src["side"] == "real" and src["generator"] == "real_commons"
    assert download.DOWNLOADERS[src["downloader"]] is download.commons
    groups = [f"commons:{i}" for i in range(500)]
    split = assign_splits(pd.DataFrame({"generator": "real_commons", "group": groups}), holdout_generator())
    assert set(split) == {"train", "val", "gate"}

"""Per-IP judge quota, budget on detector failure, judge input, the concurrency cap, and health readiness while backbones load."""
import base64
import io
import json
import sys
import threading
import time
from types import ModuleType

from fastapi.testclient import TestClient
from PIL import Image

from api.detector import FeatureEmbedder
from api.main import create_app
from ml.common.judge_backend import prepare_image
from tests.api.conftest import JUDGE_JSON, FakeClient, StubEmbedder, img_bytes, post


def _app(models_dir, tmp_path, embedder, client=None):
    return create_app(embedder, client, models_dir=models_dir,
                      budget_path=tmp_path / "budget.json")


def _count(tmp_path):
    p = tmp_path / "budget.json"
    return json.loads(p.read_text())["count"] if p.exists() else 0


def test_per_ip_judge_quota(build, tmp_path, monkeypatch):
    monkeypatch.setenv("SIXFINGERS_JUDGE_PER_IP_DAY", "1")
    fake = FakeClient(JUDGE_JSON)
    c = build(fake)
    assert post(c, img_bytes(255)).json()["source"] == "judge"
    j = post(c, img_bytes(255)).json()
    assert j["source"] == "detector_only" and j["judge_unavailable_reason"] == "budget"
    assert len(fake.calls) == 1 and _count(tmp_path) == 1  # the over-quota check spent no global budget


def test_detector_failure_keeps_paid_judge_call_spent(models_dir, tmp_path):
    class Broken(StubEmbedder):
        def embed(self, backbone, image):
            time.sleep(0.2)  # let the judge call finish first
            raise RuntimeError("boom")
    fake = FakeClient(JUDGE_JSON)
    assert post(TestClient(_app(models_dir, tmp_path, Broken(), fake)), img_bytes()).status_code == 503
    assert len(fake.calls) == 1 and _count(tmp_path) == 1  # the call happened, so it is never given back


def test_judge_gets_same_jpeg_as_from_original_bytes(build):
    b = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6  # rotated: the judge must still see it upright
    Image.new("RGB", (2400, 1200), (200, 60, 30)).save(b, "JPEG", exif=exif)
    data = b.getvalue()
    fake = FakeClient(JUDGE_JSON)
    assert post(build(fake), data, "a.jpg", "image/jpeg").json()["source"] == "judge"
    sent = fake.calls[0]["messages"][0]["content"][0]["source"]["data"]
    assert base64.b64decode(sent) == prepare_image(data)


def test_no_model_never_calls_judge(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    fake = FakeClient(JUDGE_JSON)
    assert post(TestClient(_app(empty, tmp_path, StubEmbedder(), fake)), img_bytes()).status_code == 503
    assert fake.calls == [] and _count(tmp_path) == 0


def test_busy_returns_503(models_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("SIXFINGERS_MAX_CONCURRENT", "1")
    monkeypatch.setattr("api.main.BUSY_WAIT_S", 0.2)
    entered, release = threading.Event(), threading.Event()

    class Slow(StubEmbedder):
        def embed(self, backbone, image):
            entered.set()
            release.wait(5)
            return super().embed(backbone, image)
    app = _app(models_dir, tmp_path, Slow())
    first = []
    t = threading.Thread(target=lambda: first.append(post(TestClient(app), img_bytes()).status_code))
    t.start()
    assert entered.wait(5)
    r = post(TestClient(app), img_bytes())
    release.set()
    t.join(5)
    assert r.status_code == 503 and "busy" in r.json()["detail"]
    assert first == [200]


def test_health_not_ready_until_backbones_load(models_dir, tmp_path):
    class Lazy(StubEmbedder):
        def __init__(self):
            self.done = set()

        def load(self, backbone):
            self.done.add(backbone)

        def loaded(self, backbone):
            return backbone in self.done
    app = _app(models_dir, tmp_path, Lazy())
    r = TestClient(app).get("/v1/health")
    assert r.status_code == 503 and r.json()["status"] == "loading"
    with TestClient(app) as c:  # startup runs the warm-up
        for _ in range(50):
            if c.get("/v1/health").status_code == 200:
                break
            time.sleep(0.05)
        assert c.get("/v1/health").json()["status"] == "ok"


def test_feature_embedder_loads_once_and_normalizes_once(monkeypatch):
    loads, norms = [], []

    class FakeBackbone:
        def embed(self, images):
            return [[float(images[0].size[0])]]

    def load_backbone(name):
        loads.append(name)
        time.sleep(0.1)  # widen the race window
        return FakeBackbone()
    fake = ModuleType("ml.features.backbones")
    fake.load_backbone = load_backbone
    monkeypatch.setitem(sys.modules, "ml.features.backbones", fake)
    import ml.data.normalize as norm
    real = norm.normalize_image
    monkeypatch.setattr(norm, "normalize_image", lambda im: norms.append(1) or real(im))

    e = FeatureEmbedder()
    ts = [threading.Thread(target=e.load, args=("clip_vitl14",)) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert loads == ["clip_vitl14"]
    img = Image.new("RGB", (600, 600), (10, 20, 30))
    assert e.embed("clip_vitl14", img).shape == (1, 1)
    e.embed("dinov2_vitl14", img)
    assert norms == [1]  # both backbones reuse one normalized copy
    e.embed("clip_vitl14", Image.new("RGB", (600, 600)))
    assert norms == [1, 1]

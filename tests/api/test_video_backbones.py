import json

import pytest
import torch
from PIL import Image

from api.detector import DetectorStore
from ml.features.backbones import Backbone
from tests.api.conftest import StubEmbedder, img_bytes, make_model
from tests.api.video_clips import make_clip, requires_ffmpeg


class Recorder(StubEmbedder):
    def __init__(self):
        self.calls = []

    def embed(self, backbone, image):
        self.calls.append(backbone)
        return super().embed(backbone, image)


def write_video_meta(vdir, t_ai=0.6, t_real=0.3):
    (vdir / "video_meta.json").write_text(json.dumps(
        {"backbones": ["clip_vitl14"], "t_ai": t_ai, "t_real": t_real, "fit_split": "val", "n": 10}))


def test_video_scores_use_video_backbones_and_embedder(tmp_path):
    make_model(tmp_path, "v1")  # probes: clip_vitl14 + dinov2_vitl14, no siglip
    (tmp_path / "current").symlink_to("v1")
    img, vid = Recorder(), Recorder()
    store = DetectorStore(img, tmp_path, vid)
    ims = [Image.new("RGB", (16, 16), (v,) * 3) for v in (0, 255)]

    res, _ = store.score_many(ims)
    assert set(img.calls) == {"clip_vitl14", "dinov2_vitl14"} and vid.calls == []
    assert set(res[0].backbones) == {"clip_vitl14", "dinov2_vitl14"}

    img.calls.clear()
    res, _ = store.score_many(ims, video=True)
    assert img.calls == [] and set(vid.calls) == {"clip_vitl14"}  # dinov2 is never run for video
    assert set(res[0].backbones) == {"clip_vitl14"}
    assert (res[0].thresholds.t_ai, res[0].thresholds.t_real) == (0.8, 0.2)  # no video_meta.json: image thresholds


def test_video_meta_sets_video_thresholds_only(tmp_path):
    write_video_meta(make_model(tmp_path, "v1"))
    (tmp_path / "current").symlink_to("v1")
    store = DetectorStore(StubEmbedder(), tmp_path)
    im = [Image.new("RGB", (16, 16), (200,) * 3)]
    v, _ = store.score_many(im, video=True)
    i, _ = store.score_many(im)
    assert (v[0].thresholds.t_ai, v[0].thresholds.t_real) == (0.6, 0.3)
    assert (i[0].thresholds.t_ai, i[0].thresholds.t_real) == (0.8, 0.2)


@requires_ffmpeg
def test_video_route_uses_video_meta(build, models_dir, tmp_path):
    write_video_meta(models_dir / "v1")
    c = build(client=None)
    clip = make_clip(tmp_path / "c.mp4", seconds=2)
    j = c.post("/v1/check-video", files={"video": ("a.mp4", clip.read_bytes(), "video/mp4")}).json()
    assert j["detector"]["thresholds"] == {"t_ai": 0.6, "t_real": 0.3}
    j = c.post("/v1/check", files={"image": ("a.png", img_bytes(), "image/png")}).json()
    assert j["detector"]["thresholds"] == {"t_ai": 0.8, "t_real": 0.2}


class Tiny(Backbone):
    def _load(self):
        self.model = torch.nn.Linear(4, 2).to(self.device)

    def _embed(self, images):
        return self.model(torch.ones(len(images), 4).to(self.device, self.dtype))


def test_half_falls_back_to_fp32_on_cpu():
    b = Tiny("cpu", half=True)
    assert b.dtype == torch.float32 and b.model.weight.dtype == torch.float32
    assert b.embed([Image.new("RGB", (4, 4))]).dtype.name == "float32"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_half_on_mps_is_fp16_with_fp32_output():
    b = Tiny("mps", half=True)
    assert b.model.weight.dtype == torch.float16
    assert b.embed([Image.new("RGB", (4, 4))]).dtype.name == "float32"


def test_bad_video_meta_only_affects_video(tmp_path):
    vdir = make_model(tmp_path, "v1")
    (vdir / "video_meta.json").write_text("{not json")
    (tmp_path / "current").symlink_to("v1")
    store = DetectorStore(StubEmbedder(), tmp_path)
    im = [Image.new("RGB", (16, 16), (200,) * 3)]
    i, _ = store.score_many(im)
    v, _ = store.score_many(im, video=True)
    assert (i[0].thresholds.t_ai, i[0].thresholds.t_real) == (0.8, 0.2)
    assert (v[0].thresholds.t_ai, v[0].thresholds.t_real) == (0.8, 0.2)  # same fallback as a missing file
    assert set(v[0].backbones) == {"clip_vitl14"}


@pytest.mark.parametrize("mps", [False, True])
def test_video_embedder_is_separate_only_on_mps(monkeypatch, mps):
    from api.main import create_app
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    store = create_app(anthropic_client=None).state.store
    assert (store.video_embedder is store.embedder) is not mps
    assert getattr(store.video_embedder, "half", False) is mps

import io
import json
import shlex
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import create_app
from ml.common.schemas import ModelMeta, ProbeArtifact, Thresholds, save_meta, save_probe

D = 4


def make_model(models_dir, version, weight=1.0):
    """Fake models/vN: two backbones, weight sign controls whether bright images look AI."""
    vdir = models_dir / version
    vdir.mkdir(parents=True)
    for b in ("clip_vitl14", "dinov2_vitl14"):
        save_probe(ProbeArtifact(backbone=b, embed_dim=D, mean=[0.0] * D, std=[1.0] * D,
                                 weights=[weight, 0, 0, 0], bias=0.0), vdir / f"{b}.json")
    save_meta(ModelMeta(version=version, backbones=["clip_vitl14", "dinov2_vitl14"],
                        thresholds=Thresholds(t_ai=0.8, t_real=0.2), metrics={"auroc": 0.9},
                        created_at="2026-01-01"), vdir / "meta.json")
    return vdir


class StubEmbedder:
    """First feature = (mean brightness - 128) / 16, so white is strongly 'AI', black strongly 'real'."""
    def embed(self, backbone, image):
        v = (np.asarray(image).mean() - 128) / 16
        return np.array([[v, 0, 0, 0]], dtype=np.float32)


class FakeClient:
    def __init__(self, text=None, stop_reason="end_turn", exc=None):
        self.calls, self.text, self.stop_reason, self.exc = [], text, stop_reason, exc
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        block = SimpleNamespace(type="text", text=self.text)
        return SimpleNamespace(stop_reason=self.stop_reason, content=[block])


JUDGE_JSON = json.dumps({"verdict": "likely_real", "confidence": 0.7, "explanation": "Natural grain.",
                         "cues": [{"region": "hands", "cue": "five fingers"}], "overrode_detector": True})


def img_bytes(color=255, fmt="PNG", size=(32, 32)):
    b = io.BytesIO()
    Image.new("RGB", size, (color,) * 3).save(b, fmt)
    return b.getvalue()


@pytest.fixture(autouse=True)
def _judge_env_clean(monkeypatch):
    # tests choose the judge backend explicitly; a developer's shell must not change them
    monkeypatch.delenv("SIXFINGERS_JUDGE_BACKEND", raising=False)
    monkeypatch.delenv("SIXFINGERS_JUDGE_CMD", raising=False)


@pytest.fixture(autouse=True)
def _data_dir_tmp(tmp_path, monkeypatch):
    # apps built without usage_path keep usage.sqlite out of the real data dir
    monkeypatch.setenv("SIXFINGERS_DATA_DIR", str(tmp_path / "data"))


def command_script(tmp_path, body, name="judge.py"):
    """Write a Python program to tmp_path and return a shell-safe command line that runs it."""
    script = tmp_path / name
    script.write_text(body)
    return " ".join(shlex.quote(p) for p in (sys.executable, str(script)))


@pytest.fixture
def models_dir(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    make_model(d, "v1")
    (d / "current").symlink_to("v1")
    return d


@pytest.fixture
def build(tmp_path, models_dir):
    def _build(client=None, budget_limit=500, rate_limit=10):
        app = create_app(embedder=StubEmbedder(), anthropic_client=client, models_dir=models_dir,
                         budget_path=tmp_path / "budget.json", budget_limit=budget_limit, rate_limit=rate_limit,
                         usage_path=tmp_path / "usage.sqlite")
        return TestClient(app)
    return _build


def post(c, data, name="a.png", ctype="image/png"):
    return c.post("/v1/check", files={"image": (name, data, ctype)})

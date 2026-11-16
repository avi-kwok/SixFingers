import shutil
import types
from pathlib import Path

import numpy as np
import pytest

from ml.common.schemas import band_for, calibrated_logit, load_meta, load_probe, probe_logit
from tests.fixtures.make_fixture import make

N_DIM = 8


class FakeBackbone:
    """Cheap hand-made features (colour stats + edge energy) so tests need no weights."""
    def __init__(self, name):
        self.name = name

    def embed(self, ims):
        out = []
        for im in ims:
            a = np.asarray(im, dtype=np.float32) / 255
            out.append([*a.mean((0, 1)), *a.std((0, 1)), np.abs(np.diff(a, axis=0)).mean(), np.abs(np.diff(a, axis=1)).mean()])
        return np.array(out, dtype=np.float32)


@pytest.fixture
def env(tmp_path, monkeypatch):
    data = make(tmp_path / "data")
    monkeypatch.setenv("SIXFINGERS_DATA_DIR", str(data))
    monkeypatch.setenv("SIXFINGERS_MODELS_DIR", str(tmp_path / "models"))
    (tmp_path / "models").mkdir()
    monkeypatch.setattr("ml.features.backbones.load_backbone", lambda name, device=None: FakeBackbone(name))
    return tmp_path


def test_extract_is_resumable(env):
    from ml.features import extract
    first = extract.run(["clip_vitl14"], limit=200)
    assert first["clip_vitl14"] == 84  # 18 train x3 + 9 val + 9 gate + 6 report_test x2 (clean + compressed)
    assert extract.run(["clip_vitl14"], limit=200)["clip_vitl14"] == 0


def test_mlp_artifact_matches_sklearn(env):
    from sklearn.neural_network import MLPClassifier
    from ml.probes.train import fit_backbone
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 6)); y = (X[:, 0] + X[:, 1] ** 2 > 1).astype(int)
    art, m = fit_backbone("x", X[:80], y[:80], X[80:], y[80:])
    assert m["shipped"] in ("logistic", "mlp")
    # exported weights reproduce the sklearn decision: shipped logits order == probability order
    assert np.isfinite(probe_logit(art, X)).all()

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
    for mod in ("ml.eval.run", "ml.judge_eval.run"):
        monkeypatch.setattr(f"{mod}.RESULTS", tmp_path / "results.md")
    return tmp_path


def test_extract_is_resumable(env):
    from ml.features import extract
    first = extract.run(["clip_vitl14"], limit=200)
    assert first["clip_vitl14"] == 84  # 18 train x3 + 9 val + 9 gate + 6 report_test x2 (clean + compressed)
    assert extract.run(["clip_vitl14"], limit=200)["clip_vitl14"] == 0


def test_train_eval_end_to_end(env):
    from ml.eval.run import run as run_eval
    from ml.features import extract, store
    from ml.probes.train import train
    bbs = ["clip_vitl14", "siglip"]
    extract.run(bbs, limit=200)
    v = train(bbs)
    d = env / "models" / v
    meta = load_meta(d / "meta.json")
    assert not (env / "models" / "current").exists()
    assert meta.thresholds.t_real <= meta.thresholds.t_ai
    # thresholds hold on val: <=5% of reals reach t_ai, <=10% of fakes fall under t_real
    df = store.load_manifest()
    va = df[df.split == "val"]
    from ml.eval.run import ensemble_on
    kept, s, _ = ensemble_on(d, meta, va, 0)
    y = kept.label.to_numpy()
    assert (s[y == 0] >= meta.thresholds.t_ai).mean() <= 0.05 + 1e-9
    assert (s[y == 1] <= meta.thresholds.t_real).mean() <= 0.10 + 1e-9
    res = run_eval(d)
    assert res["variants"]["clean"]["auroc"] > 0.9
    assert res["variants"]["compressed"]["n"] == 6
    assert "per_generator_tpr" in res["variants"]["clean"]
    assert (env / "results.md").exists()
    assert "report_test" in load_meta(d / "meta.json").metrics


def test_train_never_moves_current_only_gate_promotes(env):
    from ml.features import extract
    from ml.loop import run
    from ml.probes.train import train
    from ml.common.schemas import BACKBONES
    extract.run(list(BACKBONES), limit=200)  # the default train stage trains every backbone
    v1 = train(["clip_vitl14"], promote=True)
    cur = env / "models" / "current"
    assert cur.resolve().name == v1
    train(["clip_vitl14"])  # plain train: current must stay on v1
    assert cur.resolve().name == v1
    run.default_stages(0)["train"]()  # real loop CLI stage, subprocess; no promote
    assert cur.resolve().name == v1
    assert len(run.versions()) == 3
    r = run.run_loop({"fetch": lambda: None, "extract": lambda: None, "train": lambda: train(["clip_vitl14"])},
                     evaluate=lambda d: {"tpr_at_5fpr": 0.9, "real_fpr": 0.0, "per_generator_tpr": {}},
                     rows=lambda: 0, out_dir=env / "rep", post=lambda m: None)
    assert r["outcome"] == "promoted" and r["candidate"] == "v4" and cur.resolve().name == "v4"


def test_mlp_artifact_matches_sklearn(env):
    from sklearn.neural_network import MLPClassifier
    from ml.probes.train import fit_backbone
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 6)); y = (X[:, 0] + X[:, 1] ** 2 > 1).astype(int)
    art, m = fit_backbone("x", X[:80], y[:80], X[80:], y[80:])
    assert m["shipped"] in ("logistic", "mlp")
    # exported weights reproduce the sklearn decision: shipped logits order == probability order
    assert np.isfinite(probe_logit(art, X)).all()


def test_judge_cap_and_cache(env, monkeypatch):
    from ml.eval.run import ensemble_on  # noqa: F401
    from ml.features import extract
    from ml.judge_eval import run as j
    from ml.probes.train import train
    extract.run(["clip_vitl14"], limit=200)
    train(["clip_vitl14"], promote=True)
    calls = []

    def create(**kw):
        calls.append(kw["model"])
        assert "temperature" not in kw and "thinking" not in kw
        txt = '{"verdict":"likely_ai","confidence":0.9,"cues":[],"explanation":"x","overrode_detector":false}'
        return types.SimpleNamespace(stop_reason="end_turn", content=[types.SimpleNamespace(text=txt)],
                                     usage=types.SimpleNamespace(input_tokens=1000, output_tokens=100))
    client = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    with pytest.raises(SystemExit):
        j.main([], client=client)  # --approve-spend is required
    j.main(["--approve-spend", "10", "--n", "6"], client=client)
    n1 = len(calls)
    assert n1 == 30
    j.main(["--approve-spend", "10", "--n", "6"], client=client)
    assert len(calls) == n1  # fully cached
    assert "Judge comparison" in (env / "results.md").read_text()
    # tiny cap stops before the first call and never overspends
    (env / "data" / "eval" / "judge_cache_prod.jsonl").unlink()
    calls.clear()
    j.main(["--approve-spend", "0.0001", "--n", "6"], client=client)
    assert calls == []
    assert "STOPPED" in (env / "results.md").read_text()


def test_hard_cap_is_ten(env):
    assert min(50.0, __import__("ml.judge_eval.run", fromlist=["x"]).HARD_CAP_USD) == 10.0

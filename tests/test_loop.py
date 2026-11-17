import numpy as np
import pandas as pd
import pytest

from ml.common.schemas import ModelMeta, ProbeArtifact, Thresholds, save_meta, save_probe
from ml.loop import gate, report, run


def M(tpr, fpr=0.03, gens=None):
    return {"tpr_at_5fpr": tpr, "real_fpr": fpr, "per_generator_tpr": gens or {"a": tpr, "b": tpr}}


def test_decide_rules():
    assert gate.decide(M(0.9), M(0.8))[0]
    assert gate.decide(M(0.9), None)[0]
    assert not gate.decide(M(0.7), M(0.8))[0]
    assert not gate.decide(M(0.9, fpr=0.06), M(0.8))[0]
    ok, why = gate.decide(M(0.9, gens={"a": 0.95, "b": 0.7}), M(0.8, gens={"a": 0.8, "b": 0.75}))
    assert not ok and "generator b" in why[0]
    assert gate.decide(M(0.9, gens={"a": 0.95, "b": 0.74}), M(0.8, gens={"a": 0.8, "b": 0.75}))[0]


def test_metrics_and_gate_split_only():
    rng = np.random.default_rng(0)
    s = np.r_[rng.uniform(0, 0.5, 200), rng.uniform(0.4, 1, 100)]
    lab = np.r_[np.zeros(200), np.ones(100)]
    gen = np.array(["real"] * 200 + ["a"] * 50 + ["b"] * 50)
    m = gate.metrics(s, lab, gen, t_ai=0.9)
    assert 0.5 < m["tpr_at_5fpr"] <= 1 and set(m["per_generator_tpr"]) == {"a", "b"}
    with pytest.raises(ValueError):
        gate.metrics(s[:200], lab[:200], gen[:200], 0.9)


def make_model(models, v, w):
    d = models / v
    d.mkdir()
    save_probe(ProbeArtifact(backbone="clip_vitl14", embed_dim=2, mean=[0, 0], std=[1, 1], weights=w, bias=0), d / "clip_vitl14.json")
    save_meta(ModelMeta(version=v, backbones=["clip_vitl14"], thresholds=Thresholds(t_ai=0.8, t_real=0.2)), d / "meta.json")
    return d


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SIXFINGERS_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("SIXFINGERS_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "models").mkdir()
    return tmp_path


def fake_loop(env, cand_w):
    models = env / "models"
    make_model(models, "v1", [3.0, 0.0])
    (models / "current").symlink_to("v1")
    rng = np.random.default_rng(1)
    n = 200
    y = np.r_[np.zeros(n), np.ones(n)]
    X = np.c_[np.r_[rng.normal(-1, 1, n), rng.normal(1, 1, n)], rng.normal(0, 1, 2 * n)]
    ids = [f"i{k}" for k in range(2 * n)]
    man = pd.DataFrame({"image_id": ids, "label": y, "generator": ["real"] * n + ["g"] * n,
                        "split": ["gate"] * (2 * n)})
    man.loc[0, "split"] = "report_test"  # must be ignored
    lookup = dict(zip(ids, X))
    loader = lambda b, i: np.stack([lookup[k] for k in i])  # noqa: E731
    seen = []

    def train():
        make_model(models, "v2", cand_w)

    def ev(d):
        seen.append(d.name)
        return gate.evaluate(d, man, loader)

    stages = {"fetch": lambda: None, "extract": lambda: None, "train": train}
    posted = []
    r = run.run_loop(stages, ev, rows=lambda: 0, out_dir=env / "rep", post=posted.append)
    return r, posted


def test_reject_worse(env):
    # a noise-only candidate must be rejected and leave the symlink alone
    r, posted = fake_loop(env, [0.0, 3.0])
    assert r["outcome"] == "rejected" and (env / "models/current").resolve().name == "v1"
    assert (env / "rep" / f"{r['date']}.json").exists() and posted and "rejected" in posted[0].lower()
    assert run.rollback("v1") == "v1"


def test_promote_swaps_symlink(env):
    r, _ = fake_loop(env, [3.0, 0.0])  # identical quality -> TPR not lower, so promoted
    assert r["gate"]["candidate_metrics"]["n_real"] == 199  # report_test row excluded
    assert r["outcome"] == "promoted" and (env / "models/current").resolve().name == "v2"
    assert run.rollback() == "v1" and (env / "models/current").resolve().name == "v1"


def test_stage_error_reported_not_raised(env):
    def boom():
        raise RuntimeError("net down")

    posted = []
    r = run.run_loop({"fetch": boom, "extract": boom, "train": boom}, None, rows=lambda: 0, out_dir=env / "rep", post=posted.append)
    assert r["outcome"] == "error" and "net down" in posted[0]


def test_redact_and_no_url():
    assert report.redact("fail https://x/y", "https://x/y") == "fail <redacted>"
    assert report.post_discord("hi", url=None) is False


def test_default_stages_grow_capped(monkeypatch):
    calls = []
    monkeypatch.setattr(run, "sh", lambda args: calls.append(args))
    st = run.default_stages(7)
    st["fetch"]()
    st["extract"]()
    st["train"]()
    assert calls == [
        ["ml.data.download", "--grow", "7"],
        ["ml.data.generate_local", "--grow", "7"],
        ["ml.data.build", "--skip-download", "--limit", str(run.ALL)],
        ["ml.features.extract", "--limit", "0"],
        ["ml.probes.train"],
    ]


def test_sh_raises_on_failure():
    with pytest.raises(RuntimeError):
        run.sh(["ml.this_module_does_not_exist"])

import json

import numpy as np
import pytest

from ml.audio import train
from ml.common.schemas import load_probe


def fake_cache(n, shift, seed, dim=8):
    """Layer 3 separates real/fake by `shift`; layer 6 is pure noise."""
    rng = np.random.default_rng(seed)
    y = (np.arange(n) % 2).astype(np.int8)
    l3 = rng.normal(size=(n, dim)) + shift * y[:, None]
    return {"y": y, "L3": l3.astype(np.float16), "L6": rng.normal(size=(n, dim)).astype(np.float16),
            "path": np.array([f"{i}.wav" for i in range(n)]), "attack": np.array(["x"] * n)}


def caches():
    return {"la_train": fake_cache(200, 3.0, 0), "la_dev": fake_cache(200, 3.0, 1),
            "la_eval": fake_cache(200, 3.0, 2), "itw": fake_cache(200, 0.5, 3)}


def test_eer_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert train.eer(y, np.array([0.1, 0.2, 0.8, 0.9]))[0] == 0.0
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    assert train.eer(y, rng.random(4000))[0] == pytest.approx(0.5, abs=0.05)


def test_run_picks_informative_layer_and_reports():
    r = train.run(caches(), layers=(3, 6))
    assert r["layer"] == 3
    assert r["per_layer_dev"][6]["dev_eer"] > 0.3
    assert r["results"]["la_eval"]["eer"] < 0.1 and r["results"]["la_eval"]["auroc"] > 0.95
    assert r["results"]["itw"]["eer"] > r["results"]["la_eval"]["eer"]  # harder shift = worse
    assert 0 < r["threshold"] < 1 and r["probe"].temperature > 0
    for m in r["results"].values():
        assert set(m) >= {"eer", "auroc", "tpr_at_5fpr", "acc_at_t", "fake_recall_at_t", "real_fpr_at_t"}
    assert r["results"]["la_dev"]["real_fpr_at_t"] <= 0.05  # threshold set at 5% FPR on dev reals
    assert r["per_layer_dev"][3]["dev_auroc"] > r["per_layer_dev"][6]["dev_auroc"]


def test_threshold_at_fpr_flags_at_most_share_of_reals():
    y = np.array([0] * 100 + [1] * 10)
    p = np.concatenate([np.linspace(0, 1, 100), np.full(10, 0.5)])
    t = train.threshold_at_fpr(y, p, 0.05)
    assert (p[:100] >= t).sum() == 5
    assert train.threshold_at_fpr(y, p, 0.0) > 1.0  # flags no real clip


def test_report_accuracy_at_threshold():
    m = train.report(np.array([0, 0, 1, 1]), np.array([0.1, 0.6, 0.4, 0.9]), 0.5)
    assert m["acc_at_t"] == 0.5 and m["fake_recall_at_t"] == 0.5 and m["real_fpr_at_t"] == 0.5


def test_main_writes_probe_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("SIXFINGERS_MODELS_DIR", str(tmp_path))
    c = caches()
    monkeypatch.setattr(train, "load_cache", lambda name: c[name])
    monkeypatch.setattr(train.run, "__defaults__", ((3, 6),))
    train.main(["--version", "t1", "--rule", "dev"])
    d = tmp_path / "audio" / "t1"
    meta = json.loads((d / "meta.json").read_text())
    assert meta["layer"] == 3 and meta["results"]["itw"]["n"] == 200
    assert load_probe(d / "probe.json").backbone == "xlsr300m_L3"
    with pytest.raises(SystemExit):
        train.main(["--version", "t1"])


def test_split_half_is_stratified_disjoint_and_fixed():
    c = fake_cache(101, 3.0, 0)
    a, b = train.split_half(c)
    assert len(a["y"]) + len(b["y"]) == 101 and not set(a["path"]) & set(b["path"])
    assert abs(int(a["y"].sum()) - int(b["y"].sum())) <= 1
    assert list(train.split_half(c)[0]["path"]) == list(a["path"])


def test_run_sel_chooses_on_selection_sets_only():
    c = caches()
    tests = {"t": c["la_eval"], "itw": c["itw"]}
    r = train.run_sel({"la_train": c["la_train"]}, {"la_dev": c["la_dev"]}, tests, layers=(3, 6))
    assert r["layer"] == 3 and r["C"] in train.C_GRID and len(r["grid"]) == 2 * len(train.C_GRID)
    assert r["results"]["la_dev"]["real_fpr_at_t"] <= 0.05
    # flipping the test labels must not change any choice
    flipped = {n: {**t, "y": (1 - t["y"]).astype(np.int8)} for n, t in tests.items()}
    r2 = train.run_sel({"la_train": c["la_train"]}, {"la_dev": c["la_dev"]}, flipped, layers=(3, 6))
    assert (r2["layer"], r2["C"], r2["threshold"]) == (r["layer"], r["C"], r["threshold"])
    assert r2["results"]["t"]["auroc"] == pytest.approx(1 - r["results"]["t"]["auroc"])


def test_main_sel_rule_splits_la_eval(tmp_path, monkeypatch):
    monkeypatch.setenv("SIXFINGERS_MODELS_DIR", str(tmp_path))
    c = caches()
    monkeypatch.setattr(train, "load_cache", lambda name: c[name])
    monkeypatch.setattr(train.run_sel, "__defaults__", ((3, 6),))
    train.main(["--version", "s1"])
    meta = json.loads((tmp_path / "audio" / "s1" / "meta.json").read_text())
    assert meta["selection_sets"] == ["eval_sel", "la_dev"] and meta["test_sets"] == ["eval_test", "itw"]
    assert meta["clip_counts"]["eval_sel"]["n"] + meta["clip_counts"]["eval_test"]["n"] == 200
    assert set(meta["results"]) == {"la_dev", "eval_sel", "eval_test", "itw"}

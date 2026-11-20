"""Video probe: features, baseline rule, training, frame times and data selection, on tiny synthetic arrays."""
import io
import json
import zipfile

import numpy as np
import pytest

from api import video as vid
from ml.common.schemas import ProbeArtifact, load_probe, sigmoid
from ml.video import data, extract, probe

D = 4  # tiny embedding size


def fake_split(rng, n_videos, shift, frames=(1, 3, 5)):
    """Videos with a varying number of frames; label 1 videos are shifted by `shift` in every dimension."""
    y = np.array([i % 2 for i in range(n_videos)], np.int8)
    nf = np.array([frames[i % len(frames)] for i in range(n_videos)], np.int32)
    emb = {b: [] for b in probe.BACKBONES}
    for yi, n in zip(y, nf):
        for b in probe.BACKBONES:
            emb[b].append(rng.normal(size=(n, D)) + shift * yi)
    d = {b: np.concatenate(v).astype(np.float16) for b, v in emb.items()}
    gens = np.array(["real" if yi == 0 else ("genA", "genB")[(i // 2) % 2] for i, yi in enumerate(y)])
    return {**d, "y": y, "nf": nf, "gen": gens, "path": np.array([f"v{i}" for i in range(n_videos)])}


def unit_probe(name):
    return ProbeArtifact(backbone=name, embed_dim=D, mean=[0.0] * D, std=[1.0] * D, weights=[1.0] + [0.0] * (D - 1), bias=0.0)


def test_segments_and_cosine_steps():
    assert probe.segments(np.array([2, 1, 3])) == [slice(0, 2), slice(2, 3), slice(3, 6)]
    e = np.array([[1, 0], [1, 0], [0, 1]], np.float32)
    np.testing.assert_allclose(probe.cosine_steps(e), [0.0, 1.0], atol=1e-6)
    assert probe.cosine_steps(e[:1]).shape == (0,)


def test_baseline_is_top4_mean_of_frame_ensemble():
    probes = {b: unit_probe(b) for b in probe.BACKBONES}
    nf = np.array([6, 2])
    first = np.array([-3, -2, -1, 0, 1, 2, 5, 5], np.float32)
    d = {b: np.stack([first] + [np.zeros(8, np.float32)] * (D - 1), 1) for b in probe.BACKBONES}
    d["nf"] = nf
    s = probe.baseline_scores(d, probe.frame_logits(d, probes))
    frame = sigmoid(first)
    assert s[0] == pytest.approx(np.sort(frame[:6])[-4:].mean())
    assert s[1] == pytest.approx(frame[6:].mean())  # fewer than 4 frames: mean of what exists


def test_video_features_layout():
    rng = np.random.default_rng(0)
    d = fake_split(rng, 6, 1.0)
    probes = {b: unit_probe(b) for b in probe.BACKBONES}
    lg = probe.frame_logits(d, probes)
    X, names = probe.video_features(d, lg)
    assert X.shape == (6, 2 * D + 2 + 4) and len(names) == X.shape[1]
    assert np.all(X[0, -4:] == 0)  # video 0 has one frame: no temporal features
    i = names.index("max_logit_siglip")
    assert X[2, i] == pytest.approx(lg["siglip"][4:9].max())  # video 2 holds frames 4..8
    np.testing.assert_allclose(X[1, :D], d["clip_vitl14"][1:4].astype(np.float32).mean(0), rtol=1e-5)


def test_fit_separates_and_run_writes_artifact(tmp_path):
    rng = np.random.default_rng(1)
    splits = {"train": fake_split(rng, 80, 1.5), "val": fake_split(rng, 40, 1.5), "test": fake_split(rng, 40, 1.5)}
    fm = tmp_path / "frame_model"
    fm.mkdir()
    for b in probe.BACKBONES:
        (fm / f"{b}.json").write_text(unit_probe(b).model_dump_json())
    (fm / "video_meta.json").write_text(json.dumps({"backbones": list(probe.BACKBONES), "t_ai": 0.6, "t_real": 0.4}))
    res = probe.run(fm, tmp_path / "m1", splits.__getitem__)
    assert res["probe"]["test"]["auroc"] > 0.9
    assert res["baseline"]["test"]["auroc"] > 0.6
    assert set(res["probe"]["test"]["per_generator"]) == {"genA", "genB"}
    art = load_probe(tmp_path / "m1" / "probe.json")
    assert art.embed_dim == 2 * D + 6 and art.temperature > 0
    meta = json.loads((tmp_path / "m1" / "meta.json").read_text())
    assert meta["fit_split"] == "val" and meta["t_real"] <= meta["t_ai"]


def test_pick_times_matches_api_rule():
    assert extract.pick_times(8.0, []) == vid.sample_times(8.0)
    # 32 s video: even frames every 2 s at 1, 3, 5...; cuts 0.2 s from one are skipped, max 4 cuts added
    got = extract.pick_times(32.0, [1.2, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    assert [t for t in got if t not in vid.sample_times(32.0)] == [2.0, 4.0, 6.0, 8.0]
    assert len(got) == vid.MAX_FRAMES


def test_select_expr_one_term_per_time():
    e = extract.select_expr([0.25, 1.5])
    assert e.startswith("gt(") and e.count("isnan(prev_t)") == 2 and "gte(t,1.500)" in e


def test_pack_unpack_roundtrip():
    from PIL import Image
    ims = []
    for c in ((255, 0, 0), (0, 0, 255)):
        b = io.BytesIO()
        Image.new("RGB", (8, 8), c).save(b, "JPEG", quality=95)
        ims.append(b.getvalue())
    back = extract.unpack(extract.pack(ims, [0.5, 1.5]))
    assert len(back) == 2 and back[1].getpixel((4, 4))[2] > 200


@pytest.mark.parametrize("method", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_unpack_local_reads_one_member(method):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", method) as z:
        z.writestr("videos/a.mp4", b"first" * 100)
        z.writestr("videos/b.mp4", b"second" * 100)
    raw = buf.getvalue()
    info = zipfile.ZipFile(io.BytesIO(raw)).getinfo("videos/b.mp4")
    assert data.unpack_local(raw[info.header_offset:], info) == b"second" * 100
    with pytest.raises(OSError):
        data.unpack_local(raw[info.header_offset + 1:], info)


def test_select_splits_by_generator():
    ids = {"train": [f"tr{i}.mp4" for i in range(50)], "val": [f"va{i}.mp4" for i in range(50)],
           "test": [f"te{i}.mp4" for i in range(50)]}
    allv = [(f"videos/{i}", 10) for v in ids.values() for i in v]
    members = {data.zip_path(g): allv for g in data.OPEN + ["real"]}
    members.update({data.zip_path(g): [(f"{i:04d}.mp4", 10) for i in range(20)] for g in data.CLOSED})
    counts = {"train_fake": 32, "train_real": 10, "val_fake_per_gen": 3, "val_real": 5, "test_fake_per_gen": 4, "test_real": 6}
    rows = data.select(members, ids, counts)
    for r in rows:
        if r["gen"] == "real":
            assert r["member"].split("/")[-1].startswith(r["split"][:2])
        elif r["gen"] in data.CLOSED:
            assert r["split"] == "test"
        else:
            assert (r["split"] == "val") == (data.family(r["gen"]) in data.VAL_FAMILIES)
            assert r["member"].split("/")[-1].startswith(r["split"][:2])  # fakes use their split's ids
    n = lambda s, lab: sum(r["split"] == s and r["label"] == lab for r in rows)  # noqa: E731
    assert n("train", 1) == 32 and n("train", 0) == 10 and n("val", 0) == 5 and n("test", 0) == 6
    assert n("test", 1) == 4 * len(data.CLOSED)
    assert rows == data.select(members, ids, counts)  # seeded

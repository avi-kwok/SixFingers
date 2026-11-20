"""Video-level probe on cached frame embeddings, and the frame-rule baseline it is compared with.

    python -m ml.video.probe          # baseline + train on train, pick on val, report test; writes models/video/m1

Baseline: the API rule. Each frame gets the image ensemble score (CLIP + SigLIP probes of the frame
model), the video score is the mean of the 4 highest frame scores, and the band uses video_meta.json.

Video probe: one logistic regression per video on
  mean-pooled frame embeddings (CLIP 768 + SigLIP 768)
  + the highest calibrated frame-probe logit per backbone (2)
  + temporal features per backbone: mean and std of the cosine distance between consecutive frames (4).
C, the temperature and the thresholds are chosen on val only; test is scored once at the end.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from api.video import top_k_mean
from ml.common.paths import models_dir
from ml.common.schemas import ProbeArtifact, calibrated_logit, load_probe, probe_logit, save_probe, sigmoid
from ml.eval.metrics import tpr_at_fpr
from ml.probes.train import fit_temperature, pick_thresholds

BACKBONES = ("clip_vitl14", "siglip")
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
VERSION = "m1"


def segments(nf: np.ndarray) -> list[slice]:
    """Frames of all videos are stored back to back; nf[i] frames belong to video i."""
    off = np.concatenate([[0], np.cumsum(nf)])
    return [slice(off[i], off[i + 1]) for i in range(len(nf))]


def frame_logits(d: dict, probes: dict[str, ProbeArtifact]) -> dict[str, np.ndarray]:
    """Calibrated frame-probe logit per frame and backbone: {backbone: [sum(nf)]}."""
    return {b: calibrated_logit(probes[b], d[b].astype(np.float32)) for b in BACKBONES}


def baseline_scores(d: dict, logits: dict[str, np.ndarray], k: int = 4) -> np.ndarray:
    """The API rule: ensemble score per frame (sigmoid of mean logit), then mean of the top k frames."""
    frame = sigmoid(np.mean([logits[b] for b in BACKBONES], axis=0))
    return np.array([top_k_mean(frame[s].tolist(), k) for s in segments(d["nf"])])


def cosine_steps(e: np.ndarray) -> np.ndarray:
    """Cosine distance between each frame and the next one: [n-1]."""
    e = e.astype(np.float32)
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
    return 1.0 - np.sum(e[1:] * e[:-1], axis=1)


def video_features(d: dict, logits: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """[n_videos, 1542] features and their names. A one-frame video has temporal features 0."""
    rows = []
    for s in segments(d["nf"]):
        parts = [d[b][s].astype(np.float32).mean(0) for b in BACKBONES]
        parts.append(np.array([logits[b][s].max() for b in BACKBONES], np.float32))
        temporal = []
        for b in BACKBONES:
            c = cosine_steps(d[b][s])
            temporal += [c.mean(), c.std()] if len(c) else [0.0, 0.0]
        parts.append(np.array(temporal, np.float32))
        rows.append(np.concatenate(parts))
    names = ([f"mean_{b}_{i}" for b in BACKBONES for i in range(d[b].shape[1])] + [f"max_logit_{b}" for b in BACKBONES]
             + [f"{stat}_cos_{b}" for b in BACKBONES for stat in ("mean", "std")])
    return np.stack(rows), names


def fit(Xtr, ytr, Xva, yva, c_grid=C_GRID) -> tuple[ProbeArtifact, dict]:
    """Logistic regression on standardized features. C by val AUROC, then temperature on val."""
    mean, std = Xtr.mean(0), Xtr.std(0)
    z = lambda X: (X - mean) / (std + 1e-6)  # noqa: E731 - must match schemas.probe_logit
    best = None
    for C in c_grid:
        m = LogisticRegression(C=C, max_iter=5000).fit(z(Xtr), ytr)
        auc = float(roc_auc_score(yva, m.decision_function(z(Xva))))
        if best is None or auc > best[0]:
            best = (auc, C, m)
    auc, C, m = best
    art = ProbeArtifact(backbone=f"video_{VERSION}", kind="logistic", embed_dim=int(Xtr.shape[1]), mean=mean.tolist(),
                        std=std.tolist(), weights=m.coef_[0].tolist(), bias=float(m.intercept_[0]))
    lg = probe_logit(art, Xva)
    art.temperature = fit_temperature(lg, yva)
    p = np.clip(sigmoid(lg / art.temperature), 1e-7, 1 - 1e-7)
    return art, {"C": C, "val_auroc": auc, "val_logloss": float(log_loss(yva, p)), "temperature": art.temperature}


def report(y, s, gens, t_ai: float, t_real: float) -> dict:
    """AUROC, TPR at 5% FPR (this set's own reals), and the band rates at the fixed thresholds."""
    y, s, gens = np.asarray(y), np.asarray(s), np.asarray(gens)
    out = {"n_fake": int(y.sum()), "n_real": int((y == 0).sum()), "auroc": float(roc_auc_score(y, s)),
           "tpr_at_5fpr": tpr_at_fpr(y, s), "tpr_at_t_ai": float((s[y == 1] >= t_ai).mean()),
           "real_fpr_at_t_ai": float((s[y == 0] >= t_ai).mean()),
           "real_likely_real": float((s[y == 0] <= t_real).mean())}
    out["per_generator"] = {}
    for g in sorted(set(gens[y == 1].tolist())):
        m = (gens == g) | (y == 0)  # each generator against all reals of the split
        out["per_generator"][g] = {"n": int(((gens == g) & (y == 1)).sum()), "auroc": float(roc_auc_score(y[m], s[m])),
                                   "tpr_at_t_ai": float((s[(gens == g) & (y == 1)] >= t_ai).mean())}
    return out


def run(frame_model: Path, out_dir: Path, load) -> dict:
    """load(split) -> embeddings dict (see ml.video.extract.load_split)."""
    probes = {b: load_probe(frame_model / f"{b}.json") for b in BACKBONES}
    vm = json.loads((frame_model / "video_meta.json").read_text())
    assert list(vm["backbones"]) == list(BACKBONES), vm
    data, feats, base = {}, {}, {}
    for split in ("train", "val", "test"):
        d = load(split)
        lg = frame_logits(d, probes)
        data[split], base[split] = d, baseline_scores(d, lg)
        feats[split] = video_features(d, lg)[0]
    res = {"frame_model": frame_model.resolve().name, "counts": {}, "baseline": {}, "probe": {}}
    for split, d in data.items():
        res["counts"][split] = {"videos": int(len(d["y"])), "fake": int(d["y"].sum()), "frames_per_video": float(d["nf"].mean()),
                                "generators": sorted(set(d["gen"][d["y"] == 1].tolist()))}
    for split in ("val", "test"):  # the shipped thresholds; baseline is never fit on this data
        res["baseline"][split] = report(data[split]["y"], base[split], data[split]["gen"], vm["t_ai"], vm["t_real"])
    # the baseline rule with thresholds re-picked on this val set (fair comparison with the probe)
    bt = pick_thresholds(base["val"], data["val"]["y"])
    res["baseline_val_thresholds"] = {"t_ai": bt.t_ai, "t_real": bt.t_real,
                                      "test": report(data["test"]["y"], base["test"], data["test"]["gen"], bt.t_ai, bt.t_real)}
    art, fitinfo = fit(feats["train"], data["train"]["y"], feats["val"], data["val"]["y"])
    p = {s: sigmoid(calibrated_logit(art, feats[s])) for s in ("val", "test")}
    th = pick_thresholds(p["val"], data["val"]["y"])
    res["probe"] = {"fit": fitinfo, "t_ai": th.t_ai, "t_real": th.t_real,
                    **{s: report(data[s]["y"], p[s], data[s]["gen"], th.t_ai, th.t_real) for s in ("val", "test")}}
    out_dir.mkdir(parents=True, exist_ok=True)
    save_probe(art, out_dir / "probe.json")
    meta = {"version": VERSION, "frame_model": res["frame_model"], "backbones": list(BACKBONES),
            "features": video_features.__doc__.strip(), "t_ai": th.t_ai, "t_real": th.t_real, "fit_split": "val",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": res}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return res


def main(argv=None):
    from ml.video.extract import load_split
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-model", type=Path, default=models_dir() / "current")
    ap.add_argument("--out", type=Path, default=models_dir() / "video" / VERSION)
    a = ap.parse_args(argv)
    res = run(a.frame_model, a.out, load_split)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

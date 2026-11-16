"""Train per-backbone probes (logistic + MLP arm), calibrate, pick thresholds, write models/vN."""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier

from ml.common.paths import manifest_path, model_version_dir, models_dir, current_model_dir
from ml.common.schemas import (BACKBONES, ModelMeta, ProbeArtifact, Thresholds, atomic_symlink,
                               calibrated_logit, ensemble_score, probe_logit, save_meta, save_probe, sigmoid)
from ml.features import store

C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
MLP_ALPHA_GRID = (1e-3, 1e-1)


def _std(X, mean, std):
    return (X - mean) / (std + 1e-6)  # must match schemas.probe_logit


def fit_temperature(logits, y) -> float:
    def nll(t):
        p = np.clip(sigmoid(logits / t), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(minimize_scalar(nll, bounds=(0.05, 50.0), method="bounded").x)


def fit_backbone(name, Xtr, ytr, Xva, yva, seed=0):
    """Returns (best artifact, metrics dict). The MLP ships only if its calibrated val log-loss is lower."""
    mean, std = Xtr.mean(0), Xtr.std(0)
    Ztr, Zva = _std(Xtr, mean, std), _std(Xva, mean, std)
    base = dict(backbone=name, embed_dim=int(Xtr.shape[1]), mean=mean.tolist(), std=std.tolist())

    def finish(art):
        lg = probe_logit(art, Xva)
        art.temperature = fit_temperature(lg, yva)
        p = np.clip(sigmoid(lg / art.temperature), 1e-7, 1 - 1e-7)
        return art, float(log_loss(yva, p)), float(roc_auc_score(yva, lg))

    best = None
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=3000).fit(Ztr, ytr)
        art = ProbeArtifact(**base, kind="logistic", weights=m.coef_[0].tolist(), bias=float(m.intercept_[0]))
        r = finish(art)
        if best is None or r[1] < best[1]:
            best = (*r, C)
    lr_art, lr_ll, lr_auc, lr_C = best

    mbest = None
    for a in MLP_ALPHA_GRID:
        m = MLPClassifier((256,), alpha=a, max_iter=300, random_state=seed).fit(Ztr, ytr)
        art = ProbeArtifact(**base, kind="mlp", weights=[0.0] * Xtr.shape[1], bias=0.0,
                            mlp_w1=m.coefs_[0].tolist(), mlp_b1=m.intercepts_[0].tolist(),
                            mlp_w2=m.coefs_[1][:, 0].tolist(), mlp_b2=float(m.intercepts_[1][0]))
        r = finish(art)
        if mbest is None or r[1] < mbest[1]:
            mbest = (*r, a)
    mlp_art, mlp_ll, mlp_auc, mlp_a = mbest

    ship = mlp_art if mlp_ll < lr_ll else lr_art
    return ship, dict(logistic_val_logloss=lr_ll, logistic_val_auroc=lr_auc, logistic_C=lr_C,
                      mlp_val_logloss=mlp_ll, mlp_val_auroc=mlp_auc, mlp_alpha=mlp_a,
                      shipped=ship.kind, temperature=ship.temperature)


def pick_thresholds(score_val, yva) -> Thresholds:
    """t_ai: at most 5% of val reals score >= it. t_real: at most 10% of val fakes score <= it.
    Order-statistic form (no interpolation) so the guarantee holds exactly on val, even for tiny sets."""
    r, f = np.sort(score_val[yva == 0]), np.sort(score_val[yva == 1])
    t_ai = float(np.nextafter(r[len(r) - int(0.05 * len(r)) - 1], np.inf))
    t_real = float(np.nextafter(f[int(0.10 * len(f))], -np.inf))
    return Thresholds(t_ai=t_ai, t_real=min(t_real, t_ai))


def _xy(df, backbone, aug_list):
    Xs, ys = [], []
    for a in aug_list:
        d, X = store.load_matrix(df, backbone, a)
        if len(d):
            Xs.append(X); ys.append(d["label"].to_numpy())
    return np.concatenate(Xs), np.concatenate(ys)


def next_version() -> str:
    n = [int(p.name[1:]) for p in models_dir().glob("v*") if p.name[1:].isdigit()]
    return f"v{max(n, default=0) + 1}"


def train(backbones=BACKBONES, promote=False, version=None) -> str:
    df = store.load_manifest()
    tr, va = df[df.split == "train"], df[df.split == "val"]
    arts, metrics, val_logits = [], {}, []
    yva = None
    for b in backbones:
        Xtr, ytr = _xy(tr, b, (0, 1, 2))
        Xva, yva_b = _xy(va, b, (0,))
        art, m = fit_backbone(b, Xtr, ytr, Xva, yva_b)
        metrics[b] = m | dict(n_train=int(len(ytr)), n_val=int(len(yva_b)))
        arts.append(art); val_logits.append(calibrated_logit(art, Xva))
        yva = yva_b  # same val rows per backbone assumed (extract covers all rows)
    ens = ensemble_score(val_logits)
    th = pick_thresholds(ens, yva)
    metrics["ensemble_val_auroc"] = float(roc_auc_score(yva, ens))
    version = version or next_version()
    d = model_version_dir(version)
    d.mkdir(parents=True)
    for a in arts:
        save_probe(a, d / f"{a.backbone}.json")
    save_meta(ModelMeta(version=version, backbones=list(backbones), thresholds=th, metrics={"val": metrics},
                        created_from_manifest_hash=hashlib.sha256(manifest_path().read_bytes()).hexdigest(),
                        created_at=datetime.now(timezone.utc).isoformat()), d / "meta.json")
    if promote:
        atomic_symlink(version, current_model_dir())
    return version


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES), choices=BACKBONES)
    ap.add_argument("--promote", action="store_true", help="also point models/current at the new version (manual promote)")
    a = ap.parse_args()
    print(train(a.backbones, promote=a.promote))


if __name__ == "__main__":
    main()

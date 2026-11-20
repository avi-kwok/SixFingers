"""Fit a logistic probe per XLS-R layer and C, pick them and a threshold on selection sets, report test sets.

    python -m ml.audio.train --version a1 --rule dev     # a1: every choice on LA dev only
    python -m ml.audio.train --version a2 --train la_train la_train_aug codecfake_aug \
        --sel la_dev asv5_dev                             # a2: rule "sel" (default)

Rule "sel": LA eval is split in two halves with a fixed seed. eval_sel (unseen attacks) joins the --sel sets;
layer, C, temperature and threshold are chosen on that pool. eval_test and In-the-Wild are only scored.

Writes models/audio/<version>/{probe.json, meta.json}. Needs the caches from ml.audio.extract.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, roc_curve

from ml.audio.datasets import TEST_SETS
from ml.audio.extract import load_cache
from ml.audio.features import CROP_S, LAYERS, SR, XLSR_REPO
from ml.common.paths import models_dir
from ml.common.schemas import ProbeArtifact, probe_logit, save_probe, sigmoid
from ml.eval.metrics import tpr_at_fpr
from ml.probes.train import fit_temperature

C_GRID = (1e-3, 1e-2, 1e-1, 1.0)
DEV_FPR = 0.05  # shipped threshold flags 5% of the real clips it is chosen on
SPLIT_SEED = 0  # LA eval -> eval_sel / eval_test; never change it, or eval_test stops being untouched


def eer(y, s) -> tuple[float, float]:
    """Equal error rate and the score threshold where false accepts = false rejects. y: 1 = fake."""
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def report(y, p, t) -> dict:
    """y: 1 = fake. p: calibrated P(fake). t: shipped threshold (p >= t means fake)."""
    y, p = np.asarray(y), np.asarray(p)
    return {"n": int(len(y)), "n_fake": int(y.sum()), "eer": eer(y, p)[0], "auroc": float(roc_auc_score(y, p)),
            "tpr_at_5fpr": tpr_at_fpr(y, p, 0.05), "acc_at_t": float(((p >= t) == (y == 1)).mean()),
            "fake_recall_at_t": float((p[y == 1] >= t).mean()), "real_fpr_at_t": float((p[y == 0] >= t).mean())}


def threshold_at_fpr(y, p, fpr: float = 0.05) -> float:
    """Smallest P(fake) threshold that flags at most `fpr` of the real (y = 0) clips."""
    real = np.sort(np.asarray(p)[np.asarray(y) == 0])
    return float(np.nextafter(real[int(np.ceil(len(real) * (1 - fpr))) - 1], np.inf))


def fit_probe(Ztr, ytr, mean, std, C: float, layer: int) -> ProbeArtifact:
    # balanced: training sets have more fakes than reals (LA train ~9:1 full set, ~2:1 cached)
    m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Ztr, ytr)
    return ProbeArtifact(backbone=f"xlsr300m_L{layer}", embed_dim=int(Ztr.shape[1]), mean=mean.tolist(),
                         std=std.tolist(), weights=m.coef_[0].tolist(), bias=float(m.intercept_[0]))


def standardize(Xtr):
    mean, std = Xtr.mean(0), Xtr.std(0)
    return (Xtr - mean) / (std + 1e-6), mean, std  # must match schemas.probe_logit


def fit_layer(Xtr, ytr, Xdev, ydev, layer: int):
    """Logistic probe on one layer; C by dev log-loss after temperature scaling. Returns (artifact, dev P(fake))."""
    Ztr, mean, std = standardize(Xtr)
    best = None
    for C in C_GRID:
        art = fit_probe(Ztr, ytr, mean, std, C, layer)
        lg = probe_logit(art, Xdev)
        art.temperature = fit_temperature(lg, ydev)
        p = sigmoid(lg / art.temperature)
        ll = float(log_loss(ydev, np.clip(p, 1e-7, 1 - 1e-7)))
        if best is None or ll < best[0]:
            best = (ll, C, art, p)
    return best[2], best[3], best[1]


def run(caches: dict, layers=LAYERS) -> dict:
    """caches: {"la_train", "la_dev", "la_eval", "itw"} -> extract-cache dicts. Returns the chosen probe + all metrics."""
    tr, dev = caches["la_train"], caches["la_dev"]
    per_layer, best = {}, None
    for k in layers:
        key = f"L{k}"
        art, pdev, C = fit_layer(tr[key].astype(np.float32), tr["y"], dev[key].astype(np.float32), dev["y"], k)
        auc = float(roc_auc_score(dev["y"], pdev))
        per_layer[k] = {"C": C, "dev_eer": eer(dev["y"], pdev)[0], "dev_auroc": auc}
        print(f"layer {k}: C={C} dev AUROC={auc:.4f} EER={per_layer[k]['dev_eer']:.4f}", flush=True)
        if best is None or auc > best[0]:
            best = (auc, k, art, threshold_at_fpr(dev["y"], pdev, DEV_FPR))
    _, k, art, t = best
    results = {}
    for name in ("la_dev", "la_eval", "itw"):
        if name in caches:
            c = caches[name]
            results[name] = report(c["y"], sigmoid(probe_logit(art, c[f"L{k}"].astype(np.float32)) / art.temperature), t)
    return {"layer": k, "threshold": t, "probe": art, "per_layer_dev": per_layer, "results": results}


def take(c: dict, mask) -> dict:
    return {k: v[mask] for k, v in c.items()}


def concat(caches) -> dict:
    caches = list(caches)
    return {k: np.concatenate([c[k] for c in caches]) for k in caches[0]}


def split_half(c: dict, seed: int = SPLIT_SEED) -> tuple[dict, dict]:
    """Random halves of a cache, the same share of fakes in each: (selection half, test half)."""
    rng = np.random.default_rng(seed)
    sel = np.zeros(len(c["y"]), dtype=bool)
    for label in (0, 1):
        idx = np.flatnonzero(c["y"] == label)
        sel[rng.choice(idx, len(idx) // 2, replace=False)] = True
    return take(c, sel), take(c, ~sel)


def run_sel(train: dict, sel: dict, tests: dict, layers=LAYERS) -> dict:
    """train / sel / tests: {name: cache}. Fit on all train sets together, for every layer and C.
    Pick the (layer, C) with the highest AUROC on the pooled selection sets, then fit the temperature and set the
    threshold (5% of the pooled real selection clips flagged) on that same pool. Test sets are only scored."""
    tr, pool = concat(train.values()), concat(sel.values())
    grid, best = [], None
    for k in layers:
        key = f"L{k}"
        Ztr, mean, std = standardize(tr[key].astype(np.float32))
        for C in C_GRID:
            art = fit_probe(Ztr, tr["y"], mean, std, C, k)
            lg = probe_logit(art, pool[key].astype(np.float32))
            row = {"layer": k, "C": C, "sel_auroc": float(roc_auc_score(pool["y"], lg)), "sel_eer": eer(pool["y"], lg)[0]}
            grid.append(row)
            print(f"layer {k} C={C}: selection AUROC={row['sel_auroc']:.4f} EER={row['sel_eer']:.4f}", flush=True)
            if best is None or row["sel_auroc"] > best[0]["sel_auroc"]:
                best = (row, art, lg)
    row, art, lg = best
    art.temperature = fit_temperature(lg, pool["y"])
    t = threshold_at_fpr(pool["y"], sigmoid(lg / art.temperature), DEV_FPR)
    k = row["layer"]
    results = {n: report(c["y"], sigmoid(probe_logit(art, c[f"L{k}"].astype(np.float32)) / art.temperature), t)
               for n, c in {**sel, **tests}.items()}
    return {"layer": k, "C": row["C"], "threshold": t, "probe": art, "grid": grid, "results": results}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="writes models/audio/<version>/; refuses to overwrite")
    ap.add_argument("--rule", choices=["sel", "dev"], default="sel", help="dev: the a1 rule (LA dev only)")
    ap.add_argument("--train", nargs="+", default=["la_train"], help="rule sel: training caches")
    ap.add_argument("--sel", nargs="+", default=["la_dev"], help="rule sel: selection caches (eval_sel is always added)")
    a = ap.parse_args(argv)
    if leaked := sorted(set(a.train + a.sel) & set(TEST_SETS)):
        raise SystemExit(f"{leaked} are test sets; they can never be used for training or choosing")
    out_dir = models_dir() / "audio" / a.version
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; pick a new version")
    meta = {"version": a.version, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "embedder": XLSR_REPO, "pooling": "mean over time",
            "input": {"sample_rate": SR, "crop_s": CROP_S, "trim_silence_db": -40}}
    if a.rule == "dev":
        caches = {n: load_cache(n) for n in ("la_train", "la_dev", "la_eval", "itw")}
        r = run(caches)
        meta.update(threshold_rule="5% FPR on LA dev bona fide", per_layer_dev=r["per_layer_dev"])
    else:
        eval_sel, eval_test = split_half(load_cache("la_eval"))
        train = {n: load_cache(n) for n in a.train}
        sel = {**{n: load_cache(n) for n in a.sel}, "eval_sel": eval_sel}
        tests = {"eval_test": eval_test, "itw": load_cache("itw")}
        r = run_sel(train, sel, tests)
        caches = {**train, **sel, **tests}
        meta.update(C=r["C"], threshold_rule=f"5% FPR on the pooled bona fide clips of {sorted(sel)}",
                    train_sets=a.train, selection_sets=sorted(sel), test_sets=sorted(tests),
                    la_eval_split_seed=SPLIT_SEED, selection_grid=r["grid"])
    out_dir.mkdir(parents=True)
    save_probe(r["probe"], out_dir / "probe.json")
    meta.update(layer=r["layer"], threshold_p_fake=r["threshold"],
                clip_counts={n: {"n": int(len(c["y"])), "n_fake": int(c["y"].sum())} for n, c in caches.items()},
                results=r["results"])
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"layer": r["layer"], "threshold": r["threshold"], "results": r["results"]}, indent=2))


if __name__ == "__main__":
    main()

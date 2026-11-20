"""Fit a monotone map from the judge's raw P(AI) to a calibrated P(AI).

    python -m ml.judge_eval.calibrate

Reads data/eval/judge_cache_prod.jsonl (arm haiku_blind) and the manifest labels. Fits on val images
(python -m ml.judge_eval.run --split val --arms haiku_blind) and reports the fitted map on report_test images,
which the fit never sees. Writes api/judge_calibration.json: piecewise-linear knots the API interpolates.
'uncertain' verdicts are not fitted: the API maps them to 0.5.
"""
import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from ml.common import paths

ARM = "haiku_blind"
CLIP = (0.02, 0.98)  # never claim certainty from 300 images
OUT = Path(__file__).resolve().parents[2] / "api" / "judge_calibration.json"


def raw_p_ai(verdict: str, confidence: float) -> float:
    return {"likely_ai": confidence, "likely_real": 1 - confidence}.get(verdict, 0.5)


def ece(p, y, bins=10) -> float:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def brier(p, y) -> float:
    return float(np.mean((p - y) ** 2))


def fit_iso(x, y):
    m = IsotonicRegression(y_min=CLIP[0], y_max=CLIP[1], out_of_bounds="clip").fit(x, y)
    return lambda q: m.predict(q), m


def fit_platt(x, y):
    m = LogisticRegression(C=1e6).fit(x.reshape(-1, 1), y)
    return lambda q: np.clip(m.predict_proba(q.reshape(-1, 1))[:, 1], *CLIP), m


def load(split: str, cache: Path):
    man = pd.read_parquet(paths.data_dir() / "manifest.parquet").set_index("image_id")
    man = man[man.split == split]["label"]
    rows = [r for r in map(json.loads, open(cache))
            if r["arm"] == ARM and r.get("verdict") in ("likely_ai", "likely_real") and r["image_id"] in man.index]
    x = np.array([raw_p_ai(r["verdict"], r["confidence"]) for r in rows])
    return x, man.loc[[r["image_id"] for r in rows]].to_numpy(float)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=None, help="answer cache file (default data/eval/judge_cache_prod.jsonl)")
    a = ap.parse_args()
    cache = Path(a.cache) if a.cache else paths.data_dir() / "eval" / "judge_cache_prod.jsonl"
    x, y = load("val", cache)
    xt, yt = load("report_test", cache)
    cv = list(StratifiedKFold(5, shuffle=True, random_state=0).split(x, y))
    res = {"raw": {"ece": ece(x, y), "brier": brier(x, y)}}
    for name, fit in (("isotonic", fit_iso), ("platt", fit_platt)):
        oof = np.zeros(len(x))
        for tr, te in cv:
            oof[te] = fit(x[tr], y[tr])[0](x[te])
        res[name] = {"ece": ece(oof, y), "brier": brier(oof, y)}
    best = min(("isotonic", "platt"), key=lambda k: res[k]["brier"])
    f, m = (fit_iso if best == "isotonic" else fit_platt)(x, y)
    held = {"n": int(len(xt)), "raw": {"ece": ece(xt, yt), "brier": brier(xt, yt)},
            best: {"ece": ece(f(xt), yt), "brier": brier(f(xt), yt)}} if len(xt) else None
    grid = np.unique(np.concatenate([[0.0, 1.0], x]))
    pred = f(grid)
    knots = [[round(float(a), 4), round(float(b), 4)] for a, b in zip(grid, pred)]
    art = {"model": "claude-haiku-5-5", "arm": ARM, "fit_split": "val", "n": int(len(x)), "method": best,
           "evaluation": "method chosen by 5-fold stratified CV on val; then scored on report_test, "
                         "which the fit never sees. ECE with 10 equal-width bins", "cv": res, "report_test": held,
           "fitted": date.today().isoformat(), "points": knots}
    pts = json.dumps(art.pop("points"))
    OUT.write_text(json.dumps(art, indent=1)[:-2] + ',\n "points": ' + pts + "\n}\n")
    print(json.dumps({"n": art["n"], "method": best, **res, "report_test": held}, indent=1))


if __name__ == "__main__":
    main()

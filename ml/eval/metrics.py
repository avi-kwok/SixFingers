from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def tpr_at_fpr(y, s, fpr=0.05) -> float:
    f, t, _ = roc_curve(y, s)
    return float(np.interp(fpr, f, t))


def ece(y, p, bins=10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def summarize(y, s, gens, sources, t_ai: float | None = None) -> dict:
    """y: 1=ai. s: score in [0,1]. TPR@5%FPR uses report_test's own reals; the t_ai rows use the shipped threshold."""
    y, s, gens, sources = map(np.asarray, (y, s, gens, sources))
    if len(set(y.tolist())) < 2:
        return {"n": int(len(y)), "note": "need both classes"}
    out = {"n": int(len(y)), "auroc": float(roc_auc_score(y, s)), "tpr_at_5fpr": tpr_at_fpr(y, s), "ece": ece(y, s)}
    if t_ai is not None:
        out["tpr_at_t_ai"] = float((s[y == 1] >= t_ai).mean())
        out["real_fpr_at_t_ai"] = float((s[y == 0] >= t_ai).mean())
        out["per_generator_tpr"] = {g: float((s[(y == 1) & (gens == g)] >= t_ai).mean()) for g in sorted(set(gens[y == 1]))}
        raise_m = (y == 0) & np.char.find(np.char.lower(sources.astype(str)), "raise").astype(int).__ge__(0)
        if raise_m.any():
            out["raise_real_fpr_at_t_ai"] = float((s[raise_m] >= t_ai).mean())
            out["n_raise"] = int(raise_m.sum())
    return out

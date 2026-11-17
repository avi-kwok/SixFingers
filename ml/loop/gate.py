"""Gate: decide whether a freshly trained candidate replaces the live model. Uses the `gate` split only."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from ml.common.schemas import calibrated_logit, ensemble_score, load_meta, load_probe

FPR = 0.05
MAX_GEN_DROP = 0.02
MAX_REAL_FPR = 0.05
GATE_SPLIT = "gate"  # report_test must never be read here

# (backbone, image_ids) -> float embeddings [N, D]; supplied by the features module.
Loader = Callable[[str, list[str]], np.ndarray]


def default_loader(backbone: str, image_ids: list[str]) -> np.ndarray:
    from ml.features import store

    return np.stack([np.load(store.feature_path(backbone, i, 0)) for i in image_ids]).astype(np.float32)


def score_model(model_dir: Path, ids: list[str], loader: Loader) -> tuple[np.ndarray, float]:
    """Ensemble score per image plus the model's own t_ai."""
    meta = load_meta(Path(model_dir) / "meta.json")
    logits = [
        calibrated_logit(load_probe(Path(model_dir) / f"{b}.json"), loader(b, ids)) for b in meta.backbones
    ]
    return ensemble_score(logits), meta.thresholds.t_ai


def metrics(scores, labels, generators, t_ai: float) -> dict:
    """TPR at 5% FPR (own threshold from the real images), per-generator TPR at that threshold, real FPR at the shipped t_ai."""
    scores, labels, generators = np.asarray(scores), np.asarray(labels), np.asarray(generators)
    real, fake = scores[labels == 0], labels == 1
    if real.size == 0 or not fake.any():
        raise ValueError("gate split needs both real and AI images")
    thr = float(np.quantile(real, 1 - FPR))
    per_gen = {str(g): float(np.mean(scores[fake & (generators == g)] >= thr)) for g in np.unique(generators[fake])}
    return {
        "tpr_at_5fpr": float(np.mean(scores[fake] >= thr)),
        "per_generator_tpr": per_gen,
        "real_fpr": float(np.mean(real >= t_ai)),
        "n_real": int(real.size),
        "n_ai": int(fake.sum()),
    }


def decide(cand: dict, cur: dict | None) -> tuple[bool, list[str]]:
    """Return (promote, reasons). First model (no current) only needs the FPR check."""
    reasons = []
    if cand["real_fpr"] > MAX_REAL_FPR:
        reasons.append(f"real FPR {cand['real_fpr']:.3f} > {MAX_REAL_FPR}")
    if cur is not None:
        if cand["tpr_at_5fpr"] < cur["tpr_at_5fpr"]:
            reasons.append(f"TPR@5%FPR {cand['tpr_at_5fpr']:.3f} < current {cur['tpr_at_5fpr']:.3f}")
        for g, t in cur["per_generator_tpr"].items():
            c = cand["per_generator_tpr"].get(g)
            if c is not None and t - c > MAX_GEN_DROP + 1e-9:
                reasons.append(f"generator {g} dropped {t - c:.3f} (max {MAX_GEN_DROP})")
    return not reasons, reasons


def evaluate(model_dir: Path, manifest, loader: Loader) -> dict:
    rows = manifest[manifest["split"] == GATE_SPLIT]
    ids = rows["image_id"].tolist()
    scores, t_ai = score_model(model_dir, ids, loader)
    return metrics(scores, rows["label"].to_numpy(), rows["generator"].to_numpy(), t_ai)

"""Shortcut audit: can content-free properties of the normalized images predict real-vs-AI?

Gated features are pipeline-leak features: image width and height, file format, JPEG quantization
tables and metadata presence. The normalizer controls all of them, so a difference between classes
means the normalizer failed. A cross-validated logistic probe on these features must stay at or
below MAX_AUROC.

File size is reported only, never gated. After identical re-encoding it tracks how much fine texture
an image has, so it measures content, not the pipeline."""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.common.paths import manifest_path

MAX_AUROC = 0.6
FEATURES = ["width", "height", "is_jpeg", "q_luma_mean", "q_chroma_mean", "has_metadata"]
META_KEYS = ("exif", "icc_profile", "xmp", "XML:com.adobe.xmp", "comment")  # JFIF keys are not metadata


def pipeline_features(path: str) -> list[float]:
    with Image.open(path) as im:
        q = getattr(im, "quantization", None) or {}
        mean = lambda t: float(np.mean(t)) if t is not None else 0.0
        has_meta = (bool(im.getexif()) or any(k in im.info for k in META_KEYS)
                    or bool(getattr(im, "text", None)))
        return [im.width, im.height, float(im.format == "JPEG"), mean(q.get(0)), mean(q.get(1)),
                float(has_meta)]


def _auc(y: np.ndarray, v: np.ndarray) -> float:
    return float(roc_auc_score(y, v)) if v.std() > 0 else 0.5


def audit(df: pd.DataFrame) -> dict:
    X = np.array([pipeline_features(p) for p in df["path"]], dtype=float)
    size = np.array([os.path.getsize(p) for p in df["path"]], dtype=float)
    y = df["label"].to_numpy()
    folds = min(5, int(np.bincount(y).min()))
    if folds < 2:
        raise ValueError("need at least 2 images of each class to audit")
    keep = X.std(axis=0) > 0  # a constant feature cannot leak
    if keep.any():
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        pred = cross_val_predict(clf, X[:, keep], y, cv=StratifiedKFold(folds, shuffle=True, random_state=0),
                                 method="decision_function")
        auc = float(roc_auc_score(y, pred))
    else:
        auc = 0.5
    per = {f: _auc(y, X[:, i]) for i, f in enumerate(FEATURES)}
    return {"auroc": auc, "single_feature_auroc": per, "file_size_auroc": _auc(y, size),
            "n": len(df), "passed": auc <= MAX_AUROC}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(manifest_path()))
    a = ap.parse_args()
    r = audit(pd.read_parquet(a.manifest))
    print(f"shortcut audit: AUROC={r['auroc']:.3f} (limit {MAX_AUROC}) on n={r['n']}")
    for f, v in r["single_feature_auroc"].items():
        print(f"  single {f}: {v:.3f}")
    print(f"  info (not gated): file_size AUROC {r['file_size_auroc']:.3f}")
    if not r["passed"]:
        raise SystemExit("FAIL: trivial features predict the label; the dataset has a shortcut")
    print("PASS")


if __name__ == "__main__":
    main()

"""Resumable float16 feature cache: one .npy per (backbone, preprocess_version, image_id, aug_index)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.common.paths import data_dir, features_dir, manifest_path
from ml.common.schemas import PREPROCESS_VERSION


def feature_path(backbone: str, image_id: str, aug: int = 0, pv: str = PREPROCESS_VERSION):
    # shortcut: one file per vector, fine to ~1e5 files; switch to sharded npz beyond that
    return features_dir() / backbone / pv / f"{image_id}_{aug}.npy"


def has(backbone, image_id, aug=0) -> bool:
    return feature_path(backbone, image_id, aug).exists()


def put(backbone, image_id, aug, vec: np.ndarray) -> None:
    p = feature_path(backbone, image_id, aug)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.npy")
    np.save(tmp, vec.astype(np.float16))
    tmp.replace(p)


def load_manifest() -> pd.DataFrame:
    return pd.read_parquet(manifest_path())


def resolve(path: str):
    from pathlib import Path
    p = Path(path)
    return p if p.is_absolute() else data_dir() / p


def load_matrix(df: pd.DataFrame, backbone: str, aug: int = 0):
    """Rows of df that have a cached vector -> (kept_df, X float32 [n, D])."""
    keep, vecs = [], []
    for i, iid in enumerate(df["image_id"]):
        p = feature_path(backbone, iid, aug)
        if p.exists():
            keep.append(i)
            vecs.append(np.load(p))
    if not vecs:
        return df.iloc[:0], np.zeros((0, 0), np.float32)
    return df.iloc[keep], np.stack(vecs).astype(np.float32)

"""Shared data contracts: manifest columns, probe artifact, model meta."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

MANIFEST_COLUMNS = ["image_id", "path", "label", "generator", "source", "split", "license"]
SPLITS = ("train", "val", "gate", "report_test")
BACKBONES = ("clip_vitl14", "dinov2_vitl14", "siglip")
PREPROCESS_VERSION = "p1"


class ProbeArtifact(BaseModel):
    """Logistic probe on one backbone's embedding. logit = ((x-mean)/std)@weights + bias; p = sigmoid(logit/temperature)."""
    backbone: str
    kind: Literal["logistic", "mlp"] = "logistic"
    embed_dim: int
    mean: list[float]
    std: list[float]
    weights: list[float]
    bias: float
    temperature: float = 1.0
    # only for kind="mlp": hidden layer 256. W1 [dim][256] row-major, W2 [256]
    mlp_w1: list[list[float]] | None = None
    mlp_b1: list[float] | None = None
    mlp_w2: list[float] | None = None
    mlp_b2: float | None = None


class Thresholds(BaseModel):
    t_ai: float
    t_real: float


class ModelMeta(BaseModel):
    version: str
    backbones: list[str]
    preprocess_version: str = PREPROCESS_VERSION
    thresholds: Thresholds
    metrics: dict = Field(default_factory=dict)
    created_from_manifest_hash: str = ""
    created_at: str = ""


def save_probe(p: ProbeArtifact, path: str | Path) -> None:
    Path(path).write_text(p.model_dump_json(indent=2))


def load_probe(path: str | Path) -> ProbeArtifact:
    return ProbeArtifact.model_validate_json(Path(path).read_text())


def save_meta(m: ModelMeta, path: str | Path) -> None:
    Path(path).write_text(m.model_dump_json(indent=2))


def load_meta(path: str | Path) -> ModelMeta:
    return ModelMeta.model_validate_json(Path(path).read_text())


def probe_logit(p: ProbeArtifact, emb: np.ndarray) -> np.ndarray:
    """emb [N, D] -> raw logits [N] (before temperature)."""
    x = (np.asarray(emb, dtype=np.float32) - np.array(p.mean, dtype=np.float32)) / (np.array(p.std, dtype=np.float32) + 1e-6)
    if p.kind == "mlp":
        h = np.maximum(x @ np.array(p.mlp_w1, dtype=np.float32) + np.array(p.mlp_b1, dtype=np.float32), 0)
        return h @ np.array(p.mlp_w2, dtype=np.float32) + p.mlp_b2
    return x @ np.array(p.weights, dtype=np.float32) + p.bias


def calibrated_logit(p: ProbeArtifact, emb: np.ndarray) -> np.ndarray:
    return probe_logit(p, emb) / p.temperature


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def ensemble_score(logits_per_backbone: list[np.ndarray]) -> np.ndarray:
    """Mean of calibrated logits across backbones, then sigmoid -> score in [0,1]."""
    return sigmoid(np.mean(np.stack(logits_per_backbone), axis=0))


def band_for(score: float, t: Thresholds) -> Literal["likely_ai", "uncertain", "likely_real"]:
    if score >= t.t_ai:
        return "likely_ai"
    if score <= t.t_real:
        return "likely_real"
    return "uncertain"


def atomic_symlink(target: str | Path, link: str | Path) -> None:
    """Repoint `link` at `target` atomically (rename over a temp symlink)."""
    link = Path(link)
    tmp = link.with_name(link.name + ".tmp")
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    os.symlink(target, tmp)
    os.replace(tmp, link)


def save_json(obj, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2))

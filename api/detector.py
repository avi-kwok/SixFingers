"""Step 3: embeddings -> per-backbone probe -> calibrated ensemble. Reloads when models/current moves."""
import io
import logging
import os
import threading
import weakref
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from ml.common import paths
from ml.common.api_models import DetectorResult
from ml.common.api_models import Thresholds as ApiThresholds
from ml.common.schemas import (ProbeArtifact, band_for, calibrated_logit, ensemble_score,
                               load_meta, load_probe, sigmoid)

log = logging.getLogger("sixfingers")


class Embedder(Protocol):
    def embed(self, backbone: str, image: Image.Image) -> np.ndarray:
        """Return [1, D] (or [D]) embedding for one RGB image."""


class FeatureEmbedder:
    """Production embedder: loads each backbone once via ml.features (at startup, or on first use)."""

    def __init__(self):
        self._loaded: dict = {}
        self._lock = threading.Lock()
        self._last: tuple = (None, None)  # (weakref to input image, normalized copy), shared by all backbones

    def load(self, backbone):
        with self._lock:  # concurrent first requests must not load the same backbone twice
            if backbone not in self._loaded:
                # integration point: ml.features must expose load_backbone(name) -> object with embed(list[PIL]) -> ndarray
                from ml.features.backbones import load_backbone
                self._loaded[backbone] = load_backbone(backbone)
            return self._loaded[backbone]

    def loaded(self, backbone) -> bool:
        return backbone in self._loaded

    def embed(self, backbone, image):
        model = self._loaded.get(backbone) or self.load(backbone)
        ref, norm = self._last
        if ref is None or ref() is not image:
            from ml.data.normalize import encode, normalize_image  # same resize/crop/JPEG q95 as training
            norm = Image.open(io.BytesIO(encode(normalize_image(image)))).convert("RGB")
            self._last = (weakref.ref(image), norm)
        return np.asarray(model.embed([norm]), dtype=np.float32)


class LoadedModel:
    def __init__(self, directory: Path):
        self.target = str(directory.resolve())
        self.meta = load_meta(directory / "meta.json")
        self.probes: dict[str, ProbeArtifact] = {b: load_probe(directory / f"{b}.json") for b in self.meta.backbones}
        self.version = self.meta.version


class DetectorStore:
    def __init__(self, embedder: Embedder, models_dir: Path | None = None):
        self.embedder = embedder
        self.link = (models_dir / "current") if models_dir else paths.current_model_dir()
        self._model: LoadedModel | None = None
        self._lock = threading.Lock()

    def get(self) -> LoadedModel:
        """Cheap per-request check: one realpath call. Keeps the old model if the new one fails to load."""
        with self._lock:
            try:
                target = os.path.realpath(self.link)
                if self._model is None or self._model.target != target:
                    self._model = LoadedModel(Path(target))
                    log.info("model loaded", extra={"event": "model_loaded", "model_version": self._model.version})
            except Exception:
                if self._model is None:
                    raise
                log.exception("model reload failed; keeping %s", self._model.version)
            return self._model

    def warm(self) -> None:
        """Load the current model's backbones now so /v1/health turns ready and the first check is fast."""
        load = getattr(self.embedder, "load", None)
        try:
            for b in self.get().meta.backbones:
                if load:
                    load(b)
        except Exception:
            log.exception("warm-up failed; backbones load on the first check")

    def ready(self) -> bool:
        """True once every backbone of the current model is in memory. Raises if no model is loadable."""
        loaded = getattr(self.embedder, "loaded", None)  # test embedders have nothing to load
        return all(loaded(b) for b in self.get().meta.backbones) if loaded else True

    def score(self, image: Image.Image) -> tuple[DetectorResult, str]:
        m = self.get()
        logits = [calibrated_logit(p, np.atleast_2d(self.embedder.embed(b, image)))
                  for b, p in m.probes.items()]
        score = float(ensemble_score(logits)[0])
        t = m.meta.thresholds
        res = DetectorResult(
            score=score, band=band_for(score, t),
            backbones={b: float(sigmoid(lg[0])) for b, lg in zip(m.probes, logits)},
            thresholds=ApiThresholds(t_ai=t.t_ai, t_real=t.t_real))
        return res, m.version

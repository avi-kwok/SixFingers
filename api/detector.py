"""Step 3: embeddings -> per-backbone probe -> calibrated ensemble. Reloads when models/current moves."""
import io
import json
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
from ml.common.schemas import (ProbeArtifact, Thresholds, band_for, calibrated_logit, ensemble_score,
                               load_meta, load_probe, sigmoid)

log = logging.getLogger("sixfingers")
VIDEO_BACKBONES = ("clip_vitl14", "siglip")  # DINOv2 dropped for video: about half the frame scoring time


class Embedder(Protocol):
    def embed(self, backbone: str, image: Image.Image) -> np.ndarray:
        """Return [1, D] (or [D]) embedding for one RGB image."""


class FeatureEmbedder:
    """Production embedder: loads each backbone once via ml.features (at startup, or on first use)."""

    def __init__(self, half: bool = False):
        self._loaded: dict = {}
        self.half = half  # fp16 weights on MPS; CPU stays fp32
        self._lock = threading.Lock()
        self._last: tuple = (None, None)  # (weakref to input image, normalized copy), shared by all backbones

    def load(self, backbone):
        with self._lock:  # concurrent first requests must not load the same backbone twice
            if backbone not in self._loaded:
                # integration point: ml.features must expose load_backbone(name) -> object with embed(list[PIL]) -> ndarray
                from ml.features.backbones import load_backbone
                self._loaded[backbone] = load_backbone(backbone, half=True) if self.half else load_backbone(backbone)
            return self._loaded[backbone]

    def loaded(self, backbone) -> bool:
        return backbone in self._loaded

    def embed_many(self, backbone, images):
        """One backbone call for the whole batch; returns [N, D]."""
        model = self._loaded.get(backbone) or self.load(backbone)
        from ml.data.normalize import encode, normalize_image  # same resize/crop/JPEG q95 as training
        norm = [Image.open(io.BytesIO(encode(normalize_image(im)))).convert("RGB") for im in images]
        return np.asarray(model.embed(norm), dtype=np.float32)

    def embed(self, backbone, image):
        model = self._loaded.get(backbone) or self.load(backbone)
        ref, norm = self._last
        if ref is None or ref() is not image:
            from ml.data.normalize import encode, normalize_image
            norm = Image.open(io.BytesIO(encode(normalize_image(image)))).convert("RGB")
            self._last = (weakref.ref(image), norm)
        return np.asarray(model.embed([norm]), dtype=np.float32)


class LoadedModel:
    def __init__(self, directory: Path):
        self.target = str(directory.resolve())
        self.meta = load_meta(directory / "meta.json")
        self.probes: dict[str, ProbeArtifact] = {b: load_probe(directory / f"{b}.json") for b in self.meta.backbones}
        self.version = self.meta.version
        vm = directory / "video_meta.json"  # video thresholds fit on val for the video backbones
        self.video_backbones = [b for b in VIDEO_BACKBONES if b in self.probes]
        self.video_thresholds = self.meta.thresholds
        if vm.exists():
            try:  # a bad video_meta.json falls back like a missing one, so image checks keep working
                v = json.loads(vm.read_text())
                backbones = list(v["backbones"])
                self.video_thresholds = Thresholds(t_ai=v["t_ai"], t_real=v["t_real"])
                self.video_backbones = backbones
            except Exception:
                log.exception("bad %s; video uses image thresholds", vm)
        # a missing video probe fails only the video route (503), never image checks


class DetectorStore:
    def __init__(self, embedder: Embedder, models_dir: Path | None = None, video_embedder: Embedder | None = None):
        self.embedder = embedder
        self.video_embedder = video_embedder or embedder
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

    def score_many(self, images: list[Image.Image], video: bool = False) -> tuple[list[DetectorResult], str]:
        """video=True: video backbones, video embedder and video thresholds instead of the image ones."""
        m = self.get()  # one model for the whole batch, even if models/current moves mid-request
        if video:
            names, emb, t = m.video_backbones, self.video_embedder, m.video_thresholds
        else:
            names, emb, t = list(m.probes), self.embedder, m.meta.thresholds
        many = getattr(emb, "embed_many", None)
        feats = {b: (np.atleast_2d(many(b, images)) if many else
                     np.vstack([np.atleast_2d(emb.embed(b, im)) for im in images]))
                 for b in names}
        logits = [calibrated_logit(m.probes[b], feats[b]) for b in names]  # each [N]
        scores = ensemble_score(logits)
        out = [DetectorResult(
            score=float(scores[i]), band=band_for(float(scores[i]), t),
            backbones={b: float(sigmoid(lg[i])) for b, lg in zip(names, logits)},
            thresholds=ApiThresholds(t_ai=t.t_ai, t_real=t.t_real)) for i in range(len(images))]
        return out, m.version

    def warm(self) -> None:
        """Load the current model's backbones now so /v1/health turns ready and the first check is fast."""
        try:
            for b in self.get().meta.backbones:
                for emb in self._embedders():
                    if getattr(emb, "load", None):
                        emb.load(b)
        except Exception:
            log.exception("warm-up failed; backbones load on the first check")

    def ready(self) -> bool:
        """True once every backbone of the current model is in memory. Raises if no model is loadable."""
        bbs = self.get().meta.backbones
        # test embedders have nothing to load
        return all(getattr(e, "loaded", lambda b: True)(b) for e in self._embedders() for b in bbs)

    def _embedders(self):
        """Image embedder, plus the video one when it is a separate object (CPU shares one)."""
        ve = self.video_embedder
        return [self.embedder] if ve is self.embedder else [self.embedder, ve]

    def score(self, image: Image.Image) -> tuple[DetectorResult, str]:
        res, version = self.score_many([image])
        return res[0], version

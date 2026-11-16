"""Frozen image backbones behind one interface: embed(list[PIL.Image]) -> float32 [N, D]."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def pick_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Backbone:
    name = ""

    def __init__(self, device: str | None = None):
        self.device = pick_device(device)
        self._load()

    def _load(self): raise NotImplementedError

    def _embed(self, images: list[Image.Image]) -> torch.Tensor: raise NotImplementedError

    @torch.inference_mode()
    def embed(self, images: list[Image.Image]) -> np.ndarray:
        out = self._embed([im.convert("RGB") for im in images])
        return out.float().cpu().numpy()


class ClipViTL14(Backbone):
    name = "clip_vitl14"

    def _load(self):
        import open_clip
        self.model, _, self.pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        self.model.eval().to(self.device)

    def _embed(self, images):
        x = torch.stack([self.pre(im) for im in images]).to(self.device)
        return self.model.encode_image(x)


class _HFVision(Backbone):
    repo = ""

    def _load(self):
        from transformers import AutoImageProcessor, AutoModel
        self.proc = AutoImageProcessor.from_pretrained(self.repo)
        self.model = AutoModel.from_pretrained(self.repo).eval().to(self.device)

    def _vision(self):
        return self.model

    def _embed(self, images):
        px = self.proc(images=images, return_tensors="pt")["pixel_values"].to(self.device)
        return self._vision()(pixel_values=px).pooler_output


class Dinov2ViTL14(_HFVision):
    name = "dinov2_vitl14"
    repo = "facebook/dinov2-large"


class SigLIP(_HFVision):
    # base-224 chosen over so400m: ~4x cheaper on CPU; revisit if its probe lags at full scale
    name = "siglip"
    repo = "google/siglip-base-patch16-224"

    def _vision(self):
        return self.model.vision_model


REGISTRY = {c.name: c for c in (ClipViTL14, Dinov2ViTL14, SigLIP)}


def load_backbone(name: str, device: str | None = None) -> Backbone:
    return REGISTRY[name](device)

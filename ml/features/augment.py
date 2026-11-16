"""Train-time augmentation and the fixed 'compressed' eval variant (JPEG + resize)."""
from __future__ import annotations

import hashlib
import io
import random

from PIL import Image, ImageFilter

COMPRESSED_AUG = 100  # aug_index reserved for the deterministic compressed variant


def _jpeg(im: Image.Image, q: int) -> Image.Image:
    b = io.BytesIO()
    im.convert("RGB").save(b, "JPEG", quality=q)
    b.seek(0)
    return Image.open(b).convert("RGB")


def augment(im: Image.Image, image_id: str, aug_index: int) -> Image.Image:
    """Deterministic per (image_id, aug_index): crop, resize 0.5-1.5x, light blur, JPEG q30-95."""
    seed = int(hashlib.sha256(f"{image_id}:{aug_index}".encode()).hexdigest()[:12], 16)
    r = random.Random(seed)
    im = im.convert("RGB")
    w, h = im.size
    f = r.uniform(0.7, 1.0)
    cw, ch = max(8, int(w * f)), max(8, int(h * f))
    x0, y0 = r.randint(0, w - cw), r.randint(0, h - ch)
    im = im.crop((x0, y0, x0 + cw, y0 + ch))
    s = r.uniform(0.5, 1.5)
    im = im.resize((max(8, int(cw * s)), max(8, int(ch * s))), Image.BICUBIC)
    if r.random() < 0.3:
        im = im.filter(ImageFilter.GaussianBlur(r.uniform(0.3, 1.0)))
    return _jpeg(im, r.randint(30, 95))


def compress(im: Image.Image) -> Image.Image:
    im = im.convert("RGB")
    w, h = im.size
    return _jpeg(im.resize((max(8, int(w * 0.75)), max(8, int(h * 0.75))), Image.BICUBIC), 70)

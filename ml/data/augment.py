"""Train-time augmentation: light blur, resize 0.5-1.5x, random crop, JPEG q30-95. The feature extractor calls
`augment(img, seed)` with seed = hash(image_id, aug_index) so cached features stay reproducible.
Training images get N_AUG_COPIES augmented copies besides the clean one."""
import io
import random

from PIL import Image, ImageFilter

N_AUG_COPIES = 2


def augment(img: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)
    img = img.convert("RGB")
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    s = rng.uniform(0.5, 1.5)
    img = img.resize((max(32, round(img.width * s)), max(32, round(img.height * s))), Image.Resampling.BICUBIC)
    cw, ch = (max(32, round(d * rng.uniform(0.7, 1.0))) for d in img.size)
    x, y = rng.randint(0, img.width - cw), rng.randint(0, img.height - ch)
    img = img.crop((x, y, x + cw, y + ch))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(30, 95))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")

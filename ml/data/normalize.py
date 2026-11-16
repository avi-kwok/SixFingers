"""One normalizer for every image, real or AI: decode -> RGB -> shorter side to 512 -> center crop 512x512
-> JPEG q95 4:2:0 with no metadata. Identical treatment on both sides removes the "PNG = fake,
JPEG = real" file-format shortcut and strips EXIF/ICC/C2PA."""
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageOps

from ml.common.paths import normalized_dir

SIZE = 512
QUALITY = 95


def normalize_image(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    s = SIZE / min(img.size)
    img = img.resize((max(SIZE, round(img.width * s)), max(SIZE, round(img.height * s))), Image.Resampling.LANCZOS)
    left, top = (img.width - SIZE) // 2, (img.height - SIZE) // 2
    return img.crop((left, top, left + SIZE, top + SIZE))


def encode(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=QUALITY, subsampling=2, optimize=False, progressive=False)
    return buf.getvalue()


def normalize_bytes(data: bytes) -> bytes:
    return encode(normalize_image(Image.open(io.BytesIO(data))))


def store(data: bytes) -> tuple[str, Path]:
    """Write normalized bytes under normalized/<id[:2]>/<id>.jpg; image_id = sha256 of those bytes."""
    image_id = hashlib.sha256(data).hexdigest()
    p = normalized_dir() / image_id[:2] / f"{image_id}.jpg"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return image_id, p


class Cache:
    """raw file -> image_id map so reruns skip decoding. shortcut: assumes raw files are never edited in place."""

    def __init__(self) -> None:
        self.path = normalized_dir() / "_map.json"
        self.map = json.loads(self.path.read_text()) if self.path.exists() else {}

    def get(self, key: str) -> Path | None:
        i = self.map.get(key)
        p = normalized_dir() / i[:2] / f"{i}.jpg" if i else None
        return p if p and p.exists() else None

    def put(self, key: str, raw: Path) -> tuple[str, Path]:
        image_id, p = store(normalize_bytes(raw.read_bytes()))
        self.map[key] = image_id
        return image_id, p

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.map))

"""Step 1: check the upload is a real, reasonably sized JPEG/PNG/WebP. Everything stays in memory."""
import io
from dataclasses import dataclass

from PIL import Image

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 40_000_000  # blocks decompression bombs before the pixels are decoded

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class UploadError(Exception):
    def __init__(self, status: int, detail: str):
        self.status, self.detail = status, detail


@dataclass
class Checked:
    image: Image.Image  # RGB
    format: str  # jpeg|png|webp
    width: int
    height: int
    mime: str


def sniff(data: bytes) -> tuple[str, str] | None:
    """Trust the bytes, not the filename or Content-Type."""
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg", "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png", "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp", "image/webp"
    return None


def validate_upload(data: bytes) -> Checked:
    if len(data) > MAX_BYTES:
        raise UploadError(413, "Image is larger than 10 MB.")
    if not data:
        raise UploadError(400, "Empty upload.")
    kind = sniff(data)
    if kind is None:
        raise UploadError(415, "Only JPEG, PNG and WebP images are supported.")
    try:
        img = Image.open(io.BytesIO(data))
        w, h = img.size  # header only, no decode yet
        if w * h > MAX_PIXELS:
            raise UploadError(413, "Image has too many pixels (limit 40 megapixels).")
        img.load()
        return Checked(img.convert("RGB"), kind[0], w, h, kind[1])
    except UploadError:
        raise
    except Exception:  # corrupt file, PIL bomb error, truncated data
        raise UploadError(400, "Could not read this image.") from None

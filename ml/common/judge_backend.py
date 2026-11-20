"""Judge backends: the Anthropic API, or any local program that reads a JSON request on stdin and prints a verdict on stdout."""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import shlex
import subprocess

import anthropic
from PIL import Image, ImageOps

from ml.common.api_models import JudgeOutput

log = logging.getLogger("sixfingers")

MAX_EDGE = 1568
MAX_BYTES = 5 * 1024 * 1024
RETRY_EXIT = 75  # exit code meaning "temporarily unavailable", stdout may carry {"retry_after_s": N}
DEFAULT_RETRY_S = 1800
COMMAND_TIMEOUT_S = 120  # a local program may start slower than an HTTP call


class BackendError(Exception):
    def __init__(self, reason: str, retry_after_s: float | None = None):
        super().__init__(reason)
        self.reason, self.retry_after_s = reason, retry_after_s


def downscale(data: bytes) -> Image.Image:
    """Decode, turn upright and shrink to the judge's size. The full-size decode happens here."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    img.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.LANCZOS)  # never upscales
    return img


def prepare_image(data: bytes | Image.Image) -> bytes:
    """JPEG for the judge, from image bytes or from an image downscale() already made."""
    img = data if isinstance(data, Image.Image) else downscale(data)
    quality = 90
    while True:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality)
        if buf.tell() <= MAX_BYTES:
            return buf.getvalue()
        if quality > 30:
            quality -= 15
        else:  # noisy images stay large at low quality, so shrink the pixels too
            w, h = img.size
            img = img.resize((max(1, w * 3 // 4), max(1, h * 3 // 4)), Image.Resampling.LANCZOS)


def _b64_jpeg(image: bytes) -> str:
    try:
        return base64.b64encode(prepare_image(image)).decode()
    except Exception:
        raise BackendError("error") from None


class ApiBackend:
    name = "api"

    def __init__(self, client):
        self.client = client

    def ask(self, *, model, system, image, evidence, schema, max_tokens, timeout, out_model=JudgeOutput):
        data = _b64_jpeg(image)
        try:
            resp = self.client.messages.create(
                model=model, max_tokens=max_tokens, system=system, timeout=timeout,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
                    {"type": "text", "text": evidence}]}])
        except anthropic.APITimeoutError:
            raise BackendError("timeout") from None
        except Exception as e:
            log.warning("judge call failed: %s", type(e).__name__)
            raise BackendError("error") from None
        # check stop_reason before trusting content
        if resp.stop_reason == "refusal":
            raise BackendError("refusal")
        if resp.stop_reason != "end_turn":
            raise BackendError("error")
        try:
            text = next(b.text for b in resp.content if getattr(b, "type", "text") == "text")
            out = out_model.model_validate_json(text)
        except Exception:
            raise BackendError("error") from None
        usage = getattr(resp, "usage", None)
        return out, ((usage.input_tokens, usage.output_tokens) if usage is not None else (0, 0))


class CommandBackend:
    name = "command"

    def __init__(self, cmd: str, timeout_s: float = COMMAND_TIMEOUT_S):
        self.argv, self.timeout_s = shlex.split(cmd), timeout_s

    def ask(self, *, model, system, image, evidence, schema, max_tokens, timeout, out_model=JudgeOutput):  # timeout unused: self.timeout_s applies
        req = {"model": model, "system": system, "evidence": evidence,
               "image": {"media_type": "image/jpeg", "data": _b64_jpeg(image)},
               "schema": schema, "max_tokens": max_tokens}
        try:
            p = subprocess.run(self.argv, input=json.dumps(req), capture_output=True, text=True,
                               errors="replace", timeout=self.timeout_s)  # bad bytes become invalid JSON, not a crash
        except subprocess.TimeoutExpired:
            raise BackendError("timeout") from None
        except OSError:
            log.warning("judge command could not start")
            raise BackendError("error") from None
        if p.returncode == RETRY_EXIT:
            log.warning("judge command unavailable (exit %d)", p.returncode)
            try:
                retry = float(json.loads(p.stdout)["retry_after_s"])
            except (ValueError, KeyError, TypeError):
                retry = DEFAULT_RETRY_S
            raise BackendError("retry_later", retry)
        if p.returncode != 0:
            log.warning("judge command failed (exit %d)", p.returncode)
            raise BackendError("error")
        try:
            out = out_model.model_validate_json(p.stdout)
        except Exception:
            raise BackendError("error") from None
        return out, (0, 0)


def from_env(client=None):
    kind = os.environ.get("SIXFINGERS_JUDGE_BACKEND") or "api"
    if kind == "api":
        return ApiBackend(client) if client is not None else None
    if kind == "command":
        cmd = os.environ.get("SIXFINGERS_JUDGE_CMD")
        return CommandBackend(cmd) if cmd else None
    raise ValueError("SIXFINGERS_JUDGE_BACKEND must be 'api' or 'command', got an unsupported value")

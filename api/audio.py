"""Audio input (prototype, off unless SIXFINGERS_ENABLE_AUDIO=1): byte sniffing, decoding with ffmpeg,
and the XLS-R audio probe (models/audio/<SIXFINGERS_AUDIO_MODEL>, default a3). See docs/audio.md.

Like the video route, ffmpeg only reads a local file the API wrote into its own temporary folder, with the
demuxer forced from the sniffed bytes and only the file and pipe protocols allowed.
"""
import json
import os
import subprocess
import threading
from pathlib import Path

import numpy as np

from api import video as vid
from api.validate import UploadError
from ml.audio import features
from ml.common import paths
from ml.common.api_models import AudioResult
from ml.common.schemas import load_probe, probe_logit, sigmoid

MAX_BYTES = 25 * 1024 * 1024
MAX_SECONDS = 60
UNREADABLE = "Could not read this audio."
NOTE = ("Prototype. Only the first 4 seconds of speech are scored, by a model trained mostly on read English "
        "speech. There is no written explanation for audio: Claude, the explainer, cannot hear audio.")


def enabled() -> bool:
    return os.environ.get("SIXFINGERS_ENABLE_AUDIO") == "1"


def sniff_audio(head: bytes) -> tuple[str, str] | None:
    """Return (format, ffmpeg demuxer) from the file's magic bytes, or None."""
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav", "wav"
    if head[:4] == b"fLaC":
        return "flac", "flac"
    if head[:4] == b"OggS":
        return "ogg", "ogg"
    # ID3 tag, or an MPEG audio frame sync whose layer bits are not 00 (00 is ADTS AAC, not MP3)
    if head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0 and head[1] & 0x06):
        return "mp3", "mp3"
    if head[4:8] == b"ftyp":
        return "m4a", "mov"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "webm", "matroska"
    return None


def read(path: Path, demuxer: str) -> np.ndarray:
    """Decode the first audio track to 16 kHz mono float32, at most MAX_SECONDS + 1 s of it."""
    ffmpeg, _ = vid.binaries()
    try:
        x = features.decode(f"file:{path}", max_s=MAX_SECONDS + 1, fmt=demuxer, ffmpeg=ffmpeg)
    except (RuntimeError, subprocess.TimeoutExpired, OSError):
        raise UploadError(400, UNREADABLE) from None
    if len(x) == 0:
        raise UploadError(400, UNREADABLE)
    return x


def load(src, tmp: Path) -> tuple[np.ndarray, str]:
    """Save the upload, check its type and length, and return (samples, format)."""
    path = tmp / "input"
    head = vid.save_upload(src, path, MAX_BYTES, "Audio is larger than 25 MB.")
    kind = sniff_audio(head)
    if kind is None:
        raise UploadError(415, "Only WAV, MP3, FLAC, OGG, M4A and WebM audio are supported.")
    x = read(path, kind[1])
    if len(x) > (MAX_SECONDS + 0.5) * features.SR:  # 0.5 s grace, as for video
        raise UploadError(413, "Audio is longer than 60 seconds.")
    return x, kind[0]


class AudioScorer:
    """Logistic probe on one XLS-R layer. XLS-R itself loads on the first score() call."""

    def __init__(self, models_dir: Path | None = None, embedder=None):
        name = os.environ.get("SIXFINGERS_AUDIO_MODEL", "a3")
        d = (models_dir or paths.models_dir()) / "audio" / name
        meta = json.loads((d / "meta.json").read_text())
        self.probe = load_probe(d / "probe.json")
        self.layer = int(meta["layer"])
        self.threshold = float(meta["threshold_p_fake"])
        self.version = meta.get("version", name)
        self._emb, self._lock = embedder, threading.Lock()

    def score(self, samples: np.ndarray) -> AudioResult:
        # same input as training (ml.audio.features.load_clip): first 30 s decoded, silence trimmed, first 4 s
        clip = features.crop(features.trim_silence(samples[: int(features.MAX_READ_S * features.SR)]))
        with self._lock:  # loads XLS-R once, and keeps to one embedding at a time on the GPU
            if self._emb is None:
                self._emb = features.XLSR(layers=(self.layer,))
            emb = self._emb.embed(clip[None])[self.layer]
        p = float(sigmoid(probe_logit(self.probe, emb) / self.probe.temperature)[0])
        return AudioResult(p_fake=p, band="likely_ai" if p >= self.threshold else "likely_real",
                           threshold=self.threshold, model_version=self.version, note=NOTE)

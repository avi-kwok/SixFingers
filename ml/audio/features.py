"""Decode audio with ffmpeg, crop to 4 s, embed with XLS-R (several hidden layers, mean-pooled over time)."""
from __future__ import annotations

import subprocess
import tempfile
import zlib
from pathlib import Path

import numpy as np

SR = 16_000
CROP_S = 4.0
MAX_READ_S = 30  # decode at most this much; longer files only need their start
# hidden_states index k = output of transformer layer k (0 = CNN features). Layers above 12 are not run:
# on an M2 the 24-layer model embeds ~4 clips/s, half the layers about twice that.
LAYERS = (3, 6, 9, 12)
XLSR_REPO = "facebook/wav2vec2-xls-r-300m"
# Local files only: crafted playlists (HLS, concat) must never make ffmpeg fetch a URL.
SAFE = ["-protocol_whitelist", "file,pipe"]
TIMEOUT_S = 30


def decode(path: str, sr: int = SR, max_s: float = MAX_READ_S, fmt: str | None = None,
           ffmpeg: str = "ffmpeg") -> np.ndarray:
    """Any file ffmpeg reads -> float32 mono at `sr` Hz. Raises RuntimeError on a bad file.

    `fmt` forces the demuxer (the API passes the one sniffed from the file's first bytes)."""
    force = ["-f", fmt] if fmt else []
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", *SAFE, *force, "-i", str(path),
           "-t", str(max_s), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"]
    r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, timeout=TIMEOUT_S)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {r.stderr.decode(errors='replace')[-300:]}")
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def trim_silence(x: np.ndarray, sr: int = SR, db: float = -40.0) -> np.ndarray:
    """Drop leading/trailing 10 ms frames quieter than `db` below the loudest frame.

    ASVspoof 2019 LA leaks the label through the length of leading silence, so a probe
    trained on untrimmed clips partly learns "how much silence", which fails on real-world audio.
    """
    hop = sr // 100
    n = len(x) // hop
    if n == 0:
        return x
    rms = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    loud = np.flatnonzero(20 * np.log10(rms / rms.max()) > db)
    return x[loud[0] * hop: (loud[-1] + 1) * hop]


def crop(x: np.ndarray, sr: int = SR, seconds: float = CROP_S) -> np.ndarray:
    """First `seconds` of the clip. Shorter clips are repeated (tiled) up to that length so every crop is equal."""
    n = int(sr * seconds)
    if len(x) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(x) < n:
        x = np.tile(x, -(-n // len(x)))
    return x[:n].astype(np.float32)


# (ffmpeg encoder, container, bitrates in kbit/s): low to mid quality, like phone calls, voice notes and video sites
CODECS = (("libmp3lame", "mp3", (24, 32, 48, 64)), ("libopus", "ogg", (12, 16, 24, 32)), ("aac", "adts", (24, 32, 48, 64)))
LOW_RATES = (8_000, 11_025, 12_000, 14_000)


def codec_roundtrip(x: np.ndarray, encoder: str, fmt: str, kbps: int, sr: int = SR) -> np.ndarray:
    """Encode to a lossy format and decode back (temporary files; ffmpeg may only open local files)."""
    with tempfile.TemporaryDirectory() as d:
        src, enc = Path(d) / "in.f32", Path(d) / f"out.{fmt}"
        x.astype(np.float32).tofile(src)
        cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-protocol_whitelist", "file",
               "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", str(src),
               "-c:a", encoder, "-b:a", f"{kbps}k", "-f", fmt, str(enc)]
        r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, timeout=TIMEOUT_S)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg {encoder} failed: {r.stderr.decode(errors='replace')[-300:]}")
        return decode(str(enc), sr)


def colored_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """White, pink or brown noise (power falls as 1/f^0, 1/f^1 or 1/f^2 with frequency f)."""
    spec = np.fft.rfft(rng.normal(size=n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    return np.fft.irfft(spec * f ** -rng.choice([0.0, 0.5, 1.0]), n)


def augment(x: np.ndarray, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """Random real-world damage for training clips. Each step happens with probability 1/2:
    loudness change (-20 to +10 dB, clipped at full scale), background noise (10 to 30 dB SNR),
    a lower sample rate (8 to 14 kHz and back), and a lossy codec round trip (MP3, Opus or AAC)."""
    from scipy.signal import resample_poly
    x = x.astype(np.float64)
    if rng.random() < 0.5:
        x = np.clip(x * 10 ** (rng.uniform(-20, 10) / 20), -1, 1)
    if rng.random() < 0.5:
        noise = colored_noise(len(x), rng)
        snr = rng.uniform(10, 30)
        x = x + noise * np.sqrt(np.mean(x ** 2) / (np.mean(noise ** 2) + 1e-12) / 10 ** (snr / 10))
    if rng.random() < 0.5:
        low = int(rng.choice(LOW_RATES))
        x = resample_poly(resample_poly(x, low, sr), sr, low)
    if rng.random() < 0.5:
        encoder, fmt, rates = CODECS[rng.integers(len(CODECS))]
        x = codec_roundtrip(x, encoder, fmt, int(rng.choice(rates)), sr)
    return x.astype(np.float32)


def load_clip(path: str, aug_seed: int | None = None) -> np.ndarray:
    """Decode, trim silence, crop to 4 s. With `aug_seed`, also apply `augment` (same damage for the same file and seed)."""
    x = crop(trim_silence(decode(path)))
    if aug_seed is None:
        return x
    return crop(augment(x, np.random.default_rng([aug_seed, zlib.crc32(str(path).encode())])))


def normalize(batch: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance per clip (what Wav2Vec2FeatureExtractor does with do_normalize=True)."""
    m = batch.mean(axis=1, keepdims=True)
    v = batch.var(axis=1, keepdims=True)
    return (batch - m) / np.sqrt(v + 1e-7)


class XLSR:
    """Frozen XLS-R 300M. embed([N, samples]) -> {layer: float32 [N, 1024]}."""

    def __init__(self, device: str | None = None, layers=LAYERS, half: bool = True):
        import torch
        from transformers import Wav2Vec2Model
        from ml.features.backbones import pick_device
        self.torch, self.layers = torch, tuple(layers)
        self.device = pick_device(device)
        # fp16 only on MPS (~1.4x faster, cosine to fp32 > 0.9999); CPU half matmuls are slow
        self.dtype = torch.float16 if half and self.device == "mps" else torch.float32
        self.model = Wav2Vec2Model.from_pretrained(XLSR_REPO).eval().to(self.device, self.dtype)
        # skip the layers above the highest one we read (hidden_states are recorded per layer, so k keeps its meaning)
        self.model.encoder.layers = self.model.encoder.layers[: max(self.layers)]

    def embed(self, batch: np.ndarray) -> dict[int, np.ndarray]:
        with self.torch.inference_mode():
            x = self.torch.from_numpy(normalize(np.asarray(batch, dtype=np.float32))).to(self.device, self.dtype)
            hs = self.model(x, output_hidden_states=True).hidden_states
            return {k: hs[k].float().mean(dim=1).cpu().numpy() for k in self.layers}

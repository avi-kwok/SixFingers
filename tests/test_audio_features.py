import shutil

import numpy as np
import pytest
from scipy.io import wavfile

from ml.audio import datasets
from ml.audio.extract import embed_rows
from ml.audio.features import SR, crop, decode, load_clip, normalize, trim_silence

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def tone(seconds, sr=SR, hz=440.0, amp=0.5):
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_trim_silence_drops_quiet_ends():
    x = np.concatenate([np.zeros(SR), tone(1.0), np.zeros(SR // 2)])
    y = trim_silence(x)
    assert abs(len(y) - SR) <= SR // 100
    assert np.abs(y).max() == pytest.approx(0.5, abs=1e-3)


def test_trim_silence_keeps_short_and_empty():
    assert len(trim_silence(np.zeros(10, np.float32))) == 10
    assert len(trim_silence(np.zeros(0, np.float32))) == 0


def test_crop_cuts_and_tiles():
    assert crop(tone(6.0)).shape == (4 * SR,)
    short = np.arange(SR, dtype=np.float32)
    c = crop(short)
    assert c.shape == (4 * SR,) and np.array_equal(c[SR:2 * SR], short)
    assert crop(np.zeros(0, np.float32)).shape == (4 * SR,)


def test_normalize_zero_mean_unit_var():
    z = normalize(np.stack([tone(1.0) + 0.3, tone(1.0, amp=0.01)]))
    assert np.allclose(z.mean(1), 0, atol=1e-4) and np.allclose(z.std(1), 1, atol=1e-3)


@requires_ffmpeg
def test_decode_resamples_to_16k_mono(tmp_path):
    stereo = np.stack([tone(1.0, sr=8000), tone(1.0, sr=8000)], axis=1)
    p = tmp_path / "a.wav"
    wavfile.write(p, 8000, (stereo * 32767).astype(np.int16))
    x = decode(str(p))
    assert x.dtype == np.float32 and abs(len(x) - SR) < 200
    assert load_clip(str(p)).shape == (4 * SR,)


@requires_ffmpeg
def test_decode_rejects_bad_file_and_urls(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    with pytest.raises(RuntimeError):
        decode(str(bad))
    with pytest.raises(RuntimeError):
        decode("http://127.0.0.1:9/x.wav")  # http is outside the protocol whitelist


class FakeEmbedder:
    """Two 'layers': loudness and a constant, so tests need no model download."""

    def embed(self, batch):
        loud = np.abs(batch).mean(1, keepdims=True)
        return {3: np.hstack([loud, loud * 2]), 6: np.ones((len(batch), 2))}


@requires_ffmpeg
def test_embed_rows_keeps_order_and_labels(tmp_path):
    rows = []
    for i, amp in enumerate([0.1, 0.5, 0.9]):
        p = tmp_path / f"{i}.wav"
        wavfile.write(p, SR, tone(0.5, amp=amp))
        rows.append((str(p), i % 2, f"A0{i}"))
    out = embed_rows(rows, FakeEmbedder(), batch=2, workers=2, chunk=2)
    assert out["L3"].shape == (3, 2) and out["L3"].dtype == np.float16 and out["L6"].shape == (3, 2)
    assert list(out["y"]) == [0, 1, 0] and list(out["attack"]) == ["A00", "A01", "A02"]
    assert out["L3"][0, 0] < out["L3"][1, 0] < out["L3"][2, 0]


def test_asvspoof_protocol_and_itw_meta(tmp_path):
    proto = tmp_path / "asv" / "LA" / "ASVspoof2019_LA_cm_protocols"
    proto.mkdir(parents=True)
    (proto / "ASVspoof2019.LA.cm.dev.trl.txt").write_text(
        "LA_0069 LA_D_1047731 - - bonafide\nLA_0070 LA_D_1105538 - A05 spoof\n")
    rows = datasets.asvspoof_rows("la_dev", tmp_path / "asv")
    assert [(r[1], r[2]) for r in rows] == [(0, "bonafide"), (1, "A05")]
    assert rows[1][0].endswith("LA/ASVspoof2019_LA_dev/flac/LA_D_1105538.flac")

    itw = tmp_path / "itw" / "release_in_the_wild"
    itw.mkdir(parents=True)
    (itw / "meta.csv").write_text("file,speaker,label\n0.wav,Alec Guinness,spoof\n1.wav,Alec Guinness,bona-fide\n")
    rows = datasets.itw_rows(tmp_path / "itw")
    assert [r[1] for r in rows] == [1, 0] and rows[0][0].endswith("release_in_the_wild/0.wav")


def test_subsample_caps_each_class():
    rows = [(str(i), int(i < 30), "x") for i in range(100)]
    s = datasets.subsample(rows, 20)
    assert sum(r[1] for r in s) == 20 and sum(1 - r[1] for r in s) == 20
    assert s == datasets.subsample(rows, 20)  # fixed seed
    assert datasets.subsample(rows, None) == rows

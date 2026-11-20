import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.io import wavfile

from ml.audio import datasets
from ml.audio.extract import embed_rows
from ml.audio.features import CODECS, SR, augment, codec_roundtrip, load_clip

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def tone(seconds, hz=440.0, amp=0.5):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


@requires_ffmpeg
@pytest.mark.parametrize("encoder,fmt,rates", CODECS)
def test_codec_roundtrip_keeps_the_sound(encoder, fmt, rates):
    x = tone(1.0)
    y = codec_roundtrip(x, encoder, fmt, rates[-1])
    assert y.dtype == np.float32 and abs(len(y) - len(x)) < SR // 5  # codecs pad by a few ms
    spec = np.abs(np.fft.rfft(y[:SR]))
    assert abs(np.argmax(spec) * SR / len(y[:SR]) - 440) < 5  # still a 440 Hz tone


@requires_ffmpeg
def test_augment_is_seeded_finite_and_changes_some_clips():
    x = tone(4.0)
    outs = [augment(x, np.random.default_rng(s)) for s in range(12)]
    assert all(o.dtype == np.float32 and np.isfinite(o).all() for o in outs)
    assert np.array_equal(augment(x, np.random.default_rng(3)), outs[3])
    assert sum(len(o) != len(x) or not np.allclose(o, x, atol=1e-3) for o in outs) >= 6


@requires_ffmpeg
def test_load_clip_aug_is_deterministic_per_file(tmp_path):
    p = tmp_path / "a.wav"
    wavfile.write(p, SR, tone(2.0))
    a, b = load_clip(str(p), aug_seed=1), load_clip(str(p), aug_seed=1)
    assert a.shape == (4 * SR,) and np.array_equal(a, b)
    rows = [(str(p), 0, "bonafide")]
    clean = embed_rows(rows, Loudness(), workers=1)
    augs = [embed_rows(rows, Loudness(), workers=1, aug_seed=s)["L3"] for s in range(1, 9)]
    assert all(a.shape == clean["L3"].shape == (1, 1) for a in augs)
    assert any(not np.allclose(a, clean["L3"]) for a in augs)  # the damage reaches the embedding


class Loudness:
    def embed(self, batch):
        return {3: np.abs(batch).mean(1, keepdims=True)}


def test_asv5_rows_lists_only_downloaded_clips(tmp_path):
    (tmp_path / "flac_T").mkdir()
    (tmp_path / "flac_T" / "T_0000000001.flac").write_bytes(b"")
    (tmp_path / "flac_T" / "T_0000000002.flac").write_bytes(b"")
    (tmp_path / "ASVspoof5.train.tsv").write_text(
        "T_4850 T_0000000000 F - - - AC3 A05 spoof -\n"
        "T_0858 T_0000000001 M - - - AC3 A07 spoof -\n"
        "T_0001 T_0000000002 M - - - - bonafide bonafide -\n")
    rows = datasets.asv5_rows("asv5_train", tmp_path)
    assert [(r[1], r[2]) for r in rows] == [(1, "A07"), (0, "bonafide")]
    assert rows[0][0].endswith("flac_T/T_0000000001.flac")


def test_prepare_codecfake_writes_fakes_only(tmp_path):
    (tmp_path / "data").mkdir()
    for shard, codec in enumerate(["encodec_24khz", "genuine"]):
        label = "spoofing" if codec != "genuine" else "genuine"
        t = pa.table({"audio": [{"bytes": f"{shard}-{i}".encode(), "path": f"../../evil{i}.wav"} for i in range(6)],
                      "label": [label] * 6, "speaker_id": ["p225"] * 6, "codec_name": [codec] * 6})
        pq.write_table(t, tmp_path / "data" / f"train-{shard:05d}-of-00002.parquet")
    assert datasets.prepare_codecfake(4, root=tmp_path) == 4
    rows = datasets.codecfake_rows(tmp_path)
    assert len(rows) == 4 and {r[1] for r in rows} == {1} and {r[2] for r in rows} == {"encodec_24khz"}
    assert all(Path(r[0]).parent == tmp_path / "wav" for r in rows)
    assert not (tmp_path.parent / "evil0.wav").exists()


def test_rows_for_aug_uses_the_same_clips(monkeypatch):
    monkeypatch.setattr(datasets, "asvspoof_rows", lambda name: [(name, 0, "bonafide")])
    assert datasets.rows_for("la_train_aug") == datasets.rows_for("la_train")
    assert "la_eval_aug" not in datasets.AUG_SETS and "itw_aug" not in datasets.AUG_SETS

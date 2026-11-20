"""List clips of ASVspoof 2019 LA, ASVspoof 5, CodecFake and In-the-Wild as rows: (path, label, attack).
label 1 = fake (spoof), 0 = real.

A set name ending in "_aug" (e.g. "asv5_train_aug") means the same clips with random degradations
(see features.augment); it is only ever used for training clips and selection sets, never for test sets.

    python -m ml.audio.datasets codecfake --n 10000   # unpack a CodecFake sample from the downloaded shards
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np

from ml.common.paths import raw_dir

ASV_DIR = "asvspoof2019_la"  # LA.zip extracted here: <raw>/audio/asvspoof2019_la/LA/...
ITW_DIR = "in_the_wild"      # release_in_the_wild.zip extracted here: <raw>/audio/in_the_wild/release_in_the_wild/...
ASV5_DIR = "asvspoof5"    # protocol .tsv files + flac_T/, flac_D/ (from the flac_T_*.tar / flac_D_*.tar parts we downloaded)
CF_DIR = "codecfake"      # data/*.parquet shards (Hugging Face rogertseng/CodecFake); clips.csv + wav/ written by `codecfake`
ASV_SPLITS = {"la_train": ("train", "trn"), "la_dev": ("dev", "trl"), "la_eval": ("eval", "trl")}
ASV5_SPLITS = {"asv5_train": ("ASVspoof5.train.tsv", "flac_T"), "asv5_dev": ("ASVspoof5.dev.track_1.tsv", "flac_D")}
TEST_SETS = ("la_eval", "itw")  # never augmented
SETS = (*ASV_SPLITS, *ASV5_SPLITS, "codecfake", "itw")
AUG_SETS = tuple(f"{n}_aug" for n in SETS if n not in TEST_SETS)


def audio_root() -> Path:
    return raw_dir() / "audio"


def asvspoof_rows(split: str, root: Path | None = None) -> list[tuple[str, int, str]]:
    """Protocol line: 'LA_0079 LA_T_1138215 - - bonafide' or 'LA_0079 LA_T_1271820 - A01 spoof'."""
    name, ext = ASV_SPLITS[split]
    la = (root or audio_root() / ASV_DIR) / "LA"
    proto = la / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{name}.{ext}.txt"
    rows = []
    for line in proto.read_text().splitlines():
        _, utt, _, attack, key = line.split()
        rows.append((str(la / f"ASVspoof2019_LA_{name}" / "flac" / f"{utt}.flac"),
                     int(key == "spoof"), "bonafide" if attack == "-" else attack))
    return rows


def itw_rows(root: Path | None = None) -> list[tuple[str, int, str]]:
    """meta.csv columns: file, speaker, label ('spoof' or 'bona-fide')."""
    d = (root or audio_root() / ITW_DIR) / "release_in_the_wild"
    with open(d / "meta.csv", newline="") as f:
        return [(str(d / r["file"]), int(r["label"] == "spoof"), r["speaker"]) for r in csv.DictReader(f)]


def asv5_rows(split: str, root: Path | None = None) -> list[tuple[str, int, str]]:
    """Protocol line: 'T_4850 T_0000000000 F - - - AC3 A05 spoof -' (columns in the ASVspoof 5 README).
    Only clips present on disk are listed: we download a few of the tar parts, not all of them."""
    tsv, sub = ASV5_SPLITS[split]
    d = root or audio_root() / ASV5_DIR
    present = set(os.listdir(d / sub)) if (d / sub).is_dir() else set()
    rows = []
    for line in (d / tsv).read_text().splitlines():
        f = line.split()
        if f"{f[1]}.flac" in present:
            rows.append((str(d / sub / f"{f[1]}.flac"), int(f[8] == "spoof"), f[7]))
    return rows


def codecfake_rows(root: Path | None = None) -> list[tuple[str, int, str]]:
    """clips.csv (written by prepare_codecfake): file, codec. All CodecFake clips we use are fakes."""
    d = root or audio_root() / CF_DIR
    with open(d / "clips.csv", newline="") as f:
        return [(str(d / "wav" / r["file"]), 1, r["codec"]) for r in csv.DictReader(f)]


def rows_for(name: str) -> list[tuple[str, int, str]]:
    name = name.removesuffix("_aug")
    if name == "itw":
        return itw_rows()
    if name == "codecfake":
        return codecfake_rows()
    return asv5_rows(name) if name in ASV5_SPLITS else asvspoof_rows(name)


def prepare_codecfake(n: int, seed: int = 0, root: Path | None = None) -> int:
    """Write about `n` fake clips, spread evenly over the downloaded parquet shards (each shard holds one codec),
    to <root>/wav/ and list them in <root>/clips.csv. Returns the number written."""
    import pyarrow.parquet as pq
    d = root or audio_root() / CF_DIR
    labels = {s: pq.read_table(s, columns=["label"])["label"].to_numpy(zero_copy_only=False)
              for s in sorted((d / "data").glob("*.parquet"))}
    shards = {s: np.flatnonzero(lab == "spoofing") for s, lab in labels.items()}
    shards = {s: i for s, i in shards.items() if len(i)}
    per_shard = -(-n // len(shards))
    rng = np.random.default_rng(seed)
    (d / "wav").mkdir(exist_ok=True)
    out = []
    for s, idx in shards.items():
        keep = np.sort(rng.choice(idx, min(per_shard, len(idx)), replace=False))
        t = pq.read_table(s, columns=["audio", "codec_name"]).take(keep)
        for i, (a, codec) in enumerate(zip(t["audio"].to_pylist(), t["codec_name"].to_pylist())):
            name = f"{s.stem}_{keep[i]}.wav"  # our own name: never trust paths stored in the data
            (d / "wav" / name).write_bytes(a["bytes"])
            out.append({"file": name, "codec": re.sub(r"[^\w.+-]", "_", codec)})
    with open(d / "clips.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["file", "codec"])
        w.writeheader()
        w.writerows(out)
    return len(out)


def subsample(rows, per_class: int | None, seed: int = 0):
    """Keep at most `per_class` real and `per_class` fake rows (random, fixed seed). None keeps everything."""
    if per_class is None:
        return list(rows)
    rng = np.random.default_rng(seed)
    out = []
    for label in (0, 1):
        idx = [i for i, r in enumerate(rows) if r[1] == label]
        keep = idx if len(idx) <= per_class else sorted(rng.choice(idx, per_class, replace=False).tolist())
        out += [rows[i] for i in keep]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["codecfake"])
    ap.add_argument("--n", type=int, default=10_000)
    a = ap.parse_args(argv)
    print(f"codecfake: {prepare_codecfake(a.n)} clips written")


if __name__ == "__main__":
    main()

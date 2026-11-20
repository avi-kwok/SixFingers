"""Embed every clip of a set and cache it: data/features/audio/<set>.npz.

    python -m ml.audio.extract --sets la_train la_dev la_eval itw --per-class 5000
    python -m ml.audio.extract --sets asv5_train_aug --per-class 12500   # "_aug": augmented copy (training only)
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ml.audio import datasets
from ml.audio.features import load_clip
from ml.common.paths import features_dir

AUG_SEED = 1  # fixed, so an augmented cache can be rebuilt exactly


def cache_path(name: str) -> Path:
    return features_dir() / "audio" / f"{name}.npz"


def embed_rows(rows, embedder, batch: int = 16, workers: int = 8, chunk: int = 512, aug_seed: int | None = None) -> dict:
    """rows -> {"path", "y", "attack", "L<k>" (float16 [N, D]) ...}.

    ffmpeg decodes the next chunk in threads while the model embeds the current one;
    only two chunks of audio are in memory at a time. A file ffmpeg cannot read raises.
    aug_seed: apply features.augment to every clip (training data only).
    """
    feats: dict[int, list] = {}
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        submit = lambda c: [pool.submit(load_clip, r[0], aug_seed) for r in rows[c:c + chunk]]  # noqa: E731
        nxt = submit(0)
        for c in range(0, len(rows), chunk):
            cur, nxt = nxt, submit(c + chunk)
            clips = [f.result() for f in cur]
            for b in range(0, len(clips), batch):
                for k, v in embedder.embed(np.stack(clips[b:b + batch])).items():
                    feats.setdefault(k, []).append(v.astype(np.float16))
            done = c + len(clips)
            print(f"{done}/{len(rows)} clips, {done / (time.time() - t0):.1f} clips/s", flush=True)
    out = {"path": np.array([r[0] for r in rows]), "y": np.array([r[1] for r in rows], dtype=np.int8),
           "attack": np.array([r[2] for r in rows])}
    out.update({f"L{k}": np.concatenate(v) for k, v in feats.items()})
    return out


def load_cache(name: str) -> dict:
    with np.load(cache_path(name)) as z:
        return {k: z[k] for k in z.files}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["la_train", "la_dev", "la_eval", "itw"],  # the a1 sets
                     choices=datasets.SETS + datasets.AUG_SETS)
    ap.add_argument("--per-class", type=int, default=None, help="max real and max fake clips per set (random, seed 0)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--force", action="store_true", help="re-embed sets that already have a cache file")
    a = ap.parse_args(argv)
    from ml.audio.features import XLSR
    embedder = XLSR()
    for name in a.sets:
        p = cache_path(name)
        if p.exists() and not a.force:
            print(f"{name}: cached at {p}, skipping")
            continue
        rows = datasets.subsample(datasets.rows_for(name), a.per_class)
        t0 = time.time()
        out = embed_rows(rows, embedder, batch=a.batch, aug_seed=AUG_SEED if name.endswith("_aug") else None)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.stem + ".tmp.npz")
        np.savez(tmp, **out)
        tmp.replace(p)
        print(f"{name}: {len(rows)} clips in {time.time() - t0:.0f} s -> {p}", flush=True)


if __name__ == "__main__":
    main()

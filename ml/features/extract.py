"""Extract + cache embeddings: python -m ml.features.extract --limit N [--backbones ...]."""
from __future__ import annotations

import argparse

from PIL import Image
from tqdm import tqdm

from ml.common.schemas import BACKBONES
from ml.features import store
from ml.features.augment import COMPRESSED_AUG, augment, compress

TRAIN_AUGS = (1, 2)


def jobs_for(row) -> list[int]:
    if row.split == "train":
        return [0, *TRAIN_AUGS]
    if row.split == "report_test":
        return [0, COMPRESSED_AUG]
    return [0]


def run(backbones, limit=None, device=None, batch_size=16, df=None) -> dict:
    from ml.features.backbones import load_backbone
    df = store.load_manifest() if df is None else df
    if limit:  # limit is per source so every split/side stays represented
        df = df.groupby("source", group_keys=False).head(limit)
    done = {}
    for name in backbones:
        todo = [(r, a) for r in df.itertuples() for a in jobs_for(r) if not store.has(name, r.image_id, a)]
        done[name] = len(todo)
        if not todo:
            continue
        bb = load_backbone(name, device)
        for i in tqdm(range(0, len(todo), batch_size), desc=name):
            chunk = todo[i:i + batch_size]
            ims = []
            for r, a in chunk:
                im = Image.open(store.resolve(r.path)).convert("RGB")
                ims.append(im if a == 0 else compress(im) if a == COMPRESSED_AUG else augment(im, r.image_id, a))
            for (r, a), v in zip(chunk, bb.embed(ims)):
                store.put(name, r.image_id, a, v)
        del bb
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200, help="max images per source")
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES), choices=BACKBONES)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    print(run(a.backbones, a.limit, a.device, a.batch_size))


if __name__ == "__main__":
    main()

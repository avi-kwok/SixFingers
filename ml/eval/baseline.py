"""Score report_test with one open HF detector so it can be compared on identical images.

Checkpoint: umm-maybe/AI-image-detector (public, Swin-based, labels 'artificial'/'human').
It was trained on older generators, so low numbers on newer ones are expected, not a bug.
"""
from __future__ import annotations

import argparse
import json

from PIL import Image
from tqdm import tqdm

from ml.common.paths import data_dir
from ml.features import store

CHECKPOINT = "umm-maybe/AI-image-detector"


def run(limit=None, device=None):
    from transformers import pipeline
    df = store.load_manifest()
    df = df[df.split == "report_test"]
    if limit:
        df = df.head(limit)
    pipe = pipeline("image-classification", model=CHECKPOINT, top_k=None, device=device)
    scores = {}
    for r in tqdm(list(df.itertuples())):
        res = pipe(Image.open(store.resolve(r.path)).convert("RGB"))
        scores[r.image_id] = float(next(x["score"] for x in res if x["label"].lower() == "artificial"))
    out = data_dir() / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_umm-maybe_AI-image-detector.json").write_text(json.dumps(scores))
    return len(scores)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device")
    a = ap.parse_args()
    print(run(a.limit, a.device))

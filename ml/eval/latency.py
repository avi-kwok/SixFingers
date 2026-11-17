"""CPU latency per image (batch 1, backbone embed only; probe math is microseconds). Writes <model_dir>/latency.json."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from ml.common.paths import current_model_dir
from ml.common.schemas import load_meta
from ml.features import store
from ml.features.backbones import load_backbone


def measure(model_dir: Path, n: int = 5) -> dict:
    meta = load_meta(model_dir / "meta.json")
    row = store.load_manifest().iloc[0]
    im = Image.open(store.resolve(row.path)).convert("RGB").resize((512, 512))
    out = {}
    for b in meta.backbones:
        bb = load_backbone(b, "cpu")
        bb.embed([im])  # warmup
        t = time.perf_counter()
        for _ in range(n):
            bb.embed([im])
        out[b] = round((time.perf_counter() - t) / n * 1000, 1)
    out["ensemble_total"] = round(sum(out.values()), 1)
    (model_dir / "latency.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("-n", type=int, default=5)
    a = ap.parse_args()
    print(measure((Path(a.model_dir) if a.model_dir else current_model_dir()).resolve(), a.n))

"""Synthetic stand-in dataset: 'real' = noisy smooth photos-ish, 'ai' = clean banded patterns. Not meant to look real."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def make(out: Path, n_per_cell: int = 3) -> Path:
    rng = np.random.default_rng(0)
    out.mkdir(parents=True, exist_ok=True)
    plan = []  # (label, generator, source, split)
    for split in ("train", "val", "gate"):
        plan += [(0, "real", "synth_real", split), (1, "genA", "synth_ai", split), (1, "genB", "synth_ai", split)]
    plan += [(0, "real", "raise", "report_test"), (1, "genHeld", "synth_ai", "report_test")]
    rows = []
    for label, gen, src, split in plan:
        for k in range(n_per_cell * (2 if split == "train" else 1)):
            yy, xx = np.mgrid[0:96, 0:96] / 96.0
            if label == 0:
                a = rng.uniform(0.2, 1)
                img = 0.5 + 0.4 * np.sin(6 * a * xx + 4 * yy)[..., None] * rng.uniform(0.3, 1, 3) + rng.normal(0, 0.08, (96, 96, 3))
            else:
                f = 3 if gen == "genA" else 5 if gen == "genB" else 8
                img = 0.5 + 0.45 * np.sign(np.sin(f * np.pi * (xx + rng.uniform() * yy)))[..., None] * rng.uniform(0.3, 1, 3)
            iid = f"{split}_{gen}_{k}"
            p = out / f"{iid}.png"
            Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(p)
            rows.append(dict(image_id=iid, path=str(p.name), label=label, generator=gen, source=src, split=split, license="synthetic"))
    pd.DataFrame(rows).to_parquet(out / "manifest.parquet")
    return out


if __name__ == "__main__":
    make(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "synth")

import json

import numpy as np
import pytest
from PIL import Image

from ml.data import build, splits

SOURCES = [
    {"name": "real_a", "side": "real", "generator": "real_a", "downloader": "none", "enabled": True, "license": "x"},
    {"name": "gen_a", "side": "ai", "generator": "gen_a", "downloader": "none", "enabled": True, "license": "x"},
    {"name": "hold_a", "side": "ai", "generator": "hold", "downloader": "none", "enabled": True, "license": "x"},
]


def _write_source(data_dir, idx, s, n=12):
    d = data_dir / "raw" / s["name"]
    d.mkdir(parents=True)
    rows = []
    for i in range(n):
        Image.fromarray(np.random.default_rng(100 * idx + i).integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(d / f"{i}.png")
        # real_a and gen_a share a group per pair (a caption's real/fake pair); hold rows are their own groups
        group = f"hold:{i}" if s["name"] == "hold_a" else f"pair:{i}"
        rows.append({"file": f"{i}.png", "group": group})
    (d / "index.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_bigger_build_after_freeze_appends_only(tmp_path, monkeypatch):
    data = tmp_path / "data"
    for idx, s in enumerate(SOURCES):
        _write_source(data, idx, s)
    monkeypatch.setenv("SIXFINGERS_DATA_DIR", str(data))
    monkeypatch.setattr(build, "load_registry", lambda: {"holdout_generator": "hold", "sources": SOURCES})
    monkeypatch.setattr(splits, "FROZEN_DIR", tmp_path / "frozen")

    df1 = build.build_manifest(4, skip_download=True)
    _, wrote = splits.freeze(df1.loc[df1.split == "report_test", "image_id"].tolist())
    assert wrote
    frozen = splits.read_frozen()

    df2 = build.build_manifest(10, True)  # must not raise
    assert set(df2.loc[df2.split == "report_test", "image_id"]) == frozen
    old = df1[df1.split != "report_test"].set_index("image_id").split
    new = df2[df2.split != "report_test"].set_index("image_id").split
    assert set(old.index) <= set(new.index)
    assert (new.loc[old.index] == old).all()
    assert len(new) > len(old)

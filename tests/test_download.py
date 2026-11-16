import io
import sys
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from ml.data import download, generate_local
from ml.data.sources import append_index, read_index, source_dir


def test_partition_share_and_stable():
    groups = [f"rapidata:{i:04d}" for i in range(2000)]
    share = sum(download.prompt_is_holdout(g) for g in groups) / len(groups)
    assert 0.25 < share < 0.35
    assert [download.prompt_is_holdout(g) for g in groups] == [download.prompt_is_holdout(g) for g in groups]


def test_training_prompts_keep_70_15_15_split():
    from ml.data.splits import assign_splits
    import pandas as pd
    groups = [g for g in (f"rapidata:{i:05d}" for i in range(20000)) if not download.prompt_is_holdout(g)]
    split = assign_splits(pd.DataFrame({"generator": "dalle3", "group": groups}), holdout="flux")
    share = split.value_counts(normalize=True)
    assert abs(share["train"] - 0.70) < 0.02 and abs(share["val"] - 0.15) < 0.02 and abs(share["gate"] - 0.15) < 0.02


def _jpeg(v: int) -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (16, 16), (v, v, v)).save(b, "JPEG")
    return b.getvalue()


def test_rapidata_partitions_prompts_between_holdout_and_training(tmp_path, monkeypatch):
    rows = {"prompt": [], "model1": [], "model2": [], "image1": [], "image2": [],
            "image1_path": [], "image2_path": []}
    for i in range(40):
        rows["prompt"].append(f"a photo number {i}")
        rows["model1"].append("flux")
        rows["model2"].append("dalle-3")
        rows["image1"].append({"bytes": _jpeg(i)})
        rows["image2"].append({"bytes": _jpeg(255 - i)})
        rows["image1_path"].append(f"flux/{i}_a.jpg")
        rows["image2_path"].append(f"dalle/{i}_b.jpg")
    pq_path = tmp_path / "shard.parquet"
    pq.write_table(pa.table(rows), pq_path)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(pq_path))
    monkeypatch.setenv("SIXFINGERS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(download, "holdout_generator", lambda *a, **k: "flux")

    def src(name, generator, model_name):
        return {"name": name, "side": "ai", "generator": generator, "model_name": model_name,
                "license": "test", "shards": ["x.parquet"]}

    download.rapidata(src("rapidata_flux", "flux", "flux"), 1000)
    download.rapidata(src("rapidata_dalle3", "dalle3", "dalle-3"), 1000)

    flux = read_index("rapidata_flux")
    dalle = read_index("rapidata_dalle3")
    assert flux and dalle
    assert not {r["group"] for r in flux} & {r["group"] for r in dalle}
    assert all(download.prompt_is_holdout(r["group"]) for r in flux)
    for name, rows_ in (("rapidata_flux", flux), ("rapidata_dalle3", dalle)):
        on_disk = {p.name for p in source_dir(name).iterdir()} - {"index.jsonl"}
        assert on_disk == {r["file"] for r in rows_}


def test_grow_skips_report_test_sources_and_uses_index_plus_n(tmp_path, monkeypatch):
    monkeypatch.setenv("SIXFINGERS_DATA_DIR", str(tmp_path))
    append_index("normal", {"file": "a.jpg", "group": "g"})
    append_index("normal", {"file": "b.jpg", "group": "g"})
    append_index("normal", {"file": "c.jpg", "group": "g"})
    sources = [
        {"name": "holdout", "downloader": "rapidata", "generator": "flux"},
        {"name": "raise", "downloader": "raise", "generator": "real_raise"},
        {"name": "normal", "downloader": "coco", "generator": "real_coco"},
    ]
    calls = []
    monkeypatch.setattr(download, "enabled_sources", lambda *a, **k: sources)
    monkeypatch.setattr(download, "holdout_generator", lambda *a, **k: "flux")
    monkeypatch.setattr(download, "DOWNLOADERS", {
        name: (lambda s, limit, name=name: calls.append((s["name"], limit)))
        for name in ("rapidata", "raise", "coco")})
    monkeypatch.setattr(sys, "argv", ["x", "--grow", "5"])
    download.main()
    assert calls == [("normal", 8)]


def test_generate_local_grow_adds_n_images(tmp_path, monkeypatch):
    monkeypatch.setenv("SIXFINGERS_DATA_DIR", str(tmp_path))
    for i in range(6):
        append_index("coco", {"file": f"{i:012d}.jpg", "caption": f"caption {i}", "group": f"coco:{i}"})
    for i in range(2):
        append_index("sdxl_turbo", {"file": f"{i:012d}.png", "group": f"coco:{i}"})

    def fake_pipe(caption, generator=None, **kw):
        return SimpleNamespace(images=[Image.new("RGB", (64, 64), (128, 128, 128))])

    monkeypatch.setattr(generate_local, "load_pipe", lambda cfg: (fake_pipe, "cpu"))
    monkeypatch.setattr(sys, "argv", ["x", "--grow", "3"])
    generate_local.main()
    assert len({r["file"] for r in read_index("sdxl_turbo")}) == 5

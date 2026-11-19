import json

import numpy as np
import pandas as pd

from ml.eval import scorecard
from ml.features import store
from ml.features.augment import COMPRESSED_AUG
from tests.api.conftest import make_model


def test_scorecard_endpoint(build, models_dir):
    c = build()
    assert c.get("/v1/scorecard").status_code == 404
    (models_dir / "v1" / "scorecard.json").write_text(json.dumps({"model_version": "v1"}))
    r = c.get("/v1/scorecard")
    assert r.status_code == 200 and r.json() == {"model_version": "v1"}


def test_build_scorecard(tmp_path, monkeypatch):
    monkeypatch.setenv("SIXFINGERS_DATA_DIR", str(tmp_path / "data"))
    # (image_id, label, generator, source, split, first feature): +5 scores ~0.99, -5 ~0.01, t_ai is 0.8
    rows = [("a1", 1, "flux", "rapidata_flux", "report_test", 5), ("a2", 1, "flux", "rapidata_flux", "report_test", 5),
            ("r1", 0, "real_raise", "raise", "report_test", 5), ("r2", 0, "real_raise", "raise", "report_test", -5),
            ("c1", 0, "real_coco", "coco", "val", -5), ("v1", 1, "sdxl_turbo", "sdxl_turbo", "val", 5),
            ("t1", 0, "real_coco", "coco", "train", 5)]
    df = pd.DataFrame([r[:5] for r in rows], columns=["image_id", "label", "generator", "source", "split"])
    df["path"], df["license"] = "x.png", "test"
    (tmp_path / "data").mkdir()
    df.to_parquet(tmp_path / "data" / "manifest.parquet")
    for iid, *_, split, x in rows:
        for b in ("clip_vitl14", "dinov2_vitl14"):
            for aug in ((0, COMPRESSED_AUG) if split == "report_test" else (0,)):
                store.put(b, iid, aug, np.array([x, 0, 0, 0]))
    vdir = make_model(tmp_path / "models", "v1")
    before = {p.name: p.read_bytes() for p in vdir.iterdir()}
    cache = tmp_path / "cache.jsonl"
    cache.write_text("\n".join(json.dumps({"arm": a, "image_id": i, "verdict": v}) for a, i, v in [
        ("haiku_blind", "a1", "likely_ai"), ("haiku_blind", "a2", "uncertain"), ("haiku_blind", "r1", "likely_ai"),
        ("haiku_blind", "r2", "likely_real"), ("haiku_blind", "c1", "likely_ai"), ("other_arm", "a2", "likely_ai")]))
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"model": "claude-haiku-5-5", "arm": "haiku_blind", "method": "platt",
                               "report_test": {"platt": {"ece": 0.03}}}))

    sc = scorecard.build(vdir, cache, cal)

    clean, comp = sc["detector"]["clean"], sc["detector"]["compressed"]
    assert sc["model_version"] == "v1" and sc["t_ai"] == 0.8
    assert clean["overall"]["n"] == 4 and clean["overall"]["auroc"] == 0.75
    assert clean["generators"] == [{"generator": "flux", "split": "report_test", "n": 2, "catch_rate": 1.0}]
    assert clean["real_sources"] == [{"source": "raise", "split": "report_test", "n": 2, "false_alarm_rate": 0.5},
                                     {"source": "coco", "split": "val", "n": 1, "false_alarm_rate": 0.0}]
    assert [r["split"] for r in comp["real_sources"]] == ["report_test"]  # no compressed val features cached
    assert sc["judge"] == {"model": "claude-haiku-5-5", "split": "report_test", "n": 4, "accuracy": 0.5,
                           "uncertain_rate": 0.25, "calibrated_ece": 0.03, "calibration_method": "platt"}
    assert {p.name: p.read_bytes() for p in vdir.iterdir()} == before  # build() writes nothing

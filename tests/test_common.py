import numpy as np

from ml.common import api_models as a
from ml.common.schemas import (ProbeArtifact, Thresholds, band_for, ensemble_score,
                               load_probe, probe_logit, save_probe)


def test_probe_roundtrip_and_band(tmp_path):
    p = ProbeArtifact(backbone="x", embed_dim=2, mean=[0, 0], std=[1, 1], weights=[1, 0], bias=0)
    save_probe(p, tmp_path / "p.json")
    assert load_probe(tmp_path / "p.json") == p
    s = ensemble_score([probe_logit(p, np.array([[5.0, 0], [-5.0, 0]]))])
    t = Thresholds(t_ai=0.9, t_real=0.1)
    assert [band_for(x, t) for x in s] == ["likely_ai", "likely_real"]


def test_timings_alias():
    assert "validate" in a.Timings().model_dump(by_alias=True)

import json

import pytest

from api.judge import SCHEMA, SYSTEM
from ml.common.api_models import Cue
from tests.api.conftest import JUDGE_JSON, FakeClient, img_bytes, post


@pytest.mark.parametrize("raw, want", [
    ([0.1, 0.2, 0.5, 0.6], [0.1, 0.2, 0.5, 0.6]),
    ([-0.2, 0, 1.4, 1], [0.0, 0.0, 1.0, 1.0]),   # clamped into the image
    ([0, 0, 1, 1], [0, 0, 1, 1]),
    (None, None),
    ([0.5, 0.2, 0.1, 0.6], None),                 # x0 > x1
    ([0.1, 0.6, 0.5, 0.6], None),                 # zero height
    ([1.2, 0.1, 1.5, 0.4], None),                 # empty once clamped
    ([0.1, 0.2, 0.5], None),                      # wrong length
    ([0.1, "a", 0.5, 0.6], None),
    ([True, 0, 1, 1], None),
    ([float("nan"), 0, 1, 1], None),
    ("0 0 1 1", None),
])
def test_box_validation_keeps_cue(raw, want):
    c = Cue(region="hands", cue="six fingers", box=raw)
    assert (c.region, c.box) == ("hands", want)


def test_box_optional_in_schema_and_prompt():
    cue = SCHEMA["$defs"]["Cue"]
    assert "box" in cue["properties"] and "box" not in cue.get("required", [])
    assert "box" in SYSTEM


def test_response_boxes(build):
    out = json.loads(JUDGE_JSON)
    out["cues"] = [{"region": "hand", "cue": "six fingers", "box": [0.2, 0.3, 1.3, 0.5]},
                   {"region": "sign", "cue": "warped text", "box": [0.9, 0.1, 0.2, 0.4]},
                   {"region": "whole image", "cue": "plastic sheen"}]
    cues = post(build(FakeClient(json.dumps(out))), img_bytes(0)).json()["judge"]["cues"]
    assert [c["box"] for c in cues] == [[0.2, 0.3, 1.0, 0.5], None, None]
    assert [c["region"] for c in cues] == ["hand", "sign", "whole image"]

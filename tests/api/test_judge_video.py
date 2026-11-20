import io

import pytest
from PIL import Image

from api.judge import MODEL, VIDEO_SCHEMA, Budget, Judge, JudgeUnavailable, VIDEO_SYSTEM
from ml.common.api_models import VideoJudgeOutput
from ml.common.judge_backend import ApiBackend
from tests.api.conftest import FakeClient

VIDEO_JSON = (
    '{"verdict": "likely_ai", "confidence": 0.8, "explanation": "Shapes melt between frames.",'
    ' "cues": [{"frame": 3, "region": "clock", "cue": "digits warp"}], "overrode_detector": false}'
)


def jpeg():
    b = io.BytesIO()
    Image.new("RGB", (64, 36), (120, 120, 120)).save(b, "JPEG")
    return b.getvalue()


def judge_with(tmp_path, client, limit=100):
    return Judge(ApiBackend(client), Budget(tmp_path / "b.json", limit))


def test_judge_video_request_and_result(tmp_path):
    client = FakeClient(VIDEO_JSON)
    out = judge_with(tmp_path, client).judge_video(jpeg(), 16, 5.0)
    assert isinstance(out, VideoJudgeOutput) and out.cues[0].frame == 3
    call = client.calls[0]
    blocks = call["messages"][0]["content"]
    assert sorted(b["type"] for b in blocks) == ["image", "text"]
    text = next(b["text"] for b in blocks if b["type"] == "text")
    assert "frames 1 to 16" in text
    assert call["model"] == "claude-haiku-5-5" == MODEL
    assert "video" in VIDEO_SYSTEM.lower()
    banned = ("detector", "score", "probe", "t_ai", "t_real", "backbones", "clip_vitl14")
    assert not [w for w in banned if w in (call["system"] + text).lower()]


def test_judge_video_budget_limit(tmp_path):
    client = FakeClient(VIDEO_JSON)
    judge = judge_with(tmp_path, client, limit=1)
    judge.judge_video(jpeg(), 16, 5.0)
    with pytest.raises(JudgeUnavailable) as e:
        judge.judge_video(jpeg(), 16, 5.0)
    assert e.value.reason == "budget" and len(client.calls) == 1


def test_judge_video_refusal(tmp_path):
    with pytest.raises(JudgeUnavailable) as e:
        judge_with(tmp_path, FakeClient("", stop_reason="refusal")).judge_video(jpeg(), 16, 5.0)
    assert e.value.reason == "refusal"


def test_judge_video_disabled(tmp_path):
    with pytest.raises(JudgeUnavailable) as e:
        Judge(None, Budget(tmp_path / "b.json", 100)).judge_video(jpeg(), 16, 5.0)
    assert e.value.reason == "disabled"


def test_video_schema_cues_require_frame():
    schema = VIDEO_SCHEMA
    cue = schema["properties"]["cues"]["items"]
    if "$ref" in cue:
        cue = schema["$defs"][cue["$ref"].rsplit("/", 1)[-1]]
    assert "frame" in cue["required"]

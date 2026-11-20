import base64
import io
import json
import os

import pytest
from PIL import Image

import ml.common.judge_backend as jb
from ml.common.api_models import JudgeOutput, VideoJudgeOutput
from ml.common.judge_backend import ApiBackend, BackendError, CommandBackend, from_env, prepare_image
from tests.api.conftest import JUDGE_JSON, FakeClient, command_script

VIDEO_JSON = json.dumps({"verdict": "likely_ai", "confidence": 0.8, "explanation": "Shapes melt between frames.",
                         "cues": [{"frame": 3, "region": "clock", "cue": "digits warp"}], "overrode_detector": False})


def png(size, noise=False):
    b = io.BytesIO()
    img = Image.frombytes("RGB", size, os.urandom(size[0] * size[1] * 3)) if noise else Image.new("RGB", size, (90, 120, 200))
    img.save(b, "PNG")
    return b.getvalue()


def ask(backend, image=None):
    return ask_with(backend, JudgeOutput, image)


def ask_with(backend, out_model, image=None):
    return backend.ask(model="claude-haiku-5-5", system="s", image=image or png((32, 32)), evidence="e",
                       schema={"type": "object"}, max_tokens=800, timeout=15, out_model=out_model)


def test_prepare_image_downscales_to_jpeg():
    out = prepare_image(png((4000, 3000)))
    assert out[:2] == b"\xff\xd8"
    assert max(Image.open(io.BytesIO(out)).size) == 1568


def test_prepare_image_never_upscales():
    out = prepare_image(png((300, 200)))
    assert out[:2] == b"\xff\xd8" and Image.open(io.BytesIO(out)).size == (300, 200)


def test_prepare_image_respects_byte_cap(monkeypatch):
    monkeypatch.setattr(jb, "MAX_BYTES", 50_000)
    out = prepare_image(png((1568, 1568), noise=True))
    assert len(out) <= 50_000


def test_command_success_request_and_usage(tmp_path):
    saved = tmp_path / "req.json"
    cmd = command_script(tmp_path, f"import sys\nopen({str(saved)!r}, 'w').write(sys.stdin.read())\nprint({JUDGE_JSON!r})\n")
    out, usage = ask(CommandBackend(cmd), png((4000, 3000)))
    assert out == JudgeOutput.model_validate_json(JUDGE_JSON) and usage == (0, 0)
    req = json.loads(saved.read_text())
    assert set(req) == {"model", "system", "evidence", "image", "schema", "max_tokens"}
    assert req["image"]["media_type"] == "image/jpeg" and isinstance(req["schema"], dict)
    img = Image.open(io.BytesIO(base64.b64decode(req["image"]["data"])))
    assert img.format == "JPEG" and max(img.size) <= 1568


def test_command_exit_75_with_retry_after(tmp_path):
    cmd = command_script(tmp_path, "import sys\nsys.stdin.read()\nprint('{\"retry_after_s\": 7}')\nsys.exit(75)\n")
    with pytest.raises(BackendError) as e:
        ask(CommandBackend(cmd))
    assert e.value.reason == "retry_later" and e.value.retry_after_s == 7


def test_command_exit_75_without_stdout_uses_default(tmp_path):
    cmd = command_script(tmp_path, "import sys\nsys.stdin.read()\nsys.exit(75)\n")
    with pytest.raises(BackendError) as e:
        ask(CommandBackend(cmd))
    assert e.value.reason == "retry_later" and e.value.retry_after_s == 1800


def test_command_other_failures(tmp_path):
    fail = command_script(tmp_path, "import sys\nsys.stdin.read()\nsys.exit(1)\n", "fail.py")
    with pytest.raises(BackendError) as e:
        ask(CommandBackend(fail))
    assert e.value.reason == "error"
    bad = command_script(tmp_path, "import sys\nsys.stdin.read()\nprint('not json')\n", "bad.py")
    with pytest.raises(BackendError) as e:
        ask(CommandBackend(bad))
    assert e.value.reason == "error"


def test_command_non_utf8_stdout_is_error(tmp_path):
    cmd = command_script(tmp_path, "import sys\nsys.stdin.read()\nsys.stdout.buffer.write(b'\\xff\\xfex')\n", "binary.py")
    with pytest.raises(BackendError) as e:
        ask(CommandBackend(cmd))
    assert e.value.reason == "error"


def test_command_timeout(tmp_path):
    cmd = command_script(tmp_path, "import sys, time\nsys.stdin.read()\ntime.sleep(5)\n", "slow.py")
    with pytest.raises(BackendError) as e:
        ask(CommandBackend(cmd, timeout_s=0.5))
    assert e.value.reason == "timeout"


def test_from_env(monkeypatch):
    monkeypatch.delenv("SIXFINGERS_JUDGE_BACKEND", raising=False)
    monkeypatch.delenv("SIXFINGERS_JUDGE_CMD", raising=False)
    assert isinstance(from_env(object()), ApiBackend)
    assert from_env(None) is None
    monkeypatch.setenv("SIXFINGERS_JUDGE_BACKEND", "command")
    assert from_env(object()) is None
    monkeypatch.setenv("SIXFINGERS_JUDGE_BACKEND", "bogus")
    with pytest.raises(ValueError, match="SIXFINGERS_JUDGE_BACKEND"):
        from_env(object())


def test_command_video_out_model(tmp_path):
    saved = tmp_path / "req.json"
    cmd = command_script(tmp_path, f"import sys\nopen({str(saved)!r}, 'w').write(sys.stdin.read())\nprint({VIDEO_JSON!r})\n")
    out, _usage = ask_with(CommandBackend(cmd), VideoJudgeOutput)
    assert isinstance(out, VideoJudgeOutput) and out.cues[0].frame == 3
    assert set(json.loads(saved.read_text())) == {"model", "system", "evidence", "image", "schema", "max_tokens"}


def test_api_video_out_model():
    out, usage = ask_with(ApiBackend(FakeClient(VIDEO_JSON)), VideoJudgeOutput)
    assert isinstance(out, VideoJudgeOutput) and out.verdict == "likely_ai" and usage == (0, 0)


def test_default_out_model_parses_judge_output():
    backend = ApiBackend(FakeClient(JUDGE_JSON))
    out, _usage = backend.ask(model="claude-haiku-5-5", system="s", image=png((32, 32)), evidence="e",
                              schema={"type": "object"}, max_tokens=800, timeout=15)
    assert type(out) is JudgeOutput

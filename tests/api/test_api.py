import json
import os

import anthropic
import httpx

from ml.common.judge_backend import ApiBackend
from tests.api.conftest import JUDGE_JSON, FakeClient, command_script, img_bytes, make_model, post


def test_detector_only_when_no_key(build):
    r = post(build(client=None), img_bytes(255))
    assert r.status_code == 200
    j = r.json()
    assert j["source"] == "detector_only" and j["judge"] is None
    assert j["judge_unavailable_reason"] == "disabled"
    assert j["verdict"] == j["detector"]["band"] == "likely_ai"
    assert set(j["timings_ms"]) == {"validate", "provenance", "detector", "judge", "total"}
    assert j["model_version"] == "v1" and set(j["detector"]["backbones"]) == {"clip_vitl14", "dinov2_vitl14"}


def test_judge_success_and_request_shape(build):
    fake = FakeClient(JUDGE_JSON)
    j = post(build(fake), img_bytes(255)).json()
    assert j["source"] == "judge" and j["verdict"] == "likely_real"
    assert j["judge"]["overrode_detector"] is True and j["judge_unavailable_reason"] is None
    kw = fake.calls[0]
    assert kw["model"] == "claude-haiku-5-5" and kw["output_config"]["effort"] == "low"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert not {"temperature", "top_p", "top_k", "thinking"} & set(kw)


def test_judge_failures_fall_back(build):
    req = httpx.Request("POST", "http://x")
    cases = {
        "timeout": FakeClient(exc=anthropic.APITimeoutError(request=req)),
        "error": FakeClient(exc=RuntimeError("boom")),
    }
    for reason, fake in cases.items():
        j = post(build(fake), img_bytes(0)).json()
        assert j["source"] == "detector_only" and j["judge_unavailable_reason"] == reason
        assert j["verdict"] == "likely_real"
    assert post(build(FakeClient("not json")), img_bytes(0)).json()["judge_unavailable_reason"] == "error"
    assert post(build(FakeClient(JUDGE_JSON, stop_reason="max_tokens")), img_bytes(0)).json()["judge_unavailable_reason"] == "error"


def test_refusal_is_422_and_logged(build, caplog):
    with caplog.at_level("INFO", logger="sixfingers"):
        r = post(build(FakeClient("", stop_reason="refusal")), img_bytes(0))
    assert r.status_code == 422 and r.json() == {"detail": "We can't analyse this image."}
    ev = [x for x in caplog.records if getattr(x, "event", None) == "refusal"]
    assert len(ev) == 1 and ev[0].ip_hash and not hasattr(ev[0], "verdict")


def test_judge_prompt_has_no_probe_scores(build):
    fake = FakeClient(JUDGE_JSON)
    post(build(fake), img_bytes(255))
    kw = fake.calls[0]
    text = kw["messages"][0]["content"][1]["text"] + kw["system"]
    for word in ("detector", "t_ai", "t_real", "backbones", "score", "clip_vitl14"):
        assert word not in text.lower()
    assert "Provenance" in text


def test_overrode_detector_computed_from_outcome(build):
    def run(verdict, color):
        out = json.loads(JUDGE_JSON) | {"verdict": verdict, "overrode_detector": verdict == "likely_real"}
        return post(build(FakeClient(json.dumps(out))), img_bytes(color)).json()["judge"]["overrode_detector"]
    assert run("likely_real", 255) is True    # decisive and differs from likely_ai
    assert run("likely_real", 0) is False     # model claims override but agrees with band
    assert run("uncertain", 255) is False     # not decisive
    assert run("likely_ai", 0) is True


def test_detector_and_judge_run_concurrently(build, models_dir, tmp_path, monkeypatch):
    import threading
    from fastapi.testclient import TestClient
    from api.main import create_app
    from tests.api.conftest import StubEmbedder
    started = threading.Event()
    seen = []

    class WaitingEmbedder(StubEmbedder):
        def embed(self, backbone, image):
            seen.append(started.wait(2))  # only true if the judge call began while the detector runs
            return super().embed(backbone, image)

    def ask(self, **kw):
        started.set()
        return __import__("ml.common.api_models", fromlist=["JudgeOutput"]).JudgeOutput(**json.loads(JUDGE_JSON)), (0, 0)
    monkeypatch.setattr(ApiBackend, "ask", ask)
    app = create_app(embedder=WaitingEmbedder(), anthropic_client=FakeClient(), models_dir=models_dir,
                     budget_path=tmp_path / "b.json")
    assert post(TestClient(app), img_bytes(255)).status_code == 200
    assert seen and all(seen)


def test_backend_crash_falls_back(build, monkeypatch):
    def boom(self, **kw):
        raise RuntimeError("unexpected")
    monkeypatch.setattr(ApiBackend, "ask", boom)
    r = post(build(FakeClient(JUDGE_JSON)), img_bytes(0))
    assert r.status_code == 200
    j = r.json()
    assert j["source"] == "detector_only" and j["judge_unavailable_reason"] == "error"


def test_budget_exhausted_and_persisted(build, tmp_path):
    fake = FakeClient(JUDGE_JSON)
    c = build(fake, budget_limit=1)
    assert post(c, img_bytes(255)).json()["source"] == "judge"
    j = post(c, img_bytes(255)).json()
    assert j["judge_unavailable_reason"] == "budget" and len(fake.calls) == 1
    assert json.loads((tmp_path / "budget.json").read_text())["count"] == 1
    # a fresh app (restart) still sees the spent budget
    assert post(build(FakeClient(JUDGE_JSON), budget_limit=1), img_bytes(255)).json()["judge_unavailable_reason"] == "budget"


def test_rate_limit_429(build):
    c = build(rate_limit=2)
    assert [post(c, img_bytes()).status_code for _ in range(3)] == [200, 200, 429]
    assert int(post(c, img_bytes()).headers["retry-after"]) > 0


def test_bad_uploads(build):
    c = build()
    assert post(c, b"hello world", "a.png").status_code == 415
    assert post(c, b"GIF89a" + b"0" * 50, "a.png", "image/png").status_code == 415
    assert post(c, b"\xff\xd8\xff" + b"junk").status_code == 400
    assert post(c, b"x" * (10 * 1024 * 1024 + 1)).status_code == 413
    assert c.post("/v1/check").status_code == 400


def test_pixel_cap(build, monkeypatch):
    monkeypatch.setattr("api.validate.MAX_PIXELS", 100)
    assert post(build(), img_bytes(size=(50, 50))).status_code == 413


def test_hot_reload(build, models_dir):
    c = build()
    assert c.get("/v1/health").json()["model_version"] == "v1"
    make_model(models_dir, "v2", weight=-1.0)  # flipped probe: white now looks real
    (models_dir / "current.tmp").symlink_to("v2")
    os.replace(models_dir / "current.tmp", models_dir / "current")
    assert c.get("/v1/model").json()["version"] == "v2"
    assert post(c, img_bytes(255)).json()["verdict"] == "likely_real"
    # a broken new target keeps serving v2
    (models_dir / "v3").mkdir()
    (models_dir / "current.tmp").symlink_to("v3")
    os.replace(models_dir / "current.tmp", models_dir / "current")
    assert c.get("/v1/model").json()["version"] == "v2"


def test_model_health_cors(build):
    c = build()
    m = c.get("/v1/model").json()
    assert m["metrics"] == {"auroc": 0.9} and m["thresholds"]["t_ai"] == 0.8
    ok = c.get("/v1/health", headers={"Origin": "https://sixfingers.avikwok.dev"})
    assert ok.headers["access-control-allow-origin"] == "https://sixfingers.avikwok.dev"
    bad = c.get("/v1/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in bad.headers


def test_provenance_hint_and_log_has_no_image(build, caplog):
    data = img_bytes(10) + b'<x a="http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"/>'
    with caplog.at_level("INFO", logger="sixfingers"):
        j = post(build(), data).json()
    assert any("trainedalgorithmicmedia" in h for h in j["provenance"]["exif_hints"])
    assert "ip_hash" in str(caplog.records[-1].__dict__)


def test_c2pa_unverified_manifest_gets_no_ai_badge():
    from api.provenance import _info
    store = {"active_manifest": "m", "manifests": {"m": {
        "assertions": [{"data": {"actions": [{"digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"}]}}],
        "signature_info": {"issuer": "Evil"}}}}
    bad, ok = _info(store, "Invalid"), _info(store, "Valid")
    assert bad.present and bad.ai_generated is None and bad.issuer is None
    assert ok.ai_generated is True and ok.issuer == "Evil"
    store["manifests"]["m"]["assertions"][0]["data"]["actions"][0]["digitalSourceType"] += "x"
    assert _info(store, "Valid").ai_generated is None


def test_command_backend_judge(build, monkeypatch, tmp_path):
    cmd = command_script(tmp_path, "import sys\nsys.stdin.read()\nprint(" + repr(JUDGE_JSON) + ")\n")
    monkeypatch.setenv("SIXFINGERS_JUDGE_BACKEND", "command")
    monkeypatch.setenv("SIXFINGERS_JUDGE_CMD", cmd)
    j = post(build(), img_bytes(255)).json()
    assert j["source"] == "judge" and j["verdict"] == "likely_real"


def test_command_backend_retry_later_falls_back(build, monkeypatch, tmp_path):
    cmd = command_script(tmp_path, "import sys\nsys.stdin.read()\nsys.exit(75)\n")
    monkeypatch.setenv("SIXFINGERS_JUDGE_BACKEND", "command")
    monkeypatch.setenv("SIXFINGERS_JUDGE_CMD", cmd)
    j = post(build(), img_bytes(255)).json()
    assert j["source"] == "detector_only" and j["judge_unavailable_reason"] == "error"

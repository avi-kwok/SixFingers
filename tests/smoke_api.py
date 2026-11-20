"""Live smoke: start API (real detector, judge disabled), POST one real and one AI image from the manifest, check the schema."""
import os, subprocess, sys, time
import pandas as pd, requests

drop = {"ANTHROPIC_API_KEY", "SIXFINGERS_JUDGE_BACKEND", "SIXFINGERS_JUDGE_CMD"}
env = {k: v for k, v in os.environ.items() if k not in drop}
srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8123"], env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    base = "http://127.0.0.1:8123"
    for _ in range(60):
        try:
            if requests.get(base + "/v1/health", timeout=2).ok:
                break
        except requests.RequestException:
            pass
        time.sleep(1)  # also on 503 "loading"
    df = pd.read_parquet("data/manifest.parquet")
    df = df[df.split == "report_test"]
    keys = {"verdict", "source", "judge", "judge_unavailable_reason", "detector", "provenance", "model_version", "timings_ms", "disclaimer"}
    for label in (1, 0):
        p = df[df.label == label].iloc[0].path
        r = requests.post(base + "/v1/check", files={"image": open(p, "rb")}, timeout=120)
        j = r.json()
        assert r.status_code == 200 and keys <= set(j), (r.status_code, j)
        assert j["source"] == "detector_only" and j["judge"] is None and j["judge_unavailable_reason"] == "disabled"
        print("label", label, j["verdict"], round(j["detector"]["score"], 3), j["detector"]["backbones"], j["timings_ms"])
    print(requests.get(base + "/v1/model").json().keys())
finally:
    srv.terminate()

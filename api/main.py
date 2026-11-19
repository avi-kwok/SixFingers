"""FastAPI app: POST /v1/check, GET /v1/model, GET /v1/scorecard, GET /v1/health."""
import json
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.detector import DetectorStore, FeatureEmbedder
from api.fetch import fetch_image
from api.judge import MODEL, Budget, Judge, JudgeUnavailable
from api.limits import RateLimiter
from api.provenance import read_provenance
from api.validate import MAX_BYTES, UploadError, validate_upload
from ml.common import judge_backend, paths
from ml.common.api_models import CheckResponse, JudgeResult, Timings

log = logging.getLogger("sixfingers")
PROD_ORIGIN = "https://sixfingers.avikwok.dev"
DISCLAIMER = ("This is an estimate, not proof. It only covers fully AI-generated still images; edited, "
              "inpainted or upscaled images are out of scope. Do not use it as the sole basis for decisions.")
BUSY_WAIT_S = 5  # how long a check waits for a free analysis slot before answering 503


_CAL = json.loads((Path(__file__).parent / "judge_calibration.json").read_text())["points"]


def judge_probability(verdict: str, confidence: float) -> float:
    """Calibrated P(AI) from the judge's verdict and confidence (map fitted by ml/judge_eval/calibrate.py)."""
    if verdict not in ("likely_ai", "likely_real"):
        return 0.5  # 'uncertain' means no lean either way; the map is fitted on decisive verdicts only
    raw = confidence if verdict == "likely_ai" else 1 - confidence
    xs, ys = zip(*_CAL)
    return float(np.interp(raw, xs, ys))


class _JsonFmt(logging.Formatter):
    def format(self, r):
        d = {"time": self.formatTime(r, "%Y-%m-%dT%H:%M:%S"), "level": r.levelname, "msg": r.getMessage()}
        d.update({k: v for k, v in r.__dict__.items() if k in ("event", "verdict", "source", "score", "dims",
                                                            "format", "latency_ms", "model_version", "ip_hash", "reason", "input", "host")})
        return json.dumps(d)


def _setup_logging():
    if not any(isinstance(h.formatter, _JsonFmt) for h in log.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(_JsonFmt())
        log.addHandler(h)
    log.setLevel(logging.INFO)


def _make_client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic
    return anthropic.Anthropic(max_retries=0)  # key read from env by the SDK; never touched here


def create_app(embedder=None, anthropic_client="auto", models_dir: Path | None = None,
               budget_path: Path | None = None, budget_limit: int | None = None,
               rate_limit: int = 10) -> FastAPI:
    _setup_logging()
    store = DetectorStore(embedder or FeatureEmbedder(), models_dir)

    @asynccontextmanager
    async def lifespan(_app):
        threading.Thread(target=store.warm, daemon=True).start()  # /v1/health answers 503 until this finishes
        yield

    app = FastAPI(title="SixFingers", docs_url=None, redoc_url=None, lifespan=lifespan)
    origins = os.environ.get("SIXFINGERS_CORS_ORIGINS", PROD_ORIGIN).split(",")
    if os.environ.get("SIXFINGERS_DEV") == "1":
        origins += ["http://localhost:3000", "http://127.0.0.1:3000"]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.middleware("http")
    async def cap_body(request: Request, call_next):
        # shortcut: trusts Content-Length; chunked bodies are capped by the reverse proxy (Caddy max_size)
        if int(request.headers.get("content-length") or 0) > MAX_BYTES + 100_000:
            return JSONResponse({"detail": "Image is larger than 10 MB."}, 413)
        return await call_next(request)

    client = _make_client() if anthropic_client == "auto" else anthropic_client
    limit = budget_limit if budget_limit is not None else int(os.environ.get("SIXFINGERS_JUDGE_BUDGET", 500))
    judge = Judge(judge_backend.from_env(client), Budget(budget_path or paths.data_dir() / "judge_budget.json", limit))
    limiter = RateLimiter(rate_limit)
    # ip_hash changes at the UTC day change, so this rolling 24 h window is a per-day judge share per IP
    judge_quota = RateLimiter(int(os.environ.get("SIXFINGERS_JUDGE_PER_IP_DAY", 20)), window_s=86400)
    # decoding and embedding a large image costs hundreds of MB, so only a few run at once
    heavy = threading.BoundedSemaphore(int(os.environ.get("SIXFINGERS_MAX_CONCURRENT", 2)))
    pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="judge")
    app.state.store, app.state.judge, app.state.limiter = store, judge, limiter

    def client_ip(req: Request) -> str:
        # behind Caddy the last X-Forwarded-For entry is the one the proxy added
        if os.environ.get("SIXFINGERS_TRUST_PROXY") == "1" and req.headers.get("x-forwarded-for"):
            return req.headers["x-forwarded-for"].split(",")[-1].strip()
        return req.client.host if req.client else "unknown"

    @app.post("/v1/check", response_model=CheckResponse, response_model_by_alias=True)
    def check(request: Request, image: UploadFile | None = File(None),
              url: str | None = Form(None)):
        t0 = time.perf_counter()
        ms = lambda a: int((a - t0) * 1000)  # noqa: E731
        if (image is None) == (url is None):
            raise HTTPException(400, "Send either an image file or an image link.")
        ip_hash = limiter.ip_hash(client_ip(request))
        wait = limiter.allow(ip_hash)
        if wait:
            raise HTTPException(429, "Too many checks. Try again later.", headers={"Retry-After": str(wait)})
        try:
            data = fetch_image(url) if url is not None else image.file.read(MAX_BYTES + 1)  # never buffer more than the cap
        except UploadError as e:
            raise HTTPException(e.status, e.detail) from None
        if not heavy.acquire(timeout=BUSY_WAIT_S):
            raise HTTPException(503, "SixFingers is busy right now. Try again in a minute.")
        det = jfut = None
        try:
            try:
                checked = validate_upload(data)
            except UploadError as e:
                raise HTTPException(e.status, e.detail) from None
            t1 = time.perf_counter()
            src = {"input": "link", "host": urlsplit(url.strip()).hostname} if url is not None else {"input": "file"}
            prov = read_provenance(data, checked.mime, checked.image)
            t2 = time.perf_counter()
            try:
                store.get()  # no model: fail before the judge spends budget
                if not (judge.enabled and judge_quota.allow(ip_hash)):  # allow() returns 0 when under quota
                    try:
                        small = judge_backend.downscale(data) if judge.enabled else b""  # decode under the semaphore; skip when the judge is off
                    except Exception:
                        small = b""  # unreadable for the judge: it answers "error", as when its own decode failed
                    jfut = pool.submit(judge.judge, small, checked.mime, prov)  # runs while the detector scores
                det, version = store.score(checked.image)
            except Exception:
                log.exception("detector failed")
            t3 = time.perf_counter()
            checked.image = None  # free the decoded pixels while waiting for the judge
        finally:
            heavy.release()
        if det is None:
            if jfut is not None:
                jfut.cancel()  # a judge call that already started is paid for, so its budget stays spent
            raise HTTPException(503, "Detector model is not available.")
        verdict, source, jres, reason = det.band, "detector_only", None, None
        prob = det.score
        try:
            if jfut is None:
                raise JudgeUnavailable("budget")  # this IP has used its daily judge share
            out = jfut.result()
            verdict, source = out.verdict, "judge"
            prob = judge_probability(out.verdict, out.confidence)
            jres = JudgeResult(model=MODEL, confidence=out.confidence, cues=out.cues, explanation=out.explanation,
                               overrode_detector=out.verdict != "uncertain" and out.verdict != det.band)
            if jres.overrode_detector:
                log.info("judge override", extra={"event": "override", "verdict": verdict, "score": det.score,
                                                  "model_version": version, "ip_hash": ip_hash})
        except JudgeUnavailable as e:
            reason = e.reason
        t4 = time.perf_counter()
        total = ms(t4)
        if reason == "refusal":
            log.info("refusal", extra={"event": "refusal", "score": round(det.score, 4), "dims": [checked.width, checked.height],
                                       "format": checked.format, "latency_ms": total, "model_version": version,
                                       "ip_hash": ip_hash, "reason": reason, **src})
            raise HTTPException(422, "We can't analyse this image.")
        log.info("check", extra={"event": "check", "verdict": verdict, "source": source, "score": round(det.score, 4),
                                 "dims": [checked.width, checked.height], "format": checked.format,
                                 "latency_ms": total, "model_version": version, "ip_hash": ip_hash, "reason": reason, **src})
        return CheckResponse(
            verdict=verdict, source=source, probability_ai=prob, judge=jres, judge_unavailable_reason=reason, detector=det,
            provenance=prov, model_version=version, disclaimer=DISCLAIMER,
            timings_ms=Timings(validate=int((t1 - t0) * 1000), provenance=int((t2 - t1) * 1000),
                               detector=int((t3 - t2) * 1000), judge=int((t4 - t3) * 1000), total=total))

    @app.get("/v1/model")
    def model():
        try:
            m = store.get()
        except Exception:
            raise HTTPException(503, "Detector model is not available.") from None
        return {"version": m.version, "backbones": m.meta.backbones, "thresholds": m.meta.thresholds.model_dump(),
                "metrics": m.meta.metrics, "created_at": m.meta.created_at}

    @app.get("/v1/scorecard")
    def scorecard():
        try:  # written offline by ml/eval/scorecard.py next to the model it describes
            return json.loads((store.link / "scorecard.json").read_text())
        except (FileNotFoundError, NotADirectoryError):
            raise HTTPException(404, "No scorecard for the current model.") from None

    @app.get("/v1/health")
    def health():
        try:
            version = store.get().version
            status = "ok" if store.ready() else "loading"
        except Exception:
            version, status = None, "no_model"
        body = {"status": status, "model_version": version, "judge_enabled": judge.enabled}
        return body if status == "ok" else JSONResponse(body, 503)

    return app


app = create_app()

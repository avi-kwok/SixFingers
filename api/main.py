"""FastAPI app: POST /v1/check, GET /v1/model, GET /v1/health."""
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.detector import DetectorStore, FeatureEmbedder
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


class _JsonFmt(logging.Formatter):
    def format(self, r):
        d = {"time": self.formatTime(r, "%Y-%m-%dT%H:%M:%S"), "level": r.levelname, "msg": r.getMessage()}
        d.update({k: v for k, v in r.__dict__.items() if k in ("event", "verdict", "source", "score", "dims",
                                                            "format", "latency_ms", "model_version", "ip_hash", "reason")})
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
    app = FastAPI(title="SixFingers", docs_url=None, redoc_url=None)
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

    store = DetectorStore(embedder or FeatureEmbedder(), models_dir)
    client = _make_client() if anthropic_client == "auto" else anthropic_client
    limit = budget_limit if budget_limit is not None else int(os.environ.get("SIXFINGERS_JUDGE_BUDGET", 500))
    judge = Judge(judge_backend.from_env(client), Budget(budget_path or paths.data_dir() / "judge_budget.json", limit))
    limiter = RateLimiter(rate_limit)
    pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="judge")
    app.state.store, app.state.judge, app.state.limiter = store, judge, limiter

    def client_ip(req: Request) -> str:
        # behind Caddy the last X-Forwarded-For entry is the one the proxy added
        if os.environ.get("SIXFINGERS_TRUST_PROXY") == "1" and req.headers.get("x-forwarded-for"):
            return req.headers["x-forwarded-for"].split(",")[-1].strip()
        return req.client.host if req.client else "unknown"

    @app.post("/v1/check", response_model=CheckResponse, response_model_by_alias=True)
    def check(request: Request, image: UploadFile = File(...)):
        t0 = time.perf_counter()
        ms = lambda a: int((a - t0) * 1000)  # noqa: E731
        ip_hash = limiter.ip_hash(client_ip(request))
        wait = limiter.allow(ip_hash)
        if wait:
            raise HTTPException(429, "Too many checks. Try again later.", headers={"Retry-After": str(wait)})
        data = image.file.read(MAX_BYTES + 1)  # never buffer more than the cap
        try:
            checked = validate_upload(data)
        except UploadError as e:
            raise HTTPException(e.status, e.detail) from None
        t1 = time.perf_counter()
        prov = read_provenance(data, checked.mime, checked.image)
        t2 = time.perf_counter()
        jfut = pool.submit(judge.judge, data, checked.mime, prov)  # runs while the detector scores
        try:
            det, version = store.score(checked.image)
        except Exception:
            log.exception("detector failed")
            raise HTTPException(503, "Detector model is not available.") from None
        t3 = time.perf_counter()
        verdict, source, jres, reason = det.band, "detector_only", None, None
        try:
            out = jfut.result()
            verdict, source = out.verdict, "judge"
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
                                       "ip_hash": ip_hash, "reason": reason})
            raise HTTPException(422, "We can't analyse this image.")
        log.info("check", extra={"event": "check", "verdict": verdict, "source": source, "score": round(det.score, 4),
                                 "dims": [checked.width, checked.height], "format": checked.format,
                                 "latency_ms": total, "model_version": version, "ip_hash": ip_hash, "reason": reason})
        return CheckResponse(
            verdict=verdict, source=source, judge=jres, judge_unavailable_reason=reason, detector=det,
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

    @app.get("/v1/health")
    def health():
        try:
            return {"status": "ok", "model_version": store.get().version, "judge_enabled": judge.enabled}
        except Exception:
            return JSONResponse({"status": "no_model", "model_version": None, "judge_enabled": judge.enabled}, 503)

    return app


app = create_app()

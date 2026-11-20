"""FastAPI app: POST /v1/check, POST /v1/check-video, POST /v1/check-audio (only with SIXFINGERS_ENABLE_AUDIO=1),
POST /v1/feedback, GET /v1/model, GET /v1/scorecard, GET /v1/health."""
import json
import logging
import os
import sys
import tempfile
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
from api import audio as aud
from api import turnstile
from api import video as vid
from api.judge import MODEL, Budget, Judge, JudgeUnavailable
from api.limits import RateLimiter
from api.provenance import read_provenance
from api.usage import Usage
from api.validate import MAX_BYTES, UploadError, validate_upload
from ml.common import judge_backend, paths
from ml.common.api_models import (AudioCheckResponse, AudioInfo, AudioResult, AudioTimings, CheckResponse, FeedbackIn, FrameScore,
                                    JudgeResult, Timings, VideoCheckResponse, VideoCue, VideoDetectorResult,
                                    VideoJudgeResult, VideoTimings)
from ml.common.schemas import band_for

log = logging.getLogger("sixfingers")
PROD_ORIGIN = "https://sixfingers.avikwok.dev"
DISCLAIMER = ("This is an estimate, not proof. It only covers fully AI-generated still images; edited, "
              "inpainted or upscaled images are out of scope. Do not use it as the sole basis for decisions.")
VIDEO_COST = 3  # a video counts as three image checks against the rate limit
VIDEO_DISCLAIMER = ("This is an estimate, not proof. It covers fully AI-generated videos, judged from up to 20 "
                    "still frames (16 evenly spaced plus up to 4 at scene cuts); motion and audio are not analysed "
                    "yet. Do not use it as the sole basis for decisions.")
VIDEO_DISCLAIMER_AUDIO = ("This is an estimate, not proof. It covers fully AI-generated videos, judged from up to 20 "
                          "still frames (16 evenly spaced plus up to 4 at scene cuts); motion is not analysed, and the "
                          "audio score is a separate prototype. Do not use it as the sole basis for decisions.")
AUDIO_DISCLAIMER = ("This is an estimate, not proof. It is a prototype that only scores the first 4 seconds of "
                    "speech. Do not use it as the sole basis for decisions.")
BUSY_WAIT_S = 5  # how long a check waits for a free analysis slot before answering 503


_CAL = json.loads((Path(__file__).parent / "judge_calibration.json").read_text())["points"]


def judge_probability(verdict: str, confidence: float) -> float:
    """Calibrated P(AI) from the judge's verdict and confidence (map fitted by ml/judge_eval/calibrate.py)."""
    if verdict not in ("likely_ai", "likely_real"):
        return 0.5  # 'uncertain' means no lean either way; the map is fitted on decisive verdicts only
    raw = confidence if verdict == "likely_ai" else 1 - confidence
    xs, ys = zip(*_CAL)
    return float(np.interp(raw, xs, ys))


def overall_verdict(picture: str, voice: str | None) -> tuple[str, list[str]]:
    """Simple rule, not trained: either part says likely_ai -> likely_ai; both real -> likely_real; else uncertain.
    With no voice result the overall verdict is the picture verdict. Returns (verdict, parts that flagged it)."""
    flagged = [n for n, v in (("picture", picture), ("voice", voice)) if v == "likely_ai"]
    if flagged:
        return "likely_ai", flagged
    if picture == "likely_real" and voice in (None, "likely_real"):
        return "likely_real", []
    return "uncertain", []


class _JsonFmt(logging.Formatter):
    def format(self, r):
        d = {"time": self.formatTime(r, "%Y-%m-%dT%H:%M:%S"), "level": r.levelname, "msg": r.getMessage()}
        d.update({k: v for k, v in r.__dict__.items() if k in ("event", "verdict", "source", "score", "dims",
                                                            "format", "latency_ms", "model_version", "ip_hash", "reason", "input", "host",
                                                            "duration_s", "frames")})
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
               rate_limit: int = 10, audio_embedder=None, usage_path: Path | None = None) -> FastAPI:
    _setup_logging()
    audio_on = aud.enabled()
    # video frames get their own fp16 embedder only on MPS; on CPU fp16 gives no speedup, so the
    # image embedder is reused instead of loading a second fp32 copy. A test embedder serves both paths.
    from ml.features.backbones import pick_device
    video_emb = FeatureEmbedder(half=True) if embedder is None and pick_device() == "mps" else None
    store = DetectorStore(embedder or FeatureEmbedder(), models_dir, video_emb)

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
        if request.url.path == "/v1/check-video":
            cap, detail = vid.MAX_BYTES + 100_000, "Video is larger than 100 MB."
        elif audio_on and request.url.path == "/v1/check-audio":
            cap, detail = aud.MAX_BYTES + 100_000, "Audio is larger than 25 MB."
        else:
            cap, detail = MAX_BYTES + 100_000, "Image is larger than 10 MB."
        if int(request.headers.get("content-length") or 0) > cap:
            return JSONResponse({"detail": detail}, 413)
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
    usage = Usage(usage_path or paths.data_dir() / "usage.sqlite")
    feedback_limiter = RateLimiter(20)  # feedback posts per IP per hour; like checks, only a rotating hash is kept in memory
    app.state.store, app.state.judge, app.state.limiter, app.state.usage = store, judge, limiter, usage
    scorer = None
    if audio_on:
        try:  # a missing audio model fails only the audio checks (503), never images or video frames
            scorer = aud.AudioScorer(models_dir, audio_embedder)
        except Exception:
            log.exception("audio model not loadable")

    def client_ip(req: Request) -> str:
        # behind Caddy the last X-Forwarded-For entry is the one the proxy added
        if os.environ.get("SIXFINGERS_TRUST_PROXY") == "1" and req.headers.get("x-forwarded-for"):
            return req.headers["x-forwarded-for"].split(",")[-1].strip()
        return req.client.host if req.client else "unknown"

    def human_ok(req: Request) -> bool:
        """Turnstile gate for the judge: always True when the feature is off or the judge is off (nothing to protect)."""
        if not turnstile.enabled() or not judge.enabled:
            return True
        return turnstile.verify(req.headers.get(turnstile.HEADER))

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
        human = human_ok(request)
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
                if human and not (judge.enabled and judge_quota.allow(ip_hash)):  # allow() returns 0 when under quota
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
            if jfut is None:  # no token, or this IP has used its daily judge share
                raise JudgeUnavailable("budget" if human else "verification")
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
        check_id = usage.record_check(src["input"], verdict, prob, version, source == "judge")
        return CheckResponse(
            verdict=verdict, source=source, probability_ai=prob, judge=jres, judge_unavailable_reason=reason, detector=det,
            provenance=prov, model_version=version, disclaimer=DISCLAIMER, check_id=check_id,
            timings_ms=Timings(validate=int((t1 - t0) * 1000), provenance=int((t2 - t1) * 1000),
                               detector=int((t3 - t2) * 1000), judge=int((t4 - t3) * 1000), total=total))

    @app.post("/v1/feedback")
    def feedback(request: Request, body: FeedbackIn):
        wait = feedback_limiter.allow(feedback_limiter.ip_hash(client_ip(request)))
        if wait:
            raise HTTPException(429, "Too much feedback. Try again later.", headers={"Retry-After": str(wait)})
        try:
            status = usage.add_feedback(body.check_id, body.answer, body.actual, body.comment)
        except Exception:
            log.exception("feedback write failed")
            raise HTTPException(503, "Feedback could not be saved. Try again later.") from None
        if status == "unknown":
            raise HTTPException(404, "This result has expired or does not exist.")
        if status == "duplicate":
            raise HTTPException(409, "Feedback for this result was already sent.")
        return {"status": "ok"}

    @app.post("/v1/check-video", response_model=VideoCheckResponse, response_model_by_alias=True)
    def check_video(request: Request, video: UploadFile | None = File(None)):
        t0 = time.perf_counter()
        ms = lambda a: int((a - t0) * 1000)  # noqa: E731
        if video is None:  # File(None) keeps a missing field our 400, not FastAPI's 422 (which means refusal here)
            raise HTTPException(400, "Send a video file.")
        ip_hash = limiter.ip_hash(client_ip(request))
        wait = limiter.allow(ip_hash, cost=VIDEO_COST)
        if wait:
            raise HTTPException(429, "Too many checks. Try again later.", headers={"Retry-After": str(wait)})
        human = human_ok(request)
        if not heavy.acquire(timeout=BUSY_WAIT_S):
            raise HTTPException(503, "SixFingers is busy right now. Try again in a minute.")
        try:
            try:
                with tempfile.TemporaryDirectory(prefix="sixfingers-video-") as tmp:  # removed on every exit path
                    clip = vid.load(video.file, Path(tmp))
                    t1 = time.perf_counter()
                    # the audio track decodes while frames are sampled; it must finish before the temp dir goes
                    afut = pool.submit(aud.read, clip.path, clip.demuxer) if scorer and clip.has_audio else None
                    frames = vid.extract_frames(clip, Path(tmp))
                    samples = None
                    if afut and afut.exception() is None:  # an unreadable track only loses the audio score
                        samples = afut.result()
            except vid.UploadError as e:
                raise HTTPException(e.status, e.detail) from None
            grid = vid.frame_grid(frames)  # in memory; frames are already decoded
            t2 = time.perf_counter()
            try:
                store.get()  # no model: fail before the judge spends budget
            except Exception:
                log.exception("detector failed")
                raise HTTPException(503, "Detector model is not available.") from None
            jfut = None
            if human and not (judge.enabled and judge_quota.allow(ip_hash)):  # allow() returns 0 when under quota
                jfut = pool.submit(judge.judge_video, grid, len(frames), clip.duration)  # runs while the detector scores
            sfut = pool.submit(scorer.score, samples) if samples is not None else None  # runs while frames are scored
            try:
                dets, version = store.score_many([im for _, im in frames], video=True)
            except Exception:
                log.exception("detector failed")
                raise HTTPException(503, "Detector model is not available.") from None
            audio = AudioInfo()  # audio checks off: unchanged
            if scorer:
                audio = AudioInfo(note="This video has no audio track." if not clip.has_audio
                                  else "Could not read the audio track.")
                if sfut:
                    try:
                        audio = sfut.result()
                    except Exception:
                        log.exception("audio scoring failed")
                        audio = AudioInfo(note="The audio check failed.")
        finally:
            heavy.release()
        t3 = time.perf_counter()
        th = dets[0].thresholds
        score = vid.top_k_mean([d.score for d in dets])
        det = VideoDetectorResult(
            score=score, band=band_for(score, th), thresholds=th,
            frames=[FrameScore(index=i, time_s=t, score=d.score, band=d.band)
                    for i, ((t, _), d) in enumerate(zip(frames, dets), 1)])
        verdict, source, jres, reason = det.band, "detector_only", None, None
        try:
            if jfut is None:  # no token, or this IP has used its daily judge share
                raise JudgeUnavailable("budget" if human else "verification")
            out = jfut.result()
            verdict, source = out.verdict, "judge"
            cues = [VideoCue(frame=c.frame, time_s=frames[c.frame - 1][0], region=c.region, cue=c.cue)
                    for c in out.cues if 1 <= c.frame <= len(frames)]  # cues citing a missing frame are dropped
            jres = VideoJudgeResult(model=MODEL, confidence=out.confidence, cues=cues, explanation=out.explanation,
                                    overrode_detector=out.verdict != "uncertain" and out.verdict != det.band)
            if jres.overrode_detector:
                log.info("judge override", extra={"event": "override", "verdict": verdict, "score": det.score,
                                                  "model_version": version, "ip_hash": ip_hash})
        except JudgeUnavailable as e:
            reason = e.reason
        t4 = time.perf_counter()
        total = ms(t4)
        src_log = {"dims": [clip.width, clip.height], "format": clip.format, "latency_ms": total,
                   "model_version": version, "ip_hash": ip_hash, "reason": reason, "input": "video",
                   "duration_s": round(clip.duration, 2), "frames": len(frames)}
        if reason == "refusal":
            log.info("refusal", extra={"event": "refusal", "score": round(det.score, 4), **src_log})
            raise HTTPException(422, "We can't analyse this video.")
        log.info("check", extra={"event": "check", "verdict": verdict, "source": source,
                                 "score": round(det.score, 4), **src_log})
        overall, flagged_by = overall_verdict(verdict, audio.band if isinstance(audio, AudioResult) else None)
        check_id = usage.record_check("video", overall, det.score, version, source == "judge")
        return VideoCheckResponse(
            verdict=verdict, overall_verdict=overall, overall_flagged_by=flagged_by,
            source=source, judge=jres, judge_unavailable_reason=reason, detector=det, check_id=check_id,
            audio=audio, duration_s=round(clip.duration, 3), model_version=version,
            disclaimer=VIDEO_DISCLAIMER_AUDIO if scorer else VIDEO_DISCLAIMER,
            timings_ms=VideoTimings(validate=int((t1 - t0) * 1000), frames=int((t2 - t1) * 1000),
                                    detector=int((t3 - t2) * 1000), judge=int((t4 - t3) * 1000), total=total))

    if audio_on:
        @app.post("/v1/check-audio", response_model=AudioCheckResponse, response_model_by_alias=True)
        def check_audio(request: Request, audio: UploadFile | None = File(None)):
            t0 = time.perf_counter()
            if audio is None:
                raise HTTPException(400, "Send an audio file.")
            if scorer is None:
                raise HTTPException(503, "Audio checks are not available right now.")
            ip_hash = limiter.ip_hash(client_ip(request))
            wait = limiter.allow(ip_hash)
            if wait:
                raise HTTPException(429, "Too many checks. Try again later.", headers={"Retry-After": str(wait)})
            if not heavy.acquire(timeout=BUSY_WAIT_S):
                raise HTTPException(503, "SixFingers is busy right now. Try again in a minute.")
            try:
                try:
                    with tempfile.TemporaryDirectory(prefix="sixfingers-audio-") as tmp:  # removed on every exit path
                        samples, fmt = aud.load(audio.file, Path(tmp))
                except aud.UploadError as e:
                    raise HTTPException(e.status, e.detail) from None
                t1 = time.perf_counter()
                try:
                    res = scorer.score(samples)
                except Exception:
                    log.exception("audio scoring failed")
                    raise HTTPException(503, "Audio model is not available.") from None
            finally:
                heavy.release()
            t2 = time.perf_counter()
            duration = round(len(samples) / aud.features.SR, 3)
            total = int((t2 - t0) * 1000)
            log.info("check", extra={"event": "check", "verdict": res.band, "source": "detector_only",
                                     "score": round(res.p_fake, 4), "format": fmt, "latency_ms": total,
                                     "model_version": res.model_version, "ip_hash": ip_hash, "input": "audio",
                                     "duration_s": duration})
            check_id = usage.record_check("audio", res.band, res.p_fake, res.model_version, False)
            return AudioCheckResponse(check_id=check_id, verdict=res.band, audio=res, duration_s=duration, disclaimer=AUDIO_DISCLAIMER,
                                      timings_ms=AudioTimings(validate=int((t1 - t0) * 1000),
                                                              detector=int((t2 - t1) * 1000), total=total))

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
            card = json.loads((store.link / "scorecard.json").read_text())
        except (FileNotFoundError, NotADirectoryError):
            raise HTTPException(404, "No scorecard for the current model.") from None
        return {**card, "usage": usage.summary()}  # read live on every request, never stored in the file

    @app.get("/v1/health")
    def health():
        extra = {"audio_enabled": scorer is not None} if audio_on else {}  # the web page shows the audio tab from this
        try:
            version = store.get().version
            status = "ok" if store.ready() else "loading"
        except Exception:
            version, status = None, "no_model"
        body = {"status": status, "model_version": version, "judge_enabled": judge.enabled, **extra}
        return body if status == "ok" else JSONResponse(body, 503)

    return app


app = create_app()

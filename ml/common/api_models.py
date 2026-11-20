"""Pydantic models for the HTTP API. /v1/check response shape is frozen. Video* models serve /v1/check-video,
Audio* models /v1/check-audio (and the audio part of a video result when audio checks are on)."""
from __future__ import annotations

import math
import unicodedata
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Verdict = Literal["likely_ai", "uncertain", "likely_real"]


class Cue(BaseModel):
    region: str
    cue: str
    box: list[float] | None = Field(None, description=(
        "Tight box around the cue as [x0, y0, x1, y1], fractions 0-1 of image width and height from the top-left; "
        "null for whole-image cues"))

    @field_validator("box", mode="before")
    @classmethod
    def _clean_box(cls, v):
        """Clamp to [0, 1]; a malformed or empty box becomes None so the cue itself is kept."""
        if not isinstance(v, (list, tuple)) or len(v) != 4:
            return None
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v):
            return None
        x0, y0, x1, y1 = (min(max(float(x), 0.0), 1.0) for x in v)
        return [x0, y0, x1, y1] if x0 < x1 and y0 < y1 else None


class JudgeResult(BaseModel):
    model: str
    confidence: float = Field(ge=0, le=1)
    cues: list[Cue]
    explanation: str
    overrode_detector: bool


class JudgeOutput(BaseModel):
    """What Claude is asked to return (JSON schema for output_config.format)."""
    verdict: Verdict
    confidence: float = Field(ge=0, le=1)
    cues: list[Cue]
    explanation: str
    overrode_detector: bool


class Thresholds(BaseModel):
    t_ai: float
    t_real: float


class DetectorResult(BaseModel):
    score: float
    band: Verdict
    backbones: dict[str, float]
    thresholds: Thresholds


class C2PAInfo(BaseModel):
    present: bool = False
    valid: bool | None = None  # signature validated by the c2pa library
    ai_generated: bool | None = None
    issuer: str | None = None


class Provenance(BaseModel):
    c2pa: C2PAInfo = Field(default_factory=C2PAInfo)
    exif_hints: list[str] = Field(default_factory=list)


class Timings(BaseModel):
    validate_: int = Field(0, alias="validate")
    provenance: int = 0
    detector: int = 0
    judge: int = 0
    total: int = 0
    model_config = {"populate_by_name": True}


class CheckResponse(BaseModel):
    verdict: Verdict
    source: Literal["judge", "detector_only"]
    probability_ai: float = Field(ge=0, le=1)  # calibrated P(AI): judge-based when source=judge, else the detector score
    judge: JudgeResult | None = None
    judge_unavailable_reason: Literal["budget", "timeout", "error", "refusal", "disabled", "verification"] | None = None
    detector: DetectorResult
    provenance: Provenance
    model_version: str
    timings_ms: Timings
    disclaimer: str
    check_id: str | None = None  # random id for POST /v1/feedback; None when usage storage failed


class FrameScore(BaseModel):
    index: int  # 1-based, as labelled in the frame grid
    time_s: float
    score: float
    band: Verdict


class VideoDetectorResult(BaseModel):
    score: float  # mean of the 4 highest frame scores
    band: Verdict
    frames: list[FrameScore]
    thresholds: Thresholds


class FrameCue(BaseModel):
    frame: int
    region: str
    cue: str


class VideoJudgeOutput(BaseModel):
    """What Claude is asked to return for a video (JSON schema for output_config.format)."""
    verdict: Verdict
    confidence: float = Field(ge=0, le=1)
    cues: list[FrameCue]
    explanation: str
    overrode_detector: bool


class VideoCue(BaseModel):
    frame: int
    time_s: float  # looked up by the server from the frame number, never taken from the model
    region: str
    cue: str


class VideoJudgeResult(BaseModel):
    model: str
    confidence: float = Field(ge=0, le=1)
    cues: list[VideoCue]
    explanation: str
    overrode_detector: bool


class AudioInfo(BaseModel):
    analysed: bool = False
    note: str = "Audio is not analysed yet."


class AudioResult(BaseModel):
    analysed: Literal[True] = True
    p_fake: float = Field(ge=0, le=1)  # probability the speech is synthetic, from the audio probe
    band: Verdict
    threshold: float  # p_fake at or above this is likely_ai
    model_version: str
    note: str


class AudioTimings(BaseModel):
    validate_: int = Field(0, alias="validate")
    detector: int = 0
    total: int = 0
    model_config = {"populate_by_name": True}


class AudioCheckResponse(BaseModel):
    verdict: Verdict
    source: Literal["detector_only"] = "detector_only"  # no judge: Claude cannot hear audio
    audio: AudioResult
    duration_s: float
    timings_ms: AudioTimings
    disclaimer: str
    check_id: str | None = None  # random id for POST /v1/feedback; None when usage storage failed


class VideoTimings(BaseModel):
    validate_: int = Field(0, alias="validate")
    frames: int = 0
    detector: int = 0
    judge: int = 0
    total: int = 0
    model_config = {"populate_by_name": True}


class VideoCheckResponse(BaseModel):
    verdict: Verdict  # the picture verdict (frames, judge or detector); the voice result is in `audio`
    overall_verdict: Verdict  # picture and voice combined by a simple rule; equals `verdict` when there is no voice result
    overall_flagged_by: list[Literal["picture", "voice"]] = Field(default_factory=list)  # parts that said likely_ai
    source: Literal["judge", "detector_only"]
    judge: VideoJudgeResult | None = None
    judge_unavailable_reason: Literal["budget", "timeout", "error", "refusal", "disabled", "verification"] | None = None
    detector: VideoDetectorResult
    audio: AudioResult | AudioInfo = Field(default_factory=AudioInfo)
    duration_s: float
    model_version: str
    timings_ms: VideoTimings
    disclaimer: str
    check_id: str | None = None  # random id for POST /v1/feedback; None when usage storage failed


class FeedbackIn(BaseModel):
    """POST /v1/feedback body. Unknown fields are rejected."""
    model_config = {"extra": "forbid"}
    check_id: str = Field(pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
    answer: Literal["right", "wrong", "unsure"]
    actual: Literal["ai", "real", "unknown"] | None = None
    comment: str | None = Field(None, max_length=500)

    @field_validator("comment")
    @classmethod
    def _strip_controls(cls, v):
        """Drop control characters (newlines are kept); a blank comment becomes None."""
        if v is None:
            return None
        v = "".join(ch for ch in v if ch == "\n" or unicodedata.category(ch) != "Cc").strip()
        return v or None

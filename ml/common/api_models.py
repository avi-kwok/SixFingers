"""Pydantic models for the HTTP API. /v1/check response shape is frozen."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["likely_ai", "uncertain", "likely_real"]


class Cue(BaseModel):
    region: str
    cue: str


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
    judge_unavailable_reason: Literal["budget", "timeout", "error", "refusal", "disabled"] | None = None
    detector: DetectorResult
    provenance: Provenance
    model_version: str
    timings_ms: Timings
    disclaimer: str

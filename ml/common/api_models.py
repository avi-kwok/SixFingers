"""Pydantic models for the HTTP API. /v1/check response shape is frozen."""
from __future__ import annotations

import math
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
    judge_unavailable_reason: Literal["budget", "timeout", "error", "refusal", "disabled"] | None = None
    detector: DetectorResult
    provenance: Provenance
    model_version: str
    timings_ms: Timings
    disclaimer: str

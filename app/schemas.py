"""Pydantic models for what the model is allowed to say. Anything else -> InvalidModelOutput."""
import math
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt, field_validator

Status = Literal["problem_detected", "no_obvious_problem", "insufficient_visual_evidence", "unsupported_image"]
Level = Literal["low", "medium", "high"]
Verdict = Literal["holds", "weakened", "likely_wrong"]
Num = Union[StrictInt, StrictFloat]     # rejects bool and numeric strings


def _lower(v: Any) -> Any:
    return v.strip().lower() if isinstance(v, str) else v


def _strings(v: Any) -> list[str]:
    return [s for s in v if isinstance(s, str)] if isinstance(v, list) else []


def _confidence(v: Num) -> float:
    v = float(v)
    if not math.isfinite(v) or not 0.0 <= v <= 1.0:
        raise ValueError("confidence must be a finite number in [0, 1]")
    return v


class Pass1(BaseModel):
    model_config = ConfigDict(extra="ignore")
    object: str = "unknown"
    observations: list[str] = []
    evidence_sufficient: bool
    status: Status
    problem: str = ""
    confidence: Num
    severity: Level = "low"
    risk: Level = "low"
    bbox_2d: Any = None                      # a bad box only drops the location; see pipeline.to_location
    possible_causes: list[str] = []
    recommended_action: list[str] = []
    professional_help_needed: bool = False

    _norm = field_validator("status", "severity", "risk", mode="before")(_lower)
    _lists = field_validator("observations", "possible_causes", "recommended_action", mode="before")(_strings)
    _conf = field_validator("confidence")(_confidence)

    @field_validator("object", "problem", mode="before")
    @classmethod
    def _str_or_empty(cls, v):
        return v if isinstance(v, str) else ""


class ReviewOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    verdict: Verdict
    revised_confidence: Num
    strongest_counterargument: str = ""
    alternative_diagnosis: str = ""

    _norm = field_validator("verdict", mode="before")(_lower)
    _conf = field_validator("revised_confidence")(_confidence)

    @field_validator("strongest_counterargument", "alternative_diagnosis", mode="before")
    @classmethod
    def _str_or_empty(cls, v):
        return v if isinstance(v, str) else ""

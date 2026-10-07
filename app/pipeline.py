import math
import re
import time

from pydantic import ValidationError

from . import config, images, logs, parsing, prompts, safety, vlm
from .errors import InvalidModelOutput, RepairError
from .schemas import Pass1, ReviewOut

DETECTED, NO_PROBLEM = "problem_detected", "no_obvious_problem"
INSUFFICIENT, UNSUPPORTED = "insufficient_visual_evidence", "unsupported_image"   # also the legacy `problem` values
NO_PROBLEM_TEXT = "No obvious problem visible in this image"
NO_PROBLEM_NOTE = "This image alone cannot confirm the object is in good condition."
UNSUPPORTED_ACTION = "Upload a clear photo of the physical object you want checked."
REPAIR_WARNING = "Do not attempt a repair based solely on this image."
RETAKE = ["Retake the photo: closer, in good light, in focus, with the suspected area clearly in frame.",
          "Add a second angle or a close-up of the part you are worried about."]

MAX_STR = 300
CONF_CAP, CONF_NO_PROBLEM_CAP, CONF_INSUFFICIENT_CAP, CONF_MIN_DETECTED = 0.95, 0.9, 0.2, 0.25
# Words meaning "this sentence is an inference", not an observation.
INFERENCE_RE = re.compile(
    r"\b(because|therefore|probably|likely|may|might|could|suggests?|indicat(?:es|ing)|due to|caused|"
    r"result(?:s|ed|ing)? (?:in|from)|means)\b", re.I)
# Model output that looks like it is following injected instructions.
INSTRUCTION_RE = re.compile(
    r"ignore (?:all |any |the |your )?(?:previous|prior|above|earlier)?\s*instructions|system (?:prompt|message)|"
    r"developer (?:message|prompt)|you are now|disregard .{0,30}instructions|reveal .{0,20}prompt", re.I)
CERTAINTY_RE = re.compile(r"\b(definitely|certainly|undoubtedly|guaranteed|without (?:a )?doubt)\b|100 ?%", re.I)


def _text(s, limit: int = MAX_STR) -> str:
    s = " ".join(str(s).split())
    if not s or INSTRUCTION_RE.search(s):
        return ""
    return s[:limit].rstrip()


def _clean_list(xs, n: int) -> list[str]:
    out, seen = [], set()
    for s in xs:
        s = _text(s)
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out[:n]


def to_location(raw, w: int, h: int):
    """Model box -> {x, y, width, height}, normalised 0-1, top-left origin. None if unusable (never invented)."""
    if not isinstance(raw, list) or len(raw) != 4 or any(isinstance(v, bool) for v in raw):
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) and v >= 0 for v in (x1, y1, x2, y2)) or not any((x1, y1, x2, y2)):
        return None
    if max(x1, y1, x2, y2) <= 1.0 or config.BBOX_MODE == "normalized":
        pass
    elif config.BBOX_MODE == "pixel":
        x1, x2, y1, y2 = x1 / w, x2 / w, y1 / h, y2 / h
    else:
        x1, y1, x2, y2 = x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000
    if max(x1, y1, x2, y2) > 1.02:                       # tolerate 2% overshoot, reject anything wilder
        return None
    x1, y1, x2, y2 = (min(v, 1.0) for v in (x1, y1, x2, y2))
    if x2 <= x1 or y2 <= y1:                             # inverted / empty
        return None
    bw, bh = x2 - x1, y2 - y1
    if bw < 0.02 or bh < 0.02 or (bw > 0.97 and bh > 0.97):   # speck or "whole image" is not a localisation
        return None
    return {"x": round(x1, 3), "y": round(y1, 3), "width": round(bw, 3), "height": round(bh, 3)}


def _pass1(img, q: str, w: int, h: int) -> Pass1:
    raw = vlm.generate(prompts.system(), prompts.user(q, w, h), img, config.MAX_NEW_TOKENS)
    try:
        return Pass1.model_validate(parsing.extract_json(raw))
    except ValidationError as e:
        bad = sorted({str(err["loc"][0]) for err in e.errors() if err["loc"]})
        raise InvalidModelOutput(detail=f"schema validation failed on fields {bad}") from e


def _review(img, obj, problem, observations, causes, conf):
    try:
        raw = vlm.generate(prompts.review_system(),
                           prompts.review_user(obj, problem, observations, causes, round(conf, 2)),
                           img, config.REVIEW_MAX_NEW_TOKENS)
        return ReviewOut.model_validate(parsing.extract_json(raw))
    except (RepairError, ValidationError) as e:
        logs.event("review_failed", level=30, error=type(e).__name__)
        return None


def analyze(raw: bytes, question: str | None = None, review: bool | None = None) -> dict:
    t0 = time.time()
    do_review = config.SECOND_PASS if review is None else review
    img, (w, h), _orig = images.prepare(raw)
    q = prompts.clean_question(question)

    p1 = _pass1(img, q, w, h)                       # ---- pass 1: investigator

    obj = _text(p1.object, 80) or "unknown"
    causes = _clean_list(p1.possible_causes, 5)
    observations = []
    for o in _clean_list(p1.observations, 6):       # evidence discipline, enforced in code
        if INFERENCE_RE.search(o):
            if o not in causes and len(causes) < 5:
                causes.append(o)
        else:
            observations.append(o)
    problem = _text(p1.problem, 200)
    conf = min(p1.confidence, CONF_CAP)
    severity, risk = p1.severity, p1.risk
    actions = _clean_list(p1.recommended_action, 4)
    location = to_location(p1.bbox_2d, w, h)

    # ---- semantic validation: downgrade whenever the claimed state is not backed by the fields
    status = p1.status
    if status == DETECTED and (not p1.evidence_sufficient or not problem or not observations or conf < CONF_MIN_DETECTED):
        status = INSUFFICIENT
    elif status == NO_PROBLEM and (not p1.evidence_sufficient or not observations):
        status = INSUFFICIENT

    review_out = None
    if status == DETECTED:
        if len(observations) < 2:
            conf = min(conf, 0.6)                   # one observation is thin evidence
        if CERTAINTY_RE.search(problem):
            conf = min(conf, 0.7)                   # absolute language is not supported by a single photo
        if do_review:                               # ---- pass 2: skeptical reviewer (only for diagnoses)
            r = _review(img, obj, problem, observations, causes, conf)
            if r:
                initial, rc = conf, r.revised_confidence
                if r.verdict == "holds":
                    conf = (conf + rc) / 2
                elif r.verdict == "weakened":
                    conf = min(conf, rc)
                else:
                    conf = min(conf, rc, 0.4)
                review_out = {"verdict": r.verdict, "initial_confidence": round(initial, 2),
                              "strongest_counterargument": _text(r.strongest_counterargument),
                              "alternative_diagnosis": _text(r.alternative_diagnosis)}
                if r.verdict == "likely_wrong" and conf < CONF_MIN_DETECTED:
                    status = INSUFFICIENT           # evidence no longer adequate
        conf = min(conf, CONF_CAP)

    if status == DETECTED:
        pass
    elif status == NO_PROBLEM:
        problem, conf, severity, location, causes = NO_PROBLEM_TEXT, min(conf, CONF_NO_PROBLEM_CAP), "low", None, []
        actions = actions[:3] + [NO_PROBLEM_NOTE]
    elif status == UNSUPPORTED:
        problem, conf, severity, location, causes, actions = UNSUPPORTED, 0.0, "low", None, [], [UNSUPPORTED_ACTION]
    else:
        status, problem, conf, severity, location, causes = INSUFFICIENT, INSUFFICIENT, min(conf, CONF_INSUFFICIENT_CAP), "low", None, []
        actions = list(RETAKE)

    # ---- deterministic safety layer: the model has no authority over this
    blob = " ".join([q, obj, problem, *observations, *causes, *actions,
                     (review_out or {}).get("strongest_counterargument", "")])
    s = safety.apply(blob, risk, actions, p1.professional_help_needed)
    final_actions = s["actions"]
    if s["risk"] == "high" and (status == INSUFFICIENT or conf < 0.5) and REPAIR_WARNING not in final_actions:
        final_actions = final_actions[:4] + [REPAIR_WARNING]
    if s["risk"] == "high" and safety.ORDER[severity] < 1 and status == DETECTED:
        severity = "medium"

    return {
        "status": status,
        "object": obj,
        "problem": problem,
        "confidence": round(conf, 2),
        "severity": severity,
        "location": location,
        "observations": observations,
        "possible_causes": causes,
        "recommended_action": final_actions,
        "risk": s["risk"],
        "safety_warning": s["warning"],
        "professional_help_needed": s["professional"],
        "review": review_out,
        "meta": {"model": config.MODEL, "prompt_version": config.PROMPT_VERSION,
                 "second_pass": review_out is not None, "safety_flags": s["flags"],
                 "image_size": [w, h], "seconds": round(time.time() - t0, 1)},
    }

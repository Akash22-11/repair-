"""Pull a JSON object out of raw model text. Deterministic only: no extra model calls."""
import json
import re

from .errors import InvalidModelOutput

_FENCE = re.compile(r"```(?:json)?", re.I)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_dec = json.JSONDecoder()   # accepts NaN/Infinity on purpose: field validators reject them individually


def _first_object(text: str):
    i = text.find("{")
    while i != -1:
        try:
            obj, _ = _dec.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        i = text.find("{", i + 1)
    return None


def extract_json(text: str) -> dict:
    cleaned = _FENCE.sub("", text or "")
    obj = _first_object(cleaned) or _first_object(_TRAILING_COMMA.sub(r"\1", cleaned))   # one local repair attempt
    if obj is None:
        raise InvalidModelOutput(detail="no JSON object found in model output")
    return obj

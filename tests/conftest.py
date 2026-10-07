import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import vlm
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def jpeg(size=(1200, 800), color="gray") -> bytes:
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, "JPEG")
    return b.getvalue()


def post(client, raw=None, **data):
    return client.post("/analyze", files={"image": ("a.jpg", raw if raw is not None else jpeg(), "image/jpeg")}, data=data)


P1 = {"object": "bicycle", "observations": ["Rear derailleur hangs at an angle", "Chain sits on the smallest cog",
                                            "The hanger is probably bent"],
      "evidence_sufficient": True, "status": "problem_detected", "problem": "Rear derailleur appears misaligned",
      "confidence": 0.9, "severity": "medium", "risk": "low", "bbox_2d": [610, 430, 790, 650],
      "possible_causes": ["Bent derailleur hanger"], "recommended_action": ["Check hanger alignment"],
      "professional_help_needed": False}
P2 = {"strongest_counterargument": "Cable tension could explain it.", "alternative_diagnosis": "Loose cable",
      "verdict": "weakened", "revised_confidence": 0.6}


class Stub:
    """Replaces vlm.generate. Routes by system prompt; values may be dicts, raw strings or exceptions."""

    def __init__(self, p1, p2=None):
        self.p1, self.p2, self.calls = p1, p2, []

    def __call__(self, system, user, image, max_new_tokens):
        self.calls.append({"system": system, "user": user, "max_new_tokens": max_new_tokens})
        v = self.p2 if "skeptical reviewer" in system else self.p1
        if isinstance(v, Exception):
            raise v
        return v if isinstance(v, str) else json.dumps(v)


@pytest.fixture
def stub(monkeypatch):
    def install(p1, p2=None):
        s = Stub(p1, p2)
        monkeypatch.setattr(vlm, "generate", s)
        return s
    return install

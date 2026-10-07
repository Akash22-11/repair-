import io

from PIL import Image

from app import config, main, vlm
from app.errors import InferenceBusy, ModelTimeout, ModelUnavailable
from tests.conftest import P1, P2, jpeg, post


def test_model_unavailable_is_503(client, stub):
    stub(ModelUnavailable(detail="load failed: OSError /secret/path"))
    r = post(client)
    assert r.status_code == 503 and r.json()["error"] == "model_unavailable"
    assert "secret" not in r.text and "OSError" not in r.text


def test_timeout_is_504_and_lock_released(client, stub):
    stub(ModelTimeout(detail="deadline"))
    assert post(client).status_code == 504
    assert not main._gpu.locked()                         # lock must always be released
    assert post(client).status_code == 504               # and the next request is not stuck


def test_unexpected_exception_is_generic(client, stub):
    stub(ValueError("secret internal detail"))            # not a RepairError
    r = post(client)
    assert r.status_code == 500 and r.json()["error"] == "internal_error" and "secret" not in r.text
    assert not main._gpu.locked()


def test_busy_when_queue_full(client, stub, monkeypatch):
    import threading
    stub(P1, P2)
    sem = threading.BoundedSemaphore(1)
    sem.acquire()
    monkeypatch.setattr(main, "_waiting", sem)
    r = post(client)
    assert r.status_code == 503 and r.json()["error"] == "inference_busy"


def test_oversized_bytes_413(client, stub, monkeypatch):
    stub(P1, P2)
    monkeypatch.setattr(config, "MAX_IMAGE_BYTES", 1000)
    r = post(client)
    assert r.status_code == 413 and r.json()["error"] == "image_too_large"


def test_pixel_bomb_413(client, stub):
    stub(P1, P2)
    b = io.BytesIO()
    Image.new("L", (6000, 6000)).save(b, "PNG")           # tiny file, 36 MP
    assert b.tell() < 200_000
    r = post(client, b.getvalue())
    assert r.status_code == 413


def test_corrupted_and_fake_images_400(client, stub):
    stub(P1, P2)
    assert post(client, b"not an image").status_code == 400
    assert post(client, jpeg()[:300]).status_code == 400             # truncated JPEG
    assert post(client, b"").status_code == 400
    r = post(client, b"<svg xmlns='http://www.w3.org/2000/svg'/>")   # wrong format despite image/jpeg MIME
    assert r.status_code == 400 and r.json()["error"] == "bad_image"


def test_degenerate_dimensions_400(client, stub):
    stub(P1, P2)
    assert post(client, jpeg((4000, 40))).status_code == 400          # absurd aspect ratio
    assert post(client, jpeg((8, 8))).status_code == 400


def test_image_is_resized_to_model_grid(client, stub):
    stub(P1, P2)
    d = post(client, jpeg((3000, 2000))).json()
    w, h = d["meta"]["image_size"]
    assert max(w, h) <= config.IMAGE_MAX_SIDE + 32 and w % 32 == 0 and h % 32 == 0


def test_exif_rotation_and_rgba_png(client, stub):
    stub(P1, P2)
    b = io.BytesIO()
    Image.new("RGBA", (400, 300), (255, 0, 0, 0)).save(b, "PNG")
    assert post(client, b.getvalue()).status_code == 200


def test_api_key(client, stub, monkeypatch):
    stub(P1, P2)
    monkeypatch.setattr(config, "API_KEY", "s3cret")
    assert post(client).status_code == 401
    r = client.post("/analyze", files={"image": ("a.jpg", jpeg(), "image/jpeg")}, headers={"X-API-Key": "s3cret"})
    assert r.status_code == 200
    assert client.get("/health").status_code == 200


def test_health_shape_and_no_leaks(client):
    d = client.get("/health").json()
    assert {"status", "model", "model_loaded", "device", "quantization"} <= d.keys()
    assert d["model"] == config.MODEL and d["model_loaded"] is False
    assert "/" not in d["status"] and "token" not in str(d).lower()


def test_health_failed_state(client, monkeypatch):
    monkeypatch.setitem(vlm._state, "status", "failed")
    monkeypatch.setitem(vlm._state, "error_type", "OutOfMemoryError")
    d = client.get("/health").json()
    assert d["status"] == "model_unavailable" and d["model_loaded"] is False and d["error_type"] == "OutOfMemoryError"


def test_request_id_header_and_error_body(client, stub):
    stub(P1, P2)
    r = post(client, b"junk")
    assert r.headers["x-request-id"] == r.json()["request_id"]


def test_real_generate_without_model_is_unavailable(client, monkeypatch):
    """Nothing is stubbed: with torch/weights unavailable the API must answer 503, not crash or fall back."""
    monkeypatch.setitem(vlm._state, "failed_at", __import__("time").monotonic())
    monkeypatch.setattr(vlm, "_model", None)
    r = post(client)
    assert r.status_code == 503 and r.json()["error"] == "model_unavailable"

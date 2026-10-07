import hmac
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import config, logs, pipeline, vlm
from .errors import EmptyImage, ImageTooLarge, InferenceBusy, RepairError


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if config.PRELOAD_MODEL:
        threading.Thread(target=vlm.preload, name="model-preload", daemon=True).start()
    yield


app = FastAPI(title="REPAIR", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["*"], expose_headers=["X-Request-ID"])

_gpu = threading.Lock()                                   # one inference at a time on a 6 GB card
_waiting = threading.BoundedSemaphore(1 + config.MAX_QUEUE)   # 1 running + MAX_QUEUE waiting, nothing unbounded


@app.middleware("http")
async def _request_id(request: Request, call_next):
    rid = uuid.uuid4().hex[:12]
    request.state.request_id = rid
    logs.request_id_var.set(rid)
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


def _err(status: int, code: str, message: str, rid: str):
    return JSONResponse(status_code=status, content={"error": code, "message": message, "request_id": rid})


@app.get("/health")
def health():
    return vlm.health()


@app.post("/analyze")
def analyze(request: Request, image: UploadFile = File(...), question: str | None = Form(None),
            review: bool | None = Form(None)):
    rid, t0 = request.state.request_id, time.time()
    logs.request_id_var.set(rid)

    def fail(e: RepairError):
        logs.event("analyze_failed", level=30 if e.status < 500 else 40, error=e.code,
                   status_code=e.status, detail=e.detail or "-", latency=round(time.time() - t0, 1))
        return _err(e.status, e.code, e.message, rid)

    if config.API_KEY and not hmac.compare_digest(request.headers.get("x-api-key", ""), config.API_KEY):
        return _err(401, "unauthorized", "Missing or invalid API key.", rid)
    try:
        raw = image.file.read(config.MAX_IMAGE_BYTES + 1)
        if not raw:
            raise EmptyImage()
        if len(raw) > config.MAX_IMAGE_BYTES:
            raise ImageTooLarge(f"Image must be under {config.MAX_IMAGE_BYTES // (1024 * 1024)} MB.")
        if not _waiting.acquire(blocking=False):
            raise InferenceBusy()
        try:
            with _gpu:                                    # released on every exit path
                result = pipeline.analyze(raw, question, review)
        finally:
            _waiting.release()
    except RepairError as e:
        return fail(e)
    except Exception:  # noqa: BLE001 - never leak internals to the client
        logs.event("analyze_crashed", level=40, exc_info=True)
        return _err(500, "internal_error", "Something went wrong.", rid)

    logs.event("analyze_ok", model=result["meta"]["model"], status=result["status"], confidence=result["confidence"],
               severity=result["severity"], risk=result["risk"], second_pass=result["meta"]["second_pass"],
               image_size="x".join(map(str, result["meta"]["image_size"])), latency=round(time.time() - t0, 1))
    return result

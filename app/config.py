"""All runtime configuration, read once from the environment."""
import os


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or v.strip() == "" else v.strip()


def _bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


def _choice(name: str, default: str, allowed: set[str]) -> str:
    v = _env(name, default).lower()
    if v not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got {v!r}")
    return v


MODEL = _env("MODEL", "Qwen/Qwen3-VL-8B-Instruct")
MODEL_PATH = _env("MODEL_PATH", "")
DEVICE = _choice("DEVICE", "auto", {"auto", "cuda", "cpu"})
QUANTIZATION = _choice("QUANTIZATION", "4bit", {"4bit", "8bit", "none"})
DTYPE = _choice("DTYPE", "auto", {"auto", "bfloat16", "float16", "float32"})
GPU_MAX_MEMORY_GB = float(_env("GPU_MAX_MEMORY_GB", "5.0"))
CPU_MAX_MEMORY_GB = float(_env("CPU_MAX_MEMORY_GB", "24"))
PRELOAD_MODEL = _bool("PRELOAD_MODEL", True)
LOAD_RETRY_SECONDS = float(_env("LOAD_RETRY_SECONDS", "60"))

MAX_NEW_TOKENS = int(_env("MAX_NEW_TOKENS", "500"))
REVIEW_MAX_NEW_TOKENS = int(_env("REVIEW_MAX_NEW_TOKENS", "400"))
MODEL_TIMEOUT_SECONDS = float(_env("MODEL_TIMEOUT_SECONDS", "90"))
SECOND_PASS = _bool("SECOND_PASS", True)

MAX_IMAGE_BYTES = int(_env("MAX_IMAGE_BYTES", str(12 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(_env("MAX_IMAGE_PIXELS", "12000000"))
IMAGE_MAX_SIDE = int(_env("IMAGE_MAX_SIDE", "896"))
# norm1000   -> boxes on a 0-1000 grid (Qwen3-VL)
# pixel      -> absolute pixels of the image the model saw (Qwen2.5-VL style)
# normalized -> already 0-1
BBOX_MODE = _choice("BBOX_MODE", "norm1000", {"norm1000", "pixel", "normalized"})
PROMPT_VERSION = _env("PROMPT_VERSION", "v1")
QUESTION_MAX_CHARS = int(_env("QUESTION_MAX_CHARS", "500"))
MAX_QUEUE = int(_env("MAX_QUEUE", "2"))

CORS_ORIGINS = [o.strip() for o in _env("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
API_KEY = _env("REPAIR_API_KEY", "")

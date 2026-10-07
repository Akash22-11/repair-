"""Local Qwen3-VL inference (Transformers). Singleton model, thread-safe load, deadline-bounded generation.
torch/transformers are imported lazily so the API (and tests) start without them."""
import threading
import time
from typing import Any

from PIL import Image

from . import config, logs
from .errors import ModelError, ModelTimeout, ModelUnavailable, RepairError

_model: Any = None
_processor: Any = None
_load_lock = threading.Lock()
_state: dict[str, Any] = {"status": "not_loaded", "device": None, "quantization": None,
                          "cpu_offload": None, "load_seconds": None, "error_type": None, "failed_at": None}


def _pick_dtype(torch):
    if config.DTYPE != "auto":
        return getattr(torch, config.DTYPE)
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _load() -> None:
    global _model, _processor
    t0 = time.monotonic()
    _state.update(status="loading", error_type=None)
    try:
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

        has_cuda = torch.cuda.is_available()
        if config.DEVICE == "cuda" and not has_cuda:
            raise RuntimeError("DEVICE=cuda but CUDA is not available")
        use_cuda = has_cuda and config.DEVICE != "cpu"
        source = config.MODEL_PATH or config.MODEL      # standard HF cache is used; HF_HOME is honoured natively

        kwargs: dict[str, Any] = {}
        quant = "none"
        if use_cuda:
            kwargs.update(dtype=_pick_dtype(torch), device_map="auto",
                          max_memory={0: f"{config.GPU_MAX_MEMORY_GB}GiB", "cpu": f"{config.CPU_MAX_MEMORY_GB}GiB"})
            if config.QUANTIZATION != "none":
                quant = config.QUANTIZATION
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=quant == "4bit", load_in_8bit=quant == "8bit",
                    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=_pick_dtype(torch),
                    llm_int8_enable_fp32_cpu_offload=True)      # lets layers that do not fit spill to CPU
        else:
            kwargs.update(dtype=torch.bfloat16 if config.DTYPE == "auto" else getattr(torch, config.DTYPE),
                          device_map="cpu")

        logs.event("model_load_start", model=config.MODEL, device="cuda" if use_cuda else "cpu", quantization=quant)
        processor = AutoProcessor.from_pretrained(source)
        model = Qwen3VLForConditionalGeneration.from_pretrained(source, **kwargs).eval()
        offload = any(str(d) in ("cpu", "disk") for d in getattr(model, "hf_device_map", {}).values()) if use_cuda else False
        _model, _processor = model, processor
        _state.update(status="loaded", device="cuda" if use_cuda else "cpu", quantization=quant,
                      cpu_offload=offload, load_seconds=round(time.monotonic() - t0, 1), failed_at=None)
        logs.event("model_load_ok", seconds=_state["load_seconds"], cpu_offload=offload)
    except Exception as e:  # noqa: BLE001 - must never crash the backend
        _state.update(status="failed", error_type=type(e).__name__, failed_at=time.monotonic())
        logs.event("model_load_failed", level=40, exc_info=True, error_type=type(e).__name__)
        raise ModelUnavailable(detail=f"load failed: {type(e).__name__}") from e


def get_model():
    """Return (model, processor), loading once. Never blocks behind a load in progress."""
    if _model is not None:
        return _model, _processor
    if not _load_lock.acquire(blocking=False):
        raise ModelUnavailable("The diagnostic model is still loading. Try again in a minute.", detail="loading")
    try:
        if _model is None:
            failed_at = _state["failed_at"]
            if failed_at and time.monotonic() - failed_at < config.LOAD_RETRY_SECONDS:
                raise ModelUnavailable(detail="recent load failure, cooling down")
            _load()
        return _model, _processor
    finally:
        _load_lock.release()


def preload() -> None:
    try:
        get_model()
    except RepairError:
        pass  # already logged and recorded in _state; /health reports it


def generate(system: str, user: str, image: Image.Image, max_new_tokens: int) -> str:
    """One bounded, greedy generation. Raises ModelTimeout when the deadline passes between tokens."""
    model, processor = get_model()
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _Deadline(StoppingCriteria):
        def __init__(self, deadline: float):
            self.deadline, self.hit = deadline, False

        def __call__(self, input_ids, scores, **kwargs):
            if time.monotonic() > self.deadline:
                self.hit = True
            return torch.full((input_ids.shape[0],), self.hit, dtype=torch.bool, device=input_ids.device)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user}]},
    ]
    stopper = _Deadline(time.monotonic() + config.MODEL_TIMEOUT_SECONDS)
    try:
        with torch.inference_mode():
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   return_dict=True, return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 stopping_criteria=StoppingCriteriaList([stopper]))
            new_tokens = out[:, inputs["input_ids"].shape[1]:]
            text = processor.batch_decode(new_tokens, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        raise ModelError(detail="CUDA out of memory") from e
    except RepairError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ModelError(detail=f"{type(e).__name__}: {e}") from e
    if stopper.hit:
        raise ModelTimeout(detail=f"deadline {config.MODEL_TIMEOUT_SECONDS}s exceeded")
    return text


def health() -> dict:
    loaded = _state["status"] == "loaded"
    out = {"status": "ok" if loaded else _state["status"], "ok": loaded, "model": config.MODEL,
           "model_loaded": loaded, "device": _state["device"], "quantization": _state["quantization"]
           if loaded else config.QUANTIZATION}
    if loaded:
        out.update(cpu_offload=_state["cpu_offload"], load_seconds=_state["load_seconds"])
    if _state["status"] == "failed":
        out.update(status="model_unavailable", error_type=_state["error_type"])
    return out

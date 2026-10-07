# Qwen3-VL-8B-Instruct integration

## 1. Model
`Qwen/Qwen3-VL-8B-Instruct` (Hugging Face), run **locally** through Transformers (`Qwen3VLForConditionalGeneration` + `AutoProcessor`). No external inference API. If it cannot load, the backend does **not** fall back to another model: `/health` reports `model_unavailable` and `/analyze` returns 503.

## 2. Install
```
python -m venv .venv && .venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
# CUDA build of torch FIRST (pick the index that matches your driver, see pytorch.org):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
`transformers>=4.57` is required (first release with Qwen3-VL). `bitsandbytes>=0.45` provides 4-bit on CUDA (Linux and Windows).

## 3. Download / cache
Weights are public (~17 GB in bf16); no token needed. First start downloads them into the standard HF cache and later starts reuse it.
```
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct          # optional pre-download
set HF_HOME=D:\hf_cache                                      # optional: move the cache
```
Offline: download once, then `MODEL_PATH=/path/to/local/dir`. Never commit weights (`.gitignore` covers common paths).

## 4. Hardware / VRAM (~6 GB card)
8B parameters are ~4.5-5.5 GB in NF4 *before* the vision encoder, activations and KV cache, so a 6 GB card is **tight**. The loader therefore uses `device_map="auto"` with `max_memory={0: GPU_MAX_MEMORY_GB, cpu: CPU_MAX_MEMORY_GB}` and `llm_int8_enable_fp32_cpu_offload`: whatever does not fit in the GPU budget is placed in CPU RAM. Expect the model to load, but possibly partly offloaded (slower; `/health` shows `cpu_offload`). Needs ~16+ GB free system RAM for offload. If you hit CUDA OOM during inference, in this order: lower `IMAGE_MAX_SIDE` (e.g. 640), lower `MAX_NEW_TOKENS`, lower `GPU_MAX_MEMORY_GB` to leave activation headroom (e.g. 4.5), set `SECOND_PASS=false`.

## 5. CPU fallback
`DEVICE=cpu` (or no CUDA): bf16 on CPU, no quantization (bitsandbytes is CUDA-only), needs ~18 GB RAM and is slow (minutes per call). Use `MODEL_TIMEOUT_SECONDS` accordingly.

## 6. Configuration
See `.env.example`; every variable is documented there. Start with `uvicorn app.main:app --env-file .env`.

## 7. How a request flows
`/analyze` -> API key (if set) -> size check -> bounded queue (1 running + `MAX_QUEUE` waiting, else 503 `inference_busy`) -> GPU lock -> `images.prepare` (verify, bomb/size/aspect checks, EXIF rotate, RGB, resize to a multiple of 32) -> pass 1 -> parse -> Pydantic -> semantic checks -> optional pass 2 -> `safety.apply` -> response. Maximum two model calls per request; no loops, no model-based retries (JSON "repair" is deterministic string handling).

## 8. Pass 1 - investigator
Prompt in `app/prompts.py`. The model must follow a fixed decision procedure (supported object? -> evidence sufficient? -> abnormality visible?) and emit one JSON object, observations first. The user's question is placed in the *user* message between `<<< >>>` markers, length-capped and sanitised; it never enters the system prompt. Text in the image is declared untrusted visual content.

## 9. Pass 2 - skeptical reviewer
Same image, adversarial prompt: strongest counterargument, alternative diagnosis, verdict (`holds|weakened|likely_wrong`), revised confidence. `holds` = average, `weakened` = min, `likely_wrong` = min(conf, revised, 0.4) and, if that is < 0.25, status becomes `insufficient_visual_evidence`. Runs only when pass 1 says `problem_detected`. Any failure in pass 2 returns the pass-1 result with `review: null`, `meta.second_pass: false`.

## 10. Structured output & validation
`status` is one of `problem_detected | no_obvious_problem | insufficient_visual_evidence | unsupported_image`. Confidence must be a real finite number in [0, 1] (strings, NaN, Infinity, negatives, >1 -> 502 `invalid_model_output`); then capped: 0.95 max, 0.9 for `no_obvious_problem`, 0.2 for insufficient, 0 for unsupported, 0.7 if the problem text uses absolute certainty words. A `problem_detected` without sufficient evidence, observations or confidence >= 0.25 is downgraded. Observations containing inference words (because, likely, may...) move to `possible_causes`. Limits: 6/5/4 items, 300 chars each (longer strings are truncated), duplicates and instruction-like strings removed. Boxes: converted to normalised top-left `{x,y,width,height}`; invalid/inverted/NaN/tiny/whole-image boxes are dropped (`location: null`), never invented.

## 11. Safety layer
`app/safety.py` runs after the model and can only raise risk, force professional help, strip unsafe actions and prepend safe ones. For uncertain high-risk results it adds "Do not attempt a repair based solely on this image."

## 12. Errors
`bad_image`/`empty_image` 400, `image_too_large` 413, `model_unavailable` 503, `inference_busy` 503, `model_timeout` 504, `model_error` 502, `invalid_model_output` 502, `internal_error` 500, `unauthorized` 401. Bodies are `{error, message, request_id}` with generic messages; details go to the log under the same `request_id` (also in the `X-Request-ID` header).

## 13. Timeouts
A stopping criterion checks the deadline after every generated token, so a call that runs long is stopped cleanly and the GPU lock is released. Limit: a single forward pass (e.g. a very slow prefill on CPU) cannot be interrupted mid-step.

## 14. Troubleshooting
- `/health` says `model_unavailable` + `error_type`: check the server log for `model_load_failed` (stack trace). A failed load is retried after `LOAD_RETRY_SECONDS` on the next request.
- `ImportError`/`Qwen3VLForConditionalGeneration` missing: `pip install -U "transformers>=4.57"`.
- `torch.cuda.is_available()` false: you installed a CPU torch; reinstall from the CUDA index.
- OOM on load: lower `GPU_MAX_MEMORY_GB`; on inference: see section 4.
- Everything 503 `model is loading`: first start is downloading/loading; watch `/health`.

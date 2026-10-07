# REPAIR backend

REPAIR is a visual AI diagnostic assistant. Upload a photo of a physical object, optionally ask "What's wrong with this?", and get a structured, safety-checked diagnosis.

The core intelligence is a local open-weight vision-language model, **Qwen3-VL-8B-Instruct** (Transformers, 4-bit). The model's output is treated as untrusted data: it is parsed, validated and passed through a deterministic safety layer before anything is returned.

## How it works

```text
Frontend
   │  POST /analyze
   ▼
FastAPI  (auth, size limit, bounded queue, GPU lock)
   │
   ├─ images.prepare     verify, fix EXIF, convert to RGB, resize
   ├─ vlm.generate       pass 1: investigator (singleton model, deadline-bounded)
   ├─ validation         parsing + Pydantic + semantic checks
   │                     (status, confidence, bbox, observation vs inference)
   ├─ vlm.generate       pass 2: skeptical reviewer
   │                     (only for problem_detected; failure is non-fatal)
   ├─ safety.apply       can only raise risk / remove unsafe actions
   ▼
Response  {status, object, problem, confidence, severity, location, observations, ...}
```

More detail, VRAM notes and troubleshooting: [docs/qwen.md](docs/qwen.md)

### Diagnostic states

| `status` | Meaning |
|---|---|
| `problem_detected` | A relevant object is visible and there is sufficient visual evidence of a problem. |
| `no_obvious_problem` | The object is clear enough, but no visible problem was found. **This is not proof the object is healthy.** |
| `insufficient_visual_evidence` | The image is too blurry, dark, distant, obstructed or incomplete for a reliable diagnosis. Retake the photo. |
| `unsupported_image` | The image is not a physical object suitable for diagnosis (landscape, selfie, unrelated screenshot, etc.). |

REPAIR is never forced to invent a problem.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (designed around ~6 GB VRAM, see [VRAM note](#vram-note))
- ~17 GB of disk for the model download (Hugging Face cache)

## 1. Setup

**Windows (PowerShell / cmd)**

```bat
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
copy .env.example .env
```

**Linux / macOS**

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
cp .env.example .env
```

Install the CUDA build of PyTorch **first**, then the rest of the requirements.

### Pre-download the model (recommended)

The first start downloads ~17 GB and will look like it is hanging. Do it ahead of time, especially before a demo:

```bash
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct
```

### Run

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --env-file .env
```

Use `--host 0.0.0.0` only when deploying behind a reverse proxy or on a trusted network, and always set `REPAIR_API_KEY` in that case.

The model is loaded once in a background thread. Watch `GET /health` until `model_loaded` is `true`.

## 2. Configuration

Copy `.env.example` to `.env` and edit it. Never commit `.env`.

| Variable | Purpose |
|---|---|
| `REPAIR_API_KEY` | API key clients must send in `X-API-Key`. **If unset, authentication is disabled.** Always set it for anything reachable from outside your machine. |
| `CORS_ORIGINS` | Allowed browser origins. Defaults to `http://localhost:3000` (not `*`). |

See `.env.example` for the full list of settings (model, limits, timeouts, queue size).

## 3. Using the API

### Health check

```bash
curl http://localhost:8000/health
```

### Analyze an image

```bash
curl -X POST http://localhost:8000/analyze \
  -H "X-API-Key: YOUR_KEY" \
  -F "image=@bike.jpg" \
  -F "question=Why does it skip gears?"
```

| Form field | Required | Description |
|---|---|---|
| `image` | yes | The photo to analyze (size-limited, validated and re-encoded server-side). |
| `question` | no | A free-text question. Treated as untrusted input. |
| `review` | no | `true` (default) or `false`. `false` skips the second-pass review. |

Omit the `X-API-Key` header if `REPAIR_API_KEY` is not set.

### Example response

The values below are illustrative; the exact fields come from your running build.

```json
{
  "status": "problem_detected",
  "object": "bicycle rear derailleur",
  "problem": "bent derailleur hanger",
  "confidence": 0.72,
  "severity": "medium",
  "location": { "x": 0.58, "y": 0.41, "width": 0.18, "height": 0.22 },
  "observations": [
    "The derailleur cage is visibly angled relative to the rear wheel.",
    "The upper jockey wheel is not aligned with the cassette."
  ],
  "possible_causes": [
    "Impact or a fall may have bent the hanger."
  ],
  "recommended_actions": [
    "Stop riding until the derailleur is inspected.",
    "Have a bike shop check hanger alignment."
  ]
}
```

- `location` is `{x, y, width, height}` with values in 0 to 1 and a top-left origin, or `null`.
- Observations (what is visible) are kept separate from possible causes (inference).
- Errors are returned as `{ "error", "message", "request_id" }` with a 4xx or 5xx status. Raw exceptions are never exposed.

## 4. Tests

```bash
python -m pytest                                          # no GPU needed, model is stubbed
REPAIR_LIVE=1 python -m pytest tests/test_live_model.py -s  # real model
python -m tests.run_eval                                  # real model over tests/images/*
```

On Windows PowerShell, set the live flag with `$env:REPAIR_LIVE=1` before running pytest.

Prompts live in `app/prompts.py`. Safety rules live in `app/safety.py` (`RULES`).

## Security design

Defense in depth, because no single control is enough:

- **CORS** restricted to configured origins
- **API key** authentication
- **Request size limits** and a **bounded queue** (no unbounded backlog)
- **GPU lock**: one inference at a time on the single GPU
- **Image validation and safe processing**: verify, EXIF handling, RGB conversion, resize
- **Prompt boundaries**: the question and any text inside the image are treated as untrusted content, never as instructions
- **Output validation**: Pydantic schema, status checks, confidence bounds, bounding-box checks, observation vs. inference separation
- **Skeptical second pass** only when a problem is detected, and its failure is non-fatal
- **Deterministic safety layer** that can only raise risk or remove unsafe actions
- **Sanitized errors** with request IDs

## Known limitations

- The VLM can hallucinate defects, misidentify objects, misread text or overestimate confidence. Validation reduces this but cannot eliminate it. Treat results as guidance, not a professional inspection.
- Prompt injection (in the question or in text inside the image) is mitigated by prompt boundaries and output validation, not eliminated.
- `no_obvious_problem` does not mean the object is safe or defect-free.
- Authentication is a single shared API key. There is no per-user identity.
- Load protection comes from the size limit, bounded queue and GPU lock. Confirm whether per-client rate limiting is enabled in your deployment; if it is not, put a rate-limiting reverse proxy in front for public use.
- Always put the service behind HTTPS if it is exposed beyond localhost.

## VRAM note

Qwen3-VL-8B in 4-bit uses roughly 5 to 6 GB for weights alone. Image tokens, the vision encoder and the KV cache add more, so a 6 GB card can run out of memory. Keep image resolution capped and generation length bounded (both configurable), and note which model size and package versions you actually tested. If you hit OOM, Qwen3-VL-4B-Instruct is a lighter fallback.

Pin tested versions of `torch`, `transformers` and `bitsandbytes` in `requirements.txt`. 4-bit loading with `bitsandbytes` can be fussy on Windows. Make sure `python-dotenv` is installed, since `--env-file` depends on it.

## Migration notes (from the Ollama version)

- Environment variables are no longer prefixed `REPAIR_`, except `REPAIR_API_KEY`. See `.env.example` for the new names.
- `CORS_ORIGINS` now defaults to `http://localhost:3000` instead of `*`.
- The response gained a `status` field. Legacy fields are unchanged.

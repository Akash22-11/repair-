<div align="center">

# 🔧 REPAIR

### Visual AI diagnostic assistant for physical objects

Upload a photo. Ask *"What's wrong with this?"* Get an evidence-based, safety-checked answer.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)
![Model](https://img.shields.io/badge/VLM-Qwen3--VL--8B--Instruct-6f42c1)
![Runs](https://img.shields.io/badge/Inference-100%25%20local-success)
![GPU](https://img.shields.io/badge/GPU-~6%20GB%20VRAM-orange)

[Overview](#-overview) · [Quick start](#-quick-start) · [API](#-api-reference) · [How it works](#-how-it-works) · [Security](#-security-design) · [Testing](#-testing) · [Troubleshooting](#-troubleshooting)

</div>

---

## 📖 Overview

REPAIR is a **visual diagnostic assistant**. A user uploads an image of a physical object (a bike, an appliance, a wall, a laptop, a plant pot) and optionally asks a question. A local open-weight **Vision-Language Model (VLM)** inspects the image and returns a structured diagnosis:

- what the object is
- whether the photo contains enough evidence to say anything
- what is **visibly** wrong (observations)
- what **might** be causing it (inference, kept separate from observations)
- how confident it is and how risky the problem looks
- safe next steps

> **Core principle:** REPAIR is never forced to invent a problem. If the image is unclear, or nothing looks wrong, it says so.

### Why it is different from "just ask a chatbot"

| Typical chatbot | REPAIR |
|---|---|
| Free-form prose, can't be trusted programmatically | Strict JSON schema, validated with Pydantic |
| Will happily guess a defect from a blurry photo | Explicit `insufficient_visual_evidence` state |
| Mixes what it sees with what it assumes | Observations and inferences are separated in code |
| Single answer, no self-check | Optional skeptical second pass that challenges the first diagnosis |
| May give unsafe advice | Deterministic safety layer that can only add caution |
| Sends your photos to a third party | Model runs locally on your own GPU |

---

## ✨ Features

- 🧠 **Local open-weight VLM**: Qwen3-VL-8B-Instruct, 4-bit quantized, no external API calls
- 🧾 **Structured output**: object, problem, confidence, severity, bounding box, observations, causes, actions
- 🎯 **Four diagnostic states** so "nothing wrong", "can't tell" and "not my job" are never confused
- 🔍 **Skeptical second pass** that tries to disprove the first diagnosis (only when a problem is found)
- 🛡️ **Deterministic safety layer** with rule-based risk escalation and unsafe-action removal
- 🔐 **Hardened by default**: API key auth, size limits, bounded queue, GPU lock, sanitized errors, restricted CORS
- 🧪 **Test suite that runs without a GPU** (model is stubbed), plus live-model and evaluation runners

---

## 🚦 Diagnostic states

Every response carries exactly one `status`:

| Status | Meaning | Typical next step |
|---|---|---|
| ✅ `problem_detected` | A relevant object is visible and there is sufficient visual evidence of a problem | Follow the recommended actions |
| 🟢 `no_obvious_problem` | Object is clear enough, but no visible problem found. **Not proof the object is healthy** | Inspect further if symptoms persist |
| 🌫️ `insufficient_visual_evidence` | Photo is too blurry, dark, distant, obstructed or incomplete | Retake the photo |
| 🚫 `unsupported_image` | Not a physical object suitable for diagnosis (landscape, selfie, unrelated screenshot, ...) | Upload a photo of the object |

The pipeline also forces `insufficient_visual_evidence` when the model reports insufficient evidence, returns no problem, reports confidence below `0.25`, or gives no observations.

---

## 🧩 How it works

```text
Frontend
   │  POST /analyze  (image, question?, review?)
   ▼
┌──────────────────────────────────────────────────────────┐
│ FastAPI                                                  │
│  • API key check        • upload + size validation       │
│  • bounded queue        • GPU lock (1 inference at once) │
│  • sanitized errors     • request IDs                    │
└──────────────────────────────────────────────────────────┘
   │
   ▼
images.prepare       verify file, fix EXIF rotation, convert to RGB, resize
   │
   ▼
VLM pass 1           "investigator": identify object, find evidence, propose diagnosis
   │
   ▼
Validation           parse JSON → Pydantic schema → semantic checks
                     (status, confidence, bounding box, observation vs. inference)
   │
   ▼
VLM pass 2           "skeptical reviewer": challenge the diagnosis
(optional)           runs only for problem_detected; failure is non-fatal
   │
   ▼
Safety layer         can only RAISE risk or REMOVE unsafe actions, never relax them
   │
   ▼
Sanitized JSON response
```

### Evidence vs. inference

The pipeline separates *what is seen* from *what is guessed*. Statements containing hedging or causal language (`probably`, `likely`, `may`, `might`, `could`, `suggests`, `indicates`, `caused by`, `due to`) are moved out of `observations` and into `possible_causes`. This keeps hallucinated explanations from posing as visual facts.

### Confidence handling

Model-reported confidence is clamped to `0..1`, adjusted after the second pass, and used as one of the triggers for the insufficient-evidence state. The model's own confidence is never trusted blindly.

### Why a second pass?

A VLM tends to commit to its first idea. The reviewer pass is prompted to look for reasons the diagnosis is wrong, and its verdict can lower confidence or change the outcome. It is the most expensive step on a small GPU, so it is skipped unless a problem was detected, and can be disabled per request with `review=false`.

---

## 🚀 Quick start

### Requirements

| Item | Requirement |
|---|---|
| Python | 3.10 or newer |
| GPU | NVIDIA with CUDA, designed around ~6 GB VRAM (see [VRAM notes](#-vram-notes)) |
| Disk | ~17 GB for the model download |
| OS | Windows, Linux or macOS (CUDA required for the GPU path) |

### 1. Install

<details open>
<summary><b>Windows</b></summary>

```bat
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
copy .env.example .env
```
</details>

<details>
<summary><b>Linux / macOS</b></summary>

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
cp .env.example .env
```
</details>

> Install the **CUDA build of PyTorch first**, then the remaining requirements, so pip does not pull a CPU-only build.

### 2. Pre-download the model (strongly recommended)

The first start downloads ~17 GB and will look frozen. Do this ahead of time, especially before a demo:

```bash
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct
```

### 3. Configure

Edit `.env`. At minimum, set an API key if the server will be reachable by anyone but you:

```env
REPAIR_API_KEY=change-me-to-a-long-random-string
CORS_ORIGINS=http://localhost:3000
```

### 4. Run

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --env-file .env
```

The model loads **once, in a background thread**. Poll the health endpoint until it is ready:

```bash
curl http://localhost:8000/health
# wait for "model_loaded": true
```

> Use `--host 0.0.0.0` only behind a reverse proxy or on a trusted network, and always set `REPAIR_API_KEY` when you do.

---

## 🔌 API reference

### `GET /health`

Reports service status, including whether the model has finished loading.

### `POST /analyze`

`multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | file | yes | The photo to analyze. Size-limited, verified and re-encoded server-side |
| `question` | string | no | Free-text question. Treated as **untrusted** input |
| `review` | bool | no | Default `true`. Set `false` to skip the second pass (faster, cheaper) |

**Headers**

| Header | Required | Description |
|---|---|---|
| `X-API-Key` | when `REPAIR_API_KEY` is set | Shared secret |

**Example**

```bash
curl -X POST http://localhost:8000/analyze \
  -H "X-API-Key: YOUR_KEY" \
  -F "image=@bike.jpg" \
  -F "question=Why does it skip gears?" \
  -F "review=true"
```

### Response fields

| Field | Type | Description |
|---|---|---|
| `status` | string | One of the four [diagnostic states](#-diagnostic-states) |
| `object` | string | What the model thinks the object is |
| `problem` | string or null | Short description of the visible problem |
| `confidence` | number | 0 to 1, validated and adjusted by the pipeline |
| `severity` | string | Risk level, only ever raised by the safety layer |
| `location` | object or null | `{x, y, width, height}`, values 0 to 1, top-left origin |
| `observations` | string[] | Things that are **visible** in the image |
| `possible_causes` | string[] | Hypotheses, clearly separated from observations |
| `recommended_actions` | string[] | Safe next steps, filtered by the safety layer |

### Example response

> Illustrative only. Field names and values here are examples; check a real response from your build.

```json
{
  "status": "problem_detected",
  "object": "bicycle rear derailleur",
  "problem": "bent derailleur hanger",
  "confidence": 0.72,
  "severity": "medium",
  "location": { "x": 0.58, "y": 0.41, "width": 0.18, "height": 0.22 },
  "observations": [
    "The derailleur cage is angled relative to the rear wheel.",
    "The upper jockey wheel is not aligned with the cassette."
  ],
  "possible_causes": [
    "An impact or fall may have bent the hanger."
  ],
  "recommended_actions": [
    "Stop riding until the derailleur is inspected.",
    "Have a bike shop check the hanger alignment."
  ]
}
```

### Error format

All errors are structured JSON. Raw exceptions and stack traces are never returned.

```json
{ "error": "payload_too_large", "message": "Image exceeds the size limit.", "request_id": "..." }
```

| HTTP | Typical cause |
|---|---|
| 400 / 415 | Invalid or unsupported image |
| 401 / 403 | Missing or wrong API key |
| 413 | Upload larger than the limit (~12 MB) |
| 429 / 503 | Queue full or server busy, retry later |
| 504 | Inference exceeded its deadline |
| 500 | Unexpected internal error (use `request_id` to find it in the logs) |

The exact error codes and status mapping are defined in `app/main.py`.

---

## ⚙️ Configuration

Copy `.env.example` to `.env`. **Never commit `.env`.**

| Variable | Default | Purpose |
|---|---|---|
| `REPAIR_API_KEY` | unset | Key required in `X-API-Key`. **If unset, auth is disabled** |
| `CORS_ORIGINS` | `http://localhost:3000` | Allowed browser origins (no longer `*`) |

`.env.example` lists the remaining settings (model ID, upload limit, image max side, generation length, inference deadline, queue size). Keep it as the single source of truth.

---

## 🗂️ Project structure

> Approximate. Adjust to match your repository.

```text
.
├── app/
│   ├── main.py        # FastAPI app: auth, size limits, queue, GPU lock, errors
│   ├── pipeline.py    # Orchestrates passes, validation, confidence, safety
│   ├── images.py      # Image verification, EXIF, RGB, resize
│   ├── vlm.py         # Qwen3-VL singleton loader and bounded generation
│   ├── prompts.py     # Investigator and reviewer prompts
│   └── safety.py      # Deterministic safety RULES
├── docs/
│   └── qwen.md        # Model details, VRAM notes, troubleshooting
├── tests/
│   ├── images/        # Sample images for the eval runner
│   ├── test_live_model.py
│   └── run_eval.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🔐 Security design

The VLM is powerful but **untrusted**. It can hallucinate defects, misread text, follow instructions hidden in an image, return malformed data, or overstate confidence. REPAIR therefore layers defenses so that no single control has to be perfect:

```text
Untrusted client
  → CORS → Authentication → Request size limits → Bounded queue
  → Image validation → Safe image processing
  → Prompt boundaries → VLM
  → Output validation → Diagnostic-state validation → Confidence validation
  → Safety layer → Sanitized response
```

| Threat | Mitigation |
|---|---|
| Unauthorized API use | `X-API-Key` authentication |
| GPU / RAM exhaustion | Upload size cap, image resize, bounded queue, GPU lock, bounded generation length, inference deadline |
| Malicious image files | Pillow verification, re-encoding to RGB, resolution limits |
| Prompt injection via the question | Question is delimited and treated as data, never as instructions |
| Prompt injection via text in the image | Prompt tells the model that text in the image is untrusted visual content |
| Hallucinated defects becoming "facts" | Schema validation, evidence/inference separation, insufficient-evidence rules, skeptical second pass |
| Overconfident answers | Confidence clamping and adjustment |
| Unsafe recommendations | Safety layer removes unsafe actions and can only raise severity |
| Information leakage | Sanitized errors with request IDs, no stack traces, no secrets in source |
| Cross-site abuse | Restricted CORS origins |
| Runaway second pass cost | Review runs only for `problem_detected`, can be disabled, failure is non-fatal |

### ⚠️ Known limitations

REPAIR is hardened, **not invulnerable**. Be honest about what remains:

- The VLM can still hallucinate, misidentify objects or misjudge severity. Treat output as guidance, not a professional inspection.
- Prompt injection is **mitigated, not eliminated**. Output validation and the safety layer limit the damage, but cannot guarantee a model never follows a hidden instruction.
- `no_obvious_problem` is not a clean bill of health.
- Authentication is one shared key, with no per-user identity or revocation.
- Overload protection comes from the size limit, bounded queue and GPU lock. If per-client rate limiting is not enabled in your deployment, put a rate-limiting reverse proxy in front for public exposure.
- Serve over HTTPS if the service is reachable beyond localhost.
- Do not rely on REPAIR for safety-critical decisions (gas, electrical, structural, brakes). The safety layer pushes these toward "contact a professional" but is not a substitute for one.

---

## 🧪 Testing

```bash
# Fast tests, no GPU needed (model is stubbed)
python -m pytest

# Tests against the real model
REPAIR_LIVE=1 python -m pytest tests/test_live_model.py -s

# Run the real model over every image in tests/images/
python -m tests.run_eval
```

On Windows PowerShell, set the flag first: `$env:REPAIR_LIVE=1`. In cmd: `set REPAIR_LIVE=1`.

The no-GPU suite is meant to cover security-relevant behavior: auth, size limits, malformed uploads, malformed or hostile model output, status and confidence validation, and safety-layer rules.

Prompts live in `app/prompts.py`. Safety rules live in `app/safety.py` (`RULES`).

---

## 🧠 VRAM notes

Qwen3-VL-8B in 4-bit needs roughly 5 to 6 GB for weights alone. The vision encoder, activations and KV cache for image tokens add more, so a 6 GB card can run out of memory.

- Cap the image resolution (more pixels means more vision tokens)
- Keep generation length bounded
- Use `review=false` to avoid the second pass during testing
- If you still hit OOM, try **Qwen3-VL-4B-Instruct** as a lighter fallback

Record the model size and the versions of `torch`, `transformers` and `bitsandbytes` you actually tested, and pin them in `requirements.txt`. More detail is in [docs/qwen.md](docs/qwen.md).

---

## 🛠️ Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Server starts but `/analyze` fails, `model_loaded` is `false` | Model is still loading or downloading. Wait, and watch `/health` |
| First start seems frozen | ~17 GB download in progress. Pre-download with `huggingface-cli download` |
| `CUDA out of memory` | Lower the image max side, disable review, close other GPU apps, or switch to the 4B model |
| `bitsandbytes` import or CUDA errors on Windows | Use matching, pinned versions of torch, transformers and bitsandbytes |
| `--env-file` seems ignored | Install `python-dotenv` and confirm the path to `.env` |
| `401` / `403` | Send `X-API-Key` matching `REPAIR_API_KEY` |
| Browser requests blocked | Add your frontend origin to `CORS_ORIGINS` |
| `413` on upload | Image is over the size limit (~12 MB). Compress or resize it |
| Many requests return busy / `429` / `503` | Bounded queue is full by design. Retry, or run fewer parallel clients |

---

## 🔄 Migration notes (from the Ollama version)

- Environment variables are no longer prefixed `REPAIR_`, except `REPAIR_API_KEY`. See `.env.example` for the new names.
- `CORS_ORIGINS` now defaults to `http://localhost:3000` instead of `*`.
- The response gained a `status` field. Legacy fields are unchanged.
- Inference now runs locally through Transformers instead of Ollama.

---

## 🗺️ Roadmap

- [ ] Per-client rate limiting
- [ ] Multi-image input (several angles of the same object)
- [ ] Optional smaller model profile for 4 GB GPUs
- [ ] Frontend overlay of the `location` bounding box
- [ ] Larger evaluation set covering blur, darkness and adversarial images

---

## 👥 Team

Built by a two-person team at a hackathon.

| Name | Role |
|---|---|
| _Your name_ | AI / backend |
| _Teammate_ | Frontend |

## 📄 License

Add your license here (for example MIT). Note that the Qwen3-VL model weights are distributed under their own license; check it before commercial use.

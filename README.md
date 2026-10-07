# REPAIR backend

IMAGE -> local **Qwen3-VL-8B-Instruct** (Transformers, 4-bit) -> JSON -> validation -> optional skeptical 2nd pass -> deterministic safety layer -> `POST /analyze`

```
Frontend --POST /analyze--

> FastAPI (auth, size limit, bounded queue, GPU lock)
   -> images.prepare (verify, EXIF, RGB, resize)
   -> vlm.generate  [Qwen3-VL-8B-Instruct, singleton, deadline-bounded]  pass 1: investigator
   -> parsing + Pydantic + semantic checks (status, confidence, bbox, observation vs inference)
   -> vlm.generate  pass 2: skeptical reviewer (only for problem_detected; failure is non-fatal)
   -> safety.apply (can only raise risk / remove unsafe actions)
   -> response  {status, object, problem, confidence, severity, location, observations, ...}
```
Full details, VRAM notes and troubleshooting: **[docs/qwen.md](docs/q
wen.md)**.

## 1. Setup
```

python -m venv .venv && .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # CUDA build first
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --env-file .env
```
The model (~17 GB) downloads on first start into the HF cache and is loaded once in a background thread. Watch `GET /health` until `model_loaded` is `true`.

## 2. Call it
```
curl http://localhost:8000/health
curl -X POST http://localhost:8000/analyze -F "image=@bike.jpg" -F "question=Why does it skip gears?"
```
Add `-H "X-API-Key: ..."` if `REPAIR_API_KEY` is set. Optional form field `review=false` skips the second pass.

`status` is one of `problem_detected`, `no_obvious_problem` (not proof of health), `insufficient_visual_evidence`, `unsupported_image`. `location` is `{x,y,width,height}` (0-1, top-left origin) or `null`. Errors are `{error, message, request_id}` with 4xx/5xx.

## 3. Test
```
python -m pytest                                   # no GPU: model is stubbed
REPAIR_LIVE=1 python -m pytest tests/test_live_model.py -s   # real model
python -m tests.run_eval                           # real model over tests/images/*
```
Prompts: `app/prompts.py`. Safety rules: `app/safety.py` `RULES`.

## Migration notes (from the Ollama version)
Env vars are no longer `REPAIR_*` (except `REPAIR_API_KEY`); see `.env.example`. `CORS_ORIGINS` now defaults to `http://localhost:3000` instead of `*`. Response gained `status`; legacy fields are unchanged.

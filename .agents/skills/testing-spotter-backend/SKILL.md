---
name: testing-spotter-backend
description: End-to-end test the Spotter AI FastAPI backend (/analyze, /upload_example, /train, /train/status) via Swagger UI. Use when verifying backend changes or the bot's server-side flow.
---

# Testing the Spotter AI backend

The Telegram bot is a thin client that forwards multipart `photo` + GPS to the backend, so testing the backend endpoints covers the core feature. The bot itself needs a real `BOT_TOKEN` (see Secrets) and usually can't be run live.

## Run the backend
```
cd <repo> && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
- `GET /health` should return `{"status":"ok","detector_backend":"opencv-saliency"}` (or a YOLO backend if `DETECTOR_WEIGHTS` is set).
- Default detector is OpenCV saliency — no model download needed.
- Dataset dir: `dataset/images/train/`. Training status file: `models/training_status.json`.

## Adversarial test images
Generate three small JPEGs so detector output is *visibly different* per input (proves results aren't hardcoded):
- a frame with a bright object on the **left** → expect `action_required` starting "Turn left" + exactly 1 landmark, coords set, `distance_meters` 25.
- a frame with object on the right → "Turn right".
- a **blank/flat** frame → `visual_landmarks: []`, `action_required` = "No landmarks detected. Hold position and rescan the area.", coords `null`.

## Swagger UI testing (record this)
Open `http://127.0.0.1:8000/docs`. For each endpoint: expand → "Try it out" → fill fields → Execute → read the Server response body.
- File uploads: after clicking "Choose File", the OS file dialog opens. Use `ctrl+l` then type the absolute path (e.g. `/tmp/spotter_test/left.jpg`) + Enter — much more reliable than navigating the dialog.
- `/analyze` needs `latitude`, `longitude`, `accuracy` (text inputs) + `photo` (file).
- After selecting a file, re-check that you click the correct **Execute** button — the layout shifts when a response panel appears; take a screenshot if unsure.

## Training job timing
`/train` returns immediately with `status:"started"` + `job_id`; the job runs as a background task and takes ~10–30s. Wait before calling `GET /train/status`, which should eventually show `state:"completed"` with `images` = count uploaded. Empty dataset → `/train` returns 400 and status stays `idle` (regression: it must NOT get stuck at `queued`).

## Devin Secrets Needed
- `BOT_TOKEN` — only required to exercise the live Telegram bot end-to-end. Not needed for backend Swagger testing.

## Gotchas
- aiogram 3.15 constrains `pydantic<2.10` and `aiohttp<3.11`; keep `requirements.txt` pinned (`pydantic==2.9.2`, `aiohttp==3.10.11`).
- Verify `/upload_example` actually wrote the file: `ls dataset/images/train/`.

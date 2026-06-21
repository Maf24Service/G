---
name: testing-spotter-backend
description: End-to-end test the Spotter AI FastAPI backend, Telegram bot server flow, and buried-target point detection. Use when verifying /analyze, /upload_example, /train, /train/status, or bot photo-analysis changes.
---

# Testing the Spotter AI backend

The Telegram bot is a thin client that forwards multipart `photo` + GPS to the backend, so backend endpoint testing covers the core server-side flow. The bot itself needs a real `BOT_TOKEN` and usually requires the user to interact with Telegram for a live UI test.

## Run the backend
```
cd <repo> && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
- `GET /health` should return `status: ok` and a `target_backend` such as `buried-target-heuristic` or `buried-target-profile:<N>`.
- Dataset dir: `dataset/images/train/`. Training status file: `models/training_status.json`. Target profile: `models/target_profile.json`.

## Buried-target point testing
Use marked examples where the true hidden point is circled in saturated red. Upload/copy them into `dataset/images/train/`, call `/train`, then poll `/train/status`.

Pass criteria for `/train`:
- `POST /api/v1/train` returns `status: started` and a non-empty `job_id`.
- `GET /api/v1/train/status` reaches `state: completed`.
- `labeled_targets` is greater than zero and should match `models/target_profile.json.examples`.

Pass criteria for `/analyze` on a red-marked screenshot:
- `target_points` has exactly one item.
- `target_points[0].label == "marked_buried_target"`.
- The returned `center_x`/`center_y` lands inside the red-circled hidden point.
- The bot-equivalent output image should draw a red circle/cross on that point, not a green generic object box.

Pass criteria for `/analyze` on an unmarked photo:
- Returned candidates should be labeled `buried_target`, not `marked_buried_target`.
- Treat these as candidates unless ground truth is provided; exact accuracy requires a marked image or paired before/after example.

## Regression checks
For generic saliency behavior, use simple adversarial images:
- bright object on the left -> action starts with `Turn left`
- bright object on the right -> action starts with `Turn right`
- blank frame -> no target/landmark and `No landmarks detected...`

## Devin Secrets Needed
- `BOT_TOKEN` — only required to exercise the live Telegram bot end-to-end. Not needed for backend endpoint testing.

## Gotchas
- aiogram 3.15 constrains `pydantic<2.10` and `aiohttp<3.11`; keep `requirements.txt` pinned (`pydantic==2.9.2`, `aiohttp==3.10.11`).
- `opencv-contrib-python-headless` is required for the contrib saliency fallback; avoid replacing it with plain `opencv-python` unless saliency is no longer needed.
- Live Telegram bot testing needs the user to send `/start`, location, and photos; logs can be monitored from the bot process.

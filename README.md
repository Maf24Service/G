# Spotter AI

A two-part location-scouting assistant:

- **FastAPI backend** (`backend/`) — analyzes a geo-tagged photo, detects salient
  landmarks, and returns a navigation hint. Also stores training examples and
  triggers (re)training.
- **Telegram bot** (`bot/`) — an aiogram control panel that collects a GPS
  location + photo, forwards them to the backend, and draws the detected
  landmarks back onto the image for the user.

The vision pipeline ships with a dependency-light **OpenCV saliency detector** so
the whole system runs out of the box with no model downloads. You can plug in a
real **Ultralytics YOLO** model by setting `DETECTOR_WEIGHTS` (see below).

## Architecture

```
Telegram user
     │  (location + photo)
     ▼
 aiogram bot  ──HTTP──▶  FastAPI backend
     ▲                      │
     │  (annotated image)   ├─ /api/v1/analyze        detect landmarks + geo hint
     └──────────────────────┤─ /api/v1/upload_example save photo to dataset/
                            ├─ /api/v1/train          background retraining job
                            └─ /api/v1/train/status   poll job state
```

## Quick start

```bash
# 1. Install dependencies (Python 3.10+)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
#   edit .env and set BOT_TOKEN (from @BotFather)

# 3. Run the backend (terminal 1)
uvicorn backend.app.main:app --reload

# 4. Run the bot (terminal 2)
python -m bot.bot
```

Open `http://127.0.0.1:8000/docs` for the interactive API docs.

## Endpoints

| Method | Path                       | Description                                   |
| ------ | -------------------------- | --------------------------------------------- |
| GET    | `/health`                  | Liveness + active detector backend            |
| POST   | `/api/v1/analyze`          | `latitude`, `longitude`, `accuracy`, `photo`  |
| POST   | `/api/v1/upload_example`   | `photo` — saved under `dataset/images/train/` |
| POST   | `/api/v1/train`            | Start a background retraining job             |
| GET    | `/api/v1/train/status`     | Poll the latest training job                  |

### `/api/v1/analyze` response

```json
{
  "visual_landmarks": [
    {"box_pixels": [120, 80, 300, 260], "confidence": 0.82, "label": "landmark"}
  ],
  "geo_prediction": {
    "action_required": "Move straight ahead toward the landmark. (GPS accuracy ~5 m)",
    "predicted_lat": 55.75102,
    "predicted_lon": 37.61804,
    "distance_meters": 25.0
  }
}
```

## Using a real YOLO model

```bash
pip install ultralytics
export DETECTOR_WEIGHTS=./models/yolov8n.pt   # any Ultralytics .pt
```

When `DETECTOR_WEIGHTS` points to a valid model and `ultralytics` is installed,
the backend uses it automatically; otherwise it falls back to the OpenCV
heuristic. The (re)training hook lives in `backend/app/training.py` — replace
`TrainingManager.run` with your real training loop.

## Configuration

All settings are environment variables (see `.env.example`): `BOT_TOKEN`,
`BACKEND_BASE_URL`, `BACKEND_HOST`, `BACKEND_PORT`, `DATASET_DIR`, `MODELS_DIR`,
`DETECTOR_WEIGHTS`, `DETECTOR_CONFIDENCE`.

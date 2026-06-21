# Spotter AI

A two-part assistant for finding a **specific buried target point** from a geo-tagged photo.

- **FastAPI backend** (`backend/`) — analyzes GPS + photo, localizes the likely buried point, returns a navigation hint, stores marked examples, and rebuilds a target color/profile model.
- **Telegram bot** (`bot/`) — collects GPS + photo, forwards them to the backend, and draws a red circle/cross on the likely buried point.

The production detector is **not a generic object detector**. It is tuned for the target workflow: small light/gray markers hidden in grass/soil, with training examples marked by a red circle/arrow around the true point. Generic YOLO/Places365 support is optional context only.

## Architecture

```
Telegram user
     │  (location + photo)
     ▼
 aiogram bot  ──HTTP──▶  FastAPI backend
     ▲                      │
     │  (annotated image)   ├─ /api/v1/analyze        find buried target point + geo hint
     └──────────────────────┤─ /api/v1/upload_example save marked example to dataset/
                            ├─ /api/v1/train          learn target profile from red marks
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

## Training the buried-point detector

1. Take examples where the hidden point is visible or known.
2. Mark the true point with a **red circle** (best) or red arrow.
3. Upload those marked examples with the bot button **📁 Загрузить примеры для обучения** or `POST /api/v1/upload_example`.
4. Run **⚙️ Запустить переобучение ИИ** or `POST /api/v1/train`.

Training scans `dataset/images/train/`, extracts red-marked points, samples the target pixels, and writes `models/target_profile.json`. Future `/api/v1/analyze` calls use that profile to search new unmarked photos for the same kind of small buried marker.

## Endpoints

| Method | Path                       | Description                                      |
| ------ | -------------------------- | ------------------------------------------------ |
| GET    | `/health`                  | Liveness + active target backend                 |
| POST   | `/api/v1/analyze`          | `latitude`, `longitude`, `accuracy`, `photo`     |
| POST   | `/api/v1/upload_example`   | `photo` — saved under `dataset/images/train/`    |
| POST   | `/api/v1/train`            | Build target profile from red-marked examples    |
| GET    | `/api/v1/train/status`     | Poll the latest training job                     |

### `/api/v1/analyze` response

```json
{
  "visual_landmarks": [
    {"box_pixels": [430, 620, 486, 676], "confidence": 0.91, "label": "buried_target"}
  ],
  "target_points": [
    {"center_x": 458, "center_y": 648, "radius": 28, "confidence": 0.91, "label": "buried_target"}
  ],
  "geo_prediction": {
    "action_required": "Turn right and advance toward the buried_target. (GPS accuracy ~5 m)",
    "predicted_lat": 48.52172,
    "predicted_lon": 34.55754,
    "distance_meters": 25.0
  },
  "scene_tags": [],
  "place_summary": "Detected: buried_target"
}
```

## Optional open-source context models

The buried-point detector works with only `requirements.txt`. You can optionally install larger open-source models for extra context:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
```

Then set:

```bash
ENABLE_OBJECTS=1   # optional YOLO COCO object context
ENABLE_SCENE=1     # optional MIT Places365 scene labels
```

These models are not the main detection path; the main output remains `target_points`.

## Configuration

All settings are environment variables (see `.env.example`): `BOT_TOKEN`, `BACKEND_BASE_URL`, `BACKEND_HOST`, `BACKEND_PORT`, `DATASET_DIR`, `MODELS_DIR`, `TARGET_CONFIDENCE`, `ENABLE_OBJECTS`, `ENABLE_SCENE`, `DETECTOR_WEIGHTS`, `DETECTOR_CONFIDENCE`.

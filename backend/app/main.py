import logging
import os

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile

from .config import get_settings
from .detector import LandmarkDetector
from .geolocator import predict_geo
from .schemas import (
    AnalyzeResponse,
    GeoPrediction,
    TrainResponse,
    TrainStatusResponse,
    UploadResponse,
    VisualLandmark,
)
from .training import TrainingManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spotter.backend")

settings = get_settings()
detector = LandmarkDetector(
    weights=settings.detector_weights,
    confidence_threshold=settings.confidence_threshold,
)
training_manager = TrainingManager(settings.train_images_dir, settings.models_dir)

app = FastAPI(title="Spotter AI Backend", version="1.0.0")


def _decode_image(raw: bytes) -> np.ndarray:
    nparr = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return image


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "detector_backend": detector.backend}


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze(
    latitude: float = Form(...),
    longitude: float = Form(...),
    accuracy: float = Form(5.0),
    photo: UploadFile = File(...),
) -> AnalyzeResponse:
    raw = await photo.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty photo upload.")

    image = _decode_image(raw)
    height, width = image.shape[:2]

    detections = detector.detect(image)
    landmarks = [
        VisualLandmark(
            box_pixels=d.box_pixels, confidence=d.confidence, label=d.label
        )
        for d in detections
    ]
    geo: GeoPrediction = predict_geo(
        latitude, longitude, accuracy, detections, width, height
    )
    logger.info(
        "Analyzed image (%dx%d): %d landmark(s) via %s",
        width,
        height,
        len(landmarks),
        detector.backend,
    )
    return AnalyzeResponse(visual_landmarks=landmarks, geo_prediction=geo)


@app.post("/api/v1/upload_example", response_model=UploadResponse)
async def upload_example(photo: UploadFile = File(...)) -> UploadResponse:
    raw = await photo.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty photo upload.")

    settings.train_images_dir.mkdir(parents=True, exist_ok=True)
    filename = os.path.basename(photo.filename or "example.jpg")
    file_path = settings.train_images_dir / filename
    with open(file_path, "wb") as f:
        f.write(raw)

    logger.info("Saved training example to %s", file_path)
    return UploadResponse(status="saved", file=filename)


@app.post("/api/v1/train", response_model=TrainResponse)
async def train(background_tasks: BackgroundTasks) -> TrainResponse:
    job_id, images = training_manager.start()
    if images == 0:
        raise HTTPException(
            status_code=400,
            detail="No training examples found. Upload examples first.",
        )
    background_tasks.add_task(training_manager.run, job_id)
    return TrainResponse(
        status="started",
        detail=f"Retraining started on {images} example(s).",
        job_id=job_id,
    )


@app.get("/api/v1/train/status", response_model=TrainStatusResponse)
async def train_status() -> TrainStatusResponse:
    status = training_manager.read_status()
    return TrainStatusResponse(**status)

import logging
import os

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile

from .config import get_settings
from .detector import LandmarkDetector
from .geolocator import predict_geo
from .scene_classifier import SceneClassifier
from .schemas import (
    AnalyzeResponse,
    GeoPrediction,
    SceneTag,
    TargetPoint,
    TrainResponse,
    TrainStatusResponse,
    UploadResponse,
    VisualLandmark,
)
from .target_detector import BuriedTargetDetector
from .training import TrainingManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spotter.backend")

settings = get_settings()
target_detector = BuriedTargetDetector(
    settings.models_dir,
    confidence_threshold=settings.target_confidence_threshold,
)
detector = (
    LandmarkDetector(
        weights=settings.detector_weights,
        confidence_threshold=settings.confidence_threshold,
    )
    if settings.enable_objects
    else None
)
scene_classifier = (
    SceneClassifier(settings.weights_cache_dir)
    if settings.enable_scene
    else None
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
    return {
        "status": "ok",
        "target_backend": target_detector.backend,
        "detector_backend": detector.backend if detector else "disabled",
        "scene_classifier": bool(scene_classifier and scene_classifier.available),
    }


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

    target_candidates = target_detector.detect(image)
    detections = target_candidates
    if not detections and detector is not None:
        detections = detector.detect(image)
    landmarks = [
        VisualLandmark(
            box_pixels=d.box_pixels, confidence=d.confidence, label=d.label
        )
        for d in detections
    ]
    target_points = [
        TargetPoint(
            center_x=p.center_x,
            center_y=p.center_y,
            radius=p.radius,
            confidence=p.confidence,
            label=p.label,
        )
        for p in target_candidates
    ]
    geo: GeoPrediction = predict_geo(
        latitude, longitude, accuracy, detections, width, height
    )

    scene_tags: list[SceneTag] = []
    if scene_classifier is not None:
        scene_tags = [
            SceneTag(label=t.label, confidence=round(t.confidence, 4))
            for t in scene_classifier.classify(image)
        ]
    place_summary = _summarize_place(scene_tags, detections)

    logger.info(
        "Analyzed image (%dx%d): %d target(s), %d fallback landmark(s); scene=%s",
        width,
        height,
        len(target_points),
        max(0, len(landmarks) - len(target_points)),
        scene_tags[0].label if scene_tags else "n/a",
    )
    return AnalyzeResponse(
        visual_landmarks=landmarks,
        target_points=target_points,
        geo_prediction=geo,
        scene_tags=scene_tags,
        place_summary=place_summary,
    )


def _summarize_place(
    scene_tags: list[SceneTag], detections: list
) -> str | None:
    if not scene_tags and not detections:
        return None
    parts: list[str] = []
    if scene_tags:
        top = scene_tags[0]
        scene_name = top.label.replace("_", " ").replace("/", " / ")
        parts.append(f"Likely location: {scene_name} ({top.confidence:.0%})")
    if detections:
        labels: list[str] = []
        for d in detections:
            if d.label not in labels:
                labels.append(d.label)
            if len(labels) >= 5:
                break
        parts.append("Detected: " + ", ".join(labels))
    return ". ".join(parts)


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
    images = training_manager.count_images()
    if images == 0:
        raise HTTPException(
            status_code=400,
            detail="No training examples found. Upload examples first.",
        )
    job_id, images = training_manager.start()
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

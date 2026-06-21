import logging
from typing import Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Detection(Protocol):
    box_pixels: list[int]
    confidence: float
    label: str


class _Detection:
    def __init__(self, box_pixels: list[int], confidence: float, label: str) -> None:
        self.box_pixels = box_pixels
        self.confidence = confidence
        self.label = label


class LandmarkDetector:
    """Detects objects / landmarks in an image.

    Prefers a real, pretrained open-source detector (Ultralytics YOLO, COCO
    weights) which is downloaded automatically on first use. When the
    ``ultralytics``/``torch`` stack is not installed it transparently falls back
    to a dependency-light OpenCV saliency heuristic so the service still runs.
    """

    def __init__(
        self,
        weights: str | None = "yolo11n.pt",
        confidence_threshold: float = 0.35,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self._yolo = None
        self._weights = weights
        if weights:
            self._yolo = self._try_load_yolo(weights)
        self._saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

    @staticmethod
    def _try_load_yolo(weights: str):
        try:
            from ultralytics import YOLO

            model = YOLO(weights)
            logger.info("Loaded YOLO detector from %s", weights)
            return model
        except Exception as exc:
            logger.warning(
                "Could not load YOLO weights '%s' (%s). "
                "Falling back to OpenCV heuristic detector. Install the ML "
                "extras (pip install -r requirements-ml.txt) for real detection.",
                weights,
                exc,
            )
            return None

    @property
    def backend(self) -> str:
        if self._yolo is not None:
            return f"yolo:{self._weights}"
        return "opencv-saliency"

    def detect(self, image: np.ndarray) -> list[_Detection]:
        if self._yolo is not None:
            return self._detect_yolo(image)
        return self._detect_saliency(image)

    def _detect_yolo(self, image: np.ndarray) -> list[_Detection]:
        results = self._yolo(
            image, verbose=False, conf=self.confidence_threshold
        )
        detections: list[_Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                cls = int(box.cls[0])
                detections.append(
                    _Detection([x1, y1, x2, y2], conf, names.get(cls, "object"))
                )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def _detect_saliency(self, image: np.ndarray) -> list[_Detection]:
        h, w = image.shape[:2]
        success, saliency_map = self._saliency.computeSaliency(image)
        if not success:
            return []

        saliency_map = (saliency_map * 255).astype("uint8")
        thresh = cv2.threshold(
            saliency_map, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )[1]
        thresh = cv2.morphologyEx(
            thresh, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2
        )

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        image_area = float(h * w)
        candidates: list[_Detection] = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch
            if area < image_area * 0.01 or area > image_area * 0.9:
                continue
            region = saliency_map[y : y + ch, x : x + cw]
            confidence = float(np.clip(region.mean() / 255.0 + 0.25, 0.0, 0.99))
            if confidence < self.confidence_threshold:
                continue
            candidates.append(
                _Detection([x, y, x + cw, y + ch], confidence, "landmark")
            )

        candidates.sort(key=lambda d: d.confidence, reverse=True)
        return candidates[:5]

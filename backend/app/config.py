import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings:
    """Runtime configuration loaded from environment variables."""

    def __init__(self) -> None:
        self.host: str = os.getenv("BACKEND_HOST", "0.0.0.0")
        self.port: int = int(os.getenv("BACKEND_PORT", "8000"))

        dataset_dir = os.getenv("DATASET_DIR", str(BASE_DIR / "dataset"))
        self.dataset_dir: Path = Path(dataset_dir)
        self.train_images_dir: Path = self.dataset_dir / "images" / "train"

        models_dir = os.getenv("MODELS_DIR", str(BASE_DIR / "models"))
        self.models_dir: Path = Path(models_dir)

        # Detection model. Defaults to a pretrained Ultralytics YOLO (COCO)
        # checkpoint that is downloaded automatically on first use. Override with
        # a path to your own .pt weights, or set it to an empty string to force
        # the dependency-light OpenCV heuristic detector.
        weights = os.getenv("DETECTOR_WEIGHTS", "yolo11n.pt")
        self.detector_weights: str | None = weights or None

        self.confidence_threshold: float = float(
            os.getenv("DETECTOR_CONFIDENCE", "0.35")
        )
        self.target_confidence_threshold: float = float(
            os.getenv("TARGET_CONFIDENCE", "0.42")
        )

        # The production path is buried-target point detection. Generic object
        # detection is optional context and is disabled by default.
        self.enable_objects: bool = os.getenv("ENABLE_OBJECTS", "0") in (
            "1",
            "true",
            "True",
        )

        # Scene/place classification (MIT Places365). Disabled by default because
        # point localization works without large ML downloads.
        self.enable_scene: bool = os.getenv("ENABLE_SCENE", "0") in (
            "1",
            "true",
            "True",
        )
        self.weights_cache_dir: Path = self.models_dir / "weights"

        self.train_images_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()

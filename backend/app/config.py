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

        # Optional path to a real detection model (e.g. an Ultralytics YOLO .pt
        # file). When present and ultralytics is installed, it is used instead of
        # the built-in OpenCV heuristic detector.
        self.detector_weights: str | None = os.getenv("DETECTOR_WEIGHTS") or None

        self.confidence_threshold: float = float(
            os.getenv("DETECTOR_CONFIDENCE", "0.35")
        )

        self.train_images_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()

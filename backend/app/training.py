import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class TrainingManager:
    """Tracks (simulated) retraining jobs.

    Real model training is intentionally pluggable: replace ``_run`` with a call
    into your training pipeline (e.g. ``YOLO(...).train(...)``). The default
    implementation scans the dataset and records a job status file so the rest of
    the system is fully functional end to end.
    """

    def __init__(self, train_images_dir: Path, models_dir: Path) -> None:
        self.train_images_dir = train_images_dir
        self.models_dir = models_dir
        self.status_file = models_dir / "training_status.json"

    def count_images(self) -> int:
        if not self.train_images_dir.exists():
            return 0
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sum(
            1
            for p in self.train_images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )

    def _write_status(self, **payload) -> None:
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.status_file.write_text(json.dumps(payload, indent=2))

    def read_status(self) -> dict:
        if not self.status_file.exists():
            return {"state": "idle", "detail": "No training has been run yet."}
        return json.loads(self.status_file.read_text())

    def start(self) -> tuple[str, int]:
        job_id = uuid.uuid4().hex[:12]
        images = self.count_images()
        self._write_status(
            job_id=job_id,
            state="queued",
            detail="Training job queued.",
            images=images,
        )
        return job_id, images

    def run(self, job_id: str) -> None:
        """Executed in a background task."""
        try:
            images = self.count_images()
            self._write_status(
                job_id=job_id,
                state="running",
                detail="Scanning dataset and updating model weights.",
                images=images,
            )
            # Placeholder for the real training loop.
            time.sleep(2)
            self._write_status(
                job_id=job_id,
                state="completed",
                detail=f"Training finished on {images} example(s).",
                images=images,
            )
            logger.info("Training job %s completed on %d images", job_id, images)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Training job %s failed", job_id)
            self._write_status(
                job_id=job_id, state="failed", detail=str(exc), images=0
            )

import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class TargetPoint:
    def __init__(
        self,
        center_x: int,
        center_y: int,
        radius: int,
        confidence: float,
        label: str = "buried_target",
    ) -> None:
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.confidence = confidence
        self.label = label

    @property
    def box_pixels(self) -> list[int]:
        return [
            self.center_x - self.radius,
            self.center_y - self.radius,
            self.center_x + self.radius,
            self.center_y + self.radius,
        ]


class TargetProfile:
    def __init__(
        self,
        mean_bgr: tuple[float, float, float],
        std_bgr: tuple[float, float, float],
        examples: int,
    ) -> None:
        self.mean_bgr = mean_bgr
        self.std_bgr = std_bgr
        self.examples = examples


class BuriedTargetDetector:
    """Finds a small buried-cache target point, not generic objects.

    The detector is tuned for the user's workflow: marked examples contain a red
    circle/arrow around a tiny light marker in vegetation/soil. Training extracts
    that marked point and stores a color profile; inference then searches for
    small high-contrast, low-saturation patches in natural green/brown scenes and
    returns point candidates rather than object boxes.
    """

    def __init__(self, models_dir: Path, confidence_threshold: float = 0.42) -> None:
        self.models_dir = models_dir
        self.confidence_threshold = confidence_threshold
        self.profile_file = models_dir / "target_profile.json"
        self.profile = self._load_profile()

    @property
    def backend(self) -> str:
        if self.profile is None:
            return "buried-target-heuristic"
        return f"buried-target-profile:{self.profile.examples}"

    def detect(self, image: np.ndarray) -> list[TargetPoint]:
        marked = extract_marked_target(image)
        if marked is not None:
            marked.confidence = 0.99
            marked.label = "marked_buried_target"
            return [marked]
        return self._detect_unmarked(image)

    def _load_profile(self) -> TargetProfile | None:
        if not self.profile_file.exists():
            return None
        payload = json.loads(self.profile_file.read_text())
        mean = payload["mean_bgr"]
        std = payload["std_bgr"]
        return TargetProfile(
            (float(mean[0]), float(mean[1]), float(mean[2])),
            (float(std[0]), float(std[1]), float(std[2])),
            int(payload["examples"]),
        )

    def _detect_unmarked(self, image: np.ndarray) -> list[TargetPoint]:
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red = _red_mask(hsv)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        # The target is usually a tiny white/gray fragment in foliage or soil.
        candidate_mask = (
            ((saturation < 100) & (value > 115)) | ((saturation < 150) & (value > 175))
        ).astype("uint8") * 255
        candidate_mask[red > 0] = 0
        candidate_mask = cv2.morphologyEx(
            candidate_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
        )

        contours, _ = cv2.findContours(
            candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        image_area = float(h * w)
        min_area = max(6.0, image_area * 0.000004)
        max_area = max(40.0, image_area * 0.006)
        candidates: list[TargetPoint] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue
            x, y, cw, ch = cv2.boundingRect(contour)
            if cw < 3 or ch < 3:
                continue
            if cw > w * 0.18 or ch > h * 0.18:
                continue
            aspect = cw / float(ch)
            if aspect < 0.25 or aspect > 4.0:
                continue
            if y > h * 0.88 and (cw > 35 or ch > 35):
                continue

            score = self._score_component(image, hsv, x, y, cw, ch)
            if score < self.confidence_threshold:
                continue
            radius = max(12, int(max(cw, ch) * 1.8))
            candidates.append(
                TargetPoint(x + cw // 2, y + ch // 2, radius, round(score, 4))
            )

        candidates.sort(key=lambda p: p.confidence, reverse=True)
        return _dedupe_points(candidates)[:3]

    def _score_component(
        self, image: np.ndarray, hsv: np.ndarray, x: int, y: int, w: int, h: int
    ) -> float:
        patch = image[y : y + h, x : x + w]
        patch_hsv = hsv[y : y + h, x : x + w]
        brightness = float(np.mean(patch_hsv[:, :, 2]) / 255.0)
        desaturation = 1.0 - float(np.mean(patch_hsv[:, :, 1]) / 255.0)
        contrast = _local_contrast(image, x, y, w, h)
        compactness = min(1.0, (w * h) / max(1.0, float((max(w, h) ** 2))))
        profile_score = self._profile_score(patch)
        return float(
            np.clip(
                0.24 * brightness
                + 0.24 * desaturation
                + 0.24 * contrast
                + 0.18 * profile_score
                + 0.10 * compactness,
                0.0,
                0.99,
            )
        )

    def _profile_score(self, patch: np.ndarray) -> float:
        if self.profile is None:
            return 0.65
        mean = np.mean(patch.reshape(-1, 3), axis=0)
        profile_mean = np.array(self.profile.mean_bgr)
        profile_std = np.maximum(np.array(self.profile.std_bgr), 18.0)
        distance = np.linalg.norm((mean - profile_mean) / profile_std)
        return float(np.exp(-distance / 3.0))


class TargetProfileTrainer:
    def __init__(self, train_images_dir: Path, models_dir: Path) -> None:
        self.train_images_dir = train_images_dir
        self.models_dir = models_dir
        self.profile_file = models_dir / "target_profile.json"

    def build(self) -> dict:
        samples: list[np.ndarray] = []
        labeled_images = 0
        for image_path in sorted(self.train_images_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            marker = extract_marked_target(image)
            if marker is None:
                continue
            sample = _sample_target_pixels(image, marker)
            if sample.size == 0:
                continue
            samples.append(sample)
            labeled_images += 1

        if not samples:
            return {"labeled_targets": 0, "profile_written": False}

        pixels = np.concatenate(samples, axis=0)
        mean = np.mean(pixels, axis=0).tolist()
        std = np.std(pixels, axis=0).tolist()
        payload = {
            "mean_bgr": [round(float(v), 4) for v in mean],
            "std_bgr": [round(float(v), 4) for v in std],
            "examples": labeled_images,
        }
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.profile_file.write_text(json.dumps(payload, indent=2))
        logger.info("Built buried-target profile from %d labeled images", labeled_images)
        return {"labeled_targets": labeled_images, "profile_written": True}


def extract_marked_target(image: np.ndarray) -> TargetPoint | None:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = _red_mask(hsv)
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image.shape[:2]
    circle_candidates: list[tuple[float, TargetPoint]] = []
    fallback_candidates: list[tuple[float, TargetPoint]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 60:
            continue
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw < 12 or ch < 12:
            continue
        aspect = cw / float(ch)
        radius = max(12, int(max(cw, ch) * 0.6))
        point = TargetPoint(x + cw // 2, y + ch // 2, radius, 0.95)
        if 0.45 <= aspect <= 1.9 and cw < w * 0.35 and ch < h * 0.35:
            squareness = 1.0 - min(abs(1.0 - aspect), 1.0)
            right_panel_bonus = 4.0 if point.center_x > w * 0.45 else 0.0
            compact_bonus = 1.0 - min(
                max(cw, ch) / max(1.0, min(w, h) * 0.25), 1.0
            )
            score = (
                right_panel_bonus
                + squareness * 2.0
                + compact_bonus
                + min(area / 1000.0, 1.0)
            )
            circle_candidates.append((score, point))
        else:
            fallback_candidates.append((area, point))

    if circle_candidates:
        circle_candidates.sort(key=lambda item: item[0], reverse=True)
        return circle_candidates[0][1]
    if fallback_candidates:
        fallback_candidates.sort(key=lambda item: item[0], reverse=True)
        return fallback_candidates[0][1]
    return None


def _red_mask(hsv: np.ndarray) -> np.ndarray:
    # The training marker is a saturated UI-red annotation. Keep this strict so
    # rusty metal, bricks, and brown soil are not mistaken for labels.
    lower_red = cv2.inRange(hsv, (0, 135, 135), (7, 255, 255))
    upper_red = cv2.inRange(hsv, (172, 135, 135), (180, 255, 255))
    return cv2.bitwise_or(lower_red, upper_red)


def _sample_target_pixels(image: np.ndarray, marker: TargetPoint) -> np.ndarray:
    h, w = image.shape[:2]
    radius = max(6, marker.radius // 3)
    x1 = max(0, marker.center_x - radius)
    y1 = max(0, marker.center_y - radius)
    x2 = min(w, marker.center_x + radius)
    y2 = min(h, marker.center_y + radius)
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    red = _red_mask(hsv)
    target_mask = ((hsv[:, :, 1] < 130) & (hsv[:, :, 2] > 90) & (red == 0))
    pixels = patch[target_mask]
    if pixels.size == 0:
        pixels = patch[red == 0]
    return pixels.reshape(-1, 3).astype(np.float32)


def _local_contrast(image: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    ih, iw = image.shape[:2]
    pad = max(10, int(max(w, h) * 2.5))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(iw, x + w + pad)
    y2 = min(ih, y + h + pad)
    outer = image[y1:y2, x1:x2]
    inner = image[y : y + h, x : x + w]
    if outer.size == 0 or inner.size == 0:
        return 0.0
    inner_mean = np.mean(cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY))
    outer_gray = cv2.cvtColor(outer, cv2.COLOR_BGR2GRAY)
    outer_mean = np.mean(outer_gray)
    return float(np.clip(abs(inner_mean - outer_mean) / 90.0, 0.0, 1.0))


def _dedupe_points(points: list[TargetPoint]) -> list[TargetPoint]:
    kept: list[TargetPoint] = []
    for point in points:
        duplicate = False
        for existing in kept:
            distance = float(
                np.hypot(point.center_x - existing.center_x, point.center_y - existing.center_y)
            )
            if distance < max(point.radius, existing.radius):
                duplicate = True
                break
        if not duplicate:
            kept.append(point)
    return kept

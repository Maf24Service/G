import logging
import urllib.request
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_WEIGTHS_URL = (
    "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
)
_CATEGORIES_URL = (
    "https://raw.githubusercontent.com/csailvision/places365/master/"
    "categories_places365.txt"
)


class SceneTag:
    def __init__(self, label: str, confidence: float) -> None:
        self.label = label
        self.confidence = confidence


class SceneClassifier:
    """Classifies the overall scene/place type of a photo.

    Uses a ResNet-18 pretrained on the open MIT Places365 dataset, giving a
    coarse "preliminary place" label (e.g. ``street``, ``plaza``, ``parking_lot``,
    ``building_facade``). Weights and the category list are downloaded once and
    cached. If the torch stack or the weights are unavailable the classifier is
    disabled and ``classify`` returns an empty list.
    """

    def __init__(self, cache_dir: Path, top_k: int = 3) -> None:
        self.top_k = top_k
        self.cache_dir = cache_dir
        self._model = None
        self._categories: list[str] = []
        self._transform = None
        self._available = self._setup()

    @property
    def available(self) -> bool:
        return self._available

    def _setup(self) -> bool:
        try:
            import torch
            from torchvision import transforms
            from torchvision.models import resnet18

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            weights_path = self.cache_dir / "resnet18_places365.pth.tar"
            categories_path = self.cache_dir / "categories_places365.txt"
            _download(_WEIGTHS_URL, weights_path)
            _download(_CATEGORIES_URL, categories_path)

            self._categories = _load_categories(categories_path)

            model = resnet18(num_classes=365)
            checkpoint = torch.load(
                weights_path, map_location="cpu", weights_only=False
            )
            state_dict = {
                k.replace("module.", ""): v
                for k, v in checkpoint["state_dict"].items()
            }
            model.load_state_dict(state_dict)
            model.eval()
            self._model = model
            self._torch = torch

            self._transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
            logger.info("Loaded Places365 scene classifier (%d categories).", 365)
            return True
        except Exception as exc:
            logger.warning(
                "Scene classifier disabled (%s). Install the ML extras "
                "(pip install -r requirements-ml.txt) to enable scene tagging.",
                exc,
            )
            return False

    def classify(self, image_bgr: np.ndarray) -> list[SceneTag]:
        if not self._available or self._model is None:
            return []
        # torchvision transforms expect RGB.
        image_rgb = image_bgr[:, :, ::-1].copy()
        tensor = self._transform(image_rgb).unsqueeze(0)
        with self._torch.no_grad():
            logits = self._model(tensor)
            probs = self._torch.nn.functional.softmax(logits[0], dim=0)
            top_probs, top_idx = probs.topk(self.top_k)
        tags: list[SceneTag] = []
        for prob, idx in zip(top_probs.tolist(), top_idx.tolist()):
            tags.append(SceneTag(self._categories[idx], float(prob)))
        return tags


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    logger.info("Downloading %s -> %s", url, dest)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - trusted open dataset host


def _load_categories(path: Path) -> list[str]:
    categories: list[str] = []
    with open(path) as f:
        for line in f:
            # Lines look like: "/a/airfield 0"
            name = line.strip().split(" ")[0]
            categories.append(name[3:] if name.startswith("/") else name)
    return categories

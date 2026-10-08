import hashlib
import logging
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from castinspect.data.acquire import sha256_file
from castinspect.models.base import register_model

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResnetLinearProbe:
    """Frozen ResNet-18 embeddings with a logistic regression head.

    Embeddings are cached on disk keyed by image content hash, so repeated
    cross-validation runs only pay the backbone cost once.
    """

    name = "resnet18_probe"

    def __init__(
        self,
        pretrained: bool = True,
        weights_sha256: str = "",
        cache_dir: Path | None = None,
        image_size: int = 224,
        batch_size: int = 32,
        c_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
        inner_cv_folds: int = 3,
        seed: int = 42,
    ) -> None:
        self._pretrained = pretrained
        self._weights_sha256 = weights_sha256
        self._cache_dir = cache_dir
        self._image_size = image_size
        self._batch_size = batch_size
        self._c_grid = tuple(c_grid)
        self._inner_cv_folds = inner_cv_folds
        self._seed = seed
        self._classifier: object | None = None
        self._backbone: Any = None

    def _load_backbone(self) -> Any:
        if self._backbone is not None:
            return self._backbone
        import torch
        from torchvision.models import ResNet18_Weights, resnet18

        torch.manual_seed(self._seed)
        if self._pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            model = resnet18(weights=weights)
            self._verify_weights_file()
        else:
            model = resnet18(weights=None)
        model.fc = torch.nn.Identity()
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._backbone = model
        return model

    def _verify_weights_file(self) -> None:
        import torch

        checkpoints = Path(torch.hub.get_dir()) / "checkpoints"
        candidates = sorted(checkpoints.glob("resnet18-*.pth"))
        if not candidates:
            raise FileNotFoundError(f"downloaded weights not found under {checkpoints}")
        actual = sha256_file(candidates[0])
        if not self._weights_sha256:
            logger.warning(
                "weights checksum not pinned; %s has sha256 %s", candidates[0].name, actual
            )
            return
        if actual != self._weights_sha256:
            raise ValueError(
                f"weights checksum mismatch for {candidates[0].name}: "
                f"expected {self._weights_sha256}, got {actual}"
            )

    def _to_tensor_batch(self, image_paths: Sequence[Path]) -> Any:
        import torch

        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        tensors = []
        for path in image_paths:
            with Image.open(path) as img:
                resized = img.convert("L").resize(
                    (self._image_size, self._image_size), Image.Resampling.BILINEAR
                )
            array = np.asarray(resized, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).unsqueeze(0).repeat(3, 1, 1)
            tensors.append((tensor - mean) / std)
        return torch.stack(tensors)

    def _cache_path(self, image_path: Path) -> Path | None:
        if self._cache_dir is None:
            return None
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        variant = "imagenet" if self._pretrained else "random"
        return self._cache_dir / variant / f"{digest}.npy"

    def _embed(self, image_paths: Sequence[Path]) -> npt.NDArray[np.float64]:
        import torch

        vectors: dict[int, npt.NDArray[np.float64]] = {}
        missing: list[tuple[int, Path]] = []
        for index, path in enumerate(image_paths):
            cached = self._cache_path(path)
            if cached is not None and cached.is_file():
                vectors[index] = np.load(cached)
            else:
                missing.append((index, path))
        if missing:
            backbone = self._load_backbone()
            for start in range(0, len(missing), self._batch_size):
                chunk = missing[start : start + self._batch_size]
                batch = self._to_tensor_batch([path for _, path in chunk])
                with torch.no_grad():
                    embedded = backbone(batch).numpy().astype(np.float64)
                for (index, path), vector in zip(chunk, embedded, strict=True):
                    vectors[index] = vector
                    cached = self._cache_path(path)
                    if cached is not None:
                        cached.parent.mkdir(parents=True, exist_ok=True)
                        np.save(cached, vector)
        return np.stack([vectors[i] for i in range(len(image_paths))])

    def fit(self, image_paths: Sequence[Path], labels: Sequence[int]) -> None:
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.model_selection import StratifiedKFold

        embeddings = self._embed(image_paths)
        classifier = LogisticRegressionCV(
            Cs=list(self._c_grid),
            cv=StratifiedKFold(self._inner_cv_folds, shuffle=True, random_state=self._seed),
            class_weight="balanced",
            scoring="roc_auc",
            max_iter=5000,
        )
        classifier.fit(embeddings, np.asarray(labels, dtype=np.int64))
        self._classifier = classifier

    def predict_proba(self, image_paths: Sequence[Path]) -> npt.NDArray[np.float64]:
        if self._classifier is None:
            raise RuntimeError("model is not fitted")
        probabilities = self._classifier.predict_proba(self._embed(image_paths))  # type: ignore[attr-defined]
        return np.asarray(probabilities[:, 1], dtype=np.float64)

    def save(self, path: Path) -> None:
        state = {key: value for key, value in self.__dict__.items() if key != "_backbone"}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(state, fh)

    def load(self, path: Path) -> None:
        with path.open("rb") as fh:
            self.__dict__.update(pickle.load(fh))
        self._backbone = None


register_model("resnet18_probe", ResnetLinearProbe)

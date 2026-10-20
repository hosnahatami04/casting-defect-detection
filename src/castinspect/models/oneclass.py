import logging
import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from castinspect.models.base import register_model
from castinspect.models.resnet import ResnetLinearProbe

logger = logging.getLogger(__name__)


class OneClassKnn:
    """Anomaly score from frozen ResNet-18 embeddings: mean distance to the k
    nearest healthy training images.

    The reference set uses healthy images only. Defect labels in the training
    data, when present, are used solely to pick k from the grid; the evaluation
    fold is never touched (recorded in ADR 0007).
    """

    name = "oneclass_knn"

    def __init__(
        self,
        k: int = 5,
        k_grid: Sequence[int] = (1, 3, 5, 10, 20),
        pretrained: bool = True,
        weights_sha256: str = "",
        cache_dir: Path | None = None,
        image_size: int = 224,
        seed: int = 42,
    ) -> None:
        self._k = k
        self._k_grid = tuple(k_grid)
        self._embedder = ResnetLinearProbe(
            pretrained=pretrained,
            weights_sha256=weights_sha256,
            cache_dir=cache_dir,
            image_size=image_size,
            seed=seed,
        )
        self._reference: npt.NDArray[np.float64] | None = None
        self._scale = 1.0

    def _scores_against(
        self, reference: npt.NDArray[np.float64], embeddings: npt.NDArray[np.float64], k: int
    ) -> npt.NDArray[np.float64]:
        distances = np.linalg.norm(embeddings[:, None, :] - reference[None, :, :], axis=2)
        k = min(k, reference.shape[0])
        nearest = np.sort(distances, axis=1)[:, :k]
        return np.asarray(nearest.mean(axis=1), dtype=np.float64)

    def fit(self, image_paths: Sequence[Path], labels: Sequence[int]) -> None:
        from castinspect.evaluation.metrics import auroc

        y = np.asarray(labels, dtype=np.int64)
        paths = list(image_paths)
        embeddings = self._embedder._embed(paths)
        healthy = embeddings[y == 0]
        if len(healthy) == 0:
            raise ValueError("one-class fit needs healthy images")
        self._reference = healthy

        defective = embeddings[y == 1]
        if len(defective) > 0 and len(self._k_grid) > 1:
            best_k, best_auroc = self._k, -1.0
            # Healthy scores leave self out: the nearest neighbour of a
            # reference point is itself at distance zero, so the mean over k+1
            # rescaled by (k+1)/k equals the mean over the k real neighbours.
            for k in self._k_grid:
                healthy_scores = self._scores_against(healthy, healthy, k + 1) * ((k + 1) / k)
                defect_scores = self._scores_against(healthy, defective, k)
                grid_labels = np.r_[np.zeros(len(healthy)), np.ones(len(defective))]
                value = auroc(grid_labels, np.r_[healthy_scores, defect_scores])
                if value > best_auroc:
                    best_auroc, best_k = value, k
            self._k = best_k
            logger.info("one-class k selected: %d (train auroc %.4f)", best_k, best_auroc)

        # The squashing scale is fixed here, from healthy leave-self-out
        # distances, so scores stay comparable across prediction batches.
        k = self._k
        typical = self._scores_against(healthy, healthy, k + 1) * ((k + 1) / k)
        self._scale = float(typical.mean()) + 1e-9

    def predict_proba(self, image_paths: Sequence[Path]) -> npt.NDArray[np.float64]:
        if self._reference is None:
            raise RuntimeError("model is not fitted")
        embeddings = self._embedder._embed(list(image_paths))
        scores = self._scores_against(self._reference, embeddings, self._k)
        return np.asarray(1.0 - np.exp(-scores / self._scale), dtype=np.float64)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "k": self._k,
            "k_grid": self._k_grid,
            "reference": self._reference,
            "scale": self._scale,
        }
        with path.open("wb") as fh:
            pickle.dump(state, fh)

    def load(self, path: Path) -> None:
        with path.open("rb") as fh:
            state = pickle.load(fh)
        self._k = state["k"]
        self._k_grid = state["k_grid"]
        self._reference = state["reference"]
        self._scale = state["scale"]


register_model("oneclass_knn", OneClassKnn)

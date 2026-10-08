import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from castinspect.models.base import register_model


class HogClassifier:
    name = "hog"

    def __init__(
        self,
        image_size: int = 128,
        orientations: int = 9,
        pixels_per_cell: int = 8,
        cells_per_block: int = 2,
        c_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
        inner_cv_folds: int = 3,
        seed: int = 42,
    ) -> None:
        self._image_size = image_size
        self._orientations = orientations
        self._pixels_per_cell = pixels_per_cell
        self._cells_per_block = cells_per_block
        self._c_grid = tuple(c_grid)
        self._inner_cv_folds = inner_cv_folds
        self._seed = seed
        self._classifier: object | None = None

    def _features(self, image_paths: Sequence[Path]) -> npt.NDArray[np.float64]:
        from skimage.feature import hog

        rows = []
        for path in image_paths:
            with Image.open(path) as img:
                resized = img.convert("L").resize(
                    (self._image_size, self._image_size), Image.Resampling.BILINEAR
                )
            rows.append(
                hog(
                    np.asarray(resized),
                    orientations=self._orientations,
                    pixels_per_cell=(self._pixels_per_cell, self._pixels_per_cell),
                    cells_per_block=(self._cells_per_block, self._cells_per_block),
                    block_norm="L2-Hys",
                )
            )
        return np.asarray(rows, dtype=np.float64)

    def fit(self, image_paths: Sequence[Path], labels: Sequence[int]) -> None:
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.model_selection import StratifiedKFold

        features = self._features(image_paths)
        classifier = LogisticRegressionCV(
            Cs=list(self._c_grid),
            cv=StratifiedKFold(self._inner_cv_folds, shuffle=True, random_state=self._seed),
            class_weight="balanced",
            scoring="roc_auc",
            max_iter=5000,
        )
        classifier.fit(features, np.asarray(labels, dtype=np.int64))
        self._classifier = classifier

    def predict_proba(self, image_paths: Sequence[Path]) -> npt.NDArray[np.float64]:
        if self._classifier is None:
            raise RuntimeError("model is not fitted")
        probabilities = self._classifier.predict_proba(self._features(image_paths))  # type: ignore[attr-defined]
        return np.asarray(probabilities[:, 1], dtype=np.float64)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self.__dict__, fh)

    def load(self, path: Path) -> None:
        with path.open("rb") as fh:
            self.__dict__.update(pickle.load(fh))


register_model("hog", HogClassifier)

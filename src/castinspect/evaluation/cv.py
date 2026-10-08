import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from castinspect.data.dataset import LABEL_TO_INDEX
from castinspect.data.split import ManifestRow
from castinspect.models.base import ModelFactory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OofPrediction:
    path: str
    label: int
    group_id: int
    fold: int
    probability: float


def run_cross_validation(
    factory: ModelFactory,
    manifest_rows: Sequence[ManifestRow],
    image_root: Path,
    n_folds: int,
) -> list[OofPrediction]:
    dev_rows = [row for row in manifest_rows if row.split == "dev"]
    if not dev_rows:
        raise ValueError("manifest contains no development rows")
    predictions: list[OofPrediction] = []
    for fold in range(n_folds):
        train_rows = [row for row in dev_rows if row.fold != fold]
        held_out = [row for row in dev_rows if row.fold == fold]
        if not held_out:
            raise ValueError(f"fold {fold} is empty")
        model = factory()
        started = time.perf_counter()
        model.fit(
            [image_root / row.path for row in train_rows],
            [LABEL_TO_INDEX[row.label] for row in train_rows],
        )
        probabilities = model.predict_proba([image_root / row.path for row in held_out])
        logger.info(
            "fold %d: trained on %d, predicted %d in %.1fs",
            fold,
            len(train_rows),
            len(held_out),
            time.perf_counter() - started,
        )
        predictions.extend(
            OofPrediction(
                path=row.path,
                label=LABEL_TO_INDEX[row.label],
                group_id=row.group_id,
                fold=fold,
                probability=float(probability),
            )
            for row, probability in zip(held_out, probabilities, strict=True)
        )
    if len(predictions) != len(dev_rows):
        raise RuntimeError("out-of-fold predictions do not cover development exactly once")
    return sorted(predictions, key=lambda p: p.path)


def save_oof(predictions: Sequence[OofPrediction], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "path": [p.path for p in predictions],
            "label": [p.label for p in predictions],
            "group_id": [p.group_id for p in predictions],
            "fold": [p.fold for p in predictions],
            "probability": [p.probability for p in predictions],
        }
    )
    pq.write_table(table, path)


def load_oof(path: Path) -> list[OofPrediction]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    columns = table.to_pydict()
    return [
        OofPrediction(
            path=columns["path"][i],
            label=columns["label"][i],
            group_id=columns["group_id"][i],
            fold=columns["fold"][i],
            probability=columns["probability"][i],
        )
        for i in range(table.num_rows)
    ]


def oof_arrays(
    predictions: Sequence[OofPrediction],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    labels = np.asarray([p.label for p in predictions], dtype=np.int64)
    probabilities = np.asarray([p.probability for p in predictions], dtype=np.float64)
    groups = np.asarray([p.group_id for p in predictions], dtype=np.int64)
    return labels, probabilities, groups

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

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


@dataclass(frozen=True)
class CvResult:
    predictions: list[OofPrediction]
    fit_infos: list[dict[str, Any]] = field(default_factory=list)
    fold_seconds: list[float] = field(default_factory=list)


def run_cross_validation(
    factory: ModelFactory,
    manifest_rows: Sequence[ManifestRow],
    image_root: Path,
    n_folds: int,
    pass_validation: bool = False,
    fold_model_dir: Path | None = None,
    fold_model_name: str = "model",
) -> CvResult:
    """With pass_validation, each model receives the held-out fold for early
    stopping. The resulting out-of-fold scores are then mildly optimistic
    (epoch selection saw the fold); the untouched test split settles it."""
    dev_rows = [row for row in manifest_rows if row.split == "dev"]
    if not dev_rows:
        raise ValueError("manifest contains no development rows")
    predictions: list[OofPrediction] = []
    fit_infos: list[dict[str, Any]] = []
    fold_seconds: list[float] = []
    for fold in range(n_folds):
        train_rows = [row for row in dev_rows if row.fold != fold]
        held_out = [row for row in dev_rows if row.fold == fold]
        if not held_out:
            raise ValueError(f"fold {fold} is empty")
        model = factory()
        started = time.perf_counter()
        train_paths = [image_root / row.path for row in train_rows]
        train_labels = [LABEL_TO_INDEX[row.label] for row in train_rows]
        held_out_paths = [image_root / row.path for row in held_out]
        if pass_validation:
            # The protocol does not know about validation kwargs; models that
            # opt in (validation_early_stopping in their config) accept them.
            cast(Any, model).fit(
                train_paths,
                train_labels,
                val_paths=held_out_paths,
                val_labels=[LABEL_TO_INDEX[row.label] for row in held_out],
            )
        else:
            model.fit(train_paths, train_labels)
        probabilities = model.predict_proba(held_out_paths)
        if fold_model_dir is not None:
            model.save(fold_model_dir / f"{fold_model_name}_fold{fold}.pt")
        elapsed = time.perf_counter() - started
        fold_seconds.append(elapsed)
        fit_infos.append(dict(getattr(model, "fit_info", {})))
        logger.info(
            "fold %d: trained on %d, predicted %d in %.1fs",
            fold,
            len(train_rows),
            len(held_out),
            elapsed,
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
    return CvResult(
        predictions=sorted(predictions, key=lambda p: p.path),
        fit_infos=fit_infos,
        fold_seconds=fold_seconds,
    )


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

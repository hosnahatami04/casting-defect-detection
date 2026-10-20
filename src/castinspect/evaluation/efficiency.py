import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from castinspect.data.dataset import LABEL_TO_INDEX
from castinspect.data.split import ManifestRow
from castinspect.evaluation.metrics import auroc
from castinspect.models.base import ModelFactory

logger = logging.getLogger(__name__)


def subsample_defect_groups(
    rows: Sequence[ManifestRow], fraction: float, seed: int
) -> list[ManifestRow]:
    """Keep all healthy rows and roughly a fraction of defective images,
    sampled by near-duplicate group so a part never half-exists in training."""
    healthy = [r for r in rows if r.label == "ok"]
    defective = [r for r in rows if r.label == "def"]
    if fraction >= 1.0:
        return list(rows)
    groups: dict[int, list[ManifestRow]] = {}
    for row in defective:
        groups.setdefault(row.group_id, []).append(row)
    target = round(len(defective) * fraction)
    rng = np.random.default_rng(seed)
    order = rng.permutation(sorted(groups))
    kept: list[ManifestRow] = []
    for group in order:
        if len(kept) >= target:
            break
        kept.extend(groups[int(group)])
    return healthy + kept


@dataclass(frozen=True)
class EfficiencyPoint:
    fraction: float
    seed: int
    eval_fold: int
    n_defects_train: int
    auroc: float
    train_seconds: float


def run_efficiency_point(
    factory: ModelFactory,
    manifest_rows: Sequence[ManifestRow],
    image_root: Path,
    fraction: float,
    seed: int,
    n_folds: int,
    fit_kwargs: dict[str, Any] | None = None,
) -> EfficiencyPoint:
    """One training at the given defect fraction, evaluated on a held-out fold.

    The evaluation fold rotates with the seed, so the three seeds per point
    cover three different folds. A full out-of-fold run per point (5x the
    training) is deliberately not done; see ADR 0007.
    """
    eval_fold = seed % n_folds
    dev_rows = [r for r in manifest_rows if r.split == "dev"]
    train_rows = subsample_defect_groups(
        [r for r in dev_rows if r.fold != eval_fold], fraction, seed
    )
    eval_rows = [r for r in dev_rows if r.fold == eval_fold]
    n_defects = sum(1 for r in train_rows if r.label == "def")

    model = factory()
    started = time.perf_counter()
    model.fit(
        [image_root / r.path for r in train_rows],
        [LABEL_TO_INDEX[r.label] for r in train_rows],
        **(fit_kwargs or {}),
    )
    elapsed = time.perf_counter() - started
    probabilities = model.predict_proba([image_root / r.path for r in eval_rows])
    value = auroc([LABEL_TO_INDEX[r.label] for r in eval_rows], probabilities)
    logger.info(
        "fraction %.2f seed %d (fold %d): %d defects, auroc %.4f in %.0fs",
        fraction,
        seed,
        eval_fold,
        n_defects,
        value,
        elapsed,
    )
    return EfficiencyPoint(
        fraction=fraction,
        seed=seed,
        eval_fold=eval_fold,
        n_defects_train=n_defects,
        auroc=value,
        train_seconds=elapsed,
    )

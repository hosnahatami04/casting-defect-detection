"""Pure helpers behind the inspection station UI, kept apart from Streamlit so
they can be unit tested without a running app."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class BandEffect:
    threshold: float
    escapes: int
    escape_rate: float
    false_rejects: int
    false_reject_rate: float
    review_share: float


def slider_effect(
    labels: npt.ArrayLike,
    probabilities: npt.ArrayLike,
    review_threshold: float,
    reject_threshold: float,
) -> BandEffect:
    """Escape and false-reject counts the operator sees when they move the
    reject threshold, evaluated on the out-of-fold predictions."""
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    defects = int(y.sum())
    healthy = len(y) - defects
    escapes = int(np.sum((p < review_threshold) & (y == 1)))
    false_rejects = int(np.sum((p >= reject_threshold) & (y == 0)))
    reviews = int(np.sum((p >= review_threshold) & (p < reject_threshold)))
    return BandEffect(
        threshold=reject_threshold,
        escapes=escapes,
        escape_rate=escapes / defects if defects else 0.0,
        false_rejects=false_rejects,
        false_reject_rate=false_rejects / healthy if healthy else 0.0,
        review_share=reviews / len(y) if len(y) else 0.0,
    )


def load_oof_scores(path: Path) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path).to_pydict()
    return (
        np.asarray(table["label"], dtype=np.int64),
        np.asarray(table["probability"], dtype=np.float64),
    )


def sample_images(manifest_path: Path, image_root: Path, per_label: int = 4) -> list[Path]:
    from castinspect.data.split import read_manifest

    rows = [r for r in read_manifest(manifest_path) if r.split == "test"]
    picked: list[Path] = []
    for label in ("ok", "def"):
        matching = [r for r in rows if r.label == label][:per_label]
        picked.extend(image_root / r.path for r in matching)
    return picked


def read_results(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    loaded: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return loaded

from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import pytest

from castinspect.data.split import ManifestRow
from castinspect.evaluation.cv import (
    OofPrediction,
    load_oof,
    oof_arrays,
    run_cross_validation,
    save_oof,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


class RecordingModel:
    """Fake classifier that remembers its training set, to prove that no
    sample is ever predicted by a model that saw it in training."""

    name = "recording"
    instances: ClassVar[list["RecordingModel"]] = []

    def __init__(self) -> None:
        self.trained_on: set[str] = set()
        self.predicted: set[str] = set()
        RecordingModel.instances.append(self)

    def fit(self, image_paths: Sequence[Path], labels: Sequence[int]) -> None:
        self.trained_on = {p.name for p in image_paths}

    def predict_proba(self, image_paths: Sequence[Path]) -> npt.NDArray[np.float64]:
        self.predicted = {p.name for p in image_paths}
        return np.linspace(0.1, 0.9, len(image_paths))

    def save(self, path: Path) -> None:
        raise NotImplementedError

    def load(self, path: Path) -> None:
        raise NotImplementedError


def _manifest(n_folds: int = 3, per_fold: int = 4) -> list[ManifestRow]:
    rows = []
    for fold in range(n_folds):
        for i in range(per_fold):
            label = "def" if i % 2 else "ok"
            rows.append(
                ManifestRow(
                    path=f"f{fold}_{i}.png",
                    label=label,
                    group_id=fold * per_fold + i,
                    split="dev",
                    fold=fold,
                )
            )
    rows.append(ManifestRow(path="t.png", label="ok", group_id=99, split="test", fold=-1))
    return rows


def test_every_dev_sample_predicted_exactly_once(tmp_path: Path) -> None:
    RecordingModel.instances = []
    rows = _manifest()
    predictions = run_cross_validation(RecordingModel, rows, tmp_path, n_folds=3)
    dev_paths = {r.path for r in rows if r.split == "dev"}
    assert sorted(p.path for p in predictions) == sorted(dev_paths)


def test_no_model_predicts_its_own_training_data(tmp_path: Path) -> None:
    RecordingModel.instances = []
    run_cross_validation(RecordingModel, _manifest(), tmp_path, n_folds=3)
    assert len(RecordingModel.instances) == 3
    for model in RecordingModel.instances:
        assert not model.trained_on & model.predicted


def test_test_split_never_touched(tmp_path: Path) -> None:
    RecordingModel.instances = []
    run_cross_validation(RecordingModel, _manifest(), tmp_path, n_folds=3)
    for model in RecordingModel.instances:
        assert "t.png" not in model.trained_on | model.predicted


def test_oof_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    predictions = [
        OofPrediction(path="a.png", label=1, group_id=3, fold=0, probability=0.75),
        OofPrediction(path="b.png", label=0, group_id=4, fold=2, probability=0.25),
    ]
    path = tmp_path / "oof.parquet"
    save_oof(predictions, path)
    assert load_oof(path) == predictions


def test_oof_arrays() -> None:
    predictions = [
        OofPrediction(path="a.png", label=1, group_id=3, fold=0, probability=0.75),
        OofPrediction(path="b.png", label=0, group_id=4, fold=2, probability=0.25),
    ]
    labels, probabilities, groups = oof_arrays(predictions)
    assert labels.tolist() == [1, 0]
    assert probabilities.tolist() == [0.75, 0.25]
    assert groups.tolist() == [3, 4]

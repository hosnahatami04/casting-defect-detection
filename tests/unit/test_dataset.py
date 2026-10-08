from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from castinspect.data.dataset import LABEL_TO_INDEX, CastingDataset
from castinspect.data.split import ManifestRow, write_manifest


@pytest.fixture()
def dataset_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    specs = [
        ("ok_front/a.png", "ok", 0, "dev", 0),
        ("ok_front/b.png", "ok", 1, "dev", 1),
        ("def_front/c.png", "def", 2, "dev", 0),
        ("def_front/d.png", "def", 3, "test", -1),
    ]
    for rel_path, label, group, split, fold in specs:
        target = tmp_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        array = rng.integers(0, 255, size=(16, 16)).astype(np.uint8)
        Image.fromarray(array, mode="L").save(target)
        rows.append(ManifestRow(path=rel_path, label=label, group_id=group, split=split, fold=fold))
    write_manifest(rows, tmp_path / "manifest.csv")
    return tmp_path


def test_split_filter(dataset_dir: Path) -> None:
    dev = CastingDataset(dataset_dir / "manifest.csv", dataset_dir, split="dev")
    test = CastingDataset(dataset_dir / "manifest.csv", dataset_dir, split="test")
    assert len(dev) == 3
    assert len(test) == 1


def test_fold_filter(dataset_dir: Path) -> None:
    fold0 = CastingDataset(dataset_dir / "manifest.csv", dataset_dir, split="dev", folds=[0])
    assert len(fold0) == 2
    assert {row.fold for row in fold0.rows} == {0}


def test_getitem_returns_image_and_label(dataset_dir: Path) -> None:
    dataset = CastingDataset(dataset_dir / "manifest.csv", dataset_dir, split="test")
    image, label = dataset[0]
    assert isinstance(image, Image.Image)
    assert image.mode == "L"
    assert label == LABEL_TO_INDEX["def"]


def test_transform_applied(dataset_dir: Path) -> None:
    dataset = CastingDataset(
        dataset_dir / "manifest.csv",
        dataset_dir,
        split="dev",
        transform=lambda img: np.asarray(img).mean(),
    )
    value, label = dataset[0]
    assert isinstance(value, float)
    assert label in (0, 1)


def test_empty_selection_raises(dataset_dir: Path) -> None:
    with pytest.raises(ValueError, match="no manifest rows"):
        CastingDataset(dataset_dir / "manifest.csv", dataset_dir, split="dev", folds=[4])

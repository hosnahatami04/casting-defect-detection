from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.ml

SIZE = 64


def _write_images(root: Path, n_per_class: int = 8) -> tuple[list[Path], list[int]]:
    rng = np.random.default_rng(0)
    paths: list[Path] = []
    labels: list[int] = []
    root.mkdir(parents=True, exist_ok=True)
    for label in (0, 1):
        for i in range(n_per_class):
            array = rng.integers(0, 60, size=(SIZE, SIZE)).astype(np.uint8)
            if label == 1:
                array[20:40, 20:40] = 255
            path = root / f"img_{label}_{i}.png"
            Image.fromarray(array, mode="L").save(path)
            paths.append(path)
            labels.append(label)
    return paths, labels


@pytest.fixture(scope="module")
def image_data(tmp_path_factory: pytest.TempPathFactory) -> tuple[list[Path], list[int]]:
    return _write_images(tmp_path_factory.mktemp("train_images"))


def test_training_reduces_loss(image_data: tuple[list[Path], list[int]]) -> None:
    import torch

    from castinspect.models.train import (
        PathsDataset,
        TrainSettings,
        build_eval_transform,
        train_network,
    )

    paths, labels = image_data
    torch.manual_seed(0)
    network = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 5, stride=4),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(4, 1),
    )
    data = PathsDataset(paths, labels, build_eval_transform(SIZE))

    def total_loss() -> float:
        criterion = torch.nn.BCEWithLogitsLoss()
        batch = torch.stack([data[i][0] for i in range(len(data))])
        targets = torch.tensor([data[i][1] for i in range(len(data))])
        with torch.no_grad():
            return float(criterion(network(batch).squeeze(1), targets))

    before = total_loss()
    train_network(network, data, TrainSettings(max_epochs=3, batch_size=4, seed=0))
    after = total_loss()
    assert after < before


def test_train_transform_is_random_but_seedable() -> None:
    import torch

    from castinspect.models.train import build_train_transform

    rng = np.random.default_rng(1)
    image = Image.fromarray(rng.integers(0, 255, size=(SIZE, SIZE)).astype(np.uint8), mode="L")
    transform = build_train_transform(SIZE)

    torch.manual_seed(5)
    first = transform(image)
    second = transform(image)
    assert not torch.equal(first, second)

    torch.manual_seed(5)
    replay = transform(image)
    assert torch.equal(first, replay)


def test_eval_transform_deterministic_and_normalised() -> None:
    import torch

    from castinspect.models.train import IMAGENET_MEAN, IMAGENET_STD, build_eval_transform

    image = Image.new("L", (SIZE, SIZE), color=128)
    transform = build_eval_transform(SIZE)
    first = transform(image)
    second = transform(image)
    assert torch.equal(first, second)
    assert first.shape == (3, SIZE, SIZE)
    # Grayscale replicated into three channels, each normalised with its own
    # ImageNet statistics.
    for channel in range(3):
        expected = (128 / 255 - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
        assert float(first[channel, 0, 0]) == pytest.approx(expected, abs=1e-4)


def test_efficientnet_checkpoint_roundtrip(
    image_data: tuple[list[Path], list[int]], tmp_path: Path
) -> None:
    from castinspect.models.efficientnet import EfficientNetClassifier

    paths, labels = image_data
    model = EfficientNetClassifier(
        pretrained=False, image_size=SIZE, batch_size=4, max_epochs=1, seed=0
    )
    model.fit(paths, labels)
    probabilities = model.predict_proba(paths[:4])

    checkpoint = tmp_path / "model.pt"
    model.save(checkpoint)
    fresh = EfficientNetClassifier(pretrained=False, image_size=SIZE)
    fresh.load(checkpoint)
    reloaded = fresh.predict_proba(paths[:4])
    assert np.array_equal(probabilities, reloaded)


def test_early_stopping_respects_patience(image_data: tuple[list[Path], list[int]]) -> None:
    import torch

    from castinspect.models.train import (
        PathsDataset,
        TrainSettings,
        build_eval_transform,
        train_network,
    )

    paths, labels = image_data
    torch.manual_seed(0)
    # A frozen network cannot improve, so training must stop after patience.
    network = torch.nn.Sequential(
        torch.nn.Conv2d(3, 2, 7, stride=8),
        torch.nn.Flatten(),
        torch.nn.Linear(2 * 8 * 8, 1),
    )
    data = PathsDataset(paths, labels, build_eval_transform(SIZE))
    settings = TrainSettings(max_epochs=10, patience=2, batch_size=4, seed=0, learning_rate=0.0)
    result = train_network(network, data, settings, val_data=data)
    assert result.epochs_run <= 3
    assert result.best_epoch == 1

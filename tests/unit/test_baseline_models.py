from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from castinspect.evaluation.metrics import auroc

pytestmark = pytest.mark.ml

SIZE = 64


def _separable_dataset(root: Path, n_per_class: int = 24) -> tuple[list[Path], list[int]]:
    """Synthetic parts: 'ok' is a clean disc, 'defect' has a bright notch."""
    rng = np.random.default_rng(0)
    paths: list[Path] = []
    labels: list[int] = []
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    disc = ((yy - SIZE / 2) ** 2 + (xx - SIZE / 2) ** 2) < (SIZE / 3) ** 2
    for label in (0, 1):
        for i in range(n_per_class):
            image = rng.integers(0, 30, size=(SIZE, SIZE)).astype(np.uint8)
            image[disc] = 120
            if label == 1:
                r, c = rng.integers(SIZE // 2 - 8, SIZE // 2 + 8, size=2)
                image[r : r + 6, c : c + 6] = 255
            path = root / f"img_{label}_{i}.png"
            Image.fromarray(image, mode="L").save(path)
            paths.append(path)
            labels.append(label)
    return paths, labels


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[list[Path], list[int]]:
    root = tmp_path_factory.mktemp("separable")
    return _separable_dataset(root)


def _fit_and_check(model: object, dataset: tuple[list[Path], list[int]], tmp_path: Path) -> None:
    paths, labels = dataset
    model.fit(paths, labels)  # type: ignore[attr-defined]
    probabilities = model.predict_proba(paths)  # type: ignore[attr-defined]
    assert auroc(labels, probabilities) > 0.95

    saved = tmp_path / "model.pkl"
    model.save(saved)  # type: ignore[attr-defined]
    fresh = type(model)()
    fresh.load(saved)
    reloaded = fresh.predict_proba(paths)
    assert np.array_equal(probabilities, reloaded)


def test_hog_classifier_protocol(dataset: tuple[list[Path], list[int]], tmp_path: Path) -> None:
    from castinspect.models.base import DefectClassifier
    from castinspect.models.hog import HogClassifier

    model = HogClassifier(image_size=64)
    assert isinstance(model, DefectClassifier)
    _fit_and_check(model, dataset, tmp_path)


def test_resnet_probe_protocol(dataset: tuple[list[Path], list[int]], tmp_path: Path) -> None:
    from castinspect.models.base import DefectClassifier
    from castinspect.models.resnet import ResnetLinearProbe

    # Random frozen backbone: no network access, still separable features.
    model = ResnetLinearProbe(pretrained=False, image_size=64)
    assert isinstance(model, DefectClassifier)
    _fit_and_check(model, dataset, tmp_path)


def test_resnet_embedding_cache_hits(dataset: tuple[list[Path], list[int]], tmp_path: Path) -> None:
    from castinspect.models.resnet import ResnetLinearProbe

    paths, _ = dataset
    cache_dir = tmp_path / "cache"
    model = ResnetLinearProbe(pretrained=False, image_size=64, cache_dir=cache_dir)
    first = model._embed(paths[:4])
    cached_files = list(cache_dir.rglob("*.npy"))
    assert len(cached_files) == 4
    second = model._embed(paths[:4])
    assert np.array_equal(first, second)


def test_hog_deterministic(dataset: tuple[list[Path], list[int]]) -> None:
    from castinspect.models.hog import HogClassifier

    paths, labels = dataset
    first = HogClassifier(image_size=64)
    second = HogClassifier(image_size=64)
    first.fit(paths, labels)
    second.fit(paths, labels)
    assert np.array_equal(first.predict_proba(paths), second.predict_proba(paths))


def test_auroc_matches_sklearn(dataset: tuple[list[Path], list[int]]) -> None:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(3)
    labels = rng.integers(0, 2, size=200)
    labels[:3] = 1
    labels[3:6] = 0
    scores = rng.random(200)
    # Inject ties to exercise the average-rank path.
    scores[::7] = 0.5
    assert auroc(labels, scores) == pytest.approx(roc_auc_score(labels, scores))

import numpy as np
import pytest
from PIL import Image

from castinspect.deploy.preprocess import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    preprocess,
    preprocess_batch,
    sigmoid,
)


def test_shape_and_dtype() -> None:
    image = Image.new("L", (100, 80), color=128)
    array = preprocess(image)
    assert array.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert array.dtype == np.float32


def test_grayscale_replicated_and_normalised() -> None:
    image = Image.new("L", (10, 10), color=128)
    array = preprocess(image)
    for channel in range(3):
        expected = (128 / 255 - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]
        assert np.allclose(array[channel], expected, atol=1e-5)


def test_batch_stacks() -> None:
    images = [Image.new("L", (32, 32), color=c) for c in (0, 128, 255)]
    batch = preprocess_batch(images)
    assert batch.shape == (3, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_batch_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no images"):
        preprocess_batch([])


def test_sigmoid_known_values() -> None:
    assert sigmoid([0.0])[0] == pytest.approx(0.5)
    assert sigmoid([100.0])[0] == pytest.approx(1.0, abs=1e-6)
    assert sigmoid([-100.0])[0] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.ml
def test_preprocess_matches_training_transform() -> None:
    pytest.importorskip("torch")
    from castinspect.models.train import build_eval_transform

    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 255, size=(200, 160)).astype(np.uint8), mode="L")
    numpy_version = preprocess(image)
    torch_version = build_eval_transform(IMAGE_SIZE)(image).numpy()
    assert np.abs(numpy_version - torch_version).max() < 1e-4

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.ml

SIZE = 64


def test_export_and_parity_small_model(tmp_path: Path) -> None:
    """Train a tiny EfficientNet for one step, export it, and confirm PyTorch
    and ONNX agree to under 1e-4 — the parity gate, with a small model so CI
    needs no pretrained weights."""
    from castinspect.deploy.export_onnx import check_parity, export_to_onnx
    from castinspect.models.efficientnet import EfficientNetClassifier

    rng = np.random.default_rng(0)
    paths, labels = [], []
    for i in range(8):
        label = i % 2
        array = rng.integers(0, 255, size=(SIZE, SIZE)).astype(np.uint8)
        if label:
            array[20:40, 20:40] = 255
        p = tmp_path / f"img_{i}.png"
        Image.fromarray(array, mode="L").save(p)
        paths.append(p)
        labels.append(label)

    model = EfficientNetClassifier(
        pretrained=False, image_size=SIZE, batch_size=4, max_epochs=1, seed=0
    )
    model.fit(paths, labels)
    checkpoint = tmp_path / "model.pt"
    model.save(checkpoint)

    onnx_path = tmp_path / "model.onnx"
    export_to_onnx(checkpoint, onnx_path, image_size=SIZE)
    assert onnx_path.is_file()

    images = [Image.open(p).convert("L") for p in paths]
    max_diff = check_parity(checkpoint, onnx_path, images, image_size=SIZE)
    assert max_diff < 1e-4


def test_onnx_dynamic_batch(tmp_path: Path) -> None:
    from castinspect.deploy.export_onnx import export_to_onnx
    from castinspect.deploy.runtime import OnnxInspector
    from castinspect.models.efficientnet import EfficientNetClassifier

    rng = np.random.default_rng(1)
    paths, labels = [], []
    for i in range(6):
        array = rng.integers(0, 255, size=(SIZE, SIZE)).astype(np.uint8)
        p = tmp_path / f"img_{i}.png"
        Image.fromarray(array, mode="L").save(p)
        paths.append(p)
        labels.append(i % 2)
    model = EfficientNetClassifier(pretrained=False, image_size=SIZE, max_epochs=1, seed=0)
    model.fit(paths, labels)
    checkpoint = tmp_path / "m.pt"
    model.save(checkpoint)
    onnx_path = tmp_path / "m.onnx"
    export_to_onnx(checkpoint, onnx_path, image_size=SIZE)

    inspector = OnnxInspector(onnx_path, image_size=SIZE)
    images = [Image.open(p).convert("L") for p in paths]
    assert inspector.predict_proba(images[:1]).shape == (1,)
    assert inspector.predict_proba(images).shape == (6,)

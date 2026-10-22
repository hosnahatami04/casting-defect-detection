from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from tests.fixtures.tiny_onnx import write_tiny_onnx

from castinspect.deploy.preprocess import preprocess_batch, sigmoid
from castinspect.deploy.runtime import OnnxInspector

pytestmark = pytest.mark.ml


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_tiny_onnx(tmp_path_factory.mktemp("onnx") / "tiny.onnx")


def test_predict_proba_in_range(tiny_model: Path) -> None:
    inspector = OnnxInspector(tiny_model)
    images = [Image.new("L", (64, 64), color=c) for c in (10, 128, 240)]
    probs = inspector.predict_proba(images)
    assert probs.shape == (3,)
    assert np.all((probs >= 0) & (probs <= 1))
    # Brighter image -> higher mean pixel -> higher logit -> higher probability.
    assert probs[0] < probs[1] < probs[2]


def test_predict_one_matches_batch(tiny_model: Path) -> None:
    inspector = OnnxInspector(tiny_model)
    image = Image.new("L", (64, 64), color=100)
    assert inspector.predict_one(image) == pytest.approx(float(inspector.predict_proba([image])[0]))


def test_logits_match_manual_sigmoid(tiny_model: Path) -> None:
    inspector = OnnxInspector(tiny_model)
    images = [Image.new("L", (64, 64), color=c) for c in (50, 200)]
    batch = preprocess_batch(images)
    assert np.allclose(inspector.predict_proba(images), sigmoid(inspector.logits(batch)))

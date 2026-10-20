from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.ml


def _quadrant_network() -> object:
    import torch

    class QuadrantOnly(torch.nn.Module):
        """Only the top-left quadrant of the input reaches the conv layer."""

        def __init__(self) -> None:
            super().__init__()
            self.conv_head = torch.nn.Conv2d(3, 4, 3, padding=1, bias=False)
            torch.nn.init.constant_(self.conv_head.weight, 0.1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            mask = torch.zeros_like(x)
            h, w = x.shape[-2:]
            mask[..., : h // 2, : w // 2] = 1.0
            features = torch.relu(self.conv_head(x * mask))
            return features.mean(dim=(1, 2, 3)).unsqueeze(1)

    return QuadrantOnly()


def test_gradcam_shape_range_and_localisation() -> None:
    import torch

    from castinspect.explain.gradcam import GradCam

    torch.manual_seed(0)
    network = _quadrant_network()
    image = torch.rand(3, 32, 32) + 0.1
    heatmap = GradCam(network, "conv_head").heatmap(image)
    assert heatmap.shape == (32, 32)
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0
    peak_row, peak_col = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    assert peak_row < 16 and peak_col < 16
    assert heatmap[:16, :16].mean() > 5 * heatmap[16:, 16:].mean()


def test_gradcam_unknown_layer() -> None:
    from castinspect.explain.gradcam import GradCam

    with pytest.raises(ValueError, match="not found"):
        GradCam(_quadrant_network(), "missing_layer")


def test_overlay_shape() -> None:
    from castinspect.explain.gradcam import overlay

    image = Image.new("L", (40, 40), color=100)
    heatmap = np.zeros((20, 20))
    heatmap[5:10, 5:10] = 1.0
    rendered = overlay(image, heatmap)
    assert rendered.size == (40, 40)
    assert rendered.mode == "RGB"


def test_oneclass_scores_outliers_higher(tmp_path: Path) -> None:
    from castinspect.models.oneclass import OneClassKnn

    rng = np.random.default_rng(0)
    paths: list[Path] = []
    labels: list[int] = []
    for i in range(16):
        array = (120 + rng.normal(0, 4, size=(32, 32))).clip(0, 255).astype(np.uint8)
        path = tmp_path / f"healthy_{i}.png"
        Image.fromarray(array, mode="L").save(path)
        paths.append(path)
        labels.append(0)
    outliers = []
    for i in range(4):
        array = rng.integers(0, 255, size=(32, 32)).astype(np.uint8)
        path = tmp_path / f"outlier_{i}.png"
        Image.fromarray(array, mode="L").save(path)
        outliers.append(path)

    model = OneClassKnn(k=3, k_grid=(3,), pretrained=False, image_size=32)
    model.fit(paths[:12], labels[:12])
    healthy_scores = model.predict_proba(paths[12:])
    outlier_scores = model.predict_proba(outliers)
    assert outlier_scores.min() > healthy_scores.max()

    saved = tmp_path / "oneclass.pkl"
    model.save(saved)
    fresh = OneClassKnn(k=1, pretrained=False, image_size=32)
    fresh.load(saved)
    assert fresh._k == 3
    assert np.array_equal(fresh.predict_proba(outliers), outlier_scores)


def test_oneclass_scores_do_not_depend_on_batch(tmp_path: Path) -> None:
    from castinspect.models.oneclass import OneClassKnn

    rng = np.random.default_rng(2)
    paths = []
    for i in range(10):
        array = (120 + rng.normal(0, 10, size=(32, 32))).clip(0, 255).astype(np.uint8)
        path = tmp_path / f"img_{i}.png"
        Image.fromarray(array, mode="L").save(path)
        paths.append(path)
    model = OneClassKnn(k=2, k_grid=(2,), pretrained=False, image_size=32)
    model.fit(paths[:6], [0] * 6)
    together = model.predict_proba(paths[6:])
    separately = np.r_[model.predict_proba(paths[6:8]), model.predict_proba(paths[8:])]
    assert np.allclose(together, separately)


def test_oneclass_selects_k_when_defects_available(tmp_path: Path) -> None:
    from castinspect.models.oneclass import OneClassKnn

    rng = np.random.default_rng(1)
    paths: list[Path] = []
    labels: list[int] = []
    for i in range(14):
        label = int(i >= 10)
        spread = 60 if label else 4
        array = (120 + rng.normal(0, spread, size=(32, 32))).clip(0, 255).astype(np.uint8)
        path = tmp_path / f"img_{i}.png"
        Image.fromarray(array, mode="L").save(path)
        paths.append(path)
        labels.append(label)
    model = OneClassKnn(k=1, k_grid=(1, 3), pretrained=False, image_size=32)
    model.fit(paths, labels)
    assert model._k in (1, 3)
    with pytest.raises(ValueError, match="healthy"):
        OneClassKnn(pretrained=False, image_size=32).fit(paths[10:], labels[10:])

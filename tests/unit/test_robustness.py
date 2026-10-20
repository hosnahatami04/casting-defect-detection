from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from castinspect.evaluation.robustness import (
    PERTURBATION_NAMES,
    LabelledImage,
    PerturbationSpec,
    apply_perturbation,
    background_auroc,
    centre_disk_mask,
    load_perturbations,
    mask_out_part,
    max_recall_drop,
    policy_outcome,
    run_robustness,
)

SIZE = 64
# Ordered mild -> severe, as in configs/robustness.yaml; JPEG quality falls.
LEVELS = {
    "brightness": (0.1, 0.25, 0.4),
    "contrast": (0.1, 0.25, 0.4),
    "gaussian_blur": (0.5, 1.5, 3.0),
    "gaussian_noise": (4, 10, 20),
    "jpeg": (60, 35, 15),
    "translate": (0.02, 0.06, 0.12),
    "rotate": (2, 5, 10),
}


def _textured_image(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    base = 120 + 60 * np.sin(xx / 3.0) * np.cos(yy / 5.0)
    noise = rng.normal(0, 8, size=(SIZE, SIZE))
    return Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8), mode="L")


def _change(a: Image.Image, b: Image.Image) -> float:
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).mean())


@pytest.mark.parametrize("name", PERTURBATION_NAMES)
def test_zero_severity_is_identity(name: str) -> None:
    image = _textured_image()
    assert np.array_equal(np.asarray(apply_perturbation(image, name, 0)), np.asarray(image))


@pytest.mark.parametrize("name", PERTURBATION_NAMES)
def test_change_grows_with_severity(name: str) -> None:
    image = _textured_image()
    changes = [_change(image, apply_perturbation(image, name, s)) for s in LEVELS[name]]
    assert changes[0] < changes[1] < changes[2]


def test_noise_is_seedable() -> None:
    image = _textured_image()
    first = apply_perturbation(image, "gaussian_noise", 10, seed=3)
    second = apply_perturbation(image, "gaussian_noise", 10, seed=3)
    other = apply_perturbation(image, "gaussian_noise", 10, seed=4)
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert not np.array_equal(np.asarray(first), np.asarray(other))


def test_unknown_perturbation_rejected() -> None:
    with pytest.raises(ValueError, match="unknown perturbation"):
        apply_perturbation(_textured_image(), "fog", 1)


def test_load_perturbations(tmp_path: Path) -> None:
    path = tmp_path / "robustness.yaml"
    path.write_text("perturbations:\n  jpeg:\n    severities: [60, 35, 15]\nseed: 7\n")
    specs, seed = load_perturbations(path)
    assert specs == [PerturbationSpec(name="jpeg", severities=(60, 35, 15))]
    assert seed == 7
    path.write_text("perturbations:\n  fog:\n    severities: [1]\n")
    with pytest.raises(ValueError, match="unknown perturbations"):
        load_perturbations(path)


def test_repository_config_loads() -> None:
    specs, _ = load_perturbations(Path("configs/robustness.yaml"))
    assert {s.name for s in specs} == set(PERTURBATION_NAMES)
    assert all(len(s.severities) == 3 for s in specs)


def test_disk_mask_covers_centred_part_and_keeps_corners() -> None:
    size = 128
    yy, xx = np.mgrid[0:size, 0:size]
    part = ((yy - 63.5) ** 2 + (xx - 63.5) ** 2) <= (0.48 * size) ** 2
    mask = centre_disk_mask(size, size)
    assert mask[part].all()
    assert not mask[0, 0] and not mask[0, -1] and not mask[-1, 0] and not mask[-1, -1]


def test_mask_out_part_keeps_background_pixels_only() -> None:
    image = _textured_image()
    masked = np.asarray(mask_out_part(image))
    original = np.asarray(image)
    disk = centre_disk_mask(SIZE, SIZE)
    assert np.all(masked[disk] == 200)
    assert np.array_equal(masked[~disk], original[~disk])


def test_policy_outcome_hand_computed() -> None:
    labels = [1, 1, 1, 0, 0, 0]
    probabilities = [0.9, 0.3, 0.05, 0.8, 0.3, 0.01]
    outcome = policy_outcome(labels, probabilities, review_threshold=0.1, reject_threshold=0.5)
    assert outcome.escapes == 1
    assert outcome.defect_recall == pytest.approx(2 / 3)
    assert outcome.false_rejects == 1
    assert outcome.false_reject_rate == pytest.approx(1 / 3)
    assert outcome.review_share == pytest.approx(2 / 6)


def _items() -> list[LabelledImage]:
    items = []
    for i in range(12):
        label = i % 2
        array = np.full((SIZE, SIZE), 60 if label else 180, dtype=np.uint8)
        items.append(LabelledImage(fold=i % 3, label=label, image=Image.fromarray(array, mode="L")))
    return items


def _darkness_predictor(fold: int, images: Sequence[Image.Image]) -> np.ndarray:
    return np.asarray([1 - np.asarray(im, np.float64).mean() / 255 for im in images])


def test_run_robustness_rows_and_drop() -> None:
    specs = [PerturbationSpec("brightness", (0.1, 0.3, 0.6))]
    rows = run_robustness(_darkness_predictor, _items(), specs, lambda p: p, 0.3, 0.5, seed=0)
    assert [(r.perturbation, r.level) for r in rows] == [
        ("none", 0),
        ("brightness", 1),
        ("brightness", 2),
        ("brightness", 3),
    ]
    assert rows[0].outcome.defect_recall == 1.0
    perturbation, drop = max_recall_drop(rows, level=2)
    assert perturbation == "brightness"
    assert drop >= 0.0


def test_background_auroc_detects_corner_shortcut() -> None:
    items = []
    for i in range(20):
        label = i % 2
        array = np.full((SIZE, SIZE), 128, dtype=np.uint8)
        array[:4, :4] = 255 if label else 0
        items.append(LabelledImage(fold=i % 2, label=label, image=Image.fromarray(array, mode="L")))

    def corner_predictor(fold: int, images: Sequence[Image.Image]) -> np.ndarray:
        return np.asarray([np.asarray(im)[0, 0] / 255 for im in images], dtype=np.float64)

    def centre_predictor(fold: int, images: Sequence[Image.Image]) -> np.ndarray:
        return np.asarray(
            [np.asarray(im)[SIZE // 2, SIZE // 2] / 255 for im in images], dtype=np.float64
        )

    assert background_auroc(corner_predictor, items) == 1.0
    assert background_auroc(centre_predictor, items) == pytest.approx(0.5)

import io
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import yaml
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

PERTURBATION_NAMES = (
    "brightness",
    "contrast",
    "gaussian_blur",
    "gaussian_noise",
    "jpeg",
    "translate",
    "rotate",
)

# Rotation and translation expose corners; fill with the conveyor grey so the
# perturbation itself does not paint artificial black edges.
_FILL = 200


def apply_perturbation(
    image: Image.Image, name: str, severity: float, seed: int = 42
) -> Image.Image:
    if severity == 0:
        return image.copy()
    if name == "brightness":
        return ImageEnhance.Brightness(image).enhance(1.0 - severity)
    if name == "contrast":
        return ImageEnhance.Contrast(image).enhance(1.0 - severity)
    if name == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=severity))
    if name == "gaussian_noise":
        rng = np.random.default_rng(seed)
        array = np.asarray(image, dtype=np.float64)
        noisy = array + rng.normal(0.0, severity, size=array.shape)
        return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), mode=image.mode)
    if name == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(severity))
        buffer.seek(0)
        with Image.open(buffer) as reloaded:
            reloaded.load()
            return reloaded.convert(image.mode)
    if name == "translate":
        shift = round(severity * image.width)
        return image.transform(
            image.size,
            Image.Transform.AFFINE,
            (1, 0, -shift, 0, 1, -shift),
            resample=Image.Resampling.BILINEAR,
            fillcolor=_FILL,
        )
    if name == "rotate":
        return image.rotate(float(severity), resample=Image.Resampling.BILINEAR, fillcolor=_FILL)
    raise ValueError(f"unknown perturbation {name!r}")


@dataclass(frozen=True)
class PerturbationSpec:
    name: str
    severities: tuple[float, ...]


def load_perturbations(path: Path) -> tuple[list[PerturbationSpec], int]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    specs = [
        PerturbationSpec(name=name, severities=tuple(entry["severities"]))
        for name, entry in raw["perturbations"].items()
    ]
    unknown = [s.name for s in specs if s.name not in PERTURBATION_NAMES]
    if unknown:
        raise ValueError(f"unknown perturbations in config: {unknown}")
    return specs, int(raw.get("seed", 42))


@dataclass(frozen=True)
class PolicyOutcome:
    """Decision policy counts over a labelled population."""

    n_defects: int
    n_ok: int
    escapes: int
    defect_recall: float
    false_rejects: int
    false_reject_rate: float
    review_share: float


def policy_outcome(
    labels: npt.ArrayLike,
    probabilities: npt.ArrayLike,
    review_threshold: float,
    reject_threshold: float,
) -> PolicyOutcome:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    defects = int(y.sum())
    healthy = len(y) - defects
    escapes = int(np.sum((p < review_threshold) & (y == 1)))
    false_rejects = int(np.sum((p >= reject_threshold) & (y == 0)))
    reviews = int(np.sum((p >= review_threshold) & (p < reject_threshold)))
    return PolicyOutcome(
        n_defects=defects,
        n_ok=healthy,
        escapes=escapes,
        defect_recall=(defects - escapes) / defects if defects else 1.0,
        false_rejects=false_rejects,
        false_reject_rate=false_rejects / healthy if healthy else 0.0,
        review_share=reviews / len(y),
    )


FoldPredictor = Callable[[int, Sequence[Image.Image]], npt.NDArray[np.float64]]


@dataclass(frozen=True)
class LabelledImage:
    fold: int
    label: int
    image: Image.Image


@dataclass(frozen=True)
class RobustnessRow:
    perturbation: str
    level: int
    severity: float
    outcome: PolicyOutcome
    auroc: float


def _predict_by_fold(
    predict: FoldPredictor, items: Sequence[LabelledImage], images: Sequence[Image.Image]
) -> npt.NDArray[np.float64]:
    """Each image is scored by the model of the fold that held it out."""
    scores = np.empty(len(items), dtype=np.float64)
    for fold in sorted({item.fold for item in items}):
        indices = [i for i, item in enumerate(items) if item.fold == fold]
        scores[indices] = predict(fold, [images[i] for i in indices])
    return scores


def run_robustness(
    predict: FoldPredictor,
    items: Sequence[LabelledImage],
    specs: Sequence[PerturbationSpec],
    calibrate: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    review_threshold: float,
    reject_threshold: float,
    seed: int,
) -> list[RobustnessRow]:
    from castinspect.evaluation.metrics import auroc

    labels = np.asarray([item.label for item in items], dtype=np.int64)

    def evaluate(
        name: str, level: int, severity: float, images: Sequence[Image.Image]
    ) -> RobustnessRow:
        probabilities = calibrate(_predict_by_fold(predict, items, images))
        outcome = policy_outcome(labels, probabilities, review_threshold, reject_threshold)
        logger.info(
            "%s level %d: recall %.4f, false rejects %d",
            name,
            level,
            outcome.defect_recall,
            outcome.false_rejects,
        )
        return RobustnessRow(name, level, severity, outcome, auroc(labels, probabilities))

    rows = [evaluate("none", 0, 0.0, [item.image for item in items])]
    for spec in specs:
        for level, severity in enumerate(spec.severities, start=1):
            perturbed = [
                apply_perturbation(item.image, spec.name, severity, seed=seed + index)
                for index, item in enumerate(items)
            ]
            rows.append(evaluate(spec.name, level, severity, perturbed))
    return rows


def max_recall_drop(rows: Sequence[RobustnessRow], level: int = 2) -> tuple[str, float]:
    """Largest drop in defect recall, in percentage points, at one severity
    level; level 2 is the medium severity the KPI is defined on."""
    baseline = next(r for r in rows if r.perturbation == "none").outcome.defect_recall
    candidates = [r for r in rows if r.level == level]
    worst = max(candidates, key=lambda r: baseline - r.outcome.defect_recall)
    return worst.perturbation, 100.0 * (baseline - worst.outcome.defect_recall)


def background_auroc(predict: FoldPredictor, items: Sequence[LabelledImage]) -> float:
    from castinspect.evaluation.metrics import auroc

    masked = [mask_out_part(item.image) for item in items]
    scores = _predict_by_fold(predict, items, masked)
    return auroc([item.label for item in items], scores)


SHORTCUT_DISK_FRACTION = 0.56


def centre_disk_mask(
    height: int, width: int, radius_fraction: float = SHORTCUT_DISK_FRACTION
) -> npt.NDArray[np.bool_]:
    """Centred disk with radius radius_fraction times the shorter side.

    A fixed disk rather than a per-image part detector: on this dataset every
    detector tried (intensity, radial profile, Hough circles, texture) left rim
    or burrs visible on some images or erased nearly all background on others,
    and either failure corrupts the shortcut check. At 0.56 the disk contains
    the part on all but the very largest framings and keeps ~10% of the image
    as pure corner background (ADR 0007).
    """
    radius = radius_fraction * min(height, width)
    yy, xx = np.mgrid[0:height, 0:width]
    mask = ((yy - (height - 1) / 2) ** 2 + (xx - (width - 1) / 2) ** 2) <= radius**2
    return np.asarray(mask, dtype=np.bool_)


def mask_out_part(
    image: Image.Image,
    radius_fraction: float = SHORTCUT_DISK_FRACTION,
    fill: int = _FILL,
) -> Image.Image:
    """Background-only version of the image: the central disk holding the part
    is painted over, so any remaining signal must come from the corners."""
    array = np.asarray(image.convert("L")).copy()
    array[centre_disk_mask(array.shape[0], array.shape[1], radius_fraction)] = fill
    return Image.fromarray(array, mode="L").convert(image.mode)

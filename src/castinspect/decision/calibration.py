from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar

FloatArray = npt.NDArray[np.float64]

_PROB_EPS = 1e-7
_TEMPERATURE_BOUNDS = (0.05, 20.0)


def probabilities_to_logits(probabilities: npt.ArrayLike) -> FloatArray:
    p = np.clip(np.asarray(probabilities, dtype=np.float64), _PROB_EPS, 1 - _PROB_EPS)
    return np.asarray(np.log(p / (1 - p)), dtype=np.float64)


def apply_temperature(logits: npt.ArrayLike, temperature: float) -> FloatArray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    return np.asarray(1.0 / (1.0 + np.exp(-scaled)), dtype=np.float64)


def negative_log_likelihood(
    labels: npt.ArrayLike, logits: npt.ArrayLike, temperature: float
) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = np.clip(apply_temperature(logits, temperature), _PROB_EPS, 1 - _PROB_EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(labels: npt.ArrayLike, logits: npt.ArrayLike) -> float:
    result = minimize_scalar(
        lambda t: negative_log_likelihood(labels, logits, float(t)),
        bounds=_TEMPERATURE_BOUNDS,
        method="bounded",
    )
    return float(result.x)


@dataclass(frozen=True)
class ReliabilityBin:
    mean_probability: float
    fraction_defective: float
    count: int


def reliability_bins(
    labels: npt.ArrayLike, probabilities: npt.ArrayLike, n_bins: int = 15
) -> list[ReliabilityBin]:
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    for low, high in pairwise(edges):
        mask = (p >= low) & (p < high) if high < 1.0 else (p >= low) & (p <= high)
        count = int(mask.sum())
        if count == 0:
            continue
        bins.append(
            ReliabilityBin(
                mean_probability=float(p[mask].mean()),
                fraction_defective=float(y[mask].mean()),
                count=count,
            )
        )
    return bins


def expected_calibration_error(
    labels: npt.ArrayLike, probabilities: npt.ArrayLike, n_bins: int = 15
) -> float:
    """Positive-class formulation: per bin, |fraction defective - mean predicted
    probability|, weighted by bin size."""
    total = len(np.asarray(labels))
    bins = reliability_bins(labels, probabilities, n_bins)
    return float(
        sum(abs(b.fraction_defective - b.mean_probability) * b.count / total for b in bins)
    )

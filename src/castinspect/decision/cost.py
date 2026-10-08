from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class CostModel:
    escape: float = 20.0
    false_reject: float = 1.0
    manual_review: float = 0.5


def three_way_cost(
    labels: npt.ArrayLike,
    probabilities: npt.ArrayLike,
    review_threshold: float,
    reject_threshold: float,
    costs: CostModel,
) -> float:
    """Expected cost per part. Manual review is assumed to catch what it sees,
    so a reviewed defect costs only the review itself."""
    if review_threshold > reject_threshold:
        raise ValueError("review threshold must not exceed reject threshold")
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    passed = p < review_threshold
    reviewed = (p >= review_threshold) & (p < reject_threshold)
    rejected = p >= reject_threshold
    escapes = int(np.sum(passed & (y == 1)))
    false_rejects = int(np.sum(rejected & (y == 0)))
    reviews = int(reviewed.sum())
    total = (
        escapes * costs.escape + false_rejects * costs.false_reject + reviews * costs.manual_review
    )
    return float(total / len(y))


def two_way_cost(
    labels: npt.ArrayLike,
    probabilities: npt.ArrayLike,
    threshold: float,
    costs: CostModel,
) -> float:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    escapes = int(np.sum((p < threshold) & (y == 1)))
    false_rejects = int(np.sum((p >= threshold) & (y == 0)))
    return float((escapes * costs.escape + false_rejects * costs.false_reject) / len(y))


def two_way_cost_curve(
    labels: npt.ArrayLike, probabilities: npt.ArrayLike, costs: CostModel
) -> tuple[FloatArray, FloatArray]:
    """Expected two-way cost at every candidate threshold (unique scores plus
    the trivial accept-all and reject-all endpoints)."""
    p = np.asarray(probabilities, dtype=np.float64)
    thresholds = np.unique(np.r_[0.0, p, 1.0 + 1e-9])
    values = np.asarray(
        [two_way_cost(labels, probabilities, float(t), costs) for t in thresholds],
        dtype=np.float64,
    )
    return thresholds, values


def best_two_way(
    labels: npt.ArrayLike, probabilities: npt.ArrayLike, costs: CostModel
) -> tuple[float, float]:
    thresholds, values = two_way_cost_curve(labels, probabilities, costs)
    index = int(np.argmin(values))
    return float(thresholds[index]), float(values[index])

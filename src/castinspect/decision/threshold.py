from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from castinspect.decision.cost import CostModel, three_way_cost

FloatArray = npt.NDArray[np.float64]


def _validate(
    labels: npt.ArrayLike, probabilities: npt.ArrayLike
) -> tuple[npt.NDArray[np.int64], FloatArray]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("labels and probabilities must be 1-d arrays of equal length")
    if y.sum() == 0:
        raise ValueError("no defective samples to place thresholds on")
    return y, p


def reject_threshold(
    labels: npt.ArrayLike, probabilities: npt.ArrayLike, recall_target: float
) -> float:
    """Highest threshold whose hard-reject recall still meets the target, so the
    recall constraint binds with the fewest false rejects."""
    y, p = _validate(labels, probabilities)
    defect_scores = np.sort(p[y == 1])
    n_defects = len(defect_scores)
    max_missed = int(np.floor((1.0 - recall_target) * n_defects))
    # Rejecting at the score of the (max_missed+1)-th lowest defect keeps at
    # least recall_target of defects at or above the threshold.
    return float(defect_scores[max_missed])


def review_threshold(
    labels: npt.ArrayLike, probabilities: npt.ArrayLike, max_escape: float
) -> float:
    """Highest threshold whose automatic-pass escape share stays at or below
    max_escape; everything at or above it is at least sent to review."""
    y, p = _validate(labels, probabilities)
    defect_scores = np.sort(p[y == 1])
    n_defects = len(defect_scores)
    max_escaped = int(np.floor(max_escape * n_defects))
    return float(defect_scores[max_escaped])


def cost_aware_review_threshold(
    labels: npt.ArrayLike,
    probabilities: npt.ArrayLike,
    reject_thr: float,
    max_escape: float,
    costs: CostModel,
) -> float:
    """Cheapest review threshold among those meeting the escape constraint.

    Escape share is monotone in the threshold, so every candidate at or below
    review_threshold(max_escape) satisfies the constraint; among them the
    expected cost trades escaped defects against review load.
    """
    y, p = _validate(labels, probabilities)
    upper = min(review_threshold(labels, probabilities, max_escape), reject_thr)
    candidates = np.unique(np.r_[0.0, p[p <= upper], upper])
    best_threshold = float(upper)
    best_cost = three_way_cost(y, p, best_threshold, reject_thr, costs)
    # Descending, so ties keep the highest threshold (smallest review band).
    for candidate in sorted((float(c) for c in candidates), reverse=True):
        value = three_way_cost(y, p, candidate, reject_thr, costs)
        if value < best_cost:
            best_cost = value
            best_threshold = candidate
    return best_threshold


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    defect_recall: float
    defects_caught: int
    false_reject_rate: float
    false_rejects: int


def operating_point(
    labels: npt.ArrayLike, probabilities: npt.ArrayLike, threshold: float
) -> OperatingPoint:
    y, p = _validate(labels, probabilities)
    rejected = p >= threshold
    defects = int(y.sum())
    healthy = len(y) - defects
    caught = int(np.sum(rejected & (y == 1)))
    false_rejects = int(np.sum(rejected & (y == 0)))
    return OperatingPoint(
        threshold=float(threshold),
        defect_recall=caught / defects,
        defects_caught=caught,
        false_reject_rate=false_rejects / healthy if healthy else 0.0,
        false_rejects=false_rejects,
    )


def operating_point_table(
    labels: npt.ArrayLike,
    probabilities: npt.ArrayLike,
    recall_targets: list[float],
) -> list[OperatingPoint]:
    return [
        operating_point(labels, probabilities, reject_threshold(labels, probabilities, target))
        for target in recall_targets
    ]

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class ConfusionMatrix:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int


@dataclass(frozen=True)
class RocCurve:
    # Points include the (0, 0) origin; thresholds[i] is the lowest score
    # still classified positive at point i.
    fpr: FloatArray
    tpr: FloatArray
    thresholds: FloatArray


def _validate(labels: npt.ArrayLike, scores: npt.ArrayLike) -> tuple[IntArray, FloatArray]:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("labels and scores must be 1-d arrays of equal length")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be 0 (ok) or 1 (defect)")
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("both classes must be present")
    return y, s


def _average_ranks(sorted_values: FloatArray) -> FloatArray:
    n = len(sorted_values)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    _, starts, counts = np.unique(sorted_values, return_index=True, return_counts=True)
    for start, count in zip(starts, counts, strict=True):
        if count > 1:
            ranks[start : start + count] = ranks[start : start + count].mean()
    return ranks


def auroc(labels: npt.ArrayLike, scores: npt.ArrayLike) -> float:
    y, s = _validate(labels, scores)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = _average_ranks(s[order])
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def roc_curve(labels: npt.ArrayLike, scores: npt.ArrayLike) -> RocCurve:
    y, s = _validate(labels, scores)
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    # Keep one ROC point per distinct score value.
    last_of_value = np.nonzero(np.diff(s_sorted))[0]
    cut = np.r_[last_of_value, len(s_sorted) - 1]
    tps = np.cumsum(y_sorted)[cut].astype(np.float64)
    fps = np.cumsum(1 - y_sorted)[cut].astype(np.float64)
    tpr = np.r_[0.0, tps / tps[-1]]
    fpr = np.r_[0.0, fps / fps[-1]]
    thresholds = np.r_[np.inf, s_sorted[cut]]
    return RocCurve(fpr=fpr, tpr=tpr, thresholds=thresholds)


def recall_at_fpr(labels: npt.ArrayLike, scores: npt.ArrayLike, max_fpr: float) -> float:
    curve = roc_curve(labels, scores)
    feasible = curve.fpr <= max_fpr
    return float(curve.tpr[feasible].max())


def fpr_at_recall(labels: npt.ArrayLike, scores: npt.ArrayLike, min_recall: float) -> float:
    curve = roc_curve(labels, scores)
    feasible = curve.tpr >= min_recall
    return float(curve.fpr[feasible].min())


def threshold_at_recall(labels: npt.ArrayLike, scores: npt.ArrayLike, min_recall: float) -> float:
    """Highest score threshold whose recall still reaches min_recall."""
    curve = roc_curve(labels, scores)
    feasible = np.nonzero(curve.tpr >= min_recall)[0]
    return float(curve.thresholds[feasible[0]])


def confusion_at_threshold(
    labels: npt.ArrayLike, scores: npt.ArrayLike, threshold: float
) -> ConfusionMatrix:
    y, s = _validate(labels, scores)
    predicted = s >= threshold
    return ConfusionMatrix(
        true_positive=int(np.sum(predicted & (y == 1))),
        false_positive=int(np.sum(predicted & (y == 0))),
        true_negative=int(np.sum(~predicted & (y == 0))),
        false_negative=int(np.sum(~predicted & (y == 1))),
    )

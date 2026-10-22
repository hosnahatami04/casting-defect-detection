"""Final test-set evaluation. This is the one place the test split is read.

Everything here is a pure function of (labels, calibrated probabilities,
frozen thresholds); IO and model inference stay in the CLI command. Every rate
is reported with its raw count and an exact Clopper-Pearson interval.
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from castinspect.evaluation.intervals import Interval, bootstrap_auroc, clopper_pearson
from castinspect.evaluation.metrics import auroc


@dataclass
class RateResult:
    numerator: int
    denominator: int
    point: float
    lower: float
    upper: float

    @property
    def raw(self) -> str:
        return f"{self.numerator}/{self.denominator}"


def _rate(numerator: int, denominator: int) -> RateResult:
    interval = clopper_pearson(numerator, denominator) if denominator else Interval(0.0, 0.0, 0.0)
    return RateResult(
        numerator=numerator,
        denominator=denominator,
        point=interval.point,
        lower=interval.lower,
        upper=interval.upper,
    )


@dataclass
class FinalMetrics:
    n: int
    n_defect: int
    n_ok: int
    auroc: Interval
    ece: float
    defect_recall: RateResult
    escape_rate: RateResult
    false_reject_rate: RateResult
    manual_review_rate: RateResult
    confusion: dict[str, int] = field(default_factory=dict)


def evaluate_test(
    labels: npt.ArrayLike,
    calibrated: npt.ArrayLike,
    groups: npt.ArrayLike,
    review_threshold: float,
    reject_threshold: float,
    ece_bins: int = 15,
) -> FinalMetrics:
    from castinspect.decision.calibration import expected_calibration_error

    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(calibrated, dtype=np.float64)
    g = np.asarray(groups, dtype=np.int64)
    n_defect = int(y.sum())
    n_ok = len(y) - n_defect

    passed = p < review_threshold
    reviewed = (p >= review_threshold) & (p < reject_threshold)
    rejected = p >= reject_threshold

    escapes = int(np.sum(passed & (y == 1)))
    caught = n_defect - escapes
    false_rejects = int(np.sum(rejected & (y == 0)))
    reviews = int(reviewed.sum())

    return FinalMetrics(
        n=len(y),
        n_defect=n_defect,
        n_ok=n_ok,
        auroc=bootstrap_auroc(y, p, g),
        ece=expected_calibration_error(y, p, ece_bins),
        defect_recall=_rate(caught, n_defect),
        escape_rate=_rate(escapes, n_defect),
        false_reject_rate=_rate(false_rejects, n_ok),
        manual_review_rate=_rate(reviews, len(y)),
        confusion={
            "pass_ok": int(np.sum(passed & (y == 0))),
            "pass_defect_escape": escapes,
            "review_ok": int(np.sum(reviewed & (y == 0))),
            "review_defect": int(np.sum(reviewed & (y == 1))),
            "reject_ok_false": false_rejects,
            "reject_defect": int(np.sum(rejected & (y == 1))),
        },
    )


def baseline_auroc(labels: npt.ArrayLike, probabilities: npt.ArrayLike) -> float:
    return auroc(labels, probabilities)


def metrics_to_dict(metrics: FinalMetrics) -> dict[str, object]:
    def rate(r: RateResult) -> dict[str, object]:
        return {"point": r.point, "lower": r.lower, "upper": r.upper, "raw": r.raw}

    return {
        "n": metrics.n,
        "n_defect": metrics.n_defect,
        "n_ok": metrics.n_ok,
        "auroc": {
            "point": metrics.auroc.point,
            "lower": metrics.auroc.lower,
            "upper": metrics.auroc.upper,
        },
        "ece": metrics.ece,
        "defect_recall": rate(metrics.defect_recall),
        "escape_rate": rate(metrics.escape_rate),
        "false_reject_rate": rate(metrics.false_reject_rate),
        "manual_review_rate": rate(metrics.manual_review_rate),
        "confusion": metrics.confusion,
    }

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import beta

from castinspect.evaluation.metrics import auroc


@dataclass(frozen=True)
class Interval:
    point: float
    lower: float
    upper: float


def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> Interval:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between 0 and trials")
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    )
    return Interval(point=successes / trials, lower=lower, upper=upper)


def bootstrap_auroc(
    labels: npt.ArrayLike,
    scores: npt.ArrayLike,
    group_ids: npt.ArrayLike,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Interval:
    """Percentile bootstrap resampling whole groups, so correlated
    near-duplicates stay together in every resample."""
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    g = np.asarray(group_ids, dtype=np.int64)
    if not (len(y) == len(s) == len(g)):
        raise ValueError("labels, scores and group_ids must have equal length")
    members: dict[int, np.ndarray] = {group: np.nonzero(g == group)[0] for group in np.unique(g)}
    unique_groups = np.asarray(sorted(members))
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    while len(estimates) < n_resamples:
        chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([members[int(group)] for group in chosen])
        if y[idx].sum() in (0, len(idx)):
            continue
        estimates.append(auroc(y[idx], s[idx]))
    alpha = 1.0 - confidence
    lower, upper = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return Interval(point=auroc(y, s), lower=float(lower), upper=float(upper))

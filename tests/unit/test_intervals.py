import numpy as np
import pytest

from castinspect.evaluation.intervals import bootstrap_auroc, clopper_pearson


def test_zero_successes() -> None:
    interval = clopper_pearson(0, 60)
    assert interval.point == 0.0
    assert interval.lower == 0.0
    # Closed form for k=0: upper = 1 - (alpha/2)^(1/n).
    assert interval.upper == pytest.approx(1 - 0.025 ** (1 / 60))


def test_all_successes() -> None:
    interval = clopper_pearson(60, 60)
    assert interval.point == 1.0
    assert interval.upper == 1.0
    assert interval.lower == pytest.approx(0.025 ** (1 / 60))


def test_reference_value() -> None:
    # 2 escapes out of 156; bounds checked against the exact beta quantiles.
    interval = clopper_pearson(2, 156)
    assert interval.point == pytest.approx(2 / 156)
    assert interval.lower == pytest.approx(0.001556, abs=1e-5)
    assert interval.upper == pytest.approx(0.045543, abs=1e-5)


def test_interval_contains_point() -> None:
    for successes, trials in ((1, 10), (5, 10), (9, 10), (50, 200)):
        interval = clopper_pearson(successes, trials)
        assert interval.lower <= interval.point <= interval.upper


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        clopper_pearson(1, 0)
    with pytest.raises(ValueError):
        clopper_pearson(5, 4)


def test_bootstrap_auroc_separable() -> None:
    rng = np.random.default_rng(0)
    labels = np.r_[np.zeros(60, dtype=int), np.ones(60, dtype=int)]
    scores = np.r_[rng.normal(0.2, 0.05, 60), rng.normal(0.8, 0.05, 60)]
    groups = np.arange(120)
    interval = bootstrap_auroc(labels, scores, groups, n_resamples=200, seed=0)
    assert interval.point == pytest.approx(1.0)
    assert interval.lower > 0.95
    assert interval.lower <= interval.point <= interval.upper


def test_bootstrap_auroc_deterministic() -> None:
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, size=80)
    labels[:5] = 1
    labels[5:10] = 0
    scores = rng.random(80)
    groups = rng.integers(0, 30, size=80)
    first = bootstrap_auroc(labels, scores, groups, n_resamples=100, seed=7)
    second = bootstrap_auroc(labels, scores, groups, n_resamples=100, seed=7)
    assert first == second

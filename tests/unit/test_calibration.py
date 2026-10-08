import numpy as np
import pytest

from castinspect.decision.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    negative_log_likelihood,
    probabilities_to_logits,
    reliability_bins,
)


def test_logit_roundtrip() -> None:
    probabilities = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
    recovered = apply_temperature(probabilities_to_logits(probabilities), 1.0)
    assert np.allclose(recovered, probabilities, atol=1e-9)


def test_ece_hand_computed() -> None:
    # Two populated bins of two samples each: |0.5-0.15| and |0.5-0.85|,
    # equally weighted.
    labels = [0, 1, 0, 1]
    probabilities = [0.1, 0.2, 0.8, 0.9]
    assert expected_calibration_error(labels, probabilities, n_bins=2) == pytest.approx(0.35)


def test_ece_perfectly_calibrated_is_small() -> None:
    rng = np.random.default_rng(0)
    probabilities = rng.uniform(0, 1, size=20000)
    labels = (rng.uniform(0, 1, size=20000) < probabilities).astype(int)
    assert expected_calibration_error(labels, probabilities) < 0.02


def test_temperature_preserves_ranking() -> None:
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 3, size=100)
    scaled = apply_temperature(logits, 2.7)
    assert np.array_equal(np.argsort(scaled), np.argsort(logits))


def test_fitted_temperature_does_not_increase_nll() -> None:
    rng = np.random.default_rng(2)
    # Overconfident scores: true probabilities shrunk toward 0.5 before sampling.
    logits = rng.normal(0, 4, size=2000)
    true_p = apply_temperature(logits, 3.0)
    labels = (rng.uniform(0, 1, size=2000) < true_p).astype(int)
    fitted = fit_temperature(labels, logits)
    assert negative_log_likelihood(labels, logits, fitted) <= negative_log_likelihood(
        labels, logits, 1.0
    )
    assert fitted > 1.5


def test_reliability_bins_counts_sum_to_total() -> None:
    rng = np.random.default_rng(3)
    probabilities = rng.uniform(0, 1, size=500)
    labels = rng.integers(0, 2, size=500)
    bins = reliability_bins(labels, probabilities)
    assert sum(b.count for b in bins) == 500


def test_apply_temperature_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="positive"):
        apply_temperature(np.array([0.0]), 0.0)

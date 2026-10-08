import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from castinspect.decision.cost import CostModel, three_way_cost
from castinspect.decision.threshold import (
    cost_aware_review_threshold,
    reject_threshold,
    review_threshold,
)


@st.composite
def _scored_population(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    n = draw(st.integers(min_value=20, max_value=200))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    labels[: max(2, n // 10)] = 1
    probabilities = rng.uniform(0, 1, size=n)
    return labels, probabilities


@given(_scored_population(), st.sampled_from([0.9, 0.95, 0.99, 1.0]))
def test_reject_threshold_meets_recall_target(
    population: tuple[np.ndarray, np.ndarray], target: float
) -> None:
    labels, probabilities = population
    threshold = reject_threshold(labels, probabilities, target)
    defects = probabilities[labels == 1]
    recall = float((defects >= threshold).mean())
    assert recall >= target


@given(_scored_population(), st.sampled_from([(0.9, 0.99), (0.95, 0.99), (0.9, 1.0)]))
def test_higher_recall_target_never_raises_threshold(
    population: tuple[np.ndarray, np.ndarray], targets: tuple[float, float]
) -> None:
    labels, probabilities = population
    lower_target, higher_target = targets
    assert reject_threshold(labels, probabilities, higher_target) <= reject_threshold(
        labels, probabilities, lower_target
    )


@given(_scored_population(), st.sampled_from([0.0, 0.005, 0.01, 0.05]))
def test_review_threshold_meets_escape_target(
    population: tuple[np.ndarray, np.ndarray], max_escape: float
) -> None:
    labels, probabilities = population
    threshold = review_threshold(labels, probabilities, max_escape)
    defects = probabilities[labels == 1]
    escapes = float((defects < threshold).mean())
    assert escapes <= max_escape


@given(_scored_population())
def test_review_never_exceeds_reject_for_project_targets(
    population: tuple[np.ndarray, np.ndarray],
) -> None:
    labels, probabilities = population
    t_reject = reject_threshold(labels, probabilities, 0.99)
    t_review = review_threshold(labels, probabilities, 0.005)
    assert t_review <= t_reject


@given(_scored_population())
def test_cost_aware_review_meets_constraints_and_never_costs_more(
    population: tuple[np.ndarray, np.ndarray],
) -> None:
    labels, probabilities = population
    costs = CostModel(escape=20.0, false_reject=1.0, manual_review=0.5)
    max_escape = 0.005
    t_reject = reject_threshold(labels, probabilities, 0.99)
    chosen = cost_aware_review_threshold(labels, probabilities, t_reject, max_escape, costs)
    assert chosen <= t_reject
    defects = probabilities[labels == 1]
    assert float((defects < chosen).mean()) <= max_escape
    naive = min(review_threshold(labels, probabilities, max_escape), t_reject)
    assert three_way_cost(labels, probabilities, chosen, t_reject, costs) <= three_way_cost(
        labels, probabilities, naive, t_reject, costs
    )

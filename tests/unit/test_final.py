import numpy as np

from castinspect.evaluation.final import evaluate_test, metrics_to_dict


def test_evaluate_test_hand_computed() -> None:
    # 4 defects, 4 ok. review=0.2, reject=0.6.
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    probs = [0.9, 0.5, 0.3, 0.05, 0.7, 0.3, 0.1, 0.01]
    groups = list(range(8))
    m = evaluate_test(labels, probs, groups, 0.2, 0.6)
    assert m.n == 8 and m.n_defect == 4 and m.n_ok == 4
    # One defect at 0.05 < review -> escape; the other three are caught.
    assert m.escape_rate.numerator == 1
    assert m.defect_recall.numerator == 3
    # One ok at 0.7 >= reject -> false reject.
    assert m.false_reject_rate.numerator == 1
    # review band [0.2, 0.6): defect 0.3, defect 0.5, ok 0.3 -> 3 reviewed.
    assert m.manual_review_rate.numerator == 3


def test_confusion_sums_to_total() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=100)
    labels[:5] = 1
    labels[5:10] = 0
    probs = rng.uniform(0, 1, size=100)
    groups = rng.integers(0, 40, size=100)
    m = evaluate_test(labels, probs, groups, 0.3, 0.7)
    assert sum(m.confusion.values()) == 100


def test_rates_carry_intervals_containing_point() -> None:
    labels = [1, 1, 1, 0, 0, 0]
    probs = [0.9, 0.8, 0.1, 0.2, 0.1, 0.05]
    m = evaluate_test(labels, probs, list(range(6)), 0.3, 0.6)
    for rate in (m.defect_recall, m.escape_rate, m.false_reject_rate, m.manual_review_rate):
        assert rate.lower <= rate.point <= rate.upper


def test_metrics_to_dict_shape() -> None:
    labels = [1, 1, 0, 0]
    probs = [0.9, 0.3, 0.2, 0.05]
    d = metrics_to_dict(evaluate_test(labels, probs, list(range(4)), 0.25, 0.6))
    assert set(d) >= {"auroc", "ece", "defect_recall", "escape_rate", "confusion"}
    # Both defects are caught (0.9 rejected, 0.3 in the review band): recall 2/2.
    assert d["defect_recall"]["raw"] == "2/2"
    assert d["escape_rate"]["raw"] == "0/2"

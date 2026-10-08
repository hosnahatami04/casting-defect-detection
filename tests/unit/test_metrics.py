import numpy as np
import pytest

from castinspect.evaluation.metrics import (
    auroc,
    confusion_at_threshold,
    fpr_at_recall,
    recall_at_fpr,
    roc_curve,
    threshold_at_recall,
)

# Hand-worked example: 4 defects, 4 ok parts.
LABELS = [1, 1, 1, 1, 0, 0, 0, 0]
SCORES = [0.9, 0.8, 0.6, 0.3, 0.7, 0.4, 0.2, 0.1]


def test_auroc_hand_computed() -> None:
    # Pair count with defect scored above ok: 13 of 16 pairs.
    assert auroc(LABELS, SCORES) == pytest.approx(13 / 16)


def test_auroc_perfect_and_inverted() -> None:
    labels = [0, 0, 1, 1]
    assert auroc(labels, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc(labels, [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_auroc_ties_give_half_credit() -> None:
    assert auroc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)


def test_auroc_rejects_single_class() -> None:
    with pytest.raises(ValueError, match="both classes"):
        auroc([1, 1], [0.1, 0.2])


def test_roc_curve_monotone_and_anchored() -> None:
    curve = roc_curve(LABELS, SCORES)
    assert curve.fpr[0] == 0.0 and curve.tpr[0] == 0.0
    assert curve.fpr[-1] == 1.0 and curve.tpr[-1] == 1.0
    assert np.all(np.diff(curve.fpr) >= 0)
    assert np.all(np.diff(curve.tpr) >= 0)


def test_recall_at_fpr_hand_computed() -> None:
    # At FPR 0 the best recall is 0.5 (two defects above every ok score).
    assert recall_at_fpr(LABELS, SCORES, 0.0) == pytest.approx(0.5)
    # Allowing one false positive (FPR 0.25) catches three of four defects.
    assert recall_at_fpr(LABELS, SCORES, 0.25) == pytest.approx(0.75)


def test_fpr_at_recall_hand_computed() -> None:
    # Full recall requires threshold 0.3, admitting the 0.7 and 0.4 ok parts.
    assert fpr_at_recall(LABELS, SCORES, 1.0) == pytest.approx(0.5)
    assert fpr_at_recall(LABELS, SCORES, 0.75) == pytest.approx(0.25)


def test_threshold_at_recall_reaches_target() -> None:
    threshold = threshold_at_recall(LABELS, SCORES, 0.75)
    labels = np.asarray(LABELS)
    scores = np.asarray(SCORES)
    recall = (scores[labels == 1] >= threshold).mean()
    assert recall >= 0.75
    assert threshold == pytest.approx(0.6)


def test_confusion_at_threshold() -> None:
    confusion = confusion_at_threshold(LABELS, SCORES, 0.5)
    assert confusion.true_positive == 3
    assert confusion.false_negative == 1
    assert confusion.false_positive == 1
    assert confusion.true_negative == 3

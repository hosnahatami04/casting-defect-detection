"""KPI smoke gate: run the final-evaluation pipeline on a tiny synthetic
dataset and a small model, and check the metrics are well-formed and sane.
This is the CI gate that proves the evaluation path works without needing
pretrained weights or the real dataset. Thresholds live in
tests/fixtures/kpi_smoke.json.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.ml

SIZE = 48
FIXTURE = Path("tests/fixtures/kpi_smoke.json")


def _synthetic(root: Path, n: int = 40) -> tuple[list[Path], list[int], list[int]]:
    rng = np.random.default_rng(0)
    paths, labels, groups = [], [], []
    for i in range(n):
        label = i % 2
        array = rng.integers(60, 120, size=(SIZE, SIZE)).astype(np.uint8)
        if label:
            array[16:32, 16:32] = 255
        p = root / f"img_{i}.png"
        Image.fromarray(array, mode="L").save(p)
        paths.append(p)
        labels.append(label)
        groups.append(i)
    return paths, labels, groups


def test_kpi_smoke_gate(tmp_path: Path) -> None:
    from castinspect.decision.calibration import (
        apply_temperature,
        fit_temperature,
        probabilities_to_logits,
    )
    from castinspect.decision.threshold import reject_threshold, review_threshold
    from castinspect.evaluation.final import evaluate_test, metrics_to_dict
    from castinspect.models.efficientnet import EfficientNetClassifier

    bounds = json.loads(FIXTURE.read_text(encoding="utf-8"))
    paths, labels, groups = _synthetic(tmp_path)

    model = EfficientNetClassifier(
        pretrained=False, image_size=SIZE, batch_size=8, max_epochs=3, seed=0
    )
    model.fit(paths, labels)
    probs = model.predict_proba(paths)

    logits = probabilities_to_logits(probs)
    temperature = fit_temperature(labels, logits)
    calibrated = apply_temperature(logits, temperature)
    t_reject = reject_threshold(labels, calibrated, 0.99)
    t_review = min(review_threshold(labels, calibrated, 0.005), t_reject)

    metrics = metrics_to_dict(
        evaluate_test(labels, calibrated, groups, t_review, t_reject, ece_bins=5)
    )

    for key in bounds["required_keys"]:
        assert key in metrics
    assert metrics["auroc"]["point"] >= bounds["min_auroc"]
    assert metrics["ece"] <= bounds["max_ece"]
    assert metrics["defect_recall"]["point"] >= bounds["min_defect_recall"]
    assert metrics["escape_rate"]["point"] <= bounds["max_escape_rate"]
    assert sum(metrics["confusion"].values()) == len(labels)

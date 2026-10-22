import json
from pathlib import Path

import numpy as np

from castinspect.ui.logic import read_results, slider_effect


def test_slider_effect_hand_computed() -> None:
    labels = [1, 1, 1, 0, 0, 0]
    probabilities = [0.9, 0.4, 0.05, 0.8, 0.4, 0.01]
    effect = slider_effect(labels, probabilities, review_threshold=0.1, reject_threshold=0.5)
    assert effect.escapes == 1
    assert effect.escape_rate == 1 / 3
    assert effect.false_rejects == 1
    assert effect.false_reject_rate == 1 / 3
    assert effect.review_share == 2 / 6


def test_slider_effect_lower_reject_more_false_rejects() -> None:
    rng = np.random.default_rng(0)
    labels = np.r_[np.ones(50), np.zeros(50)].astype(int)
    probabilities = np.r_[rng.uniform(0.4, 1.0, 50), rng.uniform(0.0, 0.6, 50)]
    low = slider_effect(labels, probabilities, 0.1, 0.3)
    high = slider_effect(labels, probabilities, 0.1, 0.8)
    assert low.false_rejects >= high.false_rejects


def test_read_results_missing(tmp_path: Path) -> None:
    assert read_results(tmp_path / "none.json") == {}


def test_read_results_present(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"auroc": 0.98}))
    assert read_results(path) == {"auroc": 0.98}

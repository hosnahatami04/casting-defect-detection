import json
from pathlib import Path

import pytest

from castinspect.evaluation.report import (
    render_efficiency_curve,
    render_reliability,
    render_robustness_bars,
)

pytestmark = pytest.mark.ml


def test_render_efficiency_curve(tmp_path: Path) -> None:
    data = {
        "points": [
            {"fraction": f, "seed": s, "auroc": 0.9 + 0.05 * f, "n_defects_train": int(500 * f)}
            for f in (0.1, 0.5, 1.0)
            for s in range(3)
        ],
        "oneclass_knn": {"auroc": {"point": 0.92, "lower": 0.9, "upper": 0.94}},
    }
    source = tmp_path / "efficiency.json"
    source.write_text(json.dumps(data))
    output = tmp_path / "curve.png"
    render_efficiency_curve(source, output)
    assert output.is_file() and output.stat().st_size > 0


def test_render_robustness_bars(tmp_path: Path) -> None:
    rows = [{"perturbation": "none", "level": 0, "defect_recall": 1.0}]
    for name in ("brightness", "jpeg"):
        for level in (1, 2, 3):
            rows.append({"perturbation": name, "level": level, "defect_recall": 1.0 - 0.02 * level})
    source = tmp_path / "robustness.json"
    source.write_text(json.dumps({"rows": rows}))
    output = tmp_path / "bars.png"
    render_robustness_bars(source, output)
    assert output.is_file() and output.stat().st_size > 0


def test_render_reliability(tmp_path: Path) -> None:
    bins = [{"mean_probability": 0.2, "fraction_defective": 0.25, "count": 10}]
    data = {
        "temperature": 1.4,
        "ece_before": 0.02,
        "ece_after": 0.015,
        "reliability_before": bins,
        "reliability_after": bins,
    }
    source = tmp_path / "calibration.json"
    source.write_text(json.dumps(data))
    output = tmp_path / "reliability.png"
    render_reliability(source, output)
    assert output.is_file() and output.stat().st_size > 0

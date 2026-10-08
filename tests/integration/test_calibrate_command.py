from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from castinspect.cli import app

pytestmark = pytest.mark.ml

runner = CliRunner()


def _write_oof(artifacts: Path) -> None:
    from castinspect.evaluation.cv import OofPrediction, save_oof

    rng = np.random.default_rng(0)
    predictions = []
    for i in range(400):
        label = i % 2
        center = 0.8 if label else 0.15
        probability = float(np.clip(rng.normal(center, 0.1), 0.001, 0.999))
        predictions.append(
            OofPrediction(
                path=f"img_{i}.png", label=label, group_id=i, fold=i % 5, probability=probability
            )
        )
    save_oof(predictions, artifacts / "oof" / "efficientnet_b0.parquet")


def _write_config(tmp_path: Path) -> Path:
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "paths:",
                f"  data_raw: {(tmp_path / 'raw').as_posix()}",
                f"  data_processed: {(tmp_path / 'processed').as_posix()}",
                f"  artifacts: {(tmp_path / 'artifacts').as_posix()}",
                f"  reports: {(tmp_path / 'reports').as_posix()}",
            ]
        )
    )
    return config


def test_calibrate_freezes_policy_without_reading_any_manifest(tmp_path: Path) -> None:
    # Only the OOF parquet exists: no manifest, no test split, no images.
    # Succeeding here proves calibration cannot have touched the test data.
    _write_oof(tmp_path / "artifacts")
    config = _write_config(tmp_path)
    policy_path = tmp_path / "policy.yaml"

    result = runner.invoke(
        app, ["calibrate", "--config", str(config), "--policy-path", str(policy_path)]
    )

    assert result.exit_code == 0, result.output
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    assert 0.0 <= policy["thresholds"]["review"] <= policy["thresholds"]["reject"] <= 1.0
    assert policy["temperature"] > 0
    assert (tmp_path / "artifacts" / "calibration" / "efficientnet_b0_calibration.json").is_file()


def test_calibrated_thresholds_meet_targets_on_oof(tmp_path: Path) -> None:
    from castinspect.decision.calibration import apply_temperature, probabilities_to_logits
    from castinspect.evaluation.cv import load_oof, oof_arrays

    _write_oof(tmp_path / "artifacts")
    config = _write_config(tmp_path)
    policy_path = tmp_path / "policy.yaml"
    result = runner.invoke(
        app, ["calibrate", "--config", str(config), "--policy-path", str(policy_path)]
    )
    assert result.exit_code == 0, result.output

    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    labels, probabilities, _ = oof_arrays(
        load_oof(tmp_path / "artifacts" / "oof" / "efficientnet_b0.parquet")
    )
    calibrated = apply_temperature(probabilities_to_logits(probabilities), policy["temperature"])
    defects = calibrated[labels == 1]
    recall = float((defects >= policy["thresholds"]["reject"]).mean())
    escapes = float((defects < policy["thresholds"]["review"]).mean())
    assert recall >= policy["recall_target"]
    assert escapes <= policy["max_escape"]

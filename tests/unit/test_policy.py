from pathlib import Path

import pytest

from castinspect.decision.cost import CostModel, best_two_way, three_way_cost, two_way_cost
from castinspect.decision.policy import DecisionPolicy, PolicyConfig, load_policy

POLICY = DecisionPolicy(review_threshold=0.2, reject_threshold=0.7, version="1")


def test_three_intervals_map_to_actions() -> None:
    assert POLICY.decide(0.05).action == "pass"
    assert POLICY.decide(0.5).action == "manual_review"
    assert POLICY.decide(0.9).action == "reject"


def test_boundary_values_exact() -> None:
    # p >= reject -> reject; review <= p < reject -> manual_review; p < review -> pass.
    assert POLICY.decide(0.7).action == "reject"
    assert POLICY.decide(0.2).action == "manual_review"
    assert POLICY.decide(0.19999).action == "pass"


def test_decision_carries_version_and_reason() -> None:
    decision = POLICY.decide(0.5)
    assert decision.thresholds_version == "1"
    assert "review band" in decision.reason
    assert decision.defect_probability == 0.5


def test_invalid_probability_rejected() -> None:
    with pytest.raises(ValueError, match="out of range"):
        POLICY.decide(1.2)


def test_invalid_thresholds_rejected() -> None:
    with pytest.raises(ValueError, match="review <= reject"):
        DecisionPolicy(review_threshold=0.8, reject_threshold=0.3, version="1")


def test_policy_config_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text(
        "\n".join(
            [
                "version: '1'",
                "model: efficientnet_b0",
                "temperature: 2.5",
                "thresholds:",
                "  review: 0.1",
                "  reject: 0.4",
                "recall_target: 0.99",
                "max_escape: 0.005",
                "oof_source: artifacts/oof/efficientnet_b0.parquet",
            ]
        )
    )
    config = load_policy(path)
    assert isinstance(config, PolicyConfig)
    policy = DecisionPolicy.from_config(config)
    assert policy.decide(0.25).action == "manual_review"


def test_missing_policy_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="calibrate"):
        load_policy(tmp_path / "missing.yaml")


def test_three_way_cost_hand_computed() -> None:
    costs = CostModel(escape=20.0, false_reject=1.0, manual_review=0.5)
    labels = [1, 1, 0, 0]
    probabilities = [0.05, 0.5, 0.5, 0.9]
    # One escaped defect (20), one reviewed defect and one reviewed ok (2 x 0.5),
    # one falsely rejected ok part (1): total 22 over 4 parts.
    value = three_way_cost(labels, probabilities, 0.2, 0.7, costs)
    assert value == pytest.approx(22 / 4)


def test_two_way_cost_hand_computed() -> None:
    costs = CostModel(escape=20.0, false_reject=1.0, manual_review=0.5)
    labels = [1, 1, 0, 0]
    probabilities = [0.05, 0.5, 0.5, 0.9]
    # Threshold 0.4: one escape (20) and two rejected ok parts (2).
    assert two_way_cost(labels, probabilities, 0.4, costs) == pytest.approx(22 / 4)


def test_best_two_way_finds_minimum() -> None:
    costs = CostModel(escape=20.0, false_reject=1.0, manual_review=0.5)
    labels = [1, 1, 1, 0, 0, 0]
    probabilities = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    threshold, value = best_two_way(labels, probabilities, costs)
    # Perfectly separable: zero cost somewhere between 0.3 and 0.7.
    assert value == 0.0
    assert 0.3 < threshold <= 0.7


def test_three_way_rejects_inverted_thresholds() -> None:
    costs = CostModel()
    with pytest.raises(ValueError, match="must not exceed"):
        three_way_cost([1, 0], [0.5, 0.5], 0.8, 0.2, costs)

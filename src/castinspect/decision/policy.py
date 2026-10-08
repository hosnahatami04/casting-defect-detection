from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

Action = Literal["pass", "reject", "manual_review"]

DEFAULT_POLICY_PATH = Path("configs/policy.yaml")


@dataclass(frozen=True)
class Decision:
    action: Action
    defect_probability: float
    thresholds_version: str
    reason: str


class PolicyThresholds(BaseModel):
    review: float
    reject: float


class PolicyConfig(BaseModel):
    version: str
    model: str
    temperature: float
    thresholds: PolicyThresholds
    recall_target: float
    max_escape: float
    oof_source: str


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> PolicyConfig:
    if not path.is_file():
        raise FileNotFoundError(f"policy file not found: {path}; run castinspect calibrate")
    return PolicyConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class DecisionPolicy:
    def __init__(self, review_threshold: float, reject_threshold: float, version: str) -> None:
        if not 0.0 <= review_threshold <= reject_threshold <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= review <= reject <= 1")
        self._review = review_threshold
        self._reject = reject_threshold
        self._version = version

    @classmethod
    def from_config(cls, config: PolicyConfig) -> "DecisionPolicy":
        return cls(
            review_threshold=config.thresholds.review,
            reject_threshold=config.thresholds.reject,
            version=config.version,
        )

    def decide(self, defect_probability: float) -> Decision:
        p = float(defect_probability)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"defect probability out of range: {p}")
        if p >= self._reject:
            action: Action = "reject"
            reason = f"probability {p:.4f} >= reject threshold {self._reject:.4f}"
        elif p >= self._review:
            action = "manual_review"
            reason = f"probability {p:.4f} in review band [{self._review:.4f}, {self._reject:.4f})"
        else:
            action = "pass"
            reason = f"probability {p:.4f} < review threshold {self._review:.4f}"
        return Decision(
            action=action,
            defect_probability=p,
            thresholds_version=self._version,
            reason=reason,
        )

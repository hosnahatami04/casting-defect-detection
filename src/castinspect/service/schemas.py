from typing import Literal

from pydantic import BaseModel, Field


class InspectResponse(BaseModel):
    action: Literal["pass", "reject", "manual_review"]
    defect_probability: float = Field(ge=0.0, le=1.0)
    model_version: str
    policy_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str
    policy_version: str

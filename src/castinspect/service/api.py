import io
import logging
import time
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from PIL import Image, UnidentifiedImageError

from castinspect.decision.calibration import apply_temperature, probabilities_to_logits
from castinspect.decision.policy import DecisionPolicy, load_policy
from castinspect.deploy.runtime import OnnxInspector
from castinspect.service.schemas import HealthResponse, InspectResponse

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 16 * 1024 * 1024
MAX_SIDE = 4096


class AppState:
    def __init__(self, onnx_path: Path, policy_path: Path, threads: int = 1) -> None:
        self.policy_config = load_policy(policy_path)
        self.policy = DecisionPolicy.from_config(self.policy_config)
        self.inspector = OnnxInspector(onnx_path, threads=threads)
        self.temperature = self.policy_config.temperature
        self.model_version = self.policy_config.model
        self.policy_version = self.policy_config.version
        self.action_counts: Counter[str] = Counter()
        self.latencies_ms: list[float] = []

    def infer(self, image: Image.Image) -> InspectResponse:
        started = time.perf_counter()
        raw = self.inspector.predict_one(image)
        calibrated = float(apply_temperature(probabilities_to_logits([raw]), self.temperature)[0])
        decision = self.policy.decide(calibrated)
        latency_ms = (time.perf_counter() - started) * 1000
        self.action_counts[decision.action] += 1
        self.latencies_ms.append(latency_ms)
        return InspectResponse(
            action=decision.action,
            defect_probability=calibrated,
            model_version=self.model_version,
            policy_version=self.policy_version,
            latency_ms=round(latency_ms, 2),
        )


def _load_image(data: bytes) -> Image.Image:
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=422, detail="image exceeds the size limit")
    if not data:
        raise HTTPException(status_code=422, detail="empty upload")
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            image = img.convert("L")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=422, detail=f"not a readable image: {exc}") from exc
    if max(image.size) > MAX_SIDE:
        raise HTTPException(status_code=422, detail="image dimensions exceed the limit")
    return image


def create_app(onnx_path: Path, policy_path: Path, threads: int = 1) -> FastAPI:
    app = FastAPI(title="castinspect", version="1.0.0")
    state = AppState(onnx_path, policy_path, threads=threads)
    app.state.inspector_state = state

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_version=state.model_version,
            policy_version=state.policy_version,
        )

    @app.post("/inspect", response_model=InspectResponse)
    async def inspect(file: UploadFile = File(...)) -> InspectResponse:  # noqa: B008
        return state.infer(_load_image(await file.read()))

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        lines = [
            "# HELP castinspect_decisions_total Decisions by action.",
            "# TYPE castinspect_decisions_total counter",
        ]
        for action in ("pass", "reject", "manual_review"):
            lines.append(
                f'castinspect_decisions_total{{action="{action}"}} {state.action_counts[action]}'
            )
        latencies = sorted(state.latencies_ms)
        if latencies:
            p50 = latencies[len(latencies) // 2]
            p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
            lines += [
                "# HELP castinspect_latency_ms End-to-end inference latency.",
                "# TYPE castinspect_latency_ms summary",
                f'castinspect_latency_ms{{quantile="0.5"}} {p50:.2f}',
                f'castinspect_latency_ms{{quantile="0.95"}} {p95:.2f}',
                f"castinspect_latency_ms_count {len(latencies)}",
            ]
        return "\n".join(lines) + "\n"

    return app

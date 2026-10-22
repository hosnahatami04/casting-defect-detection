import io
from pathlib import Path

import pytest
import yaml
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("onnxruntime")

from fastapi.testclient import TestClient
from tests.fixtures.tiny_onnx import write_tiny_onnx

from castinspect.service.api import AppState, create_app

pytestmark = pytest.mark.ml


def _png_bytes(color: int, size: int = 64) -> bytes:
    buffer = io.BytesIO()
    Image.new("L", (size, size), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("api")
    onnx_path = write_tiny_onnx(root / "tiny.onnx")
    policy_path = root / "policy.yaml"
    policy_path.write_text(
        yaml.safe_dump(
            {
                "version": "1",
                "model": "efficientnet_b0",
                "temperature": 1.2,
                "thresholds": {"review": 0.3, "reject": 0.7},
                "recall_target": 0.99,
                "max_escape": 0.005,
                "oof_source": "artifacts/oof/efficientnet_b0.parquet",
            }
        )
    )
    return onnx_path, policy_path


@pytest.fixture()
def client(setup: tuple[Path, Path]) -> TestClient:
    return TestClient(create_app(*setup))


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["policy_version"] == "1"


def test_inspect_valid_image(client: TestClient) -> None:
    response = client.post("/inspect", files={"file": ("part.png", _png_bytes(200), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["action"] in ("pass", "reject", "manual_review")
    assert 0.0 <= body["defect_probability"] <= 1.0
    assert body["model_version"] == "efficientnet_b0"
    assert body["latency_ms"] >= 0


def test_inspect_rejects_non_image(client: TestClient) -> None:
    response = client.post("/inspect", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert response.status_code == 422
    assert "readable image" in response.json()["detail"]


def test_inspect_rejects_empty(client: TestClient) -> None:
    response = client.post("/inspect", files={"file": ("x.png", b"", "image/png")})
    assert response.status_code == 422


def test_inspect_rejects_oversized_dimensions(client: TestClient) -> None:
    response = client.post(
        "/inspect", files={"file": ("big.png", _png_bytes(128, size=5000), "image/png")}
    )
    assert response.status_code == 422


def test_api_matches_direct_library_call(setup: tuple[Path, Path], client: TestClient) -> None:
    onnx_path, policy_path = setup
    state = AppState(onnx_path, policy_path)
    image = Image.new("L", (64, 64), color=175)
    direct = state.infer(image)
    response = client.post("/inspect", files={"file": ("part.png", _png_bytes(175), "image/png")})
    body = response.json()
    assert body["action"] == direct.action
    assert body["defect_probability"] == pytest.approx(direct.defect_probability, abs=1e-6)


def test_metrics_counts_and_latency(client: TestClient) -> None:
    for color in (10, 128, 240):
        client.post("/inspect", files={"file": ("p.png", _png_bytes(color), "image/png")})
    response = client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    assert "castinspect_decisions_total" in text
    assert 'castinspect_latency_ms{quantile="0.5"}' in text

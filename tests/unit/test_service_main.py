from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")
pytest.importorskip("onnxruntime")

from tests.fixtures.tiny_onnx import write_tiny_onnx

pytestmark = pytest.mark.ml


def test_app_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from castinspect.service.main import app_from_env

    onnx_path = write_tiny_onnx(tmp_path / "tiny.onnx")
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        yaml.safe_dump(
            {
                "version": "2",
                "model": "efficientnet_b0",
                "temperature": 1.0,
                "thresholds": {"review": 0.3, "reject": 0.7},
                "recall_target": 0.99,
                "max_escape": 0.005,
                "oof_source": "x",
            }
        )
    )
    monkeypatch.setenv("CASTINSPECT_ONNX", str(onnx_path))
    monkeypatch.setenv("CASTINSPECT_POLICY", str(policy_path))
    monkeypatch.setenv("CASTINSPECT_THREADS", "2")

    client = TestClient(app_from_env())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["policy_version"] == "2"

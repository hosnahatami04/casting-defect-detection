"""End-to-end /inspect latency over 500 in-process requests via the FastAPI
test client (no network hop, so this isolates server-side cost). Writes
artifacts/benchmarks/api_latency.json.
"""

import io
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from fastapi.testclient import TestClient
from PIL import Image

from castinspect.config import load_settings
from castinspect.data.acquire import locate_image_root
from castinspect.data.split import read_manifest
from castinspect.service.api import create_app

N = 500


def main() -> None:
    settings = load_settings(Path("configs/default.yaml"))
    onnx_path = settings.paths.artifacts / "onnx" / "efficientnet_b0.onnx"
    client = TestClient(create_app(onnx_path, Path("configs/policy.yaml")))

    root = locate_image_root(settings.paths.data_raw)
    rows = [r for r in read_manifest(settings.paths.data_processed / "manifest.csv")][:N]
    payloads = []
    for r in rows:
        buffer = io.BytesIO()
        with Image.open(root / r.path) as img:
            img.convert("L").save(buffer, format="PNG")
        payloads.append(buffer.getvalue())

    for data in payloads[:20]:
        client.post("/inspect", files={"file": ("p.png", data, "image/png")})

    latencies = []
    for data in payloads:
        t0 = time.perf_counter()
        response = client.post("/inspect", files={"file": ("p.png", data, "image/png")})
        latencies.append((time.perf_counter() - t0) * 1000)
        assert response.status_code == 200

    result = {
        "n": len(latencies),
        "inspect_p50_ms": round(statistics.median(latencies), 2),
        "inspect_p95_ms": round(statistics.quantiles(latencies, n=20)[18], 2),
    }
    out = settings.paths.artifacts / "benchmarks" / "api_latency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

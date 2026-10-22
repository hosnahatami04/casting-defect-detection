"""ONNX Runtime latency and throughput, plus end-to-end /inspect latency.
Writes artifacts/benchmarks/onnx_latency.json. Single thread, batch 1, as the
deployment target is a cheap CPU on the line.
"""

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from PIL import Image

from castinspect.config import load_settings
from castinspect.data.acquire import locate_image_root
from castinspect.data.split import read_manifest
from castinspect.deploy.runtime import OnnxInspector

N = 500


def main() -> None:
    settings = load_settings(Path("configs/default.yaml"))
    onnx_path = settings.paths.artifacts / "onnx" / "efficientnet_b0.onnx"
    root = locate_image_root(settings.paths.data_raw)
    rows = [r for r in read_manifest(settings.paths.data_processed / "manifest.csv")]
    paths = [root / r.path for r in rows[:N]]
    images = [Image.open(p).convert("L") for p in paths]

    inspector = OnnxInspector(onnx_path, threads=1)
    for image in images[:20]:
        inspector.predict_one(image)  # warm up

    single = []
    start_all = time.perf_counter()
    for image in images:
        t0 = time.perf_counter()
        inspector.predict_one(image)
        single.append((time.perf_counter() - t0) * 1000)
    throughput = len(images) / (time.perf_counter() - start_all)

    result = {
        "n": len(images),
        "onnx_p50_ms": round(statistics.median(single), 2),
        "onnx_p95_ms": round(statistics.quantiles(single, n=20)[18], 2),
        "throughput_per_s": round(throughput, 1),
    }
    out = settings.paths.artifacts / "benchmarks" / "onnx_latency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

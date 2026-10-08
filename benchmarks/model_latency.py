"""Single-image PyTorch latency and model size for the three model tiers.

Fits the two baselines on a seeded subset (their fitted size is what matters,
not their quality here) and loads the final EfficientNet checkpoint. Writes
artifacts/benchmarks/pytorch_latency.json.
"""

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from castinspect.config import load_settings
from castinspect.data.acquire import locate_image_root
from castinspect.data.dataset import LABEL_TO_INDEX
from castinspect.data.split import read_manifest
from castinspect.models.efficientnet import EfficientNetClassifier
from castinspect.models.hog import HogClassifier
from castinspect.models.resnet import ResnetLinearProbe

N_TIMED = 50
N_FIT = 200


def time_single_image(model: object, paths: list[Path]) -> dict[str, float]:
    samples = []
    for path in paths:
        started = time.perf_counter()
        model.predict_proba([path])  # type: ignore[attr-defined]
        samples.append((time.perf_counter() - started) * 1000)
    return {
        "p50_ms": round(statistics.median(samples), 1),
        "p95_ms": round(statistics.quantiles(samples, n=20)[18], 1),
    }


def file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / 1_000_000, 2)


def main() -> None:
    settings = load_settings(Path("configs/default.yaml"))
    root = locate_image_root(settings.paths.data_raw)
    manifest = [r for r in read_manifest(settings.paths.data_processed / "manifest.csv")]
    dev = [r for r in manifest if r.split == "dev"]
    ok_rows = [r for r in dev if r.label == "ok"]
    def_rows = [r for r in dev if r.label == "def"]
    fit_rows = ok_rows[: N_FIT // 2] + def_rows[: N_FIT // 2]
    timed_rows = ok_rows[N_FIT // 2 : N_FIT // 2 + N_TIMED // 2]
    timed_rows += def_rows[N_FIT // 2 : N_FIT // 2 + N_TIMED // 2]
    timed_paths = [root / r.path for r in timed_rows]
    fit_paths = [root / r.path for r in fit_rows]
    fit_labels = [LABEL_TO_INDEX[r.label] for r in fit_rows]

    out_dir = settings.paths.artifacts / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, float]] = {}

    hog = HogClassifier()
    hog.fit(fit_paths, fit_labels)
    hog_path = out_dir / "hog.pkl"
    hog.save(hog_path)
    results["hog"] = {**time_single_image(hog, timed_paths), "size_mb": file_size_mb(hog_path)}

    probe = ResnetLinearProbe(weights_sha256="", cache_dir=None)
    probe.fit(fit_paths, fit_labels)
    probe_path = out_dir / "resnet18_probe.pkl"
    probe.save(probe_path)
    # The probe's size is dominated by the frozen backbone it needs at runtime.
    backbone_mb = 44.7
    results["resnet18_probe"] = {
        **time_single_image(probe, timed_paths),
        "size_mb": round(file_size_mb(probe_path) + backbone_mb, 2),
    }

    final_path = settings.paths.artifacts / "models" / "efficientnet_b0_final.pt"
    effnet = EfficientNetClassifier(pretrained=False)
    effnet.load(final_path)
    results["efficientnet_b0"] = {
        **time_single_image(effnet, timed_paths),
        "size_mb": file_size_mb(final_path),
    }

    output = out_dir / "pytorch_latency.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

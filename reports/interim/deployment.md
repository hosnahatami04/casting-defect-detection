# Deployment: ONNX export, service and latency

ONNX export of the final EfficientNet-B0, served by ONNX Runtime on CPU behind a FastAPI
inspection endpoint, with a Streamlit inspection station. Numbers from
`artifacts/benchmarks/` and `castinspect export`.

## ONNX export and parity

| metric | value | target |
|---|---|---|
| max \|PyTorch − ONNX\| probability, 200 images | 3.0e-06 | ≤ 1e-4 |
| ONNX file size | 16.0 MB | ≤ 25 MB |
| opset | 17 (dynamic batch axis) | — |

The graph ends at the raw logit; sigmoid and the three-way policy are applied outside, so
one ONNX file serves any downstream threshold. Preprocessing is a single shared numpy
module (`deploy/preprocess.py`) called by the API, the runtime and the parity test, pinned
against the training transform to under 1e-4.

## Latency (CPU, 1 thread, batch 1, 500 images)

| metric | p50 | p95 | target |
|---|---|---|---|
| ONNX Runtime, per image | 11.0 ms | 11.8 ms | ≤ 50 ms (p95) |
| /inspect end-to-end | 15.8 ms | 16.5 ms | ≤ 120 ms (p95) |

ONNX throughput: 90 images/s single-threaded.

## Service

- `POST /inspect`: image upload, validated for type, size (≤ 16 MB) and dimensions
  (≤ 4096 px); returns action, calibrated defect probability, model and policy versions,
  and latency. Invalid input returns 422 with a message.
- `GET /health`: liveness plus model and policy versions.
- `GET /metrics`: Prometheus-style action counters and a latency summary.
- The API image installs base + deploy + serve only (no torch); the ONNX model and policy
  are mounted read-only at run time.

## Inspection station (Streamlit)

Upload an image, see the three-way decision and probability, and move a reject-threshold
slider whose effect on escapes, false rejects and review load is read off the out-of-fold
predictions. A second page shows the evaluation figures.

## docker compose

Two-stage build (api, ui targets), both with HEALTHCHECK, ONNX model mounted from a volume.
`docker compose config` validates. The up/healthy check needs a running Docker daemon and
is a manual step.

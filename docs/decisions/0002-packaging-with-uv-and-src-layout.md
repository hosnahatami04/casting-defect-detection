# 0002 — Packaging with uv, src layout and dependency groups

## Context

The project must install reproducibly on CPU-only machines, keep heavy ML dependencies
out of the deployment path, and avoid accidental imports of the package from the
repository root during tests.

## Decision

- Python 3.11, managed with `uv` and a committed `uv.lock`; `pip install -e ".[dev]"`
  stays supported through standard optional dependencies.
- `src/` layout with a single package `castinspect`, so tests import the installed
  package rather than the working directory.
- Optional dependency groups: base runtime (numpy, Pillow, pydantic, typer), `ml`
  (torch, torchvision, timm, scikit-learn, scikit-image, matplotlib), `deploy`
  (onnx, onnxruntime), `dev` (test and lint tooling). The service image installs only
  base plus `deploy`.

## Rejected alternatives

- Flat layout: allows importing uninstalled code, which hides packaging bugs.
- Poetry: heavier resolver, no builtin Python toolchain management.
- A single dependency list: forces torch into the deployment image.

## Consequences

Contributors need `uv` (or plain pip with extras); in exchange installs are locked and
the ONNX service image stays small.

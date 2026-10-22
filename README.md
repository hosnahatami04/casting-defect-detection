# casting-defect-detection

Visual inspection for metal castings with a cost-aware pass, reject or manual-review
decision, tested against simulated camera conditions.

![ci](https://github.com/hosnahatami04/casting-defect-detection/actions/workflows/ci.yml/badge.svg)

## Why

On an inspection line an escaped defect costs far more than a good part wrongly rejected,
so raw accuracy is the wrong headline. This project builds a detector for cast pump
impellers and, more importantly, chooses a defensible operating point from the line's cost
structure: it holds defect recall at the target and reports exactly what gets through, with
raw counts and confidence intervals.

On the held-out test split (261 images, evaluated once under the frozen policy): **zero
automatic escapes (0/157)**, defect recall **100% (157/157)**, false reject rate **1.9%
(2/104)**, and ONNX inference at **11.8 ms p95** on CPU. Full numbers, with intervals and
provenance hashes, are in [reports/results.md](reports/results.md).

This is a personal project on public data, with no affiliation to any employer.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the offline and inference paths and
the diagram. In short: data is cleaned, grouped by perceptual hash and split without
leakage; three model tiers are compared with grouped cross-validation; a cost-aware
three-way policy is frozen on out-of-fold predictions; and the final model is served as
ONNX behind a validated API.

## Installation (users)

1. Download the Kaggle dataset "Casting product image data for quality inspection"
   (`ravirajsinh45/real-life-industrial-dataset-of-casting-product`) and place the archive
   in `data/raw/`.
2. Build and run the stack (needs Docker):

   ```bash
   uv run castinspect export        # writes artifacts/onnx/efficientnet_b0.onnx
   docker compose up --build
   ```

3. Open the inspection station at `http://localhost:8501`; the API is at
   `http://localhost:8000` (`/health`, `/inspect`, `/metrics`).

## Installation (developers)

```bash
uv sync --extra dev --extra ml --extra deploy --extra serve --extra ui
uv run pre-commit install

make test          # fast tests
make test-ml       # heavy tests (model, ONNX, service)
make data          # verify and process the dataset (after placing the archive)
make cov           # coverage
```

Reproduce the pipeline:

```bash
uv run castinspect cv configs/models/efficientnet_b0.yaml --save-folds
uv run castinspect train-final configs/models/efficientnet_b0.yaml
uv run castinspect calibrate
uv run castinspect evaluate      # test split, once
```

Layout: `src/castinspect/` holds `data/`, `models/`, `evaluation/`, `decision/`,
`explain/`, `deploy/`, `service/`, `ui/`. Decisions are recorded as ADRs in
`docs/decisions/`.

## Contributing

- Conventional Commits; one branch and one PR per change.
- `make lint typecheck test` must pass; coverage stays at or above 85%.
- Record non-trivial decisions as an ADR.
- Never tune thresholds on the test split. Report every rate with its raw count and a
  confidence interval.

## Known issues

- The dataset is close to saturated in the literature and shows one product only, so raw
  accuracy is a weak signal; the value is in the operating point, robustness and data
  efficiency.
- Labels are binary (ok / defect) with no defect-type annotation.
- The test split is small (261 images), so confidence intervals are wide; for example a
  zero-escape result still has an upper bound of 2.3%.
- Perturbations are simulated, not captured from a real production line.
- The final model is assumed to be represented by the out-of-fold predictions used for
  calibration; this is checked against the test split but remains an assumption.
- The manual-review rate on test (13%) is above the 10% target: the policy favours recall,
  so more ambiguous good parts go to review rather than letting a defect through.
- Robustness is weak against heavy sensor noise and aggressive JPEG compression, which were
  not part of training augmentation.

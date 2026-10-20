# 0007 — Perturbation suite, shortcut check and data-efficiency protocol

## Context

The model will meet camera conditions the dataset never shows: exposure drift, defocus,
sensor noise, aggressive compression, and small misplacements of the part. It may also
have learned something other than the casting, such as the background of a capture
session. Finally, a real plant rarely has hundreds of labelled defects, so the number of
defect images the model needs matters as much as its peak accuracy.

## Decision

**Perturbations** (`configs/robustness.yaml`, implemented with Pillow and numpy):
brightness and contrast reduction, Gaussian blur, Gaussian noise, JPEG re-encoding,
translation and small rotation, each at three severities (mild, medium, severe). Corners
exposed by geometric shifts are filled with the conveyor grey, so the perturbation does
not add artificial black edges. Every development image is scored by the fold model that
held it out, then passed through the frozen policy (temperature and both thresholds from
`configs/policy.yaml`); defect recall, escapes, false rejects and review share are
reported per perturbation and level. The KPI is the largest recall drop at medium
severity. The test-split run happens once, in the final evaluation.

**Shortcut check**: a fixed centred disk of radius 0.56 x the image side is painted
over and each held-out image is scored from the remaining corners alone. A per-image part
detector was tried first, four ways: intensity threshold, radial intensity profile from the
dark impeller cavity, circular Hough transform, and local texture. Each one, on visual
review of real images, either left part of the rim visible (burrs on the rim are defect
evidence, so leakage would inflate the check) or erased nearly all background (which would
make the check pass trivially). The fixed disk contains the part on all but the very
largest framings and keeps about 10% of the image, roughly 25,000 pixels, of pure
background. An AUROC above 0.60 stops the project for review.

**Data efficiency**: the fine-tuned model is retrained with 5%, 10%, 25%, 50% and 100% of
the defective training images, sampled by near-duplicate group, with all healthy images
kept. Training runs a fixed number of epochs (the cross-validation median) instead of
early stopping, so no held-out data steers training. Each of three seeds per point holds
out a different fold; a full five-fold run per point would cost five times the training
for a curve whose shape three folds already show. The zero-defect reference is a
one-class kNN on frozen ResNet-18 embeddings: the reference set is healthy images only and
k is fixed at 5 a priori, because selecting k by AUROC would need defect labels.

**Grad-CAM** is implemented with forward and backward hooks on the last convolution
(`conv_head`). Overlays of dataset images are written under `artifacts/` only: the dataset
licence (CC BY-NC-ND 4.0) does not allow redistributing derived images.

## Rejected alternatives

- Per-image part segmentation for the shortcut check: unreliable on this dataset, see
  above.
- Selecting k for the one-class model on training defects: violates the zero-defect premise.
- Early stopping inside the data-efficiency runs: the stopping fold would also be the
  evaluation fold.
- A synthetic "camera" model (lens distortion, vignetting): plausible, but each effect
  needs parameters that nothing in this dataset can calibrate.

## Consequences

All perturbations are simulated rather than captured on a real line; this is listed as a
known limitation. The fixed-disk check can only detect shortcuts in the corners, which is
where capture-session differences such as lighting and background texture would show.

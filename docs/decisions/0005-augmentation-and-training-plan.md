# 0005 — Augmentation policy and CPU training plan

## Context

The main model is EfficientNet-B0 fine-tuned on CPU within a 3-hour budget for five folds
plus the final model. The images are grayscale top-down photographs of a circular casting
on a uniform grey background, so the label is invariant to rotation and mirroring but the
camera may vary in exposure, focus and framing.

## Decision

Train-time augmentation only, applied on the fly:

- rotation by any angle, corners filled with the background grey (200) rather than black,
  so augmentation does not introduce wedges the camera never produces;
- horizontal and vertical flips;
- brightness and contrast jitter of ±20%;
- Gaussian blur (kernel 3, sigma 0.1 to 1.0) with probability 0.3;
- no hue or saturation changes: the images are grayscale.

Training: AdamW (lr 3e-4, weight decay 1e-4), cosine annealing, BCE-with-logits weighted by
the class ratio, batch size 16 at 224x224, deterministic algorithms with a fixed seed.
Early stopping monitors AUROC on the held-out fold with patience 3 over at most 15 epochs.
The final model trains on all development data for the median of the folds' best epochs.
A per-run time budget aborts training if the projected time would exceed its share of the
3-hour envelope, and the first measured epoch time is reported before continuing.

Early stopping on the held-out fold means epoch selection has seen that fold, so
out-of-fold scores are mildly optimistic; this is accepted, recorded here, and checked
against the untouched test split in the final evaluation.

## Rejected alternatives

- albumentations: one more dependency for transforms torchvision v2 already provides.
- A fixed epoch count without early stopping: wastes budget on folds that converge early
  and underfits folds that need longer.
- An inner validation split carved out of each fold's training data: cleaner statistically
  but burns scarce defect samples the thresholds in phase 4 depend on.

## Consequences

Out-of-fold predictions remain the calibration input (ADR 0004) with the caveat above.
If the budget guard ever fires, the recorded epoch times justify reducing the input size
to 192 or lowering max epochs in a follow-up decision.

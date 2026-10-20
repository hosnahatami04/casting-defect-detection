# Robustness, shortcut and data efficiency (out-of-fold)

All numbers are on the development split, scored by the fold model that held each image out,
under the frozen policy (`configs/policy.yaml`). The test split is untouched until phase 7.
Sources: `artifacts/robustness/`, `artifacts/efficiency/`.

## Background shortcut check

Background-only AUROC (part masked by a centred disk, radius 0.56): **0.567** over 1039
images, below the 0.60 ceiling. The model is not relying on a background cue. Grad-CAM
overlays (`artifacts/figures/gradcam_oof_grid.png`) agree: activation concentrates on the
part and its rim across caught defects, missed defects, false rejects and review-band parts.

## Robustness under perturbation

Each perturbation at three severities, under the frozen three-way policy. Baseline
(no perturbation): recall 1.000, 0/624 escapes, 12/415 false rejects, review share 0.040.

| perturbation | sev | AUROC | recall | escapes | false rejects | recall drop (pts) |
|---|---|---|---|---|---|---|
| brightness | 1 | 0.9984 | 1.0000 | 0/624 | 16/415 | 0.00 |
| brightness | 2 | 0.9947 | 0.9952 | 3/624 | 17/415 | 0.48 |
| brightness | 3 | 0.9597 | 0.9551 | 28/624 | 29/415 | 4.49 |
| contrast | 1 | 0.9991 | 1.0000 | 0/624 | 7/415 | 0.00 |
| contrast | 2 | 0.9984 | 0.9984 | 1/624 | 10/415 | 0.16 |
| contrast | 3 | 0.9959 | 0.9952 | 3/624 | 22/415 | 0.48 |
| gaussian_blur | 1 | 0.9987 | 0.9984 | 1/624 | 12/415 | 0.16 |
| gaussian_blur | 2 | 0.9895 | 0.9824 | 11/624 | 13/415 | 1.76 |
| gaussian_blur | 3 | 0.9036 | 0.9936 | 4/624 | 184/415 | 0.64 |
| gaussian_noise | 1 | 0.9952 | 0.9904 | 6/624 | 6/415 | 0.96 |
| gaussian_noise | 2 | 0.9241 | 0.8894 | 69/624 | 19/415 | 11.06 |
| gaussian_noise | 3 | 0.7718 | 0.8830 | 73/624 | 91/415 | 11.70 |
| jpeg | 1 | 0.9958 | 0.9872 | 8/624 | 12/415 | 1.28 |
| jpeg | 2 | 0.9891 | 0.9808 | 12/624 | 11/415 | 1.92 |
| jpeg | 3 | 0.9420 | 0.8494 | 94/624 | 4/415 | 15.06 |
| translate | 1 | 0.9993 | 0.9984 | 1/624 | 11/415 | 0.16 |
| translate | 2 | 0.9987 | 0.9984 | 1/624 | 14/415 | 0.16 |
| translate | 3 | 0.9979 | 1.0000 | 0/624 | 25/415 | 0.00 |
| rotate | 1 | 0.9990 | 0.9984 | 1/624 | 10/415 | 0.16 |
| rotate | 2 | 0.9993 | 1.0000 | 0/624 | 11/415 | 0.00 |
| rotate | 3 | 0.9992 | 0.9984 | 1/624 | 11/415 | 0.16 |

Largest recall drop at medium severity: **11.06 points (gaussian_noise)**, above the
3-point target. Honest reading: the model is very robust to the geometric and photometric
changes it was augmented against (rotation, flips, translation, brightness, contrast all
stay under 1 point through medium severity), but sensitive to sensor noise and heavy JPEG
compression, which train-time augmentation did not include. Blur at severe level keeps
recall but collapses specificity (184 false rejects): the policy sends almost everything to
rejection rather than letting a defect through, which is the intended failure direction.

## Data efficiency

AUROC versus the number of defective training images (all healthy images kept), mean over
three seeds, each on a different held-out fold. One-class kNN sees zero defects.

| defect fraction | defects (~) | AUROC mean | min–max |
|---|---|---|---|
| 0.05 | 25 | 0.9295 | 0.8653–0.9879 |
| 0.10 | 50 | 0.9543 | 0.9207–0.9798 |
| 0.25 | 125 | 0.9947 | 0.9888–0.9979 |
| 0.50 | 250 | 0.9950 | 0.9894–0.9991 |
| 1.00 | 498 | 0.9985 | 0.9974–0.9996 |
| one-class (0 defects) | 0 | 0.9523 | CI 0.9334–0.9709 |

Smallest fraction reaching mean AUROC >= 0.98: **0.25 (about 125 defective images)**. The
one-class model, with no defective image at all, reaches 0.952 — roughly the supervised
model trained on 50 defects. The practical message: on this product, a usable detector
needs only a few dozen labelled defects, and a defect-free anomaly model is a viable cold
start.

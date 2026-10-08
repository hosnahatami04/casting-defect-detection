# Data card

## Source

- Dataset: "Casting product image data for quality inspection", Kaggle, user ravirajsinh45
  (`real-life-industrial-dataset-of-casting-product`).
- Subject: top-view photographs of a submersible pump impeller casting on a production line.
- Classes: `ok_front` (accepted part) and `def_front` (defective part). The dataset carries no
  defect-type labels; nothing beyond the binary label is assumed anywhere in this project.
- License: CC BY-NC-ND 4.0 as listed on Kaggle. No dataset image is committed to this
  repository; the archive checksum is pinned in `configs/default.yaml` instead.

## Collection used

Only the un-augmented 512x512 collection is used. The 300x300 collection distributed in the
same archive contains augmented copies of the same photographs; splitting such copies across
train and test would leak and inflate every metric. Augmentation is instead applied in this
project at train time only. See ADR 0003.

## Integrity

| check | result |
|---|---|
| files found (`ok_front` / `def_front`) | 519 / 781 (1300 total) |
| accepted after full decode | 1300 |
| rejected (corrupt, wrong size, wrong mode) | 0 |
| dimensions | 512x512 for all files |
| file mode | RGB JPEG containers; content is grayscale (all channels identical on sampled files) |

The spec sheet described the files as grayscale; the pixels are, but the JPEG containers are
RGB. The pipeline accepts mode RGB and converts to single-channel at load time.

## Near-duplicate grouping

64-bit dHash, Hamming distance, union-find connected components. The pairwise distance
distribution rises smoothly from zero with no bimodal gap, because distinct parts of the same
product are visually near-identical by design. Threshold calibration details and a visual
review of borderline pairs: `reports/interim/phash_calibration.md` and ADR 0003.

| statistic (threshold 3) | value |
|---|---|
| pairs at distance 0 | 23 |
| pairs at distance <= 3 | 755 |
| groups | 780 |
| largest group | 54 images |
| images in multi-image groups | 680 |

## Split

Group-aware and stratified: a group is never divided across sections, so no near-duplicate
pair at the frozen threshold crosses test/dev or fold boundaries (verified on every run).
Seed 42, test fraction 0.20, 5 folds.

| section | ok | def | total |
|---|---|---|---|
| test | 104 | 157 | 261 |
| fold 0 | 83 | 122 | 205 |
| fold 1 | 83 | 127 | 210 |
| fold 2 | 83 | 129 | 212 |
| fold 3 | 83 | 120 | 203 |
| fold 4 | 83 | 126 | 209 |

Targets were 104 ok and 156 def for test; the one-image def overshoot comes from a
mixed-label group that is kept intact. Leakage check at threshold 3: 0 cross-section pairs.

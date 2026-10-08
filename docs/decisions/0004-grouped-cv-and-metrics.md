# 0004 — Grouped cross-validation, out-of-fold predictions and metric choice

## Context

The development set holds roughly a thousand images. Threshold selection at 99% defect
recall needs many defective predictions to be trustworthy; a single fixed validation split
of 10 to 20 percent would leave only a few dozen. Near-duplicate groups (ADR 0003) must also
never straddle a train/validation boundary, or validation scores inflate.

## Decision

- 5-fold cross-validation over the development split, reusing the group-aware, stratified
  fold assignment frozen in the manifest. Every development image receives exactly one
  out-of-fold prediction, produced by a model that never saw it or its group in training.
- Out-of-fold predictions are persisted per model to `artifacts/oof/<model>.parquet` and are
  the sole input for calibration and threshold selection in later phases; the test split is
  untouched until the final evaluation.
- Metrics: AUROC (rank-based, tie-aware, implemented in numpy and tested against
  scikit-learn), recall at fixed false-reject rate, false-reject rate at fixed recall, and
  the confusion matrix at a chosen threshold. Every reported rate carries raw counts and an
  exact Clopper-Pearson interval; AUROC carries a grouped bootstrap interval that resamples
  whole near-duplicate groups so correlated images do not fake precision.

## Rejected alternatives

- Single fixed validation split: too few defect samples for a 99%-recall threshold.
- Plain bootstrap over images: treats near-duplicates as independent and narrows intervals
  dishonestly.
- Accuracy and F1 as headline metrics: both hide the asymmetric cost of an escaped defect.

## Consequences

Training cost is multiplied by the fold count, so expensive models must budget for it
(phase 3). In exchange thresholds are chosen on several hundred out-of-fold defect
predictions instead of a few dozen.

# Model comparison on out-of-fold predictions

All rows are computed on the same grouped 5-fold out-of-fold predictions over the
development split (1039 images: 624 defective, 415 ok), with the test split untouched.
AUROC intervals are grouped bootstrap (2000 resamples); rates carry raw counts and exact
95% Clopper-Pearson intervals. Latency is single-image PyTorch on CPU (p50 over 50 images);
sources: `artifacts/oof/<model>_metrics.json` and `artifacts/benchmarks/pytorch_latency.json`.

| | HOG + LR | ResNet-18 probe | EfficientNet-B0 fine-tuned |
|---|---|---|---|
| AUROC [95% CI] | 0.943 [0.920, 0.963] | 0.985 [0.973, 0.994] | 0.999 [0.998, 1.000] |
| Recall at 5% false-reject rate | 73.7% | 95.8% | 99.7% |
| False-reject rate at >=99% recall | 60.7% (252/415) [55.8, 65.5] | 33.7% (140/415) [29.2, 38.5] | 2.9% (12/415) [1.5, 5.0] |
| Defect recall at that threshold | 99.0% (618/624) | 99.0% (618/624) | 99.0% (618/624) |
| Training time, 5-fold CV | 27 s | ~3 min first run (embedding pass), 2 s cached | 25.8 min (272+420+211+296+349 s) |
| Final model training | n/a | n/a | 206 s (4 epochs, median of folds) |
| Single-image latency p50 / p95 | 4.1 / 4.3 ms | 13.8 / 17.2 ms | 16.7 / 19.6 ms |
| Model size on disk | 0.84 MB | 44.8 MB (frozen backbone) | 16.3 MB |

## Reading

- The fine-tuned model is the only tier with a defensible operating point: holding defect
  recall at 99.0% (618/624), it rejects 12 of 415 good parts against 140 for the probe and
  252 for HOG. The AUROC gap to the probe (0.999 vs 0.985) looks small; the operating-point
  cost gap (2.9% vs 33.7% false rejects) is what actually matters on a line.
- Early stopping selected epochs 3, 7, 2, 4 and 5 across folds; the final model trained on
  all development data for the median, 4 epochs.
- Out-of-fold scores for the fine-tuned model are mildly optimistic because epoch selection
  saw the held-out fold (ADR 0005); the untouched test split settles this in phase 7.
- Total EfficientNet training (5 folds + final) was 29.2 min on CPU, well inside the
  3-hour budget, so no input-size reduction was needed.

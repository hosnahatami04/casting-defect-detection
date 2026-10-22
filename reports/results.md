# Final test evaluation

Evaluated once on the 261-image test split (157 defective, 104 ok) under the frozen policy.

| metric | point | raw | 95% CI |
|---|---|---|---|
| AUROC | 0.9991 | — | [0.9966, 1.0000] |
| ECE | 0.0117 | — | — |
| defect recall | 1.0000 | 157/157 | [0.9768, 1.0000] |
| escape rate | 0.0000 | 0/157 | [0.0000, 0.0232] |
| false reject rate | 0.0192 | 2/104 | [0.0023, 0.0677] |
| manual review rate | 0.1303 | 34/261 | [0.0919, 0.1773] |

## Confusion by action

| | ok | defect |
|---|---|---|
| pass | 69 | 0 (escape) |
| manual review | 33 | 1 |
| reject | 2 (false) | 156 |

## Provenance

- manifest sha256: `12fbb5e8cef5e4dedf926cfdb306fac7344386b652e0bb2017570daba145a852`
- onnx sha256: `efaa6a20f07c0116143e88630320f9546d285da6d8b5da7b85ba8b290d08bdf2`
- policy sha256: `78d224291690b3c7073ad4dc5c96aa1280d71f4f37c0464782a273197f0694dc`

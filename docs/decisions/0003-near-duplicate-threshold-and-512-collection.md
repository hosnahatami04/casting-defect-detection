# 0003 — Near-duplicate threshold and use of the 512x512 collection only

## Context

The archive ships two collections: un-augmented 512x512 originals and a 300x300 collection
that contains augmented copies. Augmented copies spread across train and test leak label
information, so the 300x300 collection is excluded and augmentation is applied at train time
only.

Within the 512x512 originals, re-captures of the same physical part exist (23 image pairs
have identical 64-bit dHashes). The pairwise Hamming distance distribution rises smoothly
with no bimodal gap: distinct parts of the same cast product are visually near-identical, so
no threshold cleanly separates "same part re-captured" from "different, similar part".

Group structure under transitive union-find grouping:

| threshold | pairs | groups | largest group |
|---|---|---|---|
| 0 | 23 | 1277 | 2 |
| 1 | 106 | 1198 | 4 |
| 2 | 328 | 1028 | 15 |
| 3 | 755 | 780 | 54 |
| 4 | 1557 | 509 | 276 |
| 5 | 3030 | 285 | 511 |
| 6 | 5595 | 139 | 683 |

Visual review of sampled pairs: at distances 0 to 2 the pairs are clearly the same part
re-photographed (matching burr texture and vane orientation); at 3 same-part and
different-part pairs mix; from 4 upward transitive chaining collapses the dataset into
groups of hundreds, which makes a stratified 20% test split impossible (at the default 6,
the two largest groups held 1064 of 1300 images).

## Decision

Freeze the dHash Hamming threshold at 3. Groups are connected components over pairs at
distance <= 3, the split is group-aware, and the leakage check (zero cross-section pairs at
the frozen threshold) runs on every `castinspect data` invocation.

## Rejected alternatives

- Threshold 6 (initial default): transitive collapse, split infeasible.
- Thresholds 4 to 5: largest groups of 276 to 511 images distort stratification.
- Threshold 0 to 2: misses visually confirmed re-captures at distance 2 to 3.

## Consequences

Some pairs at distance 4 and above are plausibly the same physical part and may land on
opposite sides of a split; this residual risk is accepted, is inherent to a dataset of
near-identical products, and is documented here rather than hidden by an unusable larger
threshold. 680 of 1300 images sit in multi-image groups and are kept intact across sections.

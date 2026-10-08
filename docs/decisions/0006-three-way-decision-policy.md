# 0006 — Three-way decision policy, targets and cost model

## Context

An escaped defect is far more expensive than a wrongly rejected good part, and a human
re-inspection is cheaper than either. The policy must hold defect recall at or above 99%,
keep automatic escapes at or below 0.5% on out-of-fold predictions, and justify its
operating point against the line's cost structure (configured ratios: escape 20, false
reject 1, manual review 0.5).

## Decision

Three actions from calibrated probability p: reject when p >= t_reject, manual review when
t_review <= p < t_reject, pass otherwise. All values frozen in `configs/policy.yaml` by
`castinspect calibrate`, which reads only out-of-fold predictions.

- Temperature scaling (one parameter, NLL-minimised on OOF logits) calibrates the
  probabilities first; ECE with 15 bins is reported before and after.
- t_reject is the highest threshold whose hard-reject recall still meets the 99% target,
  which binds the recall constraint with the fewest false rejects.
- t_review is cost-aware: among all thresholds meeting the escape constraint (escape share
  is monotone in the threshold, so the constraint defines a simple upper bound), the one
  minimising expected cost is chosen. A review band priced at 0.5 exists precisely to buy
  the last defects cheaply: on OOF this moved the threshold from 0.100 (3 escapes, 8
  reviews, cost 0.073/part) to 0.017 (0 escapes, 42 reviews, cost 0.032/part), beating the
  best two-way policy (0.046/part).
- Thresholds are stored at full precision; rounding can move a threshold past a borderline
  sample and silently break the recall guarantee.

## Rejected alternatives

- Escape-constrained-only review threshold (the smallest review band): met the constraint
  but cost 2.3x more than the cost-aware choice and lost to a plain two-way policy.
- Direct cost minimisation without the recall constraint: cheaper on paper, but the 99%
  recall floor is the contract with the line and must bind regardless of cost ratios.
- Platt scaling (two parameters): more flexible, but with one dataset-scale temperature the
  ranking is provably preserved and overfitting risk on 1039 points is lower.

## Consequences

The review band carries 4.0% of parts (42/1039) on OOF, well under the 10% ceiling. If the
line's true cost ratios differ, `configs/default.yaml` holds them and a single calibrate
run re-freezes the policy; the test split remains untouched until phase 7.

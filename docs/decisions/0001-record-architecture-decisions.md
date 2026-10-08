# 0001 — Record architecture decisions

## Context

The project makes non-obvious choices (data splits, thresholds, model trade-offs) whose
rationale is easy to lose. Reviewers need to see why a decision was made, not only what
was decided.

## Decision

Every non-trivial decision gets a short record in `docs/decisions/NNNN-title.md` with four
sections: context, decision, rejected alternatives, consequences. Records are immutable;
a reversal gets a new record that supersedes the old one.

## Rejected alternatives

- Inline comments or commit messages only: scattered, not discoverable.
- A single DECISIONS.md: grows unbounded and merges badly.

## Consequences

Small writing overhead per decision; in exchange the repository explains itself.

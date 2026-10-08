import csv
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

TEST_FOLD = -1


@dataclass(frozen=True)
class SplitAssignment:
    split: str
    fold: int


@dataclass(frozen=True)
class ManifestRow:
    path: str
    label: str
    group_id: int
    split: str
    fold: int


def _group_members(group_ids: Sequence[int]) -> dict[int, list[int]]:
    members: dict[int, list[int]] = {}
    for index, group in enumerate(group_ids):
        members.setdefault(group, []).append(index)
    return members


def _group_compositions(
    labels: Sequence[str], members: dict[int, list[int]]
) -> dict[int, Counter[str]]:
    compositions: dict[int, Counter[str]] = {}
    for group, indices in members.items():
        counts = Counter(labels[i] for i in indices)
        compositions[group] = counts
        if len(counts) > 1:
            logger.warning("group %d mixes labels %s", group, dict(counts))
    return compositions


def split_dataset(
    labels: Sequence[str],
    group_ids: Sequence[int],
    test_fraction: float,
    n_folds: int,
    seed: int,
) -> list[SplitAssignment]:
    if len(labels) != len(group_ids):
        raise ValueError("labels and group_ids must have the same length")
    rng = np.random.default_rng(seed)
    members = _group_members(group_ids)
    compositions = _group_compositions(labels, members)
    majority = {g: c.most_common(1)[0][0] for g, c in compositions.items()}
    label_names = sorted(set(labels))

    # Test selection counts actual label members per group and skips groups
    # that would overshoot, so chunky groups cannot inflate the test share.
    targets = {
        label: round(sum(1 for item in labels if item == label) * test_fraction)
        for label in label_names
    }
    test_counts = dict.fromkeys(label_names, 0)
    test_groups: set[int] = set()
    for label in label_names:
        label_groups = [g for g in members if majority[g] == label]
        order = rng.permutation(len(label_groups))
        for k in order:
            if test_counts[label] >= targets[label]:
                break
            group = label_groups[int(k)]
            if test_counts[label] + compositions[group][label] > targets[label]:
                continue
            test_groups.add(group)
            for other in label_names:
                test_counts[other] += compositions[group][other]
    for label in label_names:
        if test_counts[label] < targets[label]:
            logger.warning(
                "test split holds %d %r images, target %d; remaining groups were too large",
                test_counts[label],
                label,
                targets[label],
            )

    # Largest groups first onto the lightest fold for their label; fold counts
    # track the actual composition so mixed groups stay accounted for.
    fold_counts: list[dict[str, int]] = [dict.fromkeys(label_names, 0) for _ in range(n_folds)]
    fold_of_group: dict[int, int] = {}
    for label in label_names:
        dev_groups = [g for g in members if majority[g] == label and g not in test_groups]
        order = rng.permutation(len(dev_groups))
        shuffled = [dev_groups[int(k)] for k in order]
        shuffled.sort(key=lambda g: -len(members[g]))
        for group in shuffled:
            fold = min(range(n_folds), key=lambda f: fold_counts[f][label])
            fold_of_group[group] = fold
            for other in label_names:
                fold_counts[fold][other] += compositions[group][other]

    assignments: list[SplitAssignment] = []
    for group in group_ids:
        if group in test_groups:
            assignments.append(SplitAssignment(split="test", fold=TEST_FOLD))
        else:
            assignments.append(SplitAssignment(split="dev", fold=fold_of_group[group]))
    return assignments


def conflicting_pairs(
    distances: npt.NDArray[np.uint8],
    threshold: int,
    assignment: Sequence[str],
) -> list[tuple[int, int, int]]:
    close = np.triu(distances <= threshold, k=1)
    rows, cols = np.nonzero(close)
    labels = np.asarray(assignment)
    return [
        (int(i), int(j), int(distances[i, j]))
        for i, j in zip(rows, cols, strict=True)
        if labels[i] != labels[j]
    ]


def write_manifest(rows: Sequence[ManifestRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "label", "group_id", "split", "fold"])
        for row in rows:
            writer.writerow([row.path, row.label, row.group_id, row.split, row.fold])


def read_manifest(path: Path) -> list[ManifestRow]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [
            ManifestRow(
                path=record["path"],
                label=record["label"],
                group_id=int(record["group_id"]),
                split=record["split"],
                fold=int(record["fold"]),
            )
            for record in reader
        ]

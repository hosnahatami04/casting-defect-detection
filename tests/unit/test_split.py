from pathlib import Path

import numpy as np

from castinspect.data.split import (
    TEST_FOLD,
    ManifestRow,
    conflicting_pairs,
    read_manifest,
    split_dataset,
    write_manifest,
)

N_FOLDS = 5
TEST_FRACTION = 0.2
SEED = 42


def _synthetic_population() -> tuple[list[str], list[int]]:
    # 60% ok / 40% def; a mix of singleton groups and a few multi-image groups.
    labels: list[str] = []
    group_ids: list[int] = []
    group = 0
    for label, count in (("ok", 240), ("def", 160)):
        made = 0
        while made < count:
            size = 3 if group % 10 == 0 else 1
            size = min(size, count - made)
            labels.extend([label] * size)
            group_ids.extend([group] * size)
            made += size
            group += 1
    return labels, group_ids


def test_no_group_crosses_split_or_fold() -> None:
    labels, group_ids = _synthetic_population()
    assignments = split_dataset(labels, group_ids, TEST_FRACTION, N_FOLDS, SEED)
    sections: dict[int, set[tuple[str, int]]] = {}
    for group, assignment in zip(group_ids, assignments, strict=True):
        sections.setdefault(group, set()).add((assignment.split, assignment.fold))
    assert all(len(s) == 1 for s in sections.values())


def test_test_fraction_approximate() -> None:
    labels, group_ids = _synthetic_population()
    assignments = split_dataset(labels, group_ids, TEST_FRACTION, N_FOLDS, SEED)
    test_count = sum(1 for a in assignments if a.split == "test")
    assert abs(test_count / len(labels) - TEST_FRACTION) < 0.03


def test_class_ratio_per_fold_within_3_percent() -> None:
    labels, group_ids = _synthetic_population()
    assignments = split_dataset(labels, group_ids, TEST_FRACTION, N_FOLDS, SEED)
    dev = [(lab, a.fold) for lab, a in zip(labels, assignments, strict=True) if a.split == "dev"]
    overall = sum(1 for lab, _ in dev if lab == "def") / len(dev)
    for fold in range(N_FOLDS):
        fold_items = [lab for lab, f in dev if f == fold]
        assert fold_items, f"fold {fold} is empty"
        ratio = sum(1 for lab in fold_items if lab == "def") / len(fold_items)
        assert abs(ratio - overall) <= 0.03


def test_same_seed_same_assignment() -> None:
    labels, group_ids = _synthetic_population()
    first = split_dataset(labels, group_ids, TEST_FRACTION, N_FOLDS, SEED)
    second = split_dataset(labels, group_ids, TEST_FRACTION, N_FOLDS, SEED)
    assert first == second


def test_test_items_have_sentinel_fold() -> None:
    labels, group_ids = _synthetic_population()
    assignments = split_dataset(labels, group_ids, TEST_FRACTION, N_FOLDS, SEED)
    for assignment in assignments:
        if assignment.split == "test":
            assert assignment.fold == TEST_FOLD
        else:
            assert 0 <= assignment.fold < N_FOLDS


def test_mixed_label_group_stays_together() -> None:
    labels = ["ok", "def", "ok", "ok", "def", "def", "ok", "def"]
    group_ids = [0, 0, 1, 2, 3, 4, 5, 6]
    assignments = split_dataset(labels, group_ids, 0.25, 2, 0)
    assert (assignments[0].split, assignments[0].fold) == (
        assignments[1].split,
        assignments[1].fold,
    )


def test_conflicting_pairs_detects_cross_section_leak() -> None:
    distances = np.array(
        [
            [0, 2, 40],
            [2, 0, 40],
            [40, 40, 0],
        ],
        dtype=np.uint8,
    )
    assert conflicting_pairs(distances, 6, ["test", "dev", "dev"]) == [(0, 1, 2)]
    assert conflicting_pairs(distances, 6, ["dev", "dev", "test"]) == []


def test_manifest_roundtrip(tmp_path: Path) -> None:
    rows = [
        ManifestRow(path="ok_front/a.png", label="ok", group_id=0, split="dev", fold=2),
        ManifestRow(path="def_front/b.png", label="def", group_id=1, split="test", fold=-1),
    ]
    path = tmp_path / "manifest.csv"
    write_manifest(rows, path)
    assert read_manifest(path) == rows

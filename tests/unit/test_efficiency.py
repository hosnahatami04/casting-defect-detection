from castinspect.data.split import ManifestRow
from castinspect.evaluation.efficiency import subsample_defect_groups


def _rows() -> list[ManifestRow]:
    rows = []
    for i in range(40):
        rows.append(ManifestRow(f"ok_{i}.png", "ok", 1000 + i, "dev", 0))
    group = 0
    for i in range(60):
        # Pairs of defect images share a group.
        rows.append(ManifestRow(f"def_{i}.png", "def", group, "dev", 0))
        if i % 2:
            group += 1
    return rows


def test_keeps_all_healthy_rows() -> None:
    kept = subsample_defect_groups(_rows(), 0.25, seed=0)
    assert sum(1 for r in kept if r.label == "ok") == 40


def test_fraction_approximate_and_groups_intact() -> None:
    kept = subsample_defect_groups(_rows(), 0.25, seed=0)
    defects = [r for r in kept if r.label == "def"]
    assert 15 <= len(defects) <= 16
    groups = {r.group_id for r in defects}
    assert all(sum(1 for r in defects if r.group_id == g) == 2 for g in groups)


def test_deterministic_per_seed_and_varies_across_seeds() -> None:
    first = subsample_defect_groups(_rows(), 0.1, seed=1)
    second = subsample_defect_groups(_rows(), 0.1, seed=1)
    other = subsample_defect_groups(_rows(), 0.1, seed=2)
    assert first == second
    assert {r.path for r in first} != {r.path for r in other}


def test_full_fraction_returns_everything() -> None:
    rows = _rows()
    assert subsample_defect_groups(rows, 1.0, seed=0) == rows

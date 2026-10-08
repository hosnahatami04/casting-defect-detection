from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from castinspect.data.integrity import (
    IntegrityError,
    check_image,
    scan_dataset,
)

SIZE = 32
MODES = frozenset({"L"})


def _write_image(path: Path, size: int = SIZE, mode: str = "L", seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    if mode == "L":
        array = rng.integers(0, 255, size=(size, size)).astype(np.uint8)
    else:
        array = rng.integers(0, 255, size=(size, size, 3)).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode=mode).save(path)


def test_valid_image_accepted(tmp_path: Path) -> None:
    path = tmp_path / "ok.png"
    _write_image(path)
    record = check_image(path, "ok", SIZE, MODES)
    assert (record.width, record.height, record.mode) == (SIZE, SIZE, "L")


def test_corrupt_file_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"this is not an image at all")
    with pytest.raises(IntegrityError, match="unreadable image"):
        check_image(path, "ok", SIZE, MODES)


def test_truncated_image_rejected(tmp_path: Path) -> None:
    path = tmp_path / "truncated.png"
    _write_image(path)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(IntegrityError):
        check_image(path, "ok", SIZE, MODES)


def test_wrong_dimensions_rejected(tmp_path: Path) -> None:
    path = tmp_path / "small.png"
    _write_image(path, size=16)
    with pytest.raises(IntegrityError, match="unexpected dimensions 16x16"):
        check_image(path, "ok", SIZE, MODES)


def test_wrong_mode_rejected(tmp_path: Path) -> None:
    path = tmp_path / "rgb.png"
    _write_image(path, mode="RGB")
    with pytest.raises(IntegrityError, match="unexpected color mode 'RGB'"):
        check_image(path, "ok", SIZE, MODES)


def test_scan_dataset_collects_and_rejects(tmp_path: Path) -> None:
    _write_image(tmp_path / "ok_front" / "a.png", seed=1)
    _write_image(tmp_path / "ok_front" / "b.png", seed=2)
    _write_image(tmp_path / "def_front" / "c.png", seed=3)
    (tmp_path / "def_front" / "notes.txt").write_text("not an image")
    _write_image(tmp_path / "def_front" / "bad.png", size=16, seed=4)

    report = scan_dataset(tmp_path, SIZE, MODES)

    assert len(report.accepted) == 3
    assert {r.label for r in report.accepted} == {"ok", "def"}
    reasons = {r.path.name: r.reason for r in report.rejected}
    assert "notes.txt" in reasons
    assert "bad.png" in reasons


def test_scan_dataset_missing_class_folder(tmp_path: Path) -> None:
    _write_image(tmp_path / "ok_front" / "a.png")
    with pytest.raises(FileNotFoundError, match="def_front"):
        scan_dataset(tmp_path, SIZE, MODES)

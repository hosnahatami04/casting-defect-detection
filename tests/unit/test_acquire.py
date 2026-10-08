import zipfile
from pathlib import Path

import pytest

from castinspect.data.acquire import (
    find_archive,
    locate_image_root,
    safe_extract,
    sha256_file,
    verify_archive,
)

HELLO_SHA256 = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_sha256_known_value(tmp_path: Path) -> None:
    path = tmp_path / "hello.txt"
    path.write_bytes(b"hello")
    assert sha256_file(path) == HELLO_SHA256


def test_verify_archive_accepts_matching_hash(tmp_path: Path) -> None:
    path = tmp_path / "hello.txt"
    path.write_bytes(b"hello")
    assert verify_archive(path, HELLO_SHA256) == HELLO_SHA256


def test_verify_archive_rejects_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "hello.txt"
    path.write_bytes(b"hello")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_archive(path, "0" * 64)


def test_verify_archive_without_pin_returns_hash(tmp_path: Path) -> None:
    path = tmp_path / "hello.txt"
    path.write_bytes(b"hello")
    assert verify_archive(path, "") == HELLO_SHA256


def test_find_archive_single_zip(tmp_path: Path) -> None:
    archive = tmp_path / "dataset.zip"
    archive.write_bytes(b"")
    assert find_archive(tmp_path) == archive


def test_find_archive_none(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"no \.zip archive"):
        find_archive(tmp_path)


def test_find_archive_ambiguous(tmp_path: Path) -> None:
    (tmp_path / "a.zip").write_bytes(b"")
    (tmp_path / "b.zip").write_bytes(b"")
    with pytest.raises(ValueError, match="multiple archives"):
        find_archive(tmp_path)


def test_safe_extract_roundtrip(tmp_path: Path) -> None:
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("folder/file.txt", "content")
    dest = tmp_path / "out"
    safe_extract(archive, dest)
    assert (dest / "folder" / "file.txt").read_text() == "content"


def test_safe_extract_blocks_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../evil.txt", "escape")
    with pytest.raises(ValueError, match="escapes extraction root"):
        safe_extract(archive, tmp_path / "out")


def _make_class_dirs(root: Path) -> None:
    (root / "ok_front").mkdir(parents=True)
    (root / "def_front").mkdir(parents=True)


def test_locate_image_root_prefers_512(tmp_path: Path) -> None:
    target = tmp_path / "casting_512x512" / "casting_512x512"
    _make_class_dirs(target)
    _make_class_dirs(tmp_path / "casting_data" / "casting_data" / "train")
    _make_class_dirs(tmp_path / "casting_data" / "casting_data" / "test")
    assert locate_image_root(tmp_path) == target


def test_locate_image_root_single_candidate(tmp_path: Path) -> None:
    target = tmp_path / "images"
    _make_class_dirs(target)
    assert locate_image_root(tmp_path) == target


def test_locate_image_root_missing(tmp_path: Path) -> None:
    (tmp_path / "something").mkdir()
    with pytest.raises(FileNotFoundError, match="ok_front"):
        locate_image_root(tmp_path)


def test_locate_image_root_ambiguous(tmp_path: Path) -> None:
    _make_class_dirs(tmp_path / "set_a")
    _make_class_dirs(tmp_path / "set_b")
    with pytest.raises(ValueError, match="ambiguous"):
        locate_image_root(tmp_path)

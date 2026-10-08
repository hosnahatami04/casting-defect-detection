import hashlib
import logging
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

_READ_BLOCK = 1 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_READ_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def find_archive(raw_dir: Path) -> Path:
    archives = sorted(raw_dir.glob("*.zip")) if raw_dir.is_dir() else []
    if not archives:
        raise FileNotFoundError(
            f"no .zip archive found in {raw_dir}; download the dataset archive and place it there"
        )
    if len(archives) > 1:
        names = ", ".join(a.name for a in archives)
        raise ValueError(f"multiple archives found in {raw_dir} ({names}); pass one explicitly")
    return archives[0]


def verify_archive(path: Path, expected_sha256: str) -> str:
    actual = sha256_file(path)
    if expected_sha256 and actual != expected_sha256:
        raise ValueError(
            f"checksum mismatch for {path.name}: expected {expected_sha256}, got {actual}"
        )
    return actual


def safe_extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    resolved_dest = dest.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if not target.is_relative_to(resolved_dest):
                raise ValueError(f"archive member escapes extraction root: {member}")
        zf.extractall(dest)
    logger.info("extracted %s to %s", archive.name, dest)


def locate_image_root(search_root: Path) -> Path:
    if not search_root.is_dir():
        raise FileNotFoundError(f"extraction root not found: {search_root}")
    candidates = [
        d
        for d in sorted(p for p in search_root.rglob("*") if p.is_dir())
        if (d / "ok_front").is_dir() and (d / "def_front").is_dir()
    ]
    # The archive also ships a pre-augmented 300x300 copy; only the 512x512 originals are used.
    preferred = [d for d in candidates if any("512" in part for part in d.parts)]
    if len(preferred) == 1:
        return preferred[0]
    if not candidates:
        raise FileNotFoundError(
            f"no directory with ok_front and def_front subfolders under {search_root}"
        )
    if len(candidates) == 1:
        return candidates[0]
    names = ", ".join(str(c) for c in candidates)
    raise ValueError(f"ambiguous image roots under {search_root}: {names}; pass one explicitly")

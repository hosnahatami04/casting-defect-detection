import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png", ".bmp"})
LABEL_DIRS: tuple[tuple[str, str], ...] = (("ok_front", "ok"), ("def_front", "def"))


class IntegrityError(Exception):
    pass


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str
    width: int
    height: int
    mode: str


@dataclass(frozen=True)
class RejectedFile:
    path: Path
    reason: str


@dataclass(frozen=True)
class IntegrityReport:
    accepted: list[ImageRecord]
    rejected: list[RejectedFile]


def check_image(
    path: Path, label: str, expected_size: int, allowed_modes: frozenset[str]
) -> ImageRecord:
    try:
        with Image.open(path) as img:
            img.verify()
        # verify() leaves the parser unusable, so reopen for a full decode.
        with Image.open(path) as img:
            img.load()
            width, height = img.size
            mode = img.mode
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise IntegrityError(f"unreadable image: {exc}") from exc
    if (width, height) != (expected_size, expected_size):
        raise IntegrityError(
            f"unexpected dimensions {width}x{height}, expected {expected_size}x{expected_size}"
        )
    if mode not in allowed_modes:
        raise IntegrityError(f"unexpected color mode {mode!r}, allowed: {sorted(allowed_modes)}")
    return ImageRecord(path=path, label=label, width=width, height=height, mode=mode)


def scan_dataset(root: Path, expected_size: int, allowed_modes: frozenset[str]) -> IntegrityReport:
    accepted: list[ImageRecord] = []
    rejected: list[RejectedFile] = []
    for dirname, label in LABEL_DIRS:
        folder = root / dirname
        if not folder.is_dir():
            raise FileNotFoundError(f"expected class folder not found: {folder}")
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                rejected.append(RejectedFile(path=path, reason="not a recognised image file"))
                continue
            try:
                accepted.append(check_image(path, label, expected_size, allowed_modes))
            except IntegrityError as exc:
                rejected.append(RejectedFile(path=path, reason=str(exc)))
    for item in rejected:
        logger.warning("rejected %s: %s", item.path, item.reason)
    return IntegrityReport(accepted=accepted, rejected=rejected)

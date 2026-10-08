from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from castinspect.data.split import ManifestRow, read_manifest

LABEL_TO_INDEX = {"ok": 0, "def": 1}


class CastingDataset:
    """Map-style dataset over the split manifest.

    Deliberately not a torch.utils.data.Dataset subclass: DataLoader only needs
    __len__ and __getitem__, and keeping torch out of this module lets the data
    pipeline run and be tested without the ml extras installed.
    """

    def __init__(
        self,
        manifest_path: Path,
        image_root: Path,
        split: str = "dev",
        folds: Sequence[int] | None = None,
        transform: Callable[[Image.Image], Any] | None = None,
    ) -> None:
        rows = [r for r in read_manifest(manifest_path) if r.split == split]
        if folds is not None:
            wanted = set(folds)
            rows = [r for r in rows if r.fold in wanted]
        if not rows:
            raise ValueError(
                f"no manifest rows for split={split!r}"
                + (f" folds={sorted(set(folds))}" if folds is not None else "")
            )
        self._rows: list[ManifestRow] = rows
        self._root = image_root
        self._transform = transform

    @property
    def rows(self) -> list[ManifestRow]:
        return list(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        row = self._rows[index]
        with Image.open(self._root / row.path) as img:
            image = img.convert("L")
        if self._transform is not None:
            return self._transform(image), LABEL_TO_INDEX[row.label]
        return image, LABEL_TO_INDEX[row.label]

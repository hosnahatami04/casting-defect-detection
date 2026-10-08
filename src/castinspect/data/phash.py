from collections.abc import Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

HASH_BITS = 64
# Width 9 x height 8 gives 8 horizontal comparisons per row, 64 bits total.
_DHASH_SIZE = (9, 8)


def dhash_bits(image: Image.Image) -> npt.NDArray[np.bool_]:
    gray = image.convert("L").resize(_DHASH_SIZE, Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    return np.asarray(bits.reshape(-1), dtype=np.bool_)


def dhash_file(path: Path) -> npt.NDArray[np.bool_]:
    with Image.open(path) as img:
        return dhash_bits(img)


def hamming_distance(a: npt.NDArray[np.bool_], b: npt.NDArray[np.bool_]) -> int:
    return int(np.count_nonzero(a != b))


def pairwise_hamming(bits: npt.NDArray[np.bool_], chunk_size: int = 512) -> npt.NDArray[np.uint8]:
    n = bits.shape[0]
    distances = np.zeros((n, n), dtype=np.uint8)
    # Chunked so the boolean broadcast stays small for a few thousand images.
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        diff = bits[start:stop, None, :] != bits[None, :, :]
        distances[start:stop] = diff.sum(axis=2, dtype=np.uint8)
    return distances


def near_duplicate_pairs(
    distances: npt.NDArray[np.uint8], threshold: int
) -> list[tuple[int, int, int]]:
    close = np.triu(distances <= threshold, k=1)
    rows, cols = np.nonzero(close)
    return [(int(i), int(j), int(distances[i, j])) for i, j in zip(rows, cols, strict=True)]


class UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._rank = [0] * size

    def find(self, item: int) -> int:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        if self._rank[root_a] < self._rank[root_b]:
            root_a, root_b = root_b, root_a
        self._parent[root_b] = root_a
        if self._rank[root_a] == self._rank[root_b]:
            self._rank[root_a] += 1


def connected_groups(size: int, edges: Iterable[tuple[int, int]]) -> list[int]:
    uf = UnionFind(size)
    for a, b in edges:
        uf.union(a, b)
    # Group ids are relabelled by first occurrence so output is stable.
    mapping: dict[int, int] = {}
    groups: list[int] = []
    for i in range(size):
        root = uf.find(i)
        if root not in mapping:
            mapping[root] = len(mapping)
        groups.append(mapping[root])
    return groups

import numpy as np
from PIL import Image

from castinspect.data.phash import (
    connected_groups,
    dhash_bits,
    hamming_distance,
    near_duplicate_pairs,
    pairwise_hamming,
)


def _image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array.astype(np.uint8), mode="L")


def _random_image(seed: int, size: int = 64) -> Image.Image:
    rng = np.random.default_rng(seed)
    return _image(rng.integers(0, 255, size=(size, size)))


def test_dhash_has_64_bits() -> None:
    bits = dhash_bits(_random_image(0))
    assert bits.shape == (64,)
    assert bits.dtype == np.bool_


def test_identical_images_have_distance_zero() -> None:
    a = dhash_bits(_random_image(0))
    b = dhash_bits(_random_image(0))
    assert hamming_distance(a, b) == 0


def test_brightness_shift_keeps_distance_small() -> None:
    rng = np.random.default_rng(1)
    base = rng.integers(0, 200, size=(64, 64))
    original = dhash_bits(_image(base))
    brighter = dhash_bits(_image(np.clip(base + 15, 0, 255)))
    assert hamming_distance(original, brighter) <= 6


def test_different_images_have_large_distance() -> None:
    a = dhash_bits(_random_image(2))
    b = dhash_bits(_random_image(3))
    assert hamming_distance(a, b) >= 16


def test_pairwise_hamming_matches_naive() -> None:
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(12, 64)).astype(np.bool_)
    distances = pairwise_hamming(bits, chunk_size=5)
    for i in range(12):
        for j in range(12):
            assert distances[i, j] == np.count_nonzero(bits[i] != bits[j])
    assert np.array_equal(distances, distances.T)
    assert np.all(np.diag(distances) == 0)


def test_near_duplicate_pairs_respects_threshold() -> None:
    bits = np.zeros((3, 64), dtype=np.bool_)
    bits[1, :4] = True
    bits[2, :30] = True
    distances = pairwise_hamming(bits)
    pairs = near_duplicate_pairs(distances, threshold=6)
    assert pairs == [(0, 1, 4)]


def test_connected_groups_transitive() -> None:
    groups = connected_groups(4, [(0, 1), (1, 2)])
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_connected_groups_no_edges() -> None:
    assert connected_groups(4, []) == [0, 1, 2, 3]


def test_connected_groups_stable_ids() -> None:
    assert connected_groups(5, [(3, 4)]) == [0, 1, 2, 3, 3]

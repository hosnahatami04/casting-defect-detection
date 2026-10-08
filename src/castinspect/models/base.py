from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class DefectClassifier(Protocol):
    """Common contract so evaluation, decision and serving stay model-agnostic.

    predict_proba returns the probability of the defect class, shape (n,).
    """

    name: str

    def fit(self, image_paths: Sequence[Path], labels: Sequence[int]) -> None: ...

    def predict_proba(self, image_paths: Sequence[Path]) -> npt.NDArray[np.float64]: ...

    def save(self, path: Path) -> None: ...

    def load(self, path: Path) -> None: ...


ModelFactory = Callable[..., DefectClassifier]

_REGISTRY: dict[str, ModelFactory] = {}


def register_model(name: str, factory: ModelFactory) -> None:
    if name in _REGISTRY:
        raise ValueError(f"model {name!r} is already registered")
    _REGISTRY[name] = factory


def create_model(name: str, **params: Any) -> DefectClassifier:
    try:
        factory = _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        raise KeyError(f"unknown model {name!r}; registered: {known}") from None
    return factory(**params)


def registered_models() -> list[str]:
    return sorted(_REGISTRY)

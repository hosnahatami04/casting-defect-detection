import pytest

from castinspect.models.base import create_model, register_model, registered_models


def test_unknown_model_lists_known() -> None:
    with pytest.raises(KeyError, match="unknown model"):
        create_model("does-not-exist")


def test_register_and_create_with_params() -> None:
    created: list[int] = []

    class Dummy:
        name = "dummy-test"

        def __init__(self, value: int = 0) -> None:
            created.append(value)

        def fit(self, image_paths: object, labels: object) -> None: ...
        def predict_proba(self, image_paths: object) -> object: ...
        def save(self, path: object) -> None: ...
        def load(self, path: object) -> None: ...

    register_model("dummy-test", Dummy)
    create_model("dummy-test", value=5)
    assert created == [5]
    assert "dummy-test" in registered_models()
    with pytest.raises(ValueError, match="already registered"):
        register_model("dummy-test", Dummy)

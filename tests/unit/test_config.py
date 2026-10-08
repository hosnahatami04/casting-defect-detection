from pathlib import Path

import pytest

from castinspect.config import load_settings


def test_defaults_load_from_yaml() -> None:
    settings = load_settings(Path("configs/default.yaml"))
    assert settings.seed == 42
    assert settings.image_size == 512
    assert settings.split.test_fraction == 0.2
    assert settings.split.n_folds == 5
    assert settings.paths.data_raw == Path("data/raw")


def test_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("seed: 1\nimage_size: 128\n")
    monkeypatch.setenv("CASTINSPECT_SEED", "7")
    settings = load_settings(cfg)
    assert settings.seed == 7
    assert settings.image_size == 128


def test_nested_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CASTINSPECT_SPLIT__N_FOLDS", "3")
    settings = load_settings(Path("configs/default.yaml"))
    assert settings.split.n_folds == 3


def test_missing_config_file_raises() -> None:
    with pytest.raises(FileNotFoundError, match="config file not found"):
        load_settings(Path("does/not/exist.yaml"))

from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


class PathsConfig(BaseModel):
    data_raw: Path = Path("data/raw")
    data_processed: Path = Path("data/processed")
    artifacts: Path = Path("artifacts")
    reports: Path = Path("reports")


class DatasetConfig(BaseModel):
    # Pinned after the first manual download; empty means not recorded yet.
    archive_sha256: str = ""
    allowed_modes: tuple[str, ...] = ("L",)


class GroupingConfig(BaseModel):
    dhash_threshold: int = 6


class SplitConfig(BaseModel):
    test_fraction: float = 0.2
    n_folds: int = 5


class CostsConfig(BaseModel):
    escape: float = 20.0
    false_reject: float = 1.0
    manual_review: float = 0.5


class DecisionConfig(BaseModel):
    recall_target: float = 0.99
    max_escape: float = 0.005
    costs: CostsConfig = CostsConfig()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CASTINSPECT_",
        env_nested_delimiter="__",
    )

    seed: int = 42
    image_size: int = 512
    paths: PathsConfig = PathsConfig()
    dataset: DatasetConfig = DatasetConfig()
    grouping: GroupingConfig = GroupingConfig()
    split: SplitConfig = SplitConfig()
    decision: DecisionConfig = DecisionConfig()


def load_settings(config_path: Path | None = None) -> Settings:
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")

    class FileSettings(Settings):
        model_config = SettingsConfigDict(
            env_prefix="CASTINSPECT_",
            env_nested_delimiter="__",
            yaml_file=path,
        )

        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            # Environment variables take precedence over the YAML file.
            return (
                init_settings,
                env_settings,
                YamlConfigSettingsSource(settings_cls),
            )

    return FileSettings()

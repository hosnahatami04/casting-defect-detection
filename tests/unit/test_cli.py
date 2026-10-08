from pathlib import Path

from typer.testing import CliRunner

from castinspect import __version__
from castinspect.cli import app

runner = CliRunner()


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"castinspect {__version__}" in result.output


def test_info_prints_resolved_settings() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert '"seed": 42' in result.output
    assert '"image_size": 512' in result.output


def test_info_with_custom_config(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("seed: 9\n")
    result = runner.invoke(app, ["info", "--config", str(cfg)])
    assert result.exit_code == 0
    assert '"seed": 9' in result.output


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output

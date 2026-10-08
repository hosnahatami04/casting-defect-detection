from pathlib import Path
from typing import Annotated

import typer

from castinspect import __version__
from castinspect.config import DEFAULT_CONFIG_PATH, load_settings
from castinspect.logging import setup_logging

app = typer.Typer(name="castinspect", no_args_is_help=True)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"castinspect {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the version and exit.",
            callback=_print_version,
            is_eager=True,
        ),
    ] = False,
) -> None:
    setup_logging()


@app.command()
def info(
    config: Annotated[
        Path,
        typer.Option(help="Path to the YAML config file."),
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    settings = load_settings(config)
    typer.echo(settings.model_dump_json(indent=2))

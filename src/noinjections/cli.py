"""Command-line entry point for noinjections."""

from typing import Annotated

import typer

from noinjections import __version__

app = typer.Typer(
    name="noinjections",
    help="Scan websites, GitHub repositories, and local folders for indirect prompt injection.",
    add_completion=False,
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    """Print the version and exit when ``--version`` is passed."""
    if value:
        typer.echo(f"noinjections {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = None,
) -> None:
    """Scan a target for indirect prompt injection."""

from importlib.metadata import entry_points

import pytest
from typer.testing import CliRunner

from noinjections import __version__
from noinjections.cli import app

runner = CliRunner()


def test_version_flag_prints_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"noinjections {__version__}"


def test_short_version_flag() -> None:
    result = runner.invoke(app, ["-V"])
    assert result.exit_code == 0
    assert __version__ in result.output


@pytest.mark.parametrize("command", ["noinjections", "noinj"])
def test_console_scripts_point_to_cli_app(command: str) -> None:
    (script,) = entry_points(group="console_scripts", name=command)
    assert script.value == "noinjections.cli:app"
    assert script.load() is app

"""Task 2960 / 2982 (rename steps 1a and 4): the distribution is `cytir`, the import package is `cytir`, the one
console script is `cyt`. The old `aw` console script and import package are gone."""
import re
from pathlib import Path

APP = Path(__file__).resolve().parents[1]


def _text():
    return (APP / "pyproject.toml").read_text("utf-8")


def _scripts():
    block = re.search(r"^\[project\.scripts\]\n(.*?)(?=^\[)", _text(), re.S | re.M).group(1)
    return dict(re.findall(r'^(\w+) = "([^"]+)"', block, re.M))


def test_distribution_name_is_cytir():
    assert re.search(r'^\[project\]\nname = "cytir"$', _text(), re.M)


def test_cyt_is_the_only_console_script():
    assert _scripts() == {"cyt": "cytir.cli:main"}


def test_prog_name_is_cyt_whatever_started_us(monkeypatch):
    from cytir import cli
    for argv0 in ("/x/bin/cyt", "/x/bin/aw", "python -m cytir.cli"):  # rename-codemod: keep (a stale argv0 must still give `cyt`)
        monkeypatch.setattr("sys.argv", [argv0])
        assert cli._prog_name() == "cyt"
    assert cli.build_parser().prog == "cyt"


def test_version_is_the_release_version():
    import cytir
    assert cytir.__version__ == "0.1.0"
    assert re.search(r'^version = "0\.1\.0"$', _text(), re.M)

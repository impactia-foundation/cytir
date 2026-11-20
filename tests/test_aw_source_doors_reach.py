"""Task 2360 (S5 W7): the source/extract/news tool doors reach their modules (main(argv, prog))."""
from __future__ import annotations

import importlib

import pytest

from cytir import cli

DOORS = [
    (["source", "eric-pdf", "frontier"], "cytir.sources.eric.frontier"),
    (["source", "eric-pdf", "fetch"], "cytir.sources.eric.pdf_fetch"),
    (["source", "stackoverflow", "frontier"], "cytir.sources.stackoverflow.frontier"),
    (["source", "openlegaldata-hf"], "cytir.acquire.openlegaldata_hf"),
    (["extract", "wikipedia-dump"], "cytir.extract.wikipedia_dump"),
    (["news", "page-date-refetch"], "cytir.news.page_date_refetch"),
]


def _run(argv):
    try:
        return cli.main(argv)
    except SystemExit as e:
        return e.code


@pytest.mark.parametrize("door,module", DOORS)
def test_help_prints_door_prog(door, module, capsys):
    assert _run(door + ["--help"]) in (0, None)
    assert capsys.readouterr().out.splitlines()[0].startswith("usage: cyt " + " ".join(door))


@pytest.mark.parametrize("door,module", DOORS)
def test_exit_code_propagates(door, module, monkeypatch):
    mod = importlib.import_module(module)
    seen = {}

    def fake(argv=None, prog=None):
        seen["a"] = (argv, prog)
        return 7

    monkeypatch.setattr(mod, "main", fake)
    assert _run(door + ["--x"]) == 7
    assert seen["a"] == (["--x"], "cyt " + " ".join(door))

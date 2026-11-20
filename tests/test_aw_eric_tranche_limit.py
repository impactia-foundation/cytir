"""cytir.sources.eric.tranche_limit: the limit is pdf_complete of a complete fetch RUN.json."""
import json
import pathlib

import pytest

from cytir.sources.eric import tranche_limit as tl


def _write(root, offset, doc):
    path = tl.run_json_path(root, offset)
    path.parent.mkdir(parents=True)
    path.write_text(doc if isinstance(doc, str) else json.dumps(doc))


def test_path_spans_the_tranche():
    assert tl.run_json_path(pathlib.Path("r"), 150000).parts[-2] == \
        "official-pdf-tranche-150000-159999-v1"


def test_complete_run_gives_pdf_complete(tmp_path, capsys):
    _write(tmp_path, 60000, {"status": "complete", "counts": {"pdf_complete": 9965, "quarantine": 35}})
    assert tl.tranche_limit(tmp_path, 60000) == 9965
    assert tl.main(["--root", str(tmp_path), "--offset", "060000"]) == 0
    assert capsys.readouterr().out.strip() == "9965"


@pytest.mark.parametrize("doc", [
    {"status": "running", "counts": {"pdf_complete": 5}},
    {"status": "complete"},
    "not json",
])
def test_not_complete_or_malformed_is_not_ready(tmp_path, doc, capsys):
    _write(tmp_path, 150000, doc)
    assert tl.tranche_limit(tmp_path, 150000) is None
    assert tl.main(["--root", str(tmp_path), "--offset", "150000"]) == tl.NOT_READY
    assert capsys.readouterr().out == ""


def test_missing_run_json_is_not_ready(tmp_path):
    assert tl.tranche_limit(tmp_path, 150000) is None

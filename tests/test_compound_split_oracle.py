"""The German de-compounding rule against sealed expectations: parity of the CharSplit swap.

Study: specs/compound-split-replacement.md (ADR-121, Tim's ruling 26: option A). The rule
now lives in `cytir.core.decompound` (upstream `dtuggener/CharSplit`, MIT, commit 9710624e,
n-gram table fetched with a sha256 pin). The fixture holds single lower-cased tokens
(>= 12 chars, seen in two or more corpus families) and the parts the PREVIOUS
implementation (the GPL-3.0 package `compound_split` 1.0.2) emitted for each. Tokens are
not text. PARITY IS THE PASS RULE: one difference means 8,996,610 `lang=de` rows need a
rebuild, which nobody has authorised. Do not edit the fixture to make a test pass.

* `test_fixture_*` need nothing.
* `test_*_matches_fixture` need the pinned splitter and table: `pip install -e .[dev]` and
  `python3 -m cytir.embed.fetch_models --asset charsplit-ngram`. They FAIL (never skip) without
  them, as production does.
* `test_gpl_package_*` runs only where a developer installed `compound_split` as an oracle.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys
import types

import pytest

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "compound_split_parity.v1.jsonl"
ROWS = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line]
THRESHOLD = 0.7          # the rule's score gate
MIN_PART = 3             # the rule's part length gate
MIN_TOKEN = 12           # the rule's token length gate


def test_fixture_is_small_single_tokens():
    assert 1000 <= len(ROWS) <= 5000
    assert len({r["t"] for r in ROWS}) == len(ROWS)
    for r in ROWS:
        assert re.fullmatch(r"\w+", r["t"]) and r["t"] == r["t"].lower(), r
        assert len(r["t"]) >= MIN_TOKEN, r


def test_fixture_rows_are_self_consistent():
    kinds = {r["k"] for r in ROWS}
    assert kinds == {"split", "nosplit", "near", "nonalpha"}
    for r in ROWS:
        parts = r["p"]
        assert len(parts) in (0, 2), r
        if parts:
            assert "".join(parts) == r["t"], r           # parts are the token, cut in two
            assert all(len(p) >= MIN_PART for p in parts), r
        if r["k"] == "split":
            assert parts, r
        if r["k"] in ("nosplit", "nonalpha"):
            assert not parts, r


def test_decompound_matches_fixture():
    from cytir.core.decompound import decompound_parts

    bad = [(r["t"], r["p"], decompound_parts(r["t"])) for r in ROWS
           if list(decompound_parts(r["t"])) != r["p"]]
    assert not bad, bad[:10]


def test_write_side_common_matches_fixture():
    from cytir.core import common

    bad = [(r["t"], r["p"], common._decompound(r["t"])) for r in ROWS
           if common._decompound(r["t"]) != r["p"]]
    assert not bad, bad[:10]


def test_read_side_server_schema_matches_fixture():
    """The server's mirror (query time, title-centrality) emits the same parts."""
    path = pathlib.Path(__file__).resolve().parents[1] / "cytir" / "serve" / "schema.py"
    spec = importlib.util.spec_from_file_location("aw_server_schema_parity", path)
    schema = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema)
    bad = [(r["t"], r["p"], schema._decompound(r["t"])) for r in ROWS
           if list(schema._decompound(r["t"])) != r["p"]]
    assert not bad, bad[:10]


def test_bm25_tokens_emit_the_fixture_parts(monkeypatch):
    """End to end through the tokenizer both sides call: the whole token plus its parts."""
    from cytir.core import common

    from cytir.core import bm25_tokenizer as _tok
    monkeypatch.setattr(_tok, "_FOLD_ACCENTS", False)   # other tests may leave it on

    for r in [r for r in ROWS if r["k"] == "split"][:300]:
        toks = common.bm25_tokens(r["t"], "de")
        assert all(common._hash(p) in toks and toks[common._hash(p)] == 0.5 for p in r["p"]), (r, toks)


def test_gpl_package_oracle_matches_fixture():
    char_split = pytest.importorskip("compound_split.char_split")
    bad = []
    for r in ROWS:
        res = char_split.split_compound(r["t"])
        got = ([p.lower() for p in (res[0][1], res[0][2]) if len(p) >= MIN_PART]
               if len(r["t"]) >= MIN_TOKEN and res and res[0][0] >= THRESHOLD else [])
        if got != r["p"]:
            bad.append((r["t"], r["p"], got))
    assert not bad, bad[:10]


# --- refusal: no silent fallback to "no splitting" -------------------------------------

def _fresh_decompound(monkeypatch):
    from cytir.core import decompound

    monkeypatch.setattr(decompound, "_splitter", None)
    return decompound


def test_refuses_when_table_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("AW_CHARSPLIT_NGRAM", str(tmp_path / "absent.json"))
    d = _fresh_decompound(monkeypatch)
    with pytest.raises(d.DecompoundUnavailable, match="fetch_models"):
        d.decompound_parts("arbeitsunfaehigkeit")


def test_refuses_when_table_is_not_the_pinned_bytes(monkeypatch, tmp_path):
    """Right SIZE, wrong bytes: the size check passes, so only the sha256 check can refuse."""
    from cytir.embed.model_assets import CHARSPLIT_NGRAM

    bad = tmp_path / "ngram_probs.json"
    bad.write_bytes(b"x" * CHARSPLIT_NGRAM.size_bytes)
    assert bad.stat().st_size == CHARSPLIT_NGRAM.size_bytes
    monkeypatch.setenv("AW_CHARSPLIT_NGRAM", str(bad))
    d = _fresh_decompound(monkeypatch)
    with pytest.raises(d.DecompoundUnavailable, match="sha256 mismatch"):
        d.decompound_parts("arbeitsunfaehigkeit")


def test_refuses_when_table_is_the_wrong_size(monkeypatch, tmp_path):
    bad = tmp_path / "ngram_probs.json"
    bad.write_text('{"prefix": {}, "infix": {}, "suffix": {}}')
    monkeypatch.setenv("AW_CHARSPLIT_NGRAM", str(bad))
    d = _fresh_decompound(monkeypatch)
    with pytest.raises(d.DecompoundUnavailable, match="size mismatch"):
        d.decompound_parts("arbeitsunfaehigkeit")


def _no_table_anywhere(monkeypatch, tmp_path):
    """AW_MODELS_DIR names an empty dir; no other copy is found (hermetic on a dev box)."""
    empty = tmp_path / "models"
    empty.mkdir()
    monkeypatch.delenv("AW_CHARSPLIT_NGRAM", raising=False)
    monkeypatch.setenv("AW_MODELS_DIR", str(empty))
    monkeypatch.setenv("HOME", str(tmp_path))
    from cytir.embed import model_assets

    monkeypatch.setattr(model_assets, "REPO_ROOT", tmp_path / "no-repo")
    monkeypatch.setattr(model_assets, "operator_settings",
                        lambda: types.SimpleNamespace(fallback_models_dir=None))
    d = _fresh_decompound(monkeypatch)
    monkeypatch.setattr(d, "_package_table", lambda splitter_py: None)
    return empty, d


def test_missing_table_error_names_the_directory_searched(monkeypatch, tmp_path):
    empty, d = _no_table_anywhere(monkeypatch, tmp_path)
    with pytest.raises(d.DecompoundUnavailable) as exc:
        d.decompound_parts("arbeitsunfaehigkeit")
    assert str(empty / "ngram_probs.json") in str(exc.value)


def _server_app(monkeypatch):
    monkeypatch.syspath_prepend(str(pathlib.Path(__file__).resolve().parents[1] / "server"))
    from cytir.serve import app as server_app

    return server_app


def test_server_startup_fails_without_the_table(monkeypatch, tmp_path):
    """A deploy with no table must fail at container start, not on the first query."""
    from fastapi.testclient import TestClient

    server_app = _server_app(monkeypatch)
    _empty, d = _no_table_anywhere(monkeypatch, tmp_path)
    with pytest.raises(d.DecompoundUnavailable):
        with TestClient(server_app.api):
            pytest.fail("the server started without the n-gram table")


def test_server_starts_when_the_table_is_present(monkeypatch):
    """The verified table (fetched copy or the pip-installed one) is there: it starts."""
    from fastapi.testclient import TestClient

    server_app = _server_app(monkeypatch)
    _fresh_decompound(monkeypatch)
    with TestClient(server_app.api):
        pass


def test_pip_installed_table_is_accepted_when_no_fetched_copy(monkeypatch, tmp_path):
    """The table inside the `charsplit` package is used when its sha256 matches the pin."""
    from cytir.embed import model_assets

    empty = tmp_path / "models"
    empty.mkdir()
    monkeypatch.delenv("AW_CHARSPLIT_NGRAM", raising=False)
    monkeypatch.setenv("AW_MODELS_DIR", str(empty))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(model_assets, "REPO_ROOT", tmp_path / "no-repo")
    monkeypatch.setattr(model_assets, "operator_settings",
                        lambda: types.SimpleNamespace(fallback_models_dir=None))
    d = _fresh_decompound(monkeypatch)
    pkg = d._package_table(d._splitter_source())
    if pkg is None:
        pytest.skip("this charsplit install ships no table; the fetched asset is the path")
    split = next(r for r in ROWS if r["k"] == "split")
    assert list(d.decompound_parts(split["t"])) == split["p"]


def test_refuses_when_splitter_source_is_not_the_pinned_file(monkeypatch, tmp_path):
    pkg = tmp_path / "charsplit"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "splitter.py").write_text("class Splitter:\n    pass\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "charsplit", raising=False)
    d = _fresh_decompound(monkeypatch)
    with pytest.raises(d.DecompoundUnavailable, match="not the pinned"):
        d.decompound_parts("arbeitsunfaehigkeit")


def test_short_tokens_never_load_the_model(monkeypatch, tmp_path):
    monkeypatch.setenv("AW_CHARSPLIT_NGRAM", str(tmp_path / "absent.json"))
    d = _fresh_decompound(monkeypatch)
    assert d.decompound_parts("recht") == ()


def test_asset_is_registered_for_fetch_models():
    from cytir.embed.model_assets import ASSETS, CHARSPLIT_NGRAM

    assert ASSETS["charsplit-ngram"] is CHARSPLIT_NGRAM
    assert "9710624ef0b0df0edd12a721c33533ff1d07b557" in CHARSPLIT_NGRAM.url
    assert CHARSPLIT_NGRAM.sha256 == (
        "b145f5fba901698479d3e889255cf9342fe7e6c3593f80f7dc1932d1759be7bf")

"""Sealed 3-row fixture for the docs-gold loader (no network, no repo paths)."""
import json

from cytir.ops import github_docs_gold as g


def _row(n, **kw):
    r = {"id": f"gh-docs-00{n}", "query": f"flag {n}", "gold_url": f"https://example.org/blob/abc/docs/p{n}.md",
         "repo": f"o/r{n}", "stars_bucket": "ge20k", "stars": 25000, "page_words": 400 + n,
         "answer_sentence": "Use the flag.", "author": "haiku-5.5/openrouter",
         "verified_by": "haiku-5.5 second pass", "tim_spot_check": n == 1}
    r.update(kw)
    return r


def test_roundtrip_and_valid(tmp_path):
    p = tmp_path / "gold.jsonl"
    rows = [_row(1), _row(2), _row(3)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    loaded = g.load(p)
    assert loaded == rows
    urls = {r["gold_url"] for r in rows}
    assert g.validate(loaded, urls) == []


def test_flags_unknown_url_short_page_dupe():
    rows = [_row(1), _row(2, page_words=100), _row(3, gold_url=_row(1)["gold_url"])]
    errs = g.validate(rows, {rows[0]["gold_url"]})
    assert any("gh-docs-002: page_words" in e for e in errs)
    assert any("gh-docs-003: duplicate gold_url" in e for e in errs)
    assert any("gh-docs-002: gold_url not in" in e for e in errs)


def test_door_validates_the_committed_gold(capsys):
    import pytest
    from cytir import cli
    if not g.GOLD_PATH.is_file():
        pytest.skip("the gold set does not ship in this tree")
    assert cli.main(["ops", "github-docs-gold"]) == 0
    assert "0 problems" in capsys.readouterr().out


def test_door_validates_a_temp_gold(tmp_path, capsys):
    from cytir import cli
    p = tmp_path / "gold.jsonl"
    p.write_text("\n".join(json.dumps(_row(n)) for n in (1, 2, 3)) + "\n", encoding="utf-8")
    assert cli.main(["ops", "github-docs-gold", str(p)]) == 0
    assert "3 rows, 0 problems" in capsys.readouterr().out


def test_missing_gold_is_refused_not_a_traceback(tmp_path, capsys):
    assert g.main([str(tmp_path / "absent.jsonl")]) == 1
    assert "REFUSED: no gold set at" in capsys.readouterr().err

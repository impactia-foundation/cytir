#!/usr/bin/env python3
"""`dila_kali` never cuts an article (ADR-090 am., Tim 2026-10-07, ruling 27 T6 Q-b).

Real records only. `tests/fixtures/dila_kali_long/` holds one REAL in-force KALI article of 9,000
characters (47 paragraphs; read-only copy from the sealed KALI dump, sha in PROVENANCE.json).
`data/samples/dila-kali` holds real short articles. The old extractor cut every article at 4,000
characters (`MAX_ARTICLE_CHARS`): this one lost 55 % of its text.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

pytest.importorskip("lxml")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LONG = HERE / "fixtures" / "dila_kali_long" / "KALIARTI000052502103.xml"
SHORT = REPO / "data" / "samples" / "dila-kali" / "global" / "KALI" / "KALIARTI000005833790.xml"


def _tokens(text: str) -> int:
    return len(text) // 3          # stand-in: CI has no `transformers`; 3 chars per token


@pytest.fixture(autouse=True)
def _stand_in_tokenizer(monkeypatch):
    from cytir.sources.families import loda
    monkeypatch.setattr(loda, "embed_token_counter", lambda: _tokens)


def test_the_cap_is_gone_and_the_long_article_is_whole():
    from cytir.extract import dila_kali
    assert not hasattr(dila_kali, "MAX_ARTICLE_CHARS")
    rec, _key, why = dila_kali.parse_article_bytes(LONG.read_bytes())
    assert why == "kept"
    body = rec["text"].split("\n", 1)[1]
    assert len(body) > 8000 and not any(k.startswith("_") for k in rec)     # nothing cut, no private keys


def test_a_long_article_becomes_several_balanced_rows_that_hold_every_word():
    from cytir.extract import dila_kali
    rows, _key, why, source = dila_kali.parse_article_rows(LONG.read_bytes(), count_tokens=_tokens)
    assert why == "kept" and len(rows) >= 2
    head = source.split("\n", 1)[0]
    assert all(r["text"].startswith(head + "\n") for r in rows)             # header on every chunk
    assert all(len(r["text"]) <= 8000 and _tokens(r["text"]) <= 2560 for r in rows)
    sizes = [len(r["text"]) for r in rows]
    assert max(sizes) - min(sizes) < 0.35 * max(sizes)                      # balanced, not greedy
    assert [r["anchor"] for r in rows] == [
        f"article:KALIARTI000052502103:chunk-{n}" for n in range(1, len(rows) + 1)]
    assert len({r["url"] for r in rows}) == 1                               # one article, one url
    from cytir.core.common import mint_passage_point_id
    assert len({r["point_id"] for r in rows}) == len(rows)                  # distinct ids
    assert all(r["point_id"] == mint_passage_point_id(r["url"], r["anchor"], r["text"])
               for r in rows)                                               # text-anchor-v0
    joined = " ".join(r["text"].split("\n", 1)[1] for r in rows)
    assert joined.split() == source.split("\n", 1)[1].split()               # never cut, never reordered
    # the fewest rows: one fewer would not fit the 8,000-character row
    assert len(source) > 8000 * (len(rows) - 1)


def test_a_short_article_is_one_row_and_byte_identical_to_the_unsplit_record():
    from cytir.extract import dila_kali
    rec, _k, _w = dila_kali.parse_article_bytes(SHORT.read_bytes())
    rows, _key, _why, source = dila_kali.parse_article_rows(SHORT.read_bytes(), count_tokens=_tokens)
    assert rows == [rec] and "anchor" not in rows[0] and rows[0]["text"] == source


def test_a_long_article_without_a_token_counter_is_refused_not_guessed():
    from cytir.extract import dila_kali
    with pytest.raises(ValueError, match="count_tokens"):
        dila_kali.parse_article_rows(LONG.read_bytes())


def _dump(tmp_path, *files):
    n = len(list(tmp_path.glob("dump*")))
    top = tmp_path / f"dump{n}"
    root = top / "kali" / "global"
    root.mkdir(parents=True)
    for f in files:
        (root / f.name).write_bytes(f.read_bytes())
    return top


def _run(tmp_path, name, dump):
    pytest.importorskip("httpx")
    pytest.importorskip("qdrant_client")
    from cytir.sources import run_family
    return run_family.run_family(
        "dila_kali", out_dir=tmp_path / name, today="2026-10-07",
        consent=lambda h, d: pytest.fail(f"consent for {h}"),
        options={"root": str(dump)}, log=lambda m: None)


def test_the_family_stages_the_long_article_whole_with_zero_shortfall(tmp_path):
    man = _run(tmp_path, "o", _dump(tmp_path, LONG, SHORT))
    g = man["staged_text_guard"]
    assert man["contract"]["chunk"]["declared"] == "native-chunked"
    assert g["min_coverage"] == 1.0 and g["threshold_declared"] is True
    assert g["docs_checked"] == 2 and g["docs_shortfall"] == 0 and g["docs_unchecked"] == 0
    assert g["lowest_coverage"] == 1.0
    assert man["records"]["count"] >= 3 and "declared_caps" in g and g["declared_caps"] == []


def test_the_door_refuses_native_for_that_pack_and_lets_native_chunked_pass(tmp_path):
    from cytir.sources import family_stage as FS
    _run(tmp_path, "o", _dump(tmp_path, LONG, SHORT))
    pack = tmp_path / "o" / "pack"
    man = json.loads((pack / "manifest.json").read_text())
    man["consent"]["checked"] = True
    (pack / "manifest.json").write_text(json.dumps(man))
    d = FS.check_pack(pack, "dila_kali")["door_article_rows"]
    assert d["policy"] == "native-chunked" and d["articles"] == 2 and d["split_articles"] == 1
    man["contract"]["chunk"]["declared"] = "native"
    (pack / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(FS.StageError, match="'native' is one row per article"):
        FS.check_pack(pack, "dila_kali")


def test_the_preflight_asks_for_the_tokenizer_only_when_an_article_is_long(tmp_path, monkeypatch):
    from cytir.sources import run_family
    from cytir.sources.families import loda
    def broken():
        raise RuntimeError("no tokenizer")
    monkeypatch.setattr(loda, "embed_token_counter", broken)
    _run(tmp_path, "short-ok", _dump(tmp_path, SHORT))           # nothing long: no tokenizer needed
    with pytest.raises(run_family.RunError, match="may be over the embed window"):
        _run(tmp_path, "long-refused", _dump(tmp_path, LONG))

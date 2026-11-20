"""Task 1045 (review 1044 R5): the fedlex source-text guard compares ARTICLE BY ARTICLE.

Before: one doc per act-language, 8 forgiven 5-grams per extra row, so an act of ~80 rows forgave
~630 lost 5-grams: a whole lost article (104 or 196 words) or a lost tail window read 1.0, and an XML
with 0 articles read 1.0 too. Now: each article is compared with the rows at its base anchor, only the
5-grams on `window()` cut points are forgiven, and an act-language with 0 articles is refused.

The inputs are a REAL act (the sealed replies of one federal act, replayed by the package's tape
transport; no network). The mutants run through the real `run_family` with the real door.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest

httpx = pytest.importorskip("httpx")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cytir.sources import run_family as RF  # noqa: E402
from cytir.sources import staged_text_guard as STG  # noqa: E402
from cytir.sources.families import fedlex as FX  # noqa: E402
from cytir.sources.families import fedlex_contract as FC  # noqa: E402

TAPE = HERE / "fixtures" / "fedlex_offline" / "cc-1948-195.tape.json"
OK = {"bucket": "ok", "status": "ok", "search_ok": True, "train_ok": True,
      "checked_at": "2026-09-30T00:00:00Z"}
TODAY = "2026-09-30"
LONG = "art_1"          # inflated to ~600 extra words in the `inflate` variants: 3 windows (-c1..-c3)


def _tape(*, inflate: bool = False, blank_de: bool = False) -> dict:
    d = json.loads(TAPE.read_text(encoding="utf-8"))
    for t in d["tape"]:
        if not t["url"].endswith("de-xml.xml"):
            continue
        if inflate:      # ~600 unique extra words in the first article of the REAL German XML
            filler = " ".join(f"zusatzwort{i}" for i in range(600))
            t["body"] = t["body"].replace("</p></content></paragraph>", f" {filler}</p></content></paragraph>", 1)
            assert filler in t["body"]
        if blank_de:     # a well-formed act XML with no article at all
            t["body"] = re.sub(r"<article\b.*</article>", "", t["body"], flags=re.S)
            assert "<article" not in t["body"]
    return d


def _run(monkeypatch, tmp_path, tape: dict, *, chunk=None):
    from cytir.acquire.tape import Tape
    monkeypatch.setattr("time.sleep", lambda _s: None)
    monkeypatch.setattr(FX, "DOOR_SPLITTER", FX.DECLARED_SPLITTER)
    if chunk is not None:
        real = FX.FedlexFamily.chunk
        monkeypatch.setattr(FX.FedlexFamily, "chunk", lambda self, doc, ctx: chunk(list(real(self, doc, ctx))))
    t = Tape.load(tape)
    unexpected: list = []
    t.unexpected = unexpected
    man = RF.run_family("fedlex_exact", out_dir=tmp_path, today=TODAY, http=t.client(),
                        consent=lambda h, d: {**OK, "host": h},
                        options={"eli": tape["eli"], "langs": tape["langs"]}, log=lambda m: None)
    assert not unexpected
    return man


def _base(row) -> str:
    return FC.split_anchor(row["url"])[1]


def _drop_article(key):
    return lambda rows: [r for r in rows if _base(r) != key]


def _drop_tail_window(key):
    def f(rows):
        mine = [r for r in rows if _base(r) == key]
        if len(mine) < 3:            # only the inflated German article is windowed
            return rows
        return [r for r in rows if r is not mine[-1]]
    return f


def _cut_last_words(key, n):
    def f(rows):
        mine = [r for r in rows if _base(r) == key]
        if len(mine) >= 3:
            mine[-1]["text"] = " ".join(mine[-1]["text"].split()[:-n])
        return rows
    return f


# ---- (iv) the unmodified real act passes at 1.0 ------------------------------------------------
@pytest.mark.parametrize("inflate", [False, True])
def test_the_unmodified_real_act_passes_at_one(monkeypatch, tmp_path, inflate):
    man = _run(monkeypatch, tmp_path, _tape(inflate=inflate))
    g = man["staged_text_guard"]
    assert (g["docs_checked"], g["docs_unchecked"], g["docs_shortfall"]) == (3, 0, 0)
    assert g["lowest_coverage"] == 1.0 and g["min_coverage"] == 1.0 and g["threshold_declared"] is True
    if inflate:
        rows = [json.loads(x) for x in (tmp_path / "pack" / "records.jsonl").read_text(encoding="utf-8").splitlines()]
        de = [r for r in rows if r["lang"] == "de" and _base(r) == LONG]
        # really windowed (RULING 55, task 1040): every window at the BARE url, the place in `window`
        assert len(de) >= 3 and {FC.split_anchor(r["url"])[2] for r in de} == {None}
        assert [r["window"] for r in de] == [{"index": i, "count": len(de)} for i in range(1, len(de) + 1)]


# ---- (i)(ii) a lost article and a lost tail window are refused ---------------------------------
@pytest.mark.parametrize("name,mutant,inflate", [
    ("a whole article is dropped", _drop_article("art_3"), False),
    ("the tail window of a long article is dropped", _drop_tail_window(LONG), True),
    ("the last 3 words of the last window are cut", _cut_last_words(LONG, 3), True),
])
def test_a_lost_article_or_window_is_refused(monkeypatch, tmp_path, name, mutant, inflate):
    with pytest.raises(RF.RunError, match="0 records admitted.*staged_text_shortfall"):
        _run(monkeypatch, tmp_path, _tape(inflate=inflate), chunk=mutant)


def test_the_guard_names_the_article_that_fell_short(monkeypatch, tmp_path):
    seen: dict = {}
    real = FX.FedlexFamily.chunk

    def spy(self, doc, ctx):
        rows = list(real(self, doc, ctx))
        seen[doc.lang] = (doc, rows)
        return rows
    monkeypatch.setattr(FX.FedlexFamily, "chunk", spy)
    _run(monkeypatch, tmp_path, _tape(inflate=True))
    doc, rows = seen["de"]
    tail = [x for x in rows if _base(x) == LONG][-1]
    g = STG.StagedTextGuard("fedlex_exact", min_coverage=1.0)
    assert g.observe_parts("u", doc.source_parts, [r for r in rows if r is not tail], doc.part_of) < 1.0
    assert g.examples[0]["part"] == LONG and g.docs_shortfall == 1


# ---- (iii) 0 articles is refused at run start --------------------------------------------------
def test_an_act_language_with_no_article_is_refused(monkeypatch, tmp_path):
    with pytest.raises(RF.RunError, match="every one of 1 unit"):
        _run(monkeypatch, tmp_path, _tape(blank_de=True))


def test_fedlex_requires_records_and_the_guard_never_reads_an_empty_doc_as_covered():
    assert FX.FedlexFamily.require_records is True
    g = STG.StagedTextGuard("x", min_coverage=1.0)
    assert g.observe_parts("u", [], [], lambda r: "") == 0.0 and g.docs_shortfall == 1


# ---- the rule itself, on the review's own numbers ----------------------------------------------
def test_the_review_1044_simulation_now_reads_below_one():
    parts = [STG.SourcePart(f"art_{i}", " ".join(f"a{i}w{j}" for j in range(120))) for i in range(77)]
    rows = [{"url": f"https://x/eli/cc/1/fr#art_{i}", "text": p.text} for i, p in enumerate(parts)]
    key = lambda r: FC.split_anchor(r["url"])[1]       # noqa: E731
    g = STG.StagedTextGuard("x", min_coverage=1.0)
    assert g.observe_parts("u", parts, rows, key) == 1.0
    assert g.observe_parts("u", parts, rows[:-1], key) < 1.0      # an article of 120 words
    assert g.observe_parts("u", parts, rows[1:], key) < 1.0
    assert g.docs_checked == 3 and g.docs_shortfall == 2


def test_only_window_cut_points_are_forgiven():
    body = " ".join(f"w{i}" for i in range(700))
    windows = [t for _a, t in FX.window(body, "art_1")]
    cuts, at = [], 0
    for w in windows[:-1]:
        at += len(w.split())
        cuts.append(at)
    part = STG.SourcePart("art_1", body, tuple(cuts))
    assert STG.part_coverage(part, windows) == 1.0
    assert STG.part_coverage(part, windows[:-1]) < 1.0
    assert STG.part_coverage(part, windows[:-1] + [" ".join(windows[-1].split()[:-1])]) < 1.0
    assert STG.part_coverage(STG.SourcePart("art_1", body), windows) < 1.0    # no declared cuts: the cut costs

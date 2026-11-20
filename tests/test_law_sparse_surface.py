"""The law packs' SPARSE surface, replayed on the rows that lost a query term.

Task 562 attributed 24 COMPOSITION + 3 MIXED golds of the held-out sparse-only
leg to family composition. Task 564 measured the seven fedlex ones against the
family's own sealed pack records and found two defects in the epoch-4
re-materialization, both of them invisible to every count check:

  1. THE ANCESTOR HEADING PATH IS GONE FROM THE INDEXED SURFACE.
     `fedlex_dump.parse_akn` composes the law's sparse channel as
     `"; ".join(ancestor_heading_path(article)) + ". " + body`, and says why in
     the comment beside it (task 358b: truncating the path cuts OR art. 257's
     query-relevant "Die Miete" ancestor; it is a SPARSE-ONLY hook).
     `materialize_r0_fedlex_anchored.render()` emits the breadcrumb as FLAT
     sibling `## <label> {#anchor}` headings instead, so
     `generic_chunking`'s hierarchy channel never attaches it to the article's
     own chunk. Five of the seven golds lose exactly a word that is in the
     ancestor path and in no channel of the sealed record.

  2. THE AKN eId IDENTIFIERS ARE IN THE INDEXED TEXT.
     22,927 of 22,927 pack rows carry `{#art_N}` in `text` AND in
     `sparse_text`; the twin carries none. `aw_common.bm25_tokens` splits on
     `\\w+`, so `{#art_341}` becomes the token `art_341`, which no query can
     produce, and it counts toward the `dl` that discounts every real term.

The fixtures are the VERBATIM pack records plus the ancestor path re-derived
from the pinned AKN XML each record names (digest checked at seal time) --
task 564's fixture-sealing script. A test on a hand-written
example proves nothing about the incident it claims to prevent (task 527 §7).
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cytir.sources.families import fedlex_akn as FA  # noqa: E402
from cytir.corpus import law_sparse_surface as RS  # noqa: E402
from cytir.index import representation_contract as RC  # noqa: E402
# tok@2 (AW_FOLD_ACCENTS=1) is an aw_b1 INVARIANT and it is not this
# module's to choose: importing `aw_b1_projection` is how every write-side
# caller in the repo pins it, so the tokeniser here hashes exactly what the
# index hashes. Task 564 measured what happens without it -- the accented
# half of three golds' terms read as ABSENT from postings that hold them.
import cytir.core.projection as _PIN_TOK2  # noqa: E402,F401
from cytir.core.common import bm25_tokens, compose_index_text  # noqa: E402

FIXTURE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "tests", "fixtures",
                                        "law_sparse_pack_rows.json"))


def rows():
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)["rows"]


def _tokens(text, lang):
    return set(bm25_tokens(text, lang))


# --- the two defects, as they shipped --------------------------------------

def test_shipped_pack_surfaces_carry_the_akn_identifiers():
    """Both channels of every sealed row leak the eId scaffolding."""
    fx = rows()
    assert len(fx) == 7
    for r in fx:
        for field in ("text", "sparse_text", "canonical_text"):
            assert FA.ANCHOR_MARKUP.search(r["pack_row"][field]), (
                r["gold_id"], field)


def test_shipped_pack_surfaces_do_not_carry_the_ancestor_path():
    """The five golds whose lost word is in the ancestor path: the shipped
    surfaces do not carry it, in ANY channel -- which is why no re-ranking
    could have found it."""
    recovered = 0
    for r in rows():
        lang = r["payload"]["lang"]
        shipped = {f: _tokens(r["pack_row"][f], lang)
                   for f in ("text", "sparse_text", "canonical_text")}
        ctx = _tokens(r["akn"]["heading_ctx"], lang)
        for word in r["gold_words_lost"]:
            wk = _tokens(word, lang)
            if wk and wk <= ctx:
                recovered += 1
                assert not any(wk <= t for t in shipped.values()), (
                    r["gold_id"], word)
    assert recovered == 6, recovered   # 6 words over 5 golds


# --- the fix ---------------------------------------------------------------

def test_strip_anchor_markup_reproduces_the_body_production_composed():
    """`strip_anchor_markup` over the pack body yields the same TOKENS as
    `clean_text(article)` -- production's own body -- for every sealed row.
    Two independent derivations of one surface: the epoch-4 markdown cleaned,
    and the pinned XML read."""
    for r in rows():
        lang = r["payload"]["lang"]
        got = _tokens(FA.strip_anchor_markup(r["pack_row"]["text"]), lang)
        want = _tokens(r["akn"]["clean_text"], lang)
        assert want <= got, (r["gold_id"], sorted(want - got)[:8])


def test_strip_anchor_markup_removes_every_identifier_and_its_header_line():
    for r in rows():
        out = FA.strip_anchor_markup(r["pack_row"]["text"])
        assert not FA.ANCHOR_MARKUP.search(out), r["gold_id"]
        # the structural element names the headers carried go with them
        assert "listIntroduction" not in out
        assert "\nparagraph\n" not in out


def test_recomposed_sparse_surface_restores_the_lost_query_terms():
    """The fix, on the golds: the ancestor-path words come back, and the
    article's own body is still there.

    Composed through `cytir.corpus.law_sparse_surface.resurfaced_sparse_text` -- the
    function the stage writer calls -- so this exercises the shipped path
    rather than a second copy of it."""
    for r in rows():
        lang = r["payload"]["lang"]
        surface = RS.resurfaced_sparse_text(r["pack_row"], r["akn"]["heading_ctx"])
        composed = compose_index_text({
            "text": surface, "title": r["payload"]["title"],
            "site_name": r["payload"]["site_name"],
            "source_type": r["payload"]["source_type"]})
        got = _tokens(composed, lang)
        ctx = _tokens(r["akn"]["heading_ctx"], lang)
        for word in r["gold_words_lost"]:
            wk = _tokens(word, lang)
            if wk and wk <= ctx:
                assert wk <= got, (r["gold_id"], word)
        assert _tokens(r["akn"]["clean_text"], lang) <= got, r["gold_id"]
        assert not FA.ANCHOR_MARKUP.search(composed), r["gold_id"]


def test_compose_law_sparse_surface_is_the_live_loaders_own_format():
    """Byte-identity with the string `fedlex_dump.parse_akn` used to build
    inline. A mirror contract is only a contract if it is asserted."""
    assert FA.compose_law_sparse_surface("A; B", "body") == "A; B. body"
    assert FA.compose_law_sparse_surface("", "body") == "body"
    assert FA.compose_law_sparse_surface("   ", "body") == "body"


# --- the guard that would have caught it ------------------------------------

def test_representation_contract_refuses_the_shipped_fedlex_surface():
    """The guard fires on the rows as they shipped -- on BOTH channels."""
    fam = {"dense": "title_body", "sparse": "hierarchy_body",
           "refused_index_markup": ["akn_anchor_id"]}
    for r in rows():
        pr, pl = r["pack_row"], r["payload"]
        rec = {"text": pr["text"], "title": pl["title"],
               "site_name": pl["site_name"], "source_type": pl["source_type"],
               "heading_ctx": r["akn"]["heading_ctx"]}
        checks = {v["check"] for v in RC.check_row(
            rec, fam, embedded=pr["text"], sparse_surface_text=pr["sparse_text"])}
        assert "markup.dense" in checks, r["gold_id"]
        assert "markup.sparse" in checks, r["gold_id"]


def test_representation_contract_passes_the_recomposed_surface():
    fam = {"dense": "title_body", "sparse": "hierarchy_body",
           "refused_index_markup": ["akn_anchor_id"]}
    for r in rows():
        pr, pl = r["pack_row"], r["payload"]
        body = FA.strip_anchor_markup(pr["text"])
        rec = {"text": body, "title": pl["title"],
               "site_name": pl["site_name"], "source_type": pl["source_type"],
               "heading_ctx": r["akn"]["heading_ctx"], "sparse_body": body}
        sparse = FA.compose_law_sparse_surface(r["akn"]["heading_ctx"], body)
        dense = compose_index_text(rec)
        checks = {v["check"] for v in RC.check_row(
            rec, fam, embedded=dense, sparse_surface_text=sparse)}
        assert "markup.dense" not in checks, r["gold_id"]
        assert "markup.sparse" not in checks, r["gold_id"]
        assert "conformance.sparse" not in checks, r["gold_id"]


def test_the_instrument_hashes_what_the_index_hashes():
    """tok@2 must be ON in this process, or every accented term in these
    assertions is a different hash from the one in the collection.

    This is the check task 564 needed and did not have. Three golds were first
    reported with their lost word "present in the sealed stage text but absent
    from the stored posting" -- identical term COUNT, jaccard 0.60, and every
    disagreeing term accented (`chômage`, `fédérale`, `solidarité`). The
    measurement was the defect, not the data; the fix is the
    `aw_b1_projection` import at the head of this module, and this test is what
    stops it being dropped again.
    """
    import cytir.core.bm25_tokenizer as TK
    assert TK._FOLD_ACCENTS is True
    assert bm25_tokens("chômage", "fr") == bm25_tokens("chomage", "fr")


def test_unknown_markup_class_is_a_refusal_not_a_skip():
    fam = {"dense": "body_only", "sparse": "body_only",
           "refused_index_markup": ["no_such_class"]}
    try:
        RC.check_row({"text": "x"}, fam, embedded="x", sparse_surface_text="x")
    except RC.ContractError:
        return
    raise AssertionError("an undeclared markup class must refuse, not pass")


def test_the_declared_family_now_names_this_surface_and_refuses_the_markup():
    """The REGISTRY, not a local dict: `law_article_pack` must actually declare
    `hierarchy_body` and `akn_anchor_id`, or the checks above are exercising a
    family nothing in the pipeline resolves to."""
    fam = RC.load_registry()["families"]["law_article_pack"]
    assert fam["sparse"] == "hierarchy_body"
    assert "akn_anchor_id" in fam.get("refused_index_markup", [])
    assert fam["sparse_surface_field"] == "sparse_text"
    assert "fedlex-r0-exact-current-v1" in fam["applies_to"]


def test_a_law_row_with_no_hierarchy_cannot_pass_its_declared_sparse_surface():
    """A template that quietly fell back to the body would have PASSED the
    epoch-4 pack, whose `sparse_text` is the law title repeated per section and
    carries no ancestor path at all."""
    fam = RC.load_registry()["families"]["law_article_pack"]
    r = rows()[0]
    rec = {"text": r["pack_row"]["text"], "title": r["payload"]["title"],
           "site_name": r["payload"]["site_name"],
           "source_type": r["payload"]["source_type"]}     # no heading_ctx
    checks = {v["check"] for v in RC.check_row(
        rec, fam, embedded=rec["text"],
        sparse_surface_text=r["pack_row"]["sparse_text"])}
    assert "conformance.sparse" in checks

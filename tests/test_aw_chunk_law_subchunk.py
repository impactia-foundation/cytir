#!/usr/bin/env python3
"""`chunk.law.sub` on REAL law records, and the guard that refuses a bad cut.

Task 573 / E573-1. The fixtures under `tests/fixtures/law_subchunk/` are six
real rows sealed once from the corpus store by
the task-573 sealing script -- a long and a short fedlex article, a
long and a short legifrance article, a long gesetze-im-internet article and a
long Swiss cantonal article. Sealed inputs, not live calls (memory:
`fixes-land-in-the-package-with-guard`).

The tests here are the ones that would have caught the defects this policy can
introduce:
  * a sub-chunk that lost its article's anchor -- the passage the URL fragment
    no longer reaches;
  * a sub-chunk that lost the article's identity from its embedded text -- the
    epoch-4 fedlex surface defect (tasks 524-527), one generation later;
  * two sub-chunks of one article minting the same point_id -- the GAP A33
    collapse;
  * a SHORT article whose text or id moved at all.
"""
from __future__ import annotations

import json
import pathlib

import pytest

import cytir.core.ids as ident
from cytir.chunk import chunker as CH
import cytir.chunk.registry as REG
from cytir.chunk import law_subchunk as LS
from cytir.chunk.policy import ChunkPolicyError, load_chunk_policy

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "law_subchunk"
LONG = ["fedlex_long_de", "legifrance_long_fr", "gesetze_long_de",
        "cantonal_long_de"]
SHORT = ["fedlex_short_de", "legifrance_short_fr"]


def fx(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def cut(rec: dict, **kw):
    return LS.subchunk_article(
        rec["canonical_url"], rec["text"], anchor=rec.get("anchor"),
        article_id=rec["article_id"], law_id=rec.get("law_id"),
        title=rec.get("title"), source_type=rec.get("source_type"), **kw)


# --- the fixtures are what they claim to be --------------------------------

def test_fixtures_exist_and_are_real_records():
    names = sorted(p.stem for p in FIXTURES.glob("*.json"))
    assert names == sorted(LONG + SHORT), names
    for n in names:
        r = fx(n)
        assert r["sealed_from"]["text_sha256"], n
        assert r["url"].startswith("http"), n
        assert r["article_id"].strip(), n


@pytest.mark.parametrize("name", LONG)
def test_long_fixtures_are_above_the_policy_tail(name):
    p = LS.policy()
    assert CH.word_count(fx(name)["text"]) > p.word_hard_max


@pytest.mark.parametrize("name", SHORT)
def test_short_fixtures_are_at_or_below_the_policy_tail(name):
    p = LS.policy()
    assert CH.word_count(fx(name)["text"]) <= p.word_hard_max


# --- the policy artifact and its registry entry ----------------------------

def test_policy_is_registered_as_a_law_sub_family_and_is_not_yet_current():
    p = LS.policy()
    assert REG.family_of(p.sha256) == "chunk.law.sub"
    # Ratifying it is ADR-090 amendment 2 and is Tim's merge, not a script's.
    assert REG.status_of(p.sha256) == "proposed"
    with pytest.raises(Exception):
        REG.assert_selectable(p)                 # a normal lane may not take it
    assert REG.assert_selectable(p, experiment=True) == "chunk.law.sub"


def test_registry_and_the_artifacts_on_disk_agree_in_both_directions():
    assert REG.verify_against_files() == []


def test_policy_refuses_a_split_threshold_that_is_not_its_own_tail(tmp_path):
    body = json.loads((LS.REPO / LS.POLICY_PATH).read_text())
    body["law_article"]["split_body_above_words"] = 500
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(body))
    with pytest.raises(ChunkPolicyError):
        load_chunk_policy(bad)


def test_policy_declares_the_identity_rule_as_data_not_as_code():
    p = LS.policy()
    assert p.law_article == {
        "unit": "articles",
        "boundary": "source-native-anchor",
        "split_body_above_words": p.word_hard_max,
        "short_article": "emit-whole-unchanged",
        "anchor_carried": "every-sub-chunk",
        "identity": "anchor-plus-sub-ordinal",
    }


# --- short articles are unchanged, in text AND in id -----------------------

@pytest.mark.parametrize("name", SHORT)
def test_short_article_is_emitted_byte_identically_with_todays_point_id(name):
    r = fx(name)
    subs = cut(r)
    assert len(subs) == 1
    assert subs[0].text == r["text"]
    assert subs[0].n_sub == 1
    assert subs[0].split is False
    # The id `chunk.law` mints today, re-derived here rather than trusted.
    digest = ident.content_digest(r["text"])
    tier, comp = ident.classify_tier(r.get("source_type"), r.get("anchor"),
                                     True, digest, title=r.get("title"))
    assert subs[0].point_id == ident.mint_point_id(r["canonical_url"], comp)
    assert subs[0].anchor_component == comp


# --- long articles: split, and every slice is still that article -----------

@pytest.mark.parametrize("name", LONG)
def test_long_article_splits_and_every_slice_keeps_the_article(name):
    r = fx(name)
    subs = cut(r)
    assert len(subs) > 1, name
    LS.assert_subchunk_contract(subs, anchor=r.get("anchor"),
                                article_id=r["article_id"],
                                article_text=r["text"])
    want_anchor = None if r.get("anchor") in (None, "") else str(r["anchor"])
    assert {s.anchor for s in subs} == {want_anchor}
    assert {s.article_id for s in subs} == {r["article_id"]}
    assert {s.law_id for s in subs} == {r.get("law_id")}
    assert len({s.point_id for s in subs}) == len(subs)
    assert [s.ordinal for s in subs] == list(range(1, len(subs) + 1))


@pytest.mark.parametrize("name", LONG)
def test_every_slice_of_a_split_article_names_its_law(name):
    r = fx(name)
    subs = cut(r)
    prefix = subs[0].identity_prefix
    assert prefix, name
    for s in subs:
        assert prefix in s.text, (name, s.ordinal)


@pytest.mark.parametrize("name", LONG)
def test_no_slice_is_wider_than_the_policy_tail_by_more_than_the_prefix(name):
    p = LS.policy()
    r = fx(name)
    subs = cut(r)
    prefix_words = CH.word_count(subs[0].identity_prefix)
    for s in subs:
        assert s.n_words <= p.word_hard_max + prefix_words, (name, s.ordinal,
                                                             s.n_words)


@pytest.mark.parametrize("name", LONG)
def test_no_slice_is_a_bare_identity_line(name):
    """The legifrance fixture produced an 8-word title-only slice before the
    prefix was peeled off the body; a slice with no body is not a passage."""
    r = fx(name)
    subs = cut(r)
    for s in subs:
        body = s.text[len(s.identity_prefix):].strip() \
            if s.text.startswith(s.identity_prefix) else s.text
        assert CH.word_count(body) >= 15, (name, s.ordinal, s.text[:120])


@pytest.mark.parametrize("name", LONG)
def test_a_tier_a_article_puts_the_sub_ordinal_in_the_identity_component(name):
    r = fx(name)
    subs = cut(r)
    if subs[0].tier != "A":
        pytest.skip("tier B: the per-chunk digest already separates the slices")
    base = str(r["anchor"]).strip()
    for s in subs:
        assert s.anchor_component == f"{base}{ident.LAW_SUB_SEPARATOR}{s.ordinal}"


@pytest.mark.parametrize("name", LONG)
def test_the_control_arm_changes_the_boundary_and_nothing_else(name):
    """`compose_prefix=False` must produce the same boundaries as the full arm,
    so the experiment's third arm isolates the prefix from the width."""
    r = fx(name)
    with_prefix = cut(r)
    without = cut(r, compose_prefix=False)
    assert len(with_prefix) == len(without), name
    prefix = with_prefix[0].identity_prefix
    for a, b in zip(with_prefix, without):
        # identical boundaries, and the only difference is the re-stated prefix
        assert a.text in (b.text, f"{prefix}\n{b.text}"), (name, a.ordinal)
        assert a.ordinal == b.ordinal and a.n_sub == b.n_sub


@pytest.mark.parametrize("name", LONG)
def test_no_word_of_the_article_is_lost_in_a_split(name):
    r = fx(name)
    subs = cut(r)
    joined = set(" ".join(s.text for s in subs).split())
    lost = [w for w in r["text"].split() if w not in joined]
    assert not lost, (name, lost[:8])


@pytest.mark.parametrize("name", LONG)
def test_the_control_arm_loses_only_the_header_it_was_built_to_drop(name):
    """The `compose_prefix=False` arm is allowed to lose exactly the identity
    prefix and nothing else -- that is what it is for."""
    r = fx(name)
    full = cut(r)
    ctl = cut(r, compose_prefix=False)
    joined = set(" ".join(s.text for s in ctl).split())
    prefix_words = set(full[0].identity_prefix.split())
    lost = [w for w in r["text"].split()
            if w not in joined and w not in prefix_words]
    assert not lost, (name, lost[:8])


# --- the guard refuses what it exists to refuse ----------------------------

def _subs_for_guard(name: str = "fedlex_long_de"):
    r = fx(name)
    return r, cut(r)


def test_guard_refuses_a_sub_chunk_that_lost_its_anchor():
    r, subs = _subs_for_guard()
    broken = [subs[0]] + [
        LS.SubChunk(**{**s.as_dict(), "anchor": None}) for s in subs[1:]]
    with pytest.raises(LS.LawSubchunkError, match="lost its anchor"):
        LS.assert_subchunk_contract(broken, anchor=r["anchor"],
                                    article_id=r["article_id"],
                                    article_text=r["text"])


def test_guard_refuses_a_sub_chunk_that_lost_the_law_identity():
    r, subs = _subs_for_guard()
    broken = [subs[0]] + [
        LS.SubChunk(**{**s.as_dict(), "text": "1 Ein Satz ohne Gesetzesnamen."})
        for s in subs[1:]]
    with pytest.raises(LS.LawSubchunkError, match="identity prefix"):
        LS.assert_subchunk_contract(broken, anchor=r["anchor"],
                                    article_id=r["article_id"],
                                    article_text=r["text"])


def test_guard_refuses_two_sub_chunks_sharing_a_point_id():
    r, subs = _subs_for_guard()
    broken = [subs[0]] + [
        LS.SubChunk(**{**s.as_dict(), "point_id": subs[0].point_id})
        for s in subs[1:]]
    with pytest.raises(LS.LawSubchunkError, match="share a point_id"):
        LS.assert_subchunk_contract(broken, anchor=r["anchor"],
                                    article_id=r["article_id"],
                                    article_text=r["text"])


def test_guard_refuses_a_short_article_whose_text_moved():
    r = fx("fedlex_short_de")
    subs = cut(r)
    broken = [LS.SubChunk(**{**subs[0].as_dict(),
                             "text": subs[0].text + " extra"})]
    with pytest.raises(LS.LawSubchunkError, match="byte-identically"):
        LS.assert_subchunk_contract(broken, anchor=r["anchor"],
                                    article_id=r["article_id"],
                                    article_text=r["text"])


def test_guard_refuses_dropped_text():
    """A cut that silently drops its last slice keeps a coherent ordinal
    sequence and a balanced n_sub -- and still lost the end of the article."""
    r, subs = _subs_for_guard("gesetze_long_de")   # 3 slices: dropping one
    assert len(subs) >= 3                          # still leaves a plural cut
    kept = [LS.SubChunk(**{**s.as_dict(), "n_sub": len(subs) - 1})
            for s in subs[:-1]]
    with pytest.raises(LS.LawSubchunkError, match="text was lost"):
        LS.assert_subchunk_contract(kept, anchor=r["anchor"],
                                    article_id=r["article_id"],
                                    article_text=r["text"])


def test_guard_refuses_a_deletion_that_padding_would_have_hidden():
    """L573-1. The first version of this guard compared the SUM of the slices'
    word counts to the article's. Every slice repeats the identity prefix, so a
    slice could lose real sentences and the total still come out higher than the
    article started -- padding hiding a deletion. The multiset check does not
    care how long the slices are, only which words are in them."""
    r, subs = _subs_for_guard("gesetze_long_de")
    pad = (subs[0].identity_prefix + " ") * 40
    broken = [subs[0]] + [
        LS.SubChunk(**{**s.as_dict(), "text": subs[0].identity_prefix + "\n" + pad,
                       "n_words": CH.word_count(pad)}) for s in subs[1:]]
    assert sum(s.n_words for s in broken) > CH.word_count(r["text"])  # padded up
    with pytest.raises(LS.LawSubchunkError, match="text was lost"):
        LS.assert_subchunk_contract(broken, anchor=r.get("anchor"),
                                    article_id=r["article_id"],
                                    article_text=r["text"])


def test_guard_accepts_the_no_prefix_arm_whose_header_was_peeled():
    """The same guard must NOT refuse the control arm: peeling the article's own
    header line off before the split is the policy, not a loss. 390 real
    cantonal articles were refused by the sum check for exactly this."""
    for name in LONG:
        r = fx(name)
        subs = cut(r, compose_prefix=False)
        LS.assert_subchunk_contract(subs, anchor=r.get("anchor"),
                                    article_id=r["article_id"],
                                    article_text=r["text"])


def test_guard_refuses_an_empty_cut():
    with pytest.raises(LS.LawSubchunkError, match="zero sub-chunks"):
        LS.assert_subchunk_contract([], anchor=None, article_id="art_1",
                                    article_text="x")


def test_cutting_refuses_an_article_with_no_identity_at_all():
    with pytest.raises(LS.LawSubchunkError, match="identity prefix"):
        LS.subchunk_article("https://example.org/law", "word " * 400,
                            anchor="art_1", article_id="art_1", title=None,
                            identity_prefix="")


def test_cutting_refuses_an_article_with_no_article_id():
    r = fx("fedlex_long_de")
    with pytest.raises(LS.LawSubchunkError, match="article_id is required"):
        LS.subchunk_article(r["canonical_url"], r["text"],
                            anchor=r.get("anchor"), article_id="",
                            title=r.get("title"))


# --- the identity helper ---------------------------------------------------

def test_sub_component_is_a_no_op_for_a_single_chunk_article():
    assert ident.law_sub_anchor_component("art_336", 1, 1) == "art_336"


def test_sub_component_is_distinct_from_the_collision_salt():
    sub = ident.law_sub_anchor_component("art_336", 2, 3)
    salt = ident.salted_anchor_component("art_336", 2)
    assert sub != salt
    assert sub == "art_336~s2"


@pytest.mark.parametrize("ordinal,n", [(0, 3), (4, 3), (-1, 2), (1, 0)])
def test_sub_component_refuses_an_impossible_ordinal(ordinal, n):
    with pytest.raises(ValueError):
        ident.law_sub_anchor_component("art_336", ordinal, n)


# --- the self-check is wired and can fail ----------------------------------

def test_self_check_runs_every_fixture_and_returns_zero():
    assert LS._self_check() == 0


@pytest.fixture(autouse=True)
def _pure_algorithm_word_only(word_only_chunking_allowed):
    """These tests exercise the pure word-boundary algorithm on a box that may lack
    `transformers`. Generic `chunk_markdown` refuses without the tokenizer (task 2395),
    so the module opts in once, visibly, via the conftest fixture (task 2289)."""

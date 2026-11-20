"""Tim's register-first card ruling, replayed against the sealed Vector8 rows.

The fixtures in `testdata/task528a_vector8_card_rows.json` are the VERBATIM
preserved source-evidence rows for `2a5bc752-ce61-58e7-9453-d63302b86e06` from
`_preservation/entity-card-current-lineage-witness-v3/source-evidence`, in the
task-467 producer's own PRIORITY order.  A test that only exercises a
hand-written example proves nothing about the incident it claims to prevent
(task 527 §7), so the incident is replayed on the row that caused it.
"""
from __future__ import annotations

import json
import os

from cytir.corpus import card_channels as ECC  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata",
                       "task528a_vector8_card_rows.json")


def vector8_rows():
    with open(FIXTURE, encoding="utf-8") as fh:
        doc = json.load(fh)
    return [(r["source_label"], r["row"]) for r in doc["rows"]]


def test_website_rows_are_recognised_from_their_own_declarations():
    rows = dict(vector8_rows())
    assert ECC.is_website_derived(rows["task393_apply"]) is True
    assert ECC.is_website_derived(rows["task363_people"]) is False
    assert ECC.is_website_derived(rows["task363_register"]) is False


def test_vector8_dense_is_the_register_composition_not_the_website_blurb():
    res = ECC.resolve_channels(vector8_rows())
    assert res["dense_class"] == ECC.DENSE_REGISTER
    assert res["dense_source"] == "task363_people"
    dense = res["dense_text"]
    # the register purpose, then every officer, in full -- not truncated at
    # 1,600 chars mid-name the way the shipped surface was ("… Marc Maret (…); Ni")
    assert dense.startswith("Vector8 AG — Entwicklung massgeschneiderter")
    assert ECC.OFFICER_CLAUSE in dense
    for officer in ("Marc Maret", "Emanuele Diquattro", "Niclas Delfs",
                    "Raphael Schwarz"):
        assert officer in dense
    assert "Our mission is to elevate organizations" not in dense


def test_a_website_blurb_never_wins_the_dense_channel():
    """The refusal, isolated: the website row alone still loses to a synth."""
    website = {"adr060_phase": "task393-website-content", "title": "X AG",
               "dense_text": "X AG — marketing copy about our mission.",
               "snippet": "marketing copy"}
    register = {"title": "X AG", "register_id": "CHE-111.111.111",
                "snippet": "Handel mit Waren aller Art."}
    res = ECC.resolve_channels([("task393_apply", website),
                                ("task363_register", register)])
    assert res["dense_class"] == ECC.DENSE_SYNTH
    assert res["dense_text"] == "X AG — Handel mit Waren aller Art."
    # ...and with nothing else at all, the website text is still admissible
    # rather than the card being dropped -- it is refused only while a
    # register-, description- or synth-class surface exists.
    only = ECC.resolve_channels([("task393_apply", website)])
    assert only["dense_class"] == ECC.DENSE_WEBSITE


def test_vector8_sparse_is_the_register_hook_bag():
    res = ECC.resolve_channels(vector8_rows())
    assert res["sparse_class"] == ECC.SPARSE_REGISTER_SEALED
    assert res["sparse_source"] == "task363_people"
    sparse = res["sparse_text"]
    assert sparse == ("Vector8 AG CHE341859062 marc maret emanuele diquattro "
                      "niclas delfs raphael schwarz")
    # the two hooks Tim's live query needs and the shipped posting lost
    assert "CHE341859062" in sparse
    assert "marc maret" in sparse
    # ADR-060 §(b): hooks, never sentences
    assert "Our mission" not in sparse


def test_uid_token_folds_the_register_id_into_one_bm25_token():
    assert ECC.uid_token({"register_id": "CHE-341.859.062"}) == "CHE341859062"
    assert ECC.uid_token({"uid": "CHE341859062"}) == "CHE341859062"
    assert ECC.uid_token({"register_id": "None"}) == ""
    assert ECC.uid_token({}) == ""


def test_composed_sparse_keeps_hooks_in_their_own_bm25_segment():
    """Without the segment boundary the hook is diluted by the description --
    the length-normalisation half of the Marc Maret mechanism (task 527 §5)."""
    import cytir.core.projection as PROJ

    row = {"title": "X AG", "register_id": "CHE-111.111.111",
           "people_tokens": ["ada", "lovelace", "ada lovelace"],
           "snippet": "Handel mit Waren aller Art und Erbringung von "
                      "Dienstleistungen im Bereich der Informatik."}
    res = ECC.resolve_channels([("task363_register", row)])
    assert res["sparse_class"] == ECC.SPARSE_COMPOSED
    assert PROJ.SPARSE_SEGMENT_SEPARATOR in res["sparse_text"]
    hooks, _, rest = res["sparse_text"].partition(PROJ.SPARSE_SEGMENT_SEPARATOR)
    assert "CHE111111111" in hooks and "lovelace" in hooks
    # the description is still covered, so sparse is not a subset of dense
    assert "Informatik" in rest

    segmented = PROJ.saturated_bm25(res["sparse_text"], "de")
    flat = PROJ.saturated_bm25(
        res["sparse_text"].replace(PROJ.SPARSE_SEGMENT_SEPARATOR, " "), "de")
    import cytir.core.common as AC
    uid = AC._hash(AC._fold("che111111111"))
    seg = dict(zip(*segmented))
    fl = dict(zip(*flat))
    assert seg[uid] > fl[uid], "segmenting must not weaken the identity hook"


def test_a_card_with_no_register_identity_still_gets_its_NAME_as_a_hook():
    """No UID, no officers -- but the card has a name, and ADR-060 §(a) measured
    the name as a card's route into the pool. The measured instance: the
    LMArena card's title says "LLM" and its prose only ever says "large
    language models"."""
    import cytir.core.projection as PROJ

    row = {"title": "LMArena — Chatbot Arena LLM Leaderboard",
           "dense_text": "LMArena, formerly Chatbot Arena — the crowdsourced "
                         "leaderboard that ranks large language models."}
    res = ECC.resolve_channels([("existing_seed", row)])
    assert res["sparse_class"] == ECC.SPARSE_NAME_ONLY
    hooks, _, rest = res["sparse_text"].partition(PROJ.SPARSE_SEGMENT_SEPARATOR)
    assert "LLM" in hooks and "LMArena" in hooks
    assert "large language models" in rest


def test_a_card_with_no_name_at_all_gets_no_invented_hook_bag():
    res = ECC.resolve_channels([("existing_seed", {"text": "some prose"})])
    assert res["sparse_class"] == ECC.SPARSE_NONE
    assert res["sparse_text"] == ""


def test_tombstone_when_no_row_carries_any_text():
    assert ECC.resolve_channels([("existing_seed", {"url": "https://x/"})]) is None


# --- ADR-060a's ruled composition (task 528a-2) ------------------------------
#
# `compose_dense_with_website` bounds the website half with the PINNED Arctic
# tokenizer and REFUSES to approximate the budget when it is unavailable, so
# the tests that exercise the tail need `transformers` installed. They run for
# real on the GPU box (where the whole suite runs) and skip loudly elsewhere --
# named here so "12 passed" on a runner without the tokenizer is not read as
# coverage it does not have.


def needs_pinned_tokenizer():
    import pytest
    pytest.importorskip(
        "transformers",
        reason="ADR-060a's website tail is bounded by the pinned Arctic "
               "tokenizer, which refuses to approximate the budget; install "
               "the pinned requirements to exercise it")

def test_the_ruled_composition_puts_the_register_first_and_the_website_after():
    """Tim allowed "at most a bounded description sentence AFTER the register
    text". Both halves of his ruling hold because of the ORDER: identity, the
    register purpose and every officer are the head and come from the register;
    the website can only follow them."""
    needs_pinned_tokenizer()
    res = ECC.resolve_channels(vector8_rows(), website_tail=True)
    assert res["dense_class"] == ECC.DENSE_REGISTER_PLUS_WEBSITE
    assert res["website_tail_source"] == "task393_apply"
    dense = res["dense_text"]

    # the register half is the head, verbatim and complete
    reg = res["dense_register_only"]
    assert dense.startswith(reg)
    assert reg.startswith("Vector8 AG — Entwicklung massgeschneiderter")
    for officer in ("Marc Maret", "Emanuele Diquattro", "Niclas Delfs",
                    "Raphael Schwarz"):
        assert officer in dense
        assert dense.index(officer) < dense.index("Our mission")

    # the website half follows, and supplies description only
    assert "Our mission is to elevate organizations" in dense

    # the SPARSE channel does not move: it is still the register hook bag
    assert res["sparse_class"] == ECC.SPARSE_REGISTER_SEALED
    assert res["sparse_text"] == ("Vector8 AG CHE341859062 marc maret emanuele "
                                  "diquattro niclas delfs raphael schwarz")


def test_the_tail_is_off_by_default():
    assert ECC.resolve_channels(vector8_rows())["dense_class"] == ECC.DENSE_REGISTER


def test_the_tail_never_promotes_a_website_first_surface():
    """A card whose ONLY surface is website-derived stays at the refused class:
    the tail is a suffix on a register surface, not a way back in."""
    needs_pinned_tokenizer()
    website = {"adr060_phase": "task393-website-content", "title": "X AG",
               "dense_text": "X AG — marketing copy about our mission."}
    res = ECC.resolve_channels([("task393_apply", website)], website_tail=True)
    assert res["dense_class"] == ECC.DENSE_WEBSITE
    assert res["website_tail_source"] is None


def test_a_card_with_no_website_row_is_byte_identical_with_the_tail_on():
    rows = [(l, r) for l, r in vector8_rows() if l != "task393_apply"]
    off = ECC.resolve_channels(rows)
    on = ECC.resolve_channels(rows, website_tail=True)
    assert on["dense_text"] == off["dense_text"]
    assert on["dense_class"] == off["dense_class"] == ECC.DENSE_REGISTER

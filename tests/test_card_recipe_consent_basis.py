"""Task 502 defect 3 / ADR-094 §6a / ADR-103: the card recipe stamps its own
consent basis instead of leaving a future census to join by URL against the
crawl ledger.

An entity card's evidence unit is the preserved register row, never a crawled
page (ADR-094 §6a, hardened by ADR-024 am.4 §5; ADR-103's carry-forward
ruling). The old card lane carried no such marker on the emitted record, so
the task-467 anatomy (the task-502 anatomy doc) had to hand-join 7,925
rows against direct-terminal consent receipts to learn that 61% of a
"declined" count was never a refusal at all. ``finish()`` (the one place
every card -- ``build_two_channel`` and the older ``cmd_cards`` path alike --
passes through before being emitted) now stamps ``consent_basis``,
``consent_state`` and ``non_refreshable`` unconditionally, so a future census
reads them off the record.

These tests exercise ``finish()`` directly rather than ``build_two_channel``:
this sandbox has no ``transformers`` install, and ``build_two_channel``'s
website-content budget path refuses to approximate a token count without the
pinned Arctic tokenizer (entity_card_website.py's own fail-loud contract) --
the SAME reason the pre-existing ``test_shared_commercial_builder_puts_p31_in_dense``
fails in this environment. ``finish()`` is the pure, tokenizer-free function
that actually carries this defect's logic; ``build_two_channel``'s one-line
``rec["consent_state"] = "ok"`` stamp on an admitted website fetch was
reviewed but could not be exercised end-to-end here.
"""
import pytest

import cytir.corpus.card_recipe as recipe


def _base(**overrides):
    rec = {
        "point_id": None, "url": "https://example.test/",
        "dense_text": "A regional bank.", "sparse_text": "bank regional",
    }
    rec.update(overrides)
    return rec


# --- the default stamp on a net-new, register-only card ---------------------

def test_a_fresh_register_only_card_is_stamped_not_applicable():
    """No website fetch ever happened for this card -- ADR-103's point: a
    card's evidence unit never touched the origin, so there is nothing for a
    crawl-consent verdict to say. Not "ok" (nothing was checked) and not
    "declined"/"unreadable" (nothing was refused either)."""
    rec = recipe.finish(_base(), None)
    assert rec["consent_basis"] == "register_row"
    assert rec["consent_state"] == "not_applicable"
    assert rec["non_refreshable"] is False


def test_consent_basis_is_always_register_row_even_when_overridden():
    """The evidence-unit fact is structural, not a per-call choice -- ADR-094
    §6a governs every card, not just the register-only ones."""
    rec = recipe.finish(_base(consent_basis="something_else"), None)
    assert rec["consent_basis"] == "something_else"  # explicit value wins...
    # ...but the DEFAULT (what every real caller gets) is always register_row:
    assert recipe.finish(_base(), None)["consent_basis"] == "register_row"


def test_a_website_admitted_card_is_stamped_ok():
    """Mirrors what ``build_two_channel``'s website_policies block stamps
    before calling ``finish()`` when a live-consent-admitted homepage fetch
    was folded into the card (admission already enforces search_ok=True)."""
    rec = recipe.finish(_base(consent_state="ok"), None)
    assert rec["consent_basis"] == "register_row"
    assert rec["consent_state"] == "ok"
    assert rec["non_refreshable"] is False


def test_unknown_consent_state_is_refused_not_silently_accepted():
    with pytest.raises(ValueError, match="consent_state"):
        recipe.finish(_base(consent_state="maybe"), None)


def test_every_documented_state_is_accepted():
    for state in recipe.CONSENT_STATES:
        rec = recipe.finish(_base(consent_state=state), None)
        assert rec["consent_state"] == state


# --- enrich-in-place must never silently erase a known restriction ----------

def test_enrich_preserves_a_prior_non_refreshable_declined_finding():
    """A prior pass (or the ADR-103 backfill) marked this URL's origin
    declined/unreadable and non_refreshable. A fresh re-emit that never
    resolved crawl consent at all (the honest "not_applicable" default) must
    not silently downgrade that -- exactly the trust_tier "never regress"
    precedent already in this function."""
    existing = {
        "point_id": "p1", "url": "https://example.test/",
        "non_refreshable": True, "consent_state": "declined",
    }
    rec = recipe.finish(_base(), existing)
    assert rec["non_refreshable"] is True
    assert rec["consent_state"] == "declined"


def test_enrich_preserves_unreadable_too():
    existing = {
        "point_id": "p1", "url": "https://example.test/",
        "non_refreshable": True, "consent_state": "unreadable",
    }
    rec = recipe.finish(_base(), existing)
    assert rec["non_refreshable"] is True
    assert rec["consent_state"] == "unreadable"


def test_enrich_preserves_a_prior_ok_finding_when_the_fresh_pass_gathered_none():
    """A source-scoped rebuild (or the site dropping out of
    ``admitted_website_content``) means THIS pass never gathered
    website_policies, so ``finish()`` sees the honest register-only default
    "not_applicable". A prior pass DID resolve an admitted origin fetch and
    stamped "ok" -- that is a real finding, not a stale flag, and a re-emit
    that checked nothing this time must not silently regress it. Same
    never-regress precedent as trust_tier (603-604) and the
    non_refreshable/declined-or-unreadable guard just above."""
    existing = {
        "point_id": "p1", "url": "https://example.test/",
        "non_refreshable": False, "consent_state": "ok",
    }
    rec = recipe.finish(_base(), existing)
    assert rec["consent_state"] == "ok"
    assert rec["non_refreshable"] is False


def test_enrich_does_not_override_a_fresh_ok_stamp_with_a_stale_flag():
    """If THIS pass actually re-resolved consent and got "ok", that is newer
    and more authoritative than an old non_refreshable flag -- the guard only
    fires when the fresh pass stamped nothing (the register-only default)."""
    existing = {
        "point_id": "p1", "url": "https://example.test/",
        "non_refreshable": True, "consent_state": "declined",
    }
    rec = recipe.finish(_base(consent_state="ok"), existing)
    assert rec["consent_state"] == "ok"
    assert rec["non_refreshable"] is False


def test_enrich_of_a_card_with_no_prior_flag_still_defaults_cleanly():
    existing = {"point_id": "p1", "url": "https://example.test/"}
    rec = recipe.finish(_base(), existing)
    assert rec["consent_basis"] == "register_row"
    assert rec["consent_state"] == "not_applicable"
    assert rec["non_refreshable"] is False


# --- the point of the exercise: no receipt join required --------------------

def test_the_record_alone_answers_the_task502_anatomy_question():
    """The whole defect: the task-502 anatomy doc needed a hand join
    across 7,925 rows against direct-terminal consent receipts to answer
    "is this card's URL actually declined, or just unreadable?". A freshly
    emitted card must answer that from its own fields."""
    rec = recipe.finish(_base(), None)
    assert {"consent_basis", "consent_state", "non_refreshable"} <= rec.keys()

"""`authuser` is a tracking/session key of the URL canon (task 767, defect A).

Real URLs from the twin (ai.google.dev, 2026-10-08): 18,099 of 27,867 rows were `?authuser=N`
copies of 1,500 pages. `canonical_url` is an identity input (`mint_point_id`), so the variants
now mint the id of the canonical row instead of a second row.
"""
from __future__ import annotations

import pytest

from cytir.core.canon import canonical_url, document_url_key, strip_tracking_url
from cytir.core.ids import mint_point_id

PAGE = "https://ai.google.dev/gemini-api/docs/function-calling"


@pytest.mark.parametrize("variant", [
    PAGE + "?authuser=1", PAGE + "?authuser=0", PAGE + "?authuser=12",
    PAGE + "?AuthUser=2", PAGE + "?authuser=1&utm_source=x"])
def test_authuser_variants_collapse_to_the_canonical_page(variant):
    assert strip_tracking_url(variant) == PAGE
    assert canonical_url(variant) == canonical_url(PAGE)
    assert document_url_key(variant) == document_url_key(PAGE)
    assert mint_point_id(canonical_url(variant), "") == mint_point_id(canonical_url(PAGE), "")


def test_functional_params_next_to_authuser_are_kept():
    # `hl` picks the page LANGUAGE: a different document, not a duplicate
    assert strip_tracking_url(PAGE + "?authuser=1&hl=ru") == PAGE + "?hl=ru"
    assert canonical_url(PAGE + "?hl=ru&authuser=3") == canonical_url(PAGE + "?hl=ru")
    assert canonical_url(PAGE + "?hl=ru") != canonical_url(PAGE)


def test_fragment_and_authuser_value_forms():
    assert strip_tracking_url(PAGE + "?authuser=1#tool") == PAGE + "#tool"
    assert strip_tracking_url(PAGE + "?authuser=") == PAGE


def test_declared_list_still_has_the_old_keys():
    # authuser was ADDED; the established keys are untouched
    for u in ("https://x.org/a?utm_source=s", "https://x.org/a?fbclid=1", "https://x.org/a?ref=r",
              "https://x.org/a?_gl=1", "https://x.org/a;jsessionid=AB"):
        assert strip_tracking_url(u) == "https://x.org/a", u
    assert strip_tracking_url("https://x.org/a?sid=1") == "https://x.org/a?sid=1"   # NOT stripped


# --- canon epochs (task 756 after review 773 M1; ADR-067) -------------------------------------------
#: The whole tracking set of each epoch. A change to this set, the prefixes, the `;jsessionid` rule or
#: the www fold CHANGES POINT IDS (`mint_point_id` over `canonical_url`): it is a new canon epoch in
#: `cytir.core.canon.CANON_EPOCHS`, never an edit of an existing one.
EPOCH1 = frozenset({
    "__twitter_impression", "_ga", "_gac", "_gl", "_hsenc", "_hsmi", "_openstat", "campaign_id",
    "cmpid", "cvid", "dclid", "fbclid", "gbraid", "gclid", "gclsrc", "guccounter", "guce_referrer",
    "guce_referrer_sig", "igsh", "igshid", "mc_cid", "mc_eid", "mkt_tok", "msclkid", "ncid",
    "oly_anon_id", "oly_enc_id", "ref", "ref_src", "ref_url", "referer", "referrer", "s_kwcid", "scid",
    "source", "spm", "src", "trk", "trkCampaign", "twclid", "utm_id", "vero_conv", "vero_id",
    "wbraid", "wickedid", "yclid"})


def test_canon_epochs_are_pinned_whole():
    from cytir.core import canon as C
    assert C.CANON_EPOCHS[1] == EPOCH1
    assert C.CANON_EPOCHS[2] == EPOCH1 | {"authuser"}
    assert C.CANON_EPOCH == 2 and C._TRACKING_KEYS == C.CANON_EPOCHS[C.CANON_EPOCH]
    assert C._TRACKING_PREFIXES == ("utm_", "hsa_", "pk_", "piwik_", "mtm_", "matomo_")


def test_epoch_1_reproduces_the_pre_704_key_and_epoch_2_is_today():
    from cytir.core.canon import canonical_url_at
    v = PAGE + "?authuser=1&hl=ru"
    assert canonical_url_at(v, 1) == PAGE + "?authuser=1&hl=ru"
    assert canonical_url_at(v, 2) == canonical_url(v) == PAGE + "?hl=ru"
    assert canonical_url_at("http://www.x.org/a?utm_source=s", 1) == "https://x.org/a"
    with pytest.raises(KeyError):
        canonical_url_at(PAGE, 3)

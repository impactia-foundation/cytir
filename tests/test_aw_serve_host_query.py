"""task 2050: AW_HOST_QUERY_CARD (default OFF). Host-shaped query -> label; the card leg matches
the card URL host label; the TLD token is skipped in the df=0 zero-signal count. No Qdrant, no network."""
from types import SimpleNamespace as NS

import pytest

from cytir.core.common import host_query_parts, url_host_label
from cytir.serve.schema import bm25_tokens
from cytir.serve import legs
from cytir.serve.textmatch import _card_name_match


def _flag(monkeypatch, on):
    monkeypatch.setattr(legs, "_S", NS(query=NS(host_query_card=on)))


@pytest.mark.parametrize("q,exp", [
    ("ethz.ch", ("ethz", "ch")), ("www.ethz.ch", ("ethz", "ch")),
    ("fedlex.admin.ch", ("fedlex", "ch")), ("admin.ch", ("admin", "ch")),
    ("eth zurich", None), ("python list files", None), ("art. 336 CO", None), ("10.5", None),
    ("foo.bar", None), ("", None)])
def test_host_detection_and_label(q, exp):
    assert host_query_parts(q) == exp


def test_url_host_label():
    assert url_host_label("https://www.ethz.ch/") == "ethz"
    assert url_host_label("https://fedlex.admin.ch/eli/x") == "fedlex"
    assert url_host_label("") == ""


def test_card_match_on_vs_off():
    q = {"ethz", "ch"}
    assert _card_name_match(q, "ETH Zürich", "ETH Zurich") == 0.0          # off: task 2044 mechanism
    assert _card_name_match(q, "ETH Zürich", "ETH Zurich", host_label="ethz",
                            card_url="https://www.ethz.ch/") == 1.0               # on
    assert _card_name_match(q, "Other", "x", host_label="ethz", card_url="https://www.other.ch/") == 0.0


def test_label_and_hashes_flag_off_not_entered(monkeypatch):
    _flag(monkeypatch, False)
    assert legs.host_query_label("ethz.ch") is None
    assert legs.host_query_tld_hashes("ethz.ch", None) == frozenset()


def test_label_and_hashes_flag_on(monkeypatch):
    _flag(monkeypatch, True)
    assert legs.host_query_label("www.ethz.ch") == "ethz"
    assert legs.host_query_label("eth zurich") is None
    assert legs.host_query_tld_hashes("ethz.ch", None) == frozenset(bm25_tokens("ch", None))
    assert legs.host_query_tld_hashes("eth zurich", None) == frozenset()


def _zero_signal(monkeypatch, tld_hashes):
    """ethz sub-leg: full window (silently skipped, the 2044 shape); ch sub-leg: empty (df=0)."""
    monkeypatch.setattr(legs, "_RELFLOOR", True)
    bm = bm25_tokens("ethz.ch", None)
    ethz_h = next(iter(bm25_tokens("ethz", None)))
    full = [NS(id=i, score=1.0, payload=None) for i in range(legs._PER_TOKEN_LIMIT)]
    tok_legs = [NS(points=full if h == ethz_h else []) for h in bm]
    dense = [NS(id=1, score=0.1, payload=None)]
    res = legs.LegResults(bm=bm, qfilter=None, wp=False, soft_scoped=False, specs=[],
                          legs={"dense": dense, "bm25": dense}, auth_admit={}, tok_legs=tok_legs,
                          pertoken_batch_calls=1, stage1_request_ms=None, tld_hashes=tld_hashes)
    pool = legs.assemble_pool(res, scoped=False, q_cantons=[], q_country=None, q_country_soft=False,
                              scope_conf=0.0, base_off_country=0.0, w_canton=0.0, w_foreign=0.0,
                              query_plan=None)
    return pool.zero_signal


def test_df0_exclusion_off_vs_on(monkeypatch):
    assert _zero_signal(monkeypatch, frozenset()) is True                         # flag off: ch counts as df=0
    assert _zero_signal(monkeypatch, frozenset(bm25_tokens("ch", None))) is False  # on: TLD skipped

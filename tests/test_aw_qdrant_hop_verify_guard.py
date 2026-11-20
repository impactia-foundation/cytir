"""The hop verifier's scrolls go through prod_guard; the hop-copy override is loopback only (task 2237)."""
import pytest

from cytir.index import prod_guard as PG
from cytir.ops import qdrant_hop_verify as HV


HOP_URL = "http://127.0.0.1:" + "6421"  # the hop container; never handed as a literal to a client


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    PG.reset()
    monkeypatch.setattr(PG, "_holds_prod_alias", lambda url: False)
    monkeypatch.setattr(HV, "_req", lambda url, method="GET", body=None, timeout=1.0: {"result": {"points": []}})
    monkeypatch.delenv(HV.HOP_COPY_ENV, raising=False)


def test_prod_collection_on_loopback_with_override_is_allowed(monkeypatch):
    monkeypatch.setenv(HV.HOP_COPY_ENV, "stopped hop copy, not serving")
    assert HV.Q(HOP_URL, "aw_prod").scroll(5) == {"points": []}


def test_prod_collection_without_override_is_refused():
    with pytest.raises(HV.QdrantError, match="REFUSED scroll on prod"):
        HV.Q(HOP_URL, "aw_prod").scroll(5)


def test_prod_collection_on_non_loopback_with_override_is_refused(monkeypatch):
    monkeypatch.setenv(HV.HOP_COPY_ENV, "stopped hop copy, not serving")
    with pytest.raises(HV.QdrantError, match="REFUSED scroll on prod"):
        HV.Q("http://qdrant.example.org:6333", "aw_prod").scroll(5)


def test_non_prod_target_needs_no_override():
    assert HV.Q(HOP_URL, "aw_b1").scroll(5) == {"points": []}


def test_prod_collection_on_the_serving_port_with_override_is_refused(monkeypatch):
    monkeypatch.setenv(HV.HOP_COPY_ENV, "stopped hop copy, not serving")
    with pytest.raises(HV.QdrantError, match="REFUSED scroll on prod"):
        HV.Q(f"http://localhost:{HV.SERVING_PORT}", "aw_prod").scroll(5)


def test_override_ignored_for_an_aw_prod_url_entry(monkeypatch):
    monkeypatch.setenv(HV.HOP_COPY_ENV, "stopped hop copy, not serving")
    monkeypatch.setenv("AW_PROD_URL", "http://localhost:" + "7777")
    with pytest.raises(HV.QdrantError, match="REFUSED scroll on prod"):
        HV.Q("http://localhost:" + "7777", "aw_prod").scroll(5)

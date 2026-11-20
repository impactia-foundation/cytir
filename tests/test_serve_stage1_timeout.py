"""AW_STAGE1_TIMEOUT (task 2686): a per-call timeout for the stage-1 batch and the per-token batch.

At 0 (the code default) no `timeout` kwarg is sent: the calls are byte-identical to before."""
from types import SimpleNamespace as NS

import pytest


@pytest.fixture(autouse=True)
def _live_legs():
    """Other tests drop `cytir.serve*` from sys.modules and re-import; bind the live module."""
    import importlib
    globals()["legs"] = importlib.import_module("cytir.serve.legs")


class Client:
    def __init__(self):
        self.calls = []

    def query_batch_points(self, collection_name, requests, **kw):
        self.calls.append((len(requests), kw))
        return [NS(points=[NS(id=f"p{i}", score=0.5, payload=None)]) for i, _ in enumerate(requests)]


def _run(monkeypatch, timeout, per_token):
    c = Client()
    monkeypatch.setattr(legs.qdrant_io, "client", lambda: c)
    monkeypatch.setattr(legs, "_BATCH_LEGS", True)
    monkeypatch.setattr(legs, "_PER_TOKEN", per_token)
    monkeypatch.setattr(legs, "_TWO_STAGE", True)
    monkeypatch.setattr(legs, "_AUTHLEG", False)
    monkeypatch.setattr(legs, "_STAGE1_TIMEOUT", timeout)
    plan = NS(country_legs_enabled=False, country=None, country_dense_limit=0, country_bm25_limit=0)
    legs.run_legs(q="alpha beta", lang="fr", dense=[0.1, 0.2], lexical=None, candidates=10,
                  country=None, query_plan=plan, scoped=False, q_country_soft=None,
                  experimental=False, target_bm25_language=None, target_bm25_quota=0,
                  target_bm25_weight=0.0, translated_bm25_query=None, translated_bm25_quota=0,
                  translated_bm25_weight=0.0)
    return c.calls


def test_off_sends_no_timeout_kwarg(monkeypatch):
    calls = _run(monkeypatch, 0, per_token=True)
    assert len(calls) == 2 and all(kw == {} for _, kw in calls)


def test_on_sets_the_timeout_on_both_batches(monkeypatch):
    calls = _run(monkeypatch, 12, per_token=True)
    assert len(calls) == 2 and all(kw == {"timeout": 12} for _, kw in calls)


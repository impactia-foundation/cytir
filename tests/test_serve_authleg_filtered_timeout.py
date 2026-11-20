"""AW_AUTHLEG_FILTERED_TIMEOUT (task 2665): the filtered authority leg leaves the stage-1 batch and
runs beside it under Qdrant's server-side timeout; a timeout drops only its admits, and the
response says so (`degraded.authority_leg`, review 2671 F1).

Unfiltered requests, and every request with the knob at 0, keep the single-batch dispatch."""
import contextvars
import json
import os
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest
from qdrant_client import models as qm
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse


@pytest.fixture(autouse=True)
def _live_legs():
    """Other tests drop `cytir.serve*` from sys.modules and re-import; bind the live module."""
    import importlib
    globals()["legs"] = importlib.import_module("cytir.serve.legs")


def _point(pid, score):
    return NS(id=pid, score=score, payload=None)


class Client:
    def __init__(self, auth_error=None):
        self.batches, self.single, self.auth_error = [], [], auth_error

    def query_batch_points(self, collection_name, requests):
        self.batches.append(list(requests))
        return [NS(points=[_point(f"b{i}", 0.5)]) for i, _ in enumerate(requests)]

    def query_points(self, **kw):
        self.single.append(kw)
        if self.auth_error is not None:
            raise self.auth_error
        return NS(points=[_point("auth-1", 0.9), _point("auth-2", 0.8)])


def _run(monkeypatch, *, lang, timeout, client):
    monkeypatch.setattr(legs.qdrant_io, "client", lambda: client)
    monkeypatch.setattr(legs, "_BATCH_LEGS", True)
    monkeypatch.setattr(legs, "_PER_TOKEN", False)
    monkeypatch.setattr(legs, "_TWO_STAGE", True)
    monkeypatch.setattr(legs, "_AUTHLEG", True)
    monkeypatch.setattr(legs, "_AUTHLEG_FILTERED_TIMEOUT", timeout)
    plan = NS(country_legs_enabled=False, country=None, country_dense_limit=0, country_bm25_limit=0)
    return legs.run_legs(q="alpha beta", lang=lang, dense=[0.1, 0.2], lexical=None, candidates=10,
                         country=None, query_plan=plan, scoped=False, q_country_soft=None,
                         experimental=False, target_bm25_language=None, target_bm25_quota=0,
                         target_bm25_weight=0.0, translated_bm25_query=None, translated_bm25_quota=0,
                         translated_bm25_weight=0.0)


def _has_trust_floor(f):
    return any(isinstance(c, qm.FieldCondition) and c.key == "trust_tier" for c in (f.must or []))


@pytest.mark.parametrize("lang,timeout", [(None, 0), (None, 3), ("fr", 0)])
def test_single_batch_dispatch_is_unchanged(monkeypatch, lang, timeout):
    """Knob off, or no filter: the authority leg stays the batch's last request, no extra call."""
    c = Client()
    res = _run(monkeypatch, lang=lang, timeout=timeout, client=c)
    assert c.single == [] and len(c.batches) == 1
    reqs = c.batches[0]
    assert [r.using for r in reqs] == ["dense", "bm25", "dense"]
    assert reqs[-1].limit == legs._AUTHLEG_N and _has_trust_floor(reqs[-1].filter)
    assert list(res.auth_admit) == ["b2"]          # admits come from the batch's authority answer
    assert not res.auth_split and res.auth_leg_status is None


def test_filtered_authority_leg_runs_alone_with_server_timeout(monkeypatch):
    c = Client()
    ctx = contextvars.copy_context()
    res = ctx.run(_run, monkeypatch, lang="it", timeout=3, client=c)
    assert [r.using for r in c.batches[0]] == ["dense", "bm25"]      # left the batch
    (kw,) = c.single
    assert kw["timeout"] == 3 and kw["limit"] == legs._AUTHLEG_N and kw["using"] == legs.DENSE_VEC
    assert kw["search_params"] is legs._DENSE_SP and kw["with_payload"] is False
    f = kw["query_filter"]
    assert _has_trust_floor(f) and res.qfilter in f.must              # same lang scoping as the legs
    assert list(res.auth_admit) == ["auth-1", "auth-2"]
    assert res.auth_split and res.auth_leg_status is None
    assert legs._AUTH_LEG_KEY not in res.legs
    assert ctx.run(legs.AUTHLEG_DEGRADED.get) is None


TIMEOUT_BODY = b'{"status":{"error":"Timeout error: Operation \'Search\' timed out after 3 seconds"}}'


@pytest.mark.parametrize("err,status", [
    (UnexpectedResponse(500, "Internal Server Error", TIMEOUT_BODY, {}), "timed_out"),
    (UnexpectedResponse(500, "Internal Server Error", b'{"status":{"error":"Service internal error"}}',
                        {}), "error"),
    (ResponseHandlingException(ConnectionError("refused")), "error"),
])
def test_a_failed_authority_leg_drops_only_its_admits_and_says_why(monkeypatch, err, status):
    """Review 2671 F3: only Qdrant's timeout is `timed_out`; any other absorbed error is `error`."""
    c = Client(auth_error=err)
    ctx = contextvars.copy_context()
    res = ctx.run(_run, monkeypatch, lang="fr", timeout=3, client=c)
    assert res.auth_admit == {} and res.auth_split and res.auth_leg_status == status
    assert set(res.legs) == {"dense", "bm25"} and all(res.legs.values())   # the request survives
    assert ctx.run(legs.AUTHLEG_DEGRADED.get) == status                     # F1: the API reads this
    assert legs.AUTHLEG_DEGRADED.get() is None                              # ... this request only


def test_other_errors_still_fail_the_request(monkeypatch):
    c = Client(auth_error=RuntimeError("bug"))
    with pytest.raises(RuntimeError):
        _run(monkeypatch, lang="fr", timeout=3, client=c)


_APP_CODE = """
import json
from fastapi.testclient import TestClient
from cytir.serve import app
def fake(q, k, lang=None, country=None, **kw):
    if lang == "it":
        app._legs.AUTHLEG_DEGRADED.set("timed_out")
    return []
app.retrieval.search = fake
c = TestClient(app.api)
print(json.dumps([c.get("/search", params={"q": "x", "lang": "it"}).json(),
                  c.get("/search", params={"q": "x", "lang": "de"}).json()]))
"""


def test_the_response_says_when_the_authority_leg_was_dropped():
    """Review 2671 F1: `degraded.authority_leg` appears on the request that lost the leg only."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AW_", "CYTIR_"))}
    out = subprocess.run([sys.executable, "-c", _APP_CODE], env=env, capture_output=True,
                         text=True, timeout=300, check=True).stdout
    lost, kept = json.loads(out.strip().splitlines()[-1])
    assert lost["degraded"] == {"authority_leg": "timed_out"} and lost["count"] == 0
    assert "degraded" not in kept


def test_the_parity_gate_runs_both_arms_with_the_knob_off():
    """Review 2671 F2: a wall-clock bound must not make the two arms DIFF on a busy twin."""
    from cytir.gates import serve_parity
    assert serve_parity.TIMING_ENV["AW_AUTHLEG_FILTERED_TIMEOUT"] == "0"

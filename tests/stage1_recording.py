"""Record and replay of `legs.stage1` (S10 step 8a, task 1116): the per-stage test the serve-split
spec asks for ("per-stage tests on deep-tier recordings").

A recording is one `stage1` call: its `Stage1Request`, every Qdrant call it made (method, the
canonical encoding of the keyword arguments, the response), the module constants it ran under, and
the canonical encoding of its `Stage1` result. `replay` rebuilds the request, answers the client
calls from the recording (refusing a call whose method or arguments differ), runs the CURRENT
`stage1`, and returns the encoding of its result; the test asserts it equals the recorded text.
Maps are encoded as ordered pair lists: key order is part of the result (it decides pool order).
"""
from __future__ import annotations

import dataclasses
import json
import math
import types

from qdrant_client import models as qm

CONST_MODULES = ("legs", "fusion", "plan", "trace")
_SCALAR = (bool, int, float, str)


def enc(x):
    """Canonical JSON-able form. Order-preserving for dicts; sets sorted by repr."""
    if x is None or isinstance(x, _SCALAR):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return {"__t": "float", "v": repr(x)}
        return x
    if isinstance(x, dict):
        return {"__t": "dict", "v": [[enc(k), enc(v)] for k, v in x.items()]}
    if isinstance(x, (list,)):
        return [enc(v) for v in x]
    if isinstance(x, tuple):
        return {"__t": "tuple", "v": [enc(v) for v in x]}
    if isinstance(x, (set, frozenset)):
        return {"__t": "set", "v": sorted((enc(v) for v in x), key=repr)}
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {"__t": type(x).__name__,
                "v": [[f.name, enc(getattr(x, f.name))] for f in dataclasses.fields(x)]}
    if hasattr(x, "item") and not hasattr(x, "__len__"):      # a numpy scalar
        return enc(x.item())
    if hasattr(x, "tolist"):                                  # a numpy array
        return enc(x.tolist())
    if hasattr(x, "model_dump"):
        return {"__t": "m:" + type(x).__name__, "v": enc(x.model_dump(exclude_none=True))}
    if hasattr(x, "id") and hasattr(x, "payload"):          # ScoredPoint / Record
        return {"__t": "point", "id": enc(x.id), "score": enc(getattr(x, "score", None)),
                "payload": enc(x.payload)}
    raise TypeError(f"cannot encode {type(x).__name__}")


def dec(x):
    """Inverse of `enc` for plain data (requests, constants, responses)."""
    if isinstance(x, list):
        return [dec(v) for v in x]
    if isinstance(x, dict) and "__t" in x:
        t = x["__t"]
        if t == "dict":
            return {_hashable(dec(k)): dec(v) for k, v in x["v"]}
        if t == "tuple":
            return tuple(dec(v) for v in x["v"])
        if t == "set":
            return frozenset(_hashable(dec(v)) for v in x["v"])
        if t == "float":
            return float(x["v"])
        raise TypeError(f"cannot decode {t}")
    return x


def _hashable(v):
    return tuple(_hashable(i) for i in v) if isinstance(v, list) else v


def encode_text(x) -> str:
    return json.dumps(enc(x), separators=(",", ":"), ensure_ascii=True)


def request_to_json(req) -> dict:
    d = enc(req)
    return d


def request_from_json(d, legs, plan):
    fields = dict(dec_pair(p) for p in d["v"])
    qp = fields["query_plan"]
    fields["query_plan"] = plan.QueryPlan(**{k: v for k, v in qp.items()})
    return legs.Stage1Request(**fields)


def dec_pair(p):
    name, v = p
    if isinstance(v, dict) and v.get("__t") == "QueryPlan":
        return name, {n: dec(x) for n, x in v["v"]}
    return name, dec(v)


# ------------------------------------------------------------------------------------------------
# the client seam
# ------------------------------------------------------------------------------------------------
def _point(p):
    return types.SimpleNamespace(id=p["id"], score=p["score"], payload=p["payload"])   # what stage 1 reads


def _response(method, r):
    if method == "query_points":
        return types.SimpleNamespace(points=[_point(p) for p in r])
    if method == "query_batch_points":
        return [types.SimpleNamespace(points=[_point(p) for p in resp]) for resp in r]
    if method == "retrieve":
        return [_point(p) for p in r]
    raise AssertionError(f"unrecorded method {method}")


def _plain_points(x):
    return [{"id": p.id, "score": getattr(p, "score", None), "payload": p.payload} for p in x]


def _record_response(method, r):
    if method == "query_points":
        return _plain_points(r.points)
    if method == "query_batch_points":
        return [_plain_points(q.points) for q in r]
    return _plain_points(r)


def _norm_kwargs(kwargs):
    return json.loads(encode_text(kwargs))


class RecordingClient:
    """Wraps the real client; while `on`, records the three calls stage 1 makes."""
    METHODS = ("query_points", "query_batch_points", "retrieve")

    def __init__(self, real):
        self._real, self.on, self.calls = real, False, []

    def __getattr__(self, name):
        attr = getattr(self._real, name)
        if name not in self.METHODS or not callable(attr):
            return attr

        def call(**kwargs):
            r = attr(**kwargs)
            if self.on:
                self.calls.append({"method": name, "kwargs": _norm_kwargs(kwargs),
                                   "response": _record_response(name, r)})
            return r
        return call


class ReplayClient:
    def __init__(self, calls):
        self._calls, self._i = calls, 0

    def __getattr__(self, name):
        def call(**kwargs):
            assert self._i < len(self._calls), f"unexpected extra {name} call"
            rec = self._calls[self._i]
            self._i += 1
            assert rec["method"] == name, f"call {self._i}: {name} != recorded {rec['method']}"
            assert _norm_kwargs(kwargs) == rec["kwargs"], f"call {self._i} ({name}): arguments differ"
            return _response(name, rec["response"])
        return call

    def exhausted(self):
        return self._i == len(self._calls)


# ------------------------------------------------------------------------------------------------
# module constants a stage reads (the env-derived flags that change which path runs)
# ------------------------------------------------------------------------------------------------
def snapshot_constants(mods: dict) -> dict:
    out = {}
    for mname in CONST_MODULES:
        m = mods[mname]
        out[mname] = {n: enc(getattr(m, n)) for n in sorted(dir(m))
                      if n.lstrip("_").isupper() and isinstance(getattr(m, n), _SCALAR)}
    out["dense_search_params"] = mods["legs"]._DENSE_SP.model_dump(exclude_none=True)   # built at import from the env
    out["fusion_weights"] = {k: v for k, v in mods["fusion"].WEIGHTS.items() if isinstance(v, (int, float))}
    return out


def apply_constants(monkeypatch, mods: dict, snap: dict) -> None:
    for mname in CONST_MODULES:
        for n, v in snap[mname].items():
            monkeypatch.setattr(mods[mname], n, dec(v), raising=True)
    monkeypatch.setattr(mods["qdrant_io"], "COLLECTION", snap["legs"]["COLLECTION"])   # `_leg` reads its own
    monkeypatch.setattr(mods["legs"], "_DENSE_SP", qm.SearchParams(**snap["dense_search_params"]))
    for k, v in snap["fusion_weights"].items():
        monkeypatch.setitem(mods["fusion"].WEIGHTS, k, v)


def replay(rec, monkeypatch, mods) -> tuple[str, ReplayClient]:
    apply_constants(monkeypatch, mods, rec["constants"])
    client = ReplayClient(rec["calls"])
    monkeypatch.setattr(mods["qdrant_io"], "client", lambda: client)
    # The query tokenizer is not stage 1: it depends on the decompounder assets the machine has, so the
    # replay feeds it the recorded tokens (the first field of the recorded result); `schema` has its tests.
    # an empty pool has no result to take them from: that recording carries them (`bm25_tokens`).
    bm = (dec(rec["bm25_tokens"]) if "bm25_tokens" in rec
          else {k: v for k, v in dec(json.loads(rec["result_text"])["v"][0][1]).items()})
    monkeypatch.setattr(mods["legs"], "bm25_tokens", lambda q, lang=None: dict(bm))
    req = request_from_json(rec["request"], mods["legs"], mods["plan"])
    return encode_text(mods["legs"].stage1(req)), client

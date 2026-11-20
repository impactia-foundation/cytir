"""Record and replay of whole `retrieval.search()` calls (S10 step 8b, task 1125): the per-stage test for
stage 2, the levers and the response, which `tests/stage1_recording.py` (stage 1 only) does not reach.

A recording is one `search()` call taken on the BASE tree of the step (the code before the move), so the
fixture is a before/after witness and not a snapshot of the new code: its arguments, the embedding the
query got, every Qdrant call (method, canonical arguments, response), the module constants it ran under,
the answers of the two machine-dependent helpers, and the canonical encoding of the returned list plus the
deep-tier side channels (`_LAST_SCORE_DEBUG`, `_LAST_COMPONENTS`, `_LAST_QUERY_PLAN_TRACE`).

`replay` answers the embedder and the client from the recording, refuses a client call whose method or
arguments differ, runs the CURRENT `search()`, and returns the encodings; the test asserts they equal the
recorded text. NOT covered, by design: the tokenizer. `schema._decompound` (a function of the
decompounder assets on the machine) and `legs.bm25_tokens` answer from the recording, so a regression in
the decompounder or in the IDF table is not caught here; `schema` has its own tests.
The rank heads and the country map are absent in the recording environment (as in a checkout without the
trained artifacts), so their levers are covered by the parity gate on the twin, not by this fixture. The hub
and authority tables are the committed files; replay loads them under the recorded flags.
"""
from __future__ import annotations

import json
import sys

import stage1_recording
from stage1_recording import ReplayClient, dec, enc, encode_text  # noqa: F401

s1 = stage1_recording

MODULES = ("retrieval", "legs", "fusion", "levers", "respond", "trace", "plan", "heads", "geo", "tables", "schema")
_SCALAR = s1._SCALAR
SIDE = ("_LAST_SCORE_DEBUG", "_LAST_COMPONENTS", "_LAST_QUERY_PLAN_TRACE")


def snapshot_constants(mods: dict) -> dict:
    """Scalar upper-case constants of every module `search()` reads, FLAT by name: the base tree and the
    head tree keep a constant in different modules (it moved), so replay sets it wherever it exists."""
    flat = {}
    for m in MODULES:
        for n in sorted(dir(mods[m])):
            v = getattr(mods[m], n)
            if n.lstrip("_").isupper() and isinstance(v, _SCALAR):
                e = enc(v)
                assert flat.setdefault(n, e) == e, f"constant {n} differs between modules"
    return {"scalars": flat,
            "dense_search_params": mods["legs"]._DENSE_SP.model_dump(exclude_none=True),
            "fusion_weights": {k: v for k, v in mods["fusion"].WEIGHTS.items() if isinstance(v, (int, float))}}


def apply_constants(monkeypatch, mods: dict, snap: dict) -> None:
    for m in MODULES:
        for n, v in snap["scalars"].items():
            if isinstance(v, str) and v.startswith("/"):
                continue          # a path on the recording machine: the file of THIS checkout is read
            if hasattr(mods[m], n):
                monkeypatch.setattr(mods[m], n, dec(v))
    # The hub and authority tables are loaded at import, and only when their flag is on: re-load them under
    # the recorded flags (the files are the committed ones) and hand them to every module that holds a copy.
    for attr, loader in (("_AUTH_HOSTS", "_load_authority_hosts"), ("_HUB_HOSTS", "_load_hub_hosts")):
        table = getattr(mods["tables"], loader)()
        for m in MODULES:
            if hasattr(mods[m], attr):
                monkeypatch.setattr(mods[m], attr, table)
    monkeypatch.setattr(mods["qdrant_io"], "COLLECTION", dec(snap["scalars"]["COLLECTION"]))
    from qdrant_client import models as qm
    monkeypatch.setattr(mods["legs"], "_DENSE_SP", qm.SearchParams(**snap["dense_search_params"]))
    for k, v in snap["fusion_weights"].items():
        monkeypatch.setitem(mods["fusion"].WEIGHTS, k, v)


def side_channels(retrieval) -> dict:
    """The deep-tier side channels, read through the `retrieval` facade (the same objects on both trees)."""
    return {n: enc(getattr(retrieval, n)) for n in SIDE}


class SearchReplayClient(ReplayClient):
    """`ReplayClient`, except that the requests of the per-token COVERAGE PROBE call (the one
    `query_batch_points` call made from `levers`) match as a set. The probe builds its requests by
    iterating sets of strings, whose order depends on the interpreter (Python 3.10 on the recording
    box, anything on CI, even at PYTHONHASHSEED=0): the same requests in another order are the same
    call. The responses are handed back in the order the caller asked. Every other client call,
    stage 1's batches included, replays strictly, in order."""

    def __getattr__(self, name):
        strict = super().__getattr__(name)
        if name != "query_batch_points":
            return strict

        def call(**kwargs):
            if not sys._getframe(1).f_globals.get("__name__", "").endswith(".levers"):
                return strict(**kwargs)        # not the coverage probe: strict order and arguments
            assert self._i < len(self._calls), f"unexpected extra {name} call"
            rec = self._calls[self._i]
            self._i += 1
            assert rec["method"] == name, f"call {self._i}: {name} != recorded {rec['method']}"
            got = dict(s1._norm_kwargs(kwargs)["v"])        # canonical kwargs: {"__t": "dict", "v": [[name, value], ...]}
            want = dict(rec["kwargs"]["v"])
            key = lambda r: json.dumps(r, sort_keys=True)          # noqa: E731
            assert {k: v for k, v in got.items() if k != "requests"} == {k: v for k, v in want.items() if k != "requests"}
            assert sorted(map(key, got["requests"])) == sorted(map(key, want["requests"])), \
                f"call {self._i} ({name}): arguments differ"
            by_req = dict(zip(map(key, want["requests"]), rec["response"]))
            return s1._response(name, [by_req[key(r)] for r in got["requests"]])
        return call


def mask_timing(x):
    """The one wall-clock field of the side channels (`stage1_request_ms`) is not part of the answer."""
    if isinstance(x, list):
        if len(x) == 2 and x[0] == "stage1_request_ms":
            return [x[0], None]
        return [mask_timing(v) for v in x]
    if isinstance(x, dict):
        return {k: mask_timing(v) for k, v in x.items()}
    return x


def replay(rec, monkeypatch, mods, models, schema) -> tuple[str, dict, ReplayClient]:
    import numpy as np
    apply_constants(monkeypatch, mods, rec["constants"])
    client = SearchReplayClient(rec["calls"])
    monkeypatch.setattr(mods["qdrant_io"], "client", lambda: client)
    emb = rec["embed"]
    monkeypatch.setattr(models, "embed_query", lambda q: (dec(emb["dense"]), dec(emb["lexical"])))
    monkeypatch.setattr(models, "last_full_query_vec", lambda: np.asarray(emb["full"], dtype="float32"))
    parts = {t: tuple(p) for t, p in rec["decompound"]}
    monkeypatch.setattr(schema, "_decompound", lambda token: parts[token])        # a new token is an error
    bm = {(t, l): dec(v) for t, l, v in rec["bm25_tokens"]}
    monkeypatch.setattr(mods["legs"], "bm25_tokens", lambda q, lang=None: dict(bm[(q, lang)]))
    p = rec["params"]
    out = mods["retrieval"].search(p["q"], **{k: v for k, v in p.items() if k != "q"})
    return encode_text(out), side_channels(mods["retrieval"]), client

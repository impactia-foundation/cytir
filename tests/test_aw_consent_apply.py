"""ADR-115 backfill WRITE engine (cytir.consent.apply), proven on an in-memory Qdrant stand-in.

Rows are the REAL twin payloads of `test_consent_derive_adr115` (one per rule), the real
contract table and the real `aw_e9` projector, so the coded `consent_basis` values and the
decode round trip are the production ones.

Run: python3 -m pytest tests/test_aw_consent_apply.py
"""
from __future__ import annotations

import bisect
import copy
import uuid

import pytest

from cytir.consent import apply as A
from cytir.consent.derive import WRITTEN_BASES
from cytir.index import payload_profile as PP
import consent_adr115_rows as T  # noqa: E402  (tests/)


class FakeQdrant:
    """Just the calls the engine makes: count, scroll (ordered by id, offset = id),
    retrieve by id, set_payload. Records every write so a test can say what was NOT sent."""

    def __init__(self, rows: dict):
        self.rows = {k: copy.deepcopy(v) for k, v in rows.items()}
        self.writes: list = []
        self.scrolls = 0

    def __call__(self, method, path, body=None):
        body = body or {}
        if path.endswith("/points/count"):
            return {"result": {"count": len(self.rows)}}
        if path.endswith("/points/scroll"):
            self.scrolls += 1
            ids = sorted(self.rows)
            start = bisect.bisect_left(ids, body["offset"]) if body.get("offset") else 0
            chunk = ids[start:start + body["limit"]]
            nxt = ids[start + body["limit"]] if start + body["limit"] < len(ids) else None
            inc = (body.get("with_payload") or {}).get("include")
            pts = [{"id": i, "payload": {k: v for k, v in self.rows[i].items() if not inc or k in inc}}
                   for i in chunk]
            return {"result": {"points": pts, "next_page_offset": nxt}}
        if path.endswith("/points/payload?wait=true"):
            self.writes.append((tuple(body["points"]), dict(body["payload"])))
            for i in body["points"]:
                self.rows[i].update(body["payload"])
            return {"result": {"status": "completed"}}
        if path.endswith("/points"):
            inc = (body.get("with_payload") or {}).get("include")
            return {"result": [{"id": i, "payload": {k: v for k, v in self.rows[i].items() if not inc or k in inc}}
                               for i in body["ids"] if i in self.rows]}
        raise AssertionError(path)

    def written_ids(self):
        return {i for ids, _ in self.writes for i in ids}


def _id(n):
    return str(uuid.UUID(int=n))


def _twin_rows():
    rows = {}
    seeds = [T.WIKI, T.SE, T.ARXIV, T.PUBMED, T.HN, T.WEKO_CARRIED, T.IAIS_CARRIED, T.LODA_DUMP, T.RII,
             *T.DILA, {"url": "https://nowhere.example/x", "domain": "nowhere.example", "license": "?"},
             {**T.WIKI, "url": "https://en.wikipedia.org/wiki/Other", "search_ok": True}]                      # one flag only -> D, never coerced
    for n, row in enumerate(seeds, 1):
        rows[_id(n * (2 ** 128 // 20))] = {**row, "vec_marker": n, "title": f"t{n}"}
    return rows


CACHE = {"www.weko.admin.ch": T.rec(True, False, "2026-09-01T00:00:00Z"),
         "www.iais.org": T.rec(False, False, "2026-09-02T00:00:00Z")}
ARX = {"2503.21080": "http://creativecommons.org/licenses/by/4.0/"}


def _ctx():
    return T.ctx(CACHE, ARX)


def test_twin_apply_writes_only_a_and_b_rows_and_only_the_four_fields():
    rows = _twin_rows()
    q = FakeQdrant(rows)
    out = A.apply_twin(q, "aw_e8", _ctx(), page=4, batch=3, workers=2, checkpoint_pages=1, log=lambda m: None)
    assert out["scan_complete"] and out["rows_scanned"] == len(rows)
    assert sum(out["classes"].values()) == len(rows)          # conservation
    assert out["classes"]["E"] == 1 and out["classes"]["D"] == 2          # HN is E; unknown + one-flag are D
    assert out["classes"]["A"] + out["classes"]["B"] == len(rows) - 3
    assert out["classes"]["A"] + out["classes"]["B"] == sum(out["written_by_rule"].values())
    wrote = q.written_ids()
    for i, before in rows.items():
        after = q.rows[i]
        if i in wrote:
            assert {k: v for k, v in after.items() if k not in A.WRITE_FIELDS} == \
                   {k: v for k, v in before.items() if k not in A.WRITE_FIELDS}, "only the four fields move"
            assert isinstance(after["search_ok"], bool) and isinstance(after["train_ok"], bool)
            assert after["consent_basis"] in WRITTEN_BASES | {"official-dump-license"}
        else:
            assert after == before, "E and D rows are never touched"
    assert len(q.rows) == len(rows)                           # no point added or removed


def test_twin_apply_values_follow_the_rules():
    q = FakeQdrant(_twin_rows())
    A.apply_twin(q, "aw_e8", _ctx(), page=50, log=lambda m: None)
    by_url = {r["url"]: r for r in q.rows.values()}
    assert (by_url[T.WIKI["url"]]["search_ok"], by_url[T.WIKI["url"]]["train_ok"],
            by_url[T.WIKI["url"]]["consent_basis"]) == (True, True, "dump-licence")
    assert by_url[T.ARXIV["url"]]["train_ok"] is True                           # CC-BY paper
    assert by_url[T.PUBMED["url"]]["train_ok"] is False                         # fail closed
    assert by_url[T.IAIS_CARRIED["url"]]["search_ok"] is False                  # carried row, host declines
    assert by_url[T.IAIS_CARRIED["url"]]["consent_state"] == "carried"          # never rewritten
    assert by_url[T.WEKO_CARRIED["url"]]["consent_checked_at"] == "2026-09-01T00:00:00Z"   # the record's own
    assert by_url[T.LODA_DUMP["url"]]["consent_basis"] == "official-dump-license"          # keeps its string
    assert by_url[T.HN["url"]]["train_ok"] is False and by_url[T.HN["url"]]["search_ok"] is True


def test_twin_apply_is_idempotent_a_rerun_writes_nothing():
    q = FakeQdrant(_twin_rows())
    A.apply_twin(q, "aw_e8", _ctx(), page=5, log=lambda m: None)
    n = len(q.writes)
    out = A.apply_twin(q, "aw_e8", _ctx(), page=5, log=lambda m: None)
    assert len(q.writes) == n and out["ab_rows"] == 0 and out["classes"].get("A", 0) == 0


def test_dry_run_sends_nothing_and_plans_the_same():
    rows = _twin_rows()
    q = FakeQdrant(rows)
    dry = A.apply_twin(q, "aw_e8", _ctx(), page=5, dry=True, log=lambda m: None)
    assert q.writes == [] and q.rows == rows
    real = A.apply_twin(FakeQdrant(rows), "aw_e8", _ctx(), page=5, log=lambda m: None)
    assert dry["classes"] == real["classes"] and dry["planned_by_rule"] == real["planned_by_rule"]


def test_resume_from_a_checkpoint_loses_and_repeats_nothing():
    rows = _twin_rows()
    q = FakeQdrant(rows)
    saved: list = []
    first = A.apply_twin(q, "aw_e8", _ctx(), page=3, limit=6, checkpoint_pages=1, batch=100,
                         save_state=lambda s: saved.append(copy.deepcopy(s)), log=lambda m: None)
    assert not first["scan_complete"]
    state = saved[-1]
    state.pop("result", None)
    out = A.apply_twin(q, "aw_e8", _ctx(), page=3, state=state, batch=100, log=lambda m: None)
    assert out["scan_complete"] and out["rows_scanned"] == len(rows)
    written = [i for ids, _ in q.writes for i in ids]
    assert len(written) == len(set(written)), "no id written twice"
    full = FakeQdrant(rows)
    A.apply_twin(full, "aw_e8", _ctx(), page=50, log=lambda m: None)
    assert q.rows == full.rows


def test_a_failed_write_stops_the_run_and_does_not_advance_the_saved_offset():
    class Failing(FakeQdrant):
        def __call__(self, method, path, body=None):
            if path.endswith("/points/payload?wait=true"):
                raise RuntimeError("disk full")
            return super().__call__(method, path, body)
    saved: list = []
    with pytest.raises(RuntimeError, match="disk full"):
        A.apply_twin(Failing(_twin_rows()), "aw_e8", _ctx(), page=3, checkpoint_pages=1,
                     save_state=lambda s: saved.append(copy.deepcopy(s)), log=lambda m: None)
    assert all(s.get("offset") is None for s in saved), "an offset was saved past an unwritten id"


# ----------------------------------------------------------------------- prod leg
@pytest.fixture(scope="module")
def e9():
    return PP.Projector.for_collection("aw_e9")


def test_every_basis_the_adr_writes_has_a_code_in_the_prod_tables(e9):
    for basis in WRITTEN_BASES | {"official-dump-license"}:
        out = e9.project({"consent_basis": basis})
        assert isinstance(out["consent_basis"], int), basis
        assert PP.decode_payload(out, e9.profile, e9.tables)["consent_basis"] == basis


def _prod_from(twin: FakeQdrant, strip_flags=True, flagged=()):
    rows = {}
    for i, r in twin.rows.items():
        r = {k: v for k, v in r.items() if k not in A.WRITE_FIELDS} if strip_flags else dict(r)
        rows[i] = r
    for i in flagged:
        rows[i].update(search_ok=False, train_ok=False)
    return FakeQdrant(rows)


def _decode(e9):
    return lambda p: PP.decode_payload(p, e9.profile, e9.tables)


def test_prod_plan_is_read_only_and_classifies_by_id(e9):
    twin = FakeQdrant(_twin_rows())
    A.apply_twin(twin, "aw_e8", _ctx(), page=50, log=lambda m: None)
    wrote = sorted(twin.written_ids())
    prod = _prod_from(twin, flagged=[wrote[0]])
    before = copy.deepcopy(prod.rows)
    absent_id = wrote[1]
    del prod.rows[absent_id]
    out = A.plan_prod(twin, "aw_e8", prod, "aw_e9", _ctx().as_of, e9.project, decode=_decode(e9), log=lambda m: None)
    assert prod.writes == [] and out["mode"] == "plan"
    c = out["classes"]
    assert c["absent"] == 1 and c.get("conflict", 0) + c.get("same", 0) == 1
    assert c["write"] == len(wrote) - 2
    assert out["decode_mismatch"] == 0
    assert all(prod.rows[i] == before[i] for i in prod.rows)


def test_plan_prod_has_no_apply_argument(e9):
    """The prod-write path is deleted (task 1288); a prod backfill goes through the twin-first door."""
    twin = FakeQdrant(_twin_rows())
    prod = _prod_from(twin)
    with pytest.raises(TypeError):
        A.plan_prod(twin, "aw_e8", prod, "aw_e9", _ctx().as_of, e9.project, apply=True, log=lambda m: None)
    assert prod.writes == []



def test_the_sample_dry_run_reports_decode_and_prod_state(e9):
    twin = FakeQdrant(_twin_rows())
    A.apply_twin(twin, "aw_e8", _ctx(), page=50, log=lambda m: None)
    prod = _prod_from(twin)
    s = A.sample_prod_projection(twin, "aw_e8", prod, "aw_e9", _ctx().as_of, e9.project, _decode(e9), 40, 1)
    assert s["decode_mismatch"] == 0 and s["classes"].get("write", 0) > 0
    assert prod.writes == []


def test_twin_window_stops_at_a_checkpoint_and_resumes_to_the_same_result():
    rows = _twin_rows()
    q = FakeQdrant(rows)
    saved: list = []
    first = A.apply_twin(q, "aw_e8", _ctx(), page=3, batch=100, deadline=0.0,
                         save_state=lambda s: saved.append(copy.deepcopy(s)), log=lambda m: None)
    assert not first["scan_complete"] and saved[-1]["offset"] is not None
    state = saved[-1]
    state.pop("result", None)
    out = A.apply_twin(q, "aw_e8", _ctx(), page=3, batch=100, state=state, log=lambda m: None)
    full = FakeQdrant(rows)
    A.apply_twin(full, "aw_e8", _ctx(), page=50, log=lambda m: None)
    assert out["scan_complete"] and q.rows == full.rows



def test_refuses_while_an_aw_e10_build_holds_the_twin(tmp_path):
    """Task 756 (review 776 item 2): a twin writer outside the drain refuses the held collection
    before any read, while `cytir.index.twin_write_hold`'s hold file exists."""
    import json as _json
    import os as _os
    import pathlib as _pathlib
    import subprocess
    import sys
    hold = tmp_path / "hold.json"
    hold.write_text(_json.dumps({"collection": "aw_e10_twin", "plan_dir": "/p", "t0": "2026-10-08T09:31:18Z"}))
    app = _pathlib.Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-m", "cytir.cli", "consent", "backfill", "twin", "--twin-url", "http://127.0.0.1:9", "--collection", "aw_e10_twin",
         "--arxiv-licences", "x", "--as-of", "2026-10-08T00:00:00Z", "--state", str(tmp_path / "st.json")], cwd=app, env=dict(_os.environ, AW_TWIN_WRITE_HOLD=str(hold)),
        capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "an aw_e10 build holds" in r.stderr, r.stderr[-800:]


def _twin_args(url, collection, tmp_path):
    return ["twin", "--dry", "--twin-url", url, "--collection", collection, "--arxiv-licences", "x",
            "--as-of", "2026-10-24T00:00:00Z", "--state", str(tmp_path / "st.json")]


def test_twin_door_refuses_prod_url_and_live_collection_before_any_call(monkeypatch, tmp_path):
    """Review 1294 finding 2: `cyt consent backfill twin` is TWIN-ONLY, refused first, zero calls."""
    from cytir.index import live_target as LT
    from cytir.index import prod_guard as PG
    calls = []
    monkeypatch.setattr(A, "http_api", lambda *a, **k: calls.append(("client", a)) or (lambda *x, **y: calls.append(x)))
    live = sorted(LT.live_collection_names())[0]
    for url, coll in ((f"http://127.0.0.1:{PG.PROD_PORT}", "twin_x"), ("http://twin.invalid:6333", live)):
        with pytest.raises(SystemExit, match="REFUSED.*twin-first door"):
            A.main(_twin_args(url, coll, tmp_path))
    assert calls == []


def test_twin_door_still_runs_for_an_invalid_host_twin(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(A, "http_api", lambda url, *a, **k: seen.append(url) or object())
    monkeypatch.setattr(A, "_ctx", lambda a: None)
    monkeypatch.setattr(A, "apply_twin", lambda *a, **k: {"scan_complete": True})
    assert A.main(_twin_args("http://twin.invalid:6333", "aw_twin_scratch", tmp_path)) == 0
    assert seen == ["http://twin.invalid:6333"]

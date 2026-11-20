"""Every reader of the twin decides the same on a RAW twin and on a CODED one (task 855).

After the aw_e10 flip the twin is a coded copy of prod (ruling 27 Q1: profile `rebuild_e10`, enum
codes, a 32-hex digest, no `stamps`/`point_id`/`tier`/`build_id`). Task 847 S2 listed the readers
that assumed the raw twin (`aw_e8`). Each test below runs ONE reader twice on the same rows: (a) a
raw twin `aw_e8`, (b) the same rows as the coded twin stores them (`coded_twin_rows.declare`:
`aw_e10_twin`, `rebuild_e10`), and asserts the same decision. Where the reader used to fail on (b)
the test also shows the old failure shape (a 32-hex digest is never a 64-hex one; an int has no
`.strip()`; code 0 reads as "none").
"""
from __future__ import annotations

import bisect
import copy
import datetime as dt
import json
import types
import uuid

import pytest

import coded_twin_rows as CT  # noqa: E402  (tests/)
from cytir.core.paths import data_path
from cytir.index import payload_profile as PP
from cytir.index import row_decode as RD

RAW, CODED = CT.RAW_TWIN, CT.CODED_TWIN


@pytest.fixture(autouse=True)
def _declared(monkeypatch, tmp_path):
    CT.declare(monkeypatch, tmp_path)


def _store(rows: dict, collection: str) -> dict:
    """`{id: payload}` as `collection` STORES it: raw for aw_e8, the projection for the coded twin."""
    if RD.profile_of(collection) in (None, "prod"):
        return copy.deepcopy(rows)
    return {i: CT.coded(p, RD.profile_of(collection)) for i, p in rows.items()}


def _uid(n: int) -> str:
    return str(uuid.UUID(int=n))


# ----------------------------------------------------------------- 1. news re-capture skip
class _ScrollClient:
    """`client.scroll(collection, scroll_filter=..., ...)` over stored payloads (qdrant_client shape)."""

    def __init__(self, stored: dict):
        self.stored = stored

    def scroll(self, collection, *, scroll_filter, limit, offset, with_payload, with_vectors):
        want = set(scroll_filter.must[0].match.any)
        pts = [types.SimpleNamespace(id=i, payload={k: v for k, v in p.items() if k in with_payload})
               for i, p in self.stored.items() if p.get("url") in want]
        return pts, None


def _news_rows():
    rows = [r for r in CT.real_rows() if (r["payload"].get("provenance") == "sitemap-ingest")]
    assert len(rows) >= 3
    return rows


@pytest.mark.parametrize("case", ["unchanged", "changed", "overhang"])
def test_news_recapture_skip_decides_the_same_on_a_coded_twin(case):
    from cytir.news import compose as NC
    rows = _news_rows()[:3]
    served = {r["id"]: r["payload"] for r in rows}
    kept = [{"id": r["id"], "payload": copy.deepcopy(r["payload"])} for r in rows]
    newest = max(dt.date.fromisoformat(r["payload"]["published"]) for r in rows)
    now_ref = newest + dt.timedelta(days=2)
    if case == "changed":
        kept[0]["payload"]["content_digest"] = "ab" * 32
    if case == "overhang":
        now_ref = newest + dt.timedelta(days=400)
    out = {}
    for coll in (RAW, CODED):
        res = NC.filter_unchanged_recaptures(kept, _ScrollClient(_store(served, coll)), coll,
                                             now_ref=now_ref, overhang_days=60, log=lambda *a, **k: None)
        out[coll] = {k: sorted(v) if k.endswith("urls") else sorted(r["id"] for r in v)
                     for k, v in res.items()}
    assert out[RAW] == out[CODED]
    if case == "unchanged":
        assert len(out[CODED]["unchanged_skipped_urls"]) == len({r["payload"]["url"] for r in rows})
        # the failure this fixes: the coded twin's digest is never the pack's 64-hex one
        assert _store(served, CODED)[rows[0]["id"]]["content_digest"] != rows[0]["payload"]["content_digest"]
    if case == "changed":
        assert out[CODED]["changed_recaptured_urls"] == [rows[0]["payload"]["url"]]


# ----------------------------------------------------------------- 2. the consent backfill
class _Api:
    """count, scroll (by id order, offset = id), retrieve by id, set_payload."""

    def __init__(self, rows: dict):
        self.rows = copy.deepcopy(rows)
        self.writes: list = []

    def __call__(self, method, path, body=None):
        body = body or {}
        inc = (body.get("with_payload") or {}).get("include") if isinstance(body.get("with_payload"), dict) else None
        sel = (lambda p: {k: v for k, v in p.items() if not inc or k in inc})
        if path.endswith("/points/count"):
            return {"result": {"count": len(self.rows)}}
        if path.endswith("/points/scroll"):
            ids = sorted(self.rows)
            start = bisect.bisect_left(ids, body["offset"]) if body.get("offset") else 0
            chunk = ids[start:start + body["limit"]]
            nxt = ids[start + body["limit"]] if start + body["limit"] < len(ids) else None
            return {"result": {"points": [{"id": i, "payload": sel(self.rows[i])} for i in chunk],
                               "next_page_offset": nxt}}
        if path.endswith("/points/payload?wait=true"):
            self.writes.append((tuple(body["points"]), dict(body["payload"])))
            for i in body["points"]:
                self.rows[i].update(body["payload"])
            return {"result": {"status": "completed"}}
        if path.endswith("/points"):
            return {"result": [{"id": i, "payload": sel(self.rows[i])} for i in body["ids"] if i in self.rows]}
        raise AssertionError(path)


def _consent_ctx():
    from cytir.consent.derive import DeriveContext
    fams = json.loads(data_path("source-family-contracts.v1.json").read_text())["families"]
    cache = {"__unparseable__": 0,
             "www.weko.admin.ch": {"cc": {"status": "ok", "search_ok": True, "train_ok": False,
                                          "checked_at": "2026-09-01T00:00:00Z"}}}
    return DeriveContext(fams, cache, "2026-10-01T00:00:00Z", None, None)


def _consent_rows():
    rows = {_uid(n + 1): dict(r) for n, r in enumerate(CT.CONSENT_ROWS)}
    for n, r in enumerate(CT.real_rows()):
        rows[_uid(1000 + n)] = {k: v for k, v in r["payload"].items()}
    return rows


def test_consent_backfill_matches_the_same_family_and_class_on_a_coded_twin():
    from cytir.consent import apply as A
    rows = _consent_rows()
    keys = ("rows_scanned", "classes", "rules", "families", "planned_by_rule")
    out = {}
    for coll in (RAW, CODED):
        res = A.apply_twin(_Api(_store(rows, coll)), coll, _consent_ctx(), dry=True, page=7,
                           log=lambda m: None)
        out[coll] = {k: res[k] for k in keys}
    assert out[RAW] == out[CODED]
    fams = out[CODED]["families"]
    assert "(unmapped)" not in fams or sum(fams["(unmapped)"].values()) == sum(
        out[RAW]["families"].get("(unmapped)", {}).values())
    assert sum(out[CODED]["planned_by_rule"].values()) > 0


def test_consent_backfill_writes_a_coded_twin_coded_and_the_same_values():
    from cytir.consent import apply as A
    rows = _consent_rows()
    apis = {}
    for coll in (RAW, CODED):
        apis[coll] = _Api(_store(rows, coll))
        A.apply_twin(apis[coll], coll, _consent_ctx(), page=5, batch=3, workers=1, readback=3,
                     log=lambda m: None)
    raw_after, coded_after = apis[RAW].rows, apis[CODED].rows
    assert {i for w in apis[RAW].writes for i in w[0]} == {i for w in apis[CODED].writes for i in w[0]}
    written = {i for w in apis[CODED].writes for i in w[0]}
    assert written
    for i in written:
        # the coded twin stores codes (never a raw string in a closed key) ...
        assert isinstance(coded_after[i]["consent_basis"], int)
        # ... and decodes to exactly what the raw twin holds
        assert RD.reader(CODED)(coded_after[i]) == {
            k: v for k, v in RD.reader(RAW)(raw_after[i]).items() if k not in RD.reader(CODED).absent}


def test_consent_prod_leg_finds_the_same_written_rows_on_a_coded_twin():
    from cytir.consent import apply as A
    rows = _consent_rows()
    e9 = PP.Projector.for_collection("aw_e9")
    tallies = {}
    for coll in (RAW, CODED):
        twin = _Api(_store(rows, coll))
        A.apply_twin(twin, coll, _consent_ctx(), page=50, log=lambda m: None)
        prod = _Api({i: e9.project({k: v for k, v in p.items() if k not in A.WRITE_FIELDS})
                     for i, p in rows.items()})
        res = A.plan_prod(twin, coll, prod, "aw_e9", "2026-10-01T00:00:00Z", e9.project,
                          decode=lambda p: PP.decode_payload(p, e9.profile, e9.tables),
                          log=lambda m: None)
        tallies[coll] = (res["classes"], res["decode_mismatch"])
    assert tallies[RAW] == tallies[CODED]
    assert tallies[CODED][0].get("twin_written_rows", 0) > 0


# ----------------------------------------------------------------- 3. the German-parts tripwire
def _gp_rows():
    real = json.loads((CT.FIX / "german_parts_real_rows.json").read_text())["rows"]
    with_parts = [r for r in real if r.get("has_parts_on_twin") is True]
    without = [r for r in real if r.get("has_parts_on_twin") is False]
    assert with_parts and without
    pts = {}
    for i in range(2400):
        r = (without if i % 2 == 0 else with_parts)[i % min(len(with_parts), len(without))]
        dom = "no.example.de" if i % 2 == 0 else "yes.example.de"
        pts[f"{i:08d}-0000-0000-0000-000000000000"] = (
            {"lang": "de", "domain": dom, "source_type": "official",
             "provenance": "weko_docs" if i % 2 == 0 else "loda",       # code 0 and code 27
             "prov_id": "6370ee1d65a21add", "rule_epochs": {"extract": 1, "chunk": 1, "embed": 1,
                                                            "tok": 2, "country": 2, "trust": 1, "card": 1},
             "build_id": "aw_b1", "last_seen": "2026-09-01", "content_digest": "cd" * 32,
             "tier": "B", "point_id": f"{i:08d}-0000-0000-0000-000000000000"},
            r["bm25"])
    return pts


def _tripwire(collection, rows):
    from cytir.ops import german_parts_tripwire as T

    class Twin(T.Qdrant):
        def __init__(self):
            super().__init__("http://twin.invalid:16457", collection)
            stored = _store({i: p for i, (p, _v) in rows.items()}, collection)
            self.pts = [{"id": i, "payload": stored[i], "vector": {"bm25": rows[i][1]}} for i in sorted(rows)]

        def post(self, path, body):
            if path == "/points/count":
                return {"count": len(self.pts)}
            if path == "/facet":
                c = {}
                for p in self.pts:
                    c[p["payload"]["domain"]] = c.get(p["payload"]["domain"], 0) + 1
                return {"hits": [{"value": k, "count": v} for k, v in c.items()]}
            if path == "/points/scroll":
                flt = body["filter"]
                dom = next((x["match"]["value"] for x in flt.get("must", []) if x["key"] == "domain"), None)
                sel = [p for p in self.pts if dom is None or p["payload"]["domain"] == dom]
                off = body.get("offset")
                start = 0 if off is None else next((k for k, p in enumerate(sel) if p["id"] >= off), len(sel))
                page = sel[start:start + body["limit"]]
                nxt = sel[start + body["limit"]]["id"] if start + body["limit"] < len(sel) else None
                keys = body["with_payload"]
                return {"points": [dict(p, payload={k: v for k, v in p["payload"].items() if k in keys})
                                   for p in page], "next_page_offset": nxt}
            if path == "/points":
                want = set(body["ids"])
                return [p for p in self.pts if p["id"] in want]
            raise AssertionError(path)

    return T, Twin()


def test_tripwire_counts_the_same_german_rows_and_judges_the_same(tmp_path):
    rows = _gp_rows()
    reports = {}
    for coll in (RAW, CODED):
        T, q = _tripwire(coll, rows)
        reports[coll] = T.run(q, min_generation=1000, sample=1000, alarm_below=0.5, windows=3,
                              tail_windows=3, scan_cap=10_000, seed=1)
    raw, coded = reports[RAW], reports[CODED]
    assert raw["self_check"]["ok"] and coded["self_check"]["ok"]
    assert raw["twin_german_rows"] == coded["twin_german_rows"] == 2400

    def by_key(rep, absent_build):
        out = {}
        for g in rep["generations"]:
            f = g["generation"].split("|")
            if absent_build:
                assert f[T.BUILD_ID_FIELD] == T.ABSENT       # rebuild_e10 stores no build_id
                f[T.BUILD_ID_FIELD] = "aw_b1"
            out["|".join(f)] = {k: v for k, v in g.items() if k != "generation"}
        return out
    assert by_key(raw, False) == by_key(coded, True)
    keys = sorted(by_key(raw, False))
    assert keys[0] == "no.example.de|official|weko_docs|6370ee1d|chunk1|aw_b1|2026-09"   # code 0 named
    # the acknowledged line measured on the raw twin acknowledges the same generation on the coded one
    alarmed = [g["generation"] for g in raw["generations"] if g["verdict"] == "ALARM"]
    assert alarmed
    ack = tmp_path / "ack.txt"
    ack.write_text("".join(f"{g} ; rows=1200 ; share=0.00%  # test\n" for g in alarmed))
    lines = T.acknowledged(str(ack))
    j_raw, j_coded = T.judge(raw, lines), T.judge(coded, lines)
    assert j_raw["new_alarms"] == j_coded["new_alarms"] == 0
    assert j_raw["stale_lines"] == j_coded["stale_lines"] == []


def test_tripwire_absent_build_id_matches_any_line_but_a_present_one_does_not():
    from cytir.ops import german_parts_tripwire as T
    line = "d.de|official|loda|-|chunk1|aw_b1|2026-09"
    assert T.key_matches("d.de|official|loda|-|chunk1|~|2026-09", line)
    assert not T.key_matches("d.de|official|loda|-|chunk1|aw_b2|2026-09", line)
    assert not T.key_matches("d.de|official|legi_code|-|chunk1|~|2026-09", line)


# ----------------------------------------------------------------- 4. mirror_compare: mirror-audit + finalize
def _mirror_cases():
    mg = json.loads((CT.FIX / "mirror_guards" / "rows.v1.json").read_text())
    return [e for cls in ("equal", "prod_only_old") for e in mg[cls]]


@pytest.mark.parametrize("target", [CODED, "aw_e9"])
def test_mirror_compare_classes_a_coded_reference_like_a_raw_one(target):
    from cytir.index import mirror_compare as MC
    from cytir.ops import mirror_audit as MA
    strict = MC.Comparator(target)
    lenient = MC.Comparator(target, allow_unstamped_rule_epochs=True, projector=strict.projector)
    ref_profile = RD.profile_of(target)
    ids, refs_raw, refs_coded, stored = [], {}, {}, {}
    for e in _mirror_cases():
        tw = e["twin"]
        ids.append(e["id"])
        refs_raw[e["id"]] = {"payload": tw["payload"], "vector": tw["vector"]}
        refs_coded[e["id"]] = {"payload": CT.coded(tw["payload"], ref_profile), "vector": tw["vector"]}
        # the target holds the twin row in its own form (stamped where a rule is declared)
        stored[e["id"]] = {"payload": strict.target_form(tw["payload"]), "vector": tw["vector"]}
    bad = ids[0]
    stored[bad] = copy.deepcopy(stored[bad])
    stored[bad]["payload"]["title"] = "a different title"
    for pid in ids:
        a = MA.classify(pid, refs_raw[pid], stored[pid], strict=strict, lenient=lenient, dset=None,
                        global_allowed=set(), user_lenient=False)
        b = MA.classify(pid, refs_coded[pid], stored[pid], strict=strict, lenient=lenient, dset=None,
                        global_allowed=set(), user_lenient=False)
        assert a == b
        assert a[0] == ("DIFF-undeclared" if pid == bad else "EQUAL")
    # the finalize's call (lenient comparator, `compare`)
    fin = MC.Comparator(target, allow_unstamped_rule_epochs=True)
    assert fin.compare(ids, refs_raw, stored) == fin.compare(ids, refs_coded, stored)
    assert fin.compare(ids, refs_coded, stored)["mismatched"] == 1


def test_mirror_compare_no_longer_calls_a_coded_reference_unstorable():
    from cytir.index import mirror_compare as MC
    e = _mirror_cases()[0]
    cmp = MC.Comparator(CODED)
    ref = {"payload": CT.coded(e["twin"]["payload"]), "vector": e["twin"]["vector"]}
    stored = {"payload": cmp.target_form(e["twin"]["payload"]), "vector": e["twin"]["vector"]}
    assert cmp.diff(ref, stored) == []
    assert not any(k.startswith("<unstorable") for k in cmp.diff(ref, None))


# ----------------------------------------------------------------- 5. the drain's tombstone checks
def _loda_rows():
    base = next(r["payload"] for r in CT.real_rows() if r["payload"].get("provenance") == "loda")
    url = base["url"]
    old = dict(base)
    new = dict(base, anchor="article:LEGIARTI000099999999", content_digest="ef" * 32)
    variant = dict(base, url=url + "?authuser=1")
    canon = dict(base)                      # the canonical spelling, the variant's text
    return url, {"old": old, "new": new, "variant": variant, "canon": canon}


def _served_api(stored: dict, calls: list):
    def api(url, path, body=None, *a, **k):
        calls.append(path)
        assert path.endswith("/points/scroll")
        want = set(body["filter"]["must"][0]["match"]["any"])
        keys = set(body["with_payload"])
        return {"points": [{"id": i, "payload": {kk: v for kk, v in p.items() if kk in keys}}
                           for i, p in stored.items() if p.get("url") in want],
                "next_page_offset": None}
    return api


def _run_check(monkeypatch, coll, served_rows, fn, *args, **kw):
    from cytir.index import promote_queue as PQ
    calls: list = []
    monkeypatch.setattr(PQ.PSP, "_api", _served_api(_store(served_rows, coll), calls))
    return fn(*args, target_url="http://twin.invalid:16457", collection=coll, **kw)


@pytest.mark.parametrize("other_family", [False, True])
def test_generation_siblings_decide_the_same_on_a_coded_twin(monkeypatch, other_family):
    from cytir.index import promote_queue as PQ
    url, r = _loda_rows()
    served = {_uid(1): r["old"], _uid(2): dict(r["new"], provenance="legi_code" if other_family else "loda")}
    tombs = [{"id": _uid(1), "url": url, "reason": PQ.SUPERSEDED_GENERATION}]
    out = {}
    for coll in (RAW, CODED):
        stored_by_id = {_uid(1): _store(served, coll)[_uid(1)]}       # what fetch_payload_fields returns
        out[coll] = _run_check(monkeypatch, coll, served, PQ.check_generation_siblings, tombs,
                               stored_by_id, export_ids={_uid(1)})
    assert out[RAW] == out[CODED]
    assert bool(out[CODED]) is other_family
    assert "no `provenance`" not in " ".join(out[CODED])     # code 0/ints never read as "none"


@pytest.mark.parametrize("mixed", [False, True])
def test_canonical_duplicates_decide_the_same_on_a_coded_twin(monkeypatch, mixed):
    """`mixed`: the stored row read raw (64 hex) against coded group members (32 hex) -- the
    digest compare is on the canonical key, so the survivor is found either way."""
    from cytir.index import promote_queue as PQ
    url, r = _loda_rows()
    served = {_uid(3): r["variant"], _uid(4): r["canon"]}
    tombs = [{"id": _uid(3), "url": r["variant"]["url"], "reason": PQ.CANONICAL_DUPLICATE}]
    out = {}
    for coll in (RAW, CODED):
        src = served if mixed else _store(served, coll)
        out[coll] = _run_check(monkeypatch, coll, served, PQ.check_canonical_duplicates, tombs,
                               {_uid(3): src[_uid(3)]}, export_ids={_uid(3)})
    assert out[RAW] == out[CODED] == []
    # a survivor holding OTHER text is still refused on both
    served[_uid(4)] = dict(r["canon"], content_digest="00" * 32)
    refused = [_run_check(monkeypatch, coll, served, PQ.check_canonical_duplicates, tombs,
                          {_uid(3): _store(served, coll)[_uid(3)]}, export_ids={_uid(3)})
               for coll in (RAW, CODED)]
    assert refused[0] == refused[1] and len(refused[0]) == 1


def test_restore_rollback_identity_is_the_same_for_a_pack_row_and_a_coded_stored_row():
    from cytir.index import promote_queue as PQ
    _url, r = _loda_rows()
    pack_row = r["old"]                                   # what the restore pack wrote (64 hex)
    want = {_uid(5): PQ.row_identity(pack_row)}
    tombs = [{"id": _uid(5), "url": pack_row["url"], "reason": PQ.RESTORE_ROLLBACK}]
    for coll in (RAW, CODED):
        stored = {_uid(5): _store({_uid(5): pack_row}, coll)[_uid(5)]}
        assert PQ.check_restore_rollback(tombs, stored, want) == []
    assert PQ.row_identity(CT.coded(pack_row)) == PQ.row_identity(pack_row)


def test_url_group_supersede_names_the_same_victims_on_a_coded_twin(monkeypatch):
    from cytir.index import promote_queue as PQ
    from cytir.index import url_group_supersede as UGS
    url, r = _loda_rows()
    served = {_uid(6): r["old"], _uid(7): dict(r["old"], provenance="legi_code"),
              _uid(8): dict(r["old"], provenance="weko_docs")}              # code 0
    pack = {_uid(9): r["new"]}
    groups, cards = UGS.pack_groups(pack, "loda")
    out = {}
    for coll in (RAW, CODED):
        monkeypatch.setattr(PQ.PSP, "_api", _served_api(_store(served, coll), []))
        out[coll] = UGS.compute(pack, groups, cards, "loda", twin_url="http://twin.invalid:16457",
                                collection=coll)
    assert out[RAW] == out[CODED]
    assert out[CODED]["ids"] == [_uid(6)] and out[CODED]["kept_other_family"] == 2


def test_index_text_tombstones_plan_the_same_on_a_coded_twin(monkeypatch):
    from cytir.index import point_delete as PD
    from cytir.index import promote_queue as PQ
    from cytir.ops import index_text_tombstones as ITT
    _url, r = _loda_rows()
    served = {_uid(3): r["variant"], _uid(4): r["canon"]}
    cands = [{"id": _uid(3), "url": r["variant"]["url"]}]
    out = {}
    for coll in (RAW, CODED):
        st = _store(served, coll)
        monkeypatch.setattr(PQ.PSP, "_api", _served_api(st, []))
        monkeypatch.setattr(PD, "fetch_payload_fields",
                            lambda u, c, ids, fields, st=st: {i: {k: v for k, v in st[i].items() if k in fields}
                                                              for i in ids if i in st})
        res = ITT.classify(PQ.CANONICAL_DUPLICATE, cands, "http://twin.invalid:16457", coll)
        out[coll] = ([t["id"] for t in res["pass"]], res["failed"], res["absent"],
                     ITT.summary(PQ.CANONICAL_DUPLICATE, res, 1)["by_provenance"])
    assert out[RAW] == out[CODED]
    assert out[CODED][0] == [_uid(3)] and out[CODED][3] == {"loda": 1}


# ----------------------------------------------------------------- 6. compose's served exclusion
def test_served_exclusion_compares_the_same_fields_and_skips_what_the_profile_drops():
    from cytir.index import served_exclusion as SE
    raw = next(r["payload"] for r in CT.real_rows() if r["payload"].get("published"))
    pack = dict(raw)
    pack_changed = dict(raw, title="another title")
    for p in (pack, pack_changed):
        a = SE.compare_payloads(p, RD.reader(RAW)(raw))
        b = SE.compare_payloads(p, RD.reader(CODED)(CT.coded(raw)))
        assert a == b
    assert SE.compare_payloads(pack_changed, RD.reader(CODED)(CT.coded(raw))) == ["title"]
    # `tier` is not stored by the coded twin: its absence is not a difference
    assert SE.compare_payloads(dict(raw, tier="A"), RD.reader(CODED)(CT.coded(raw))) == []
    assert SE.compare_payloads(dict(raw, tier="A"), RD.reader(RAW)(raw)) == ["tier"]


def test_consent_backfill_refuses_to_write_an_undeclared_twin_before_any_write():
    """A twin setting moved to a collection whose declaration has not landed: the reader still
    decodes (form by form), but the WRITE refuses -- it never guesses raw for a coded collection."""
    from cytir.consent import apply as A
    api = _Api(_store(_consent_rows(), RAW))
    with pytest.raises(PP.PayloadProfileError):
        A.apply_twin(api, "aw_not_declared", _consent_ctx(), page=5, log=lambda m: None)
    assert api.writes == []
    dry = A.apply_twin(api, "aw_not_declared", _consent_ctx(), dry=True, page=5, log=lambda m: None)
    assert dry["rows_scanned"] == len(api.rows)
    assert "not checked" in dry["encode_refusals"]["<twin encoding>"]


def test_consent_dry_run_reports_a_write_the_coded_twin_cannot_encode(monkeypatch):
    """The dry run projects each distinct write set once (review of #791, S4): a value with no
    code refuses in the dry run, not after the real run wrote earlier batches."""
    from cytir.consent import apply as A
    rows = _consent_rows()
    for coll in (RAW, CODED):
        out = A.apply_twin(_Api(_store(rows, coll)), coll, _consent_ctx(), dry=True, page=7,
                           log=lambda m: None)
        assert out["encode_refusals"] == {}
    real = A.twin_encoder(CODED)

    def strict(writes):
        if writes.get("consent_basis") == "official-dump-license":
            raise PP.UnknownValueError("consent_basis", writes["consent_basis"], CODED)
        return real(writes)
    monkeypatch.setattr(A, "twin_encoder", lambda c: strict)
    out = A.apply_twin(_Api(_store(rows, CODED)), CODED, _consent_ctx(), dry=True, page=7,
                       log=lambda m: None)
    assert len(out["encode_refusals"]) >= 1
    assert all("consent_basis" in v for v in out["encode_refusals"].values())

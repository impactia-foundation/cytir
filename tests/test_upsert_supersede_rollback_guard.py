"""Task 574 -- rollback-completeness guard for supersede-on-upsert.

Task 572 exported a pre-apply rollback copy keyed by a MATERIALIZE-time
ledger (task 565's split-and-export script's scan of the roster's
own `old_point_ids`), then let `apply_pack_delta.py`'s `Superseder.run()`
delete by its OWN, live, url-group scroll -- a DIFFERENT selector. The two
disagreed by 315 points (62,023 exported vs. 62,338 deleted); the gap was
discovered only after the delete had already run and is now permanent and
unrecoverable (task 572's second-recut doc, sec 4, lesson L572-2).

These tests seal the fix entirely against a fake Qdrant HTTP-shaped client
-- no network, no live collection:

  1. `Superseder.run(expect_ids=...)` REFUSES, before any delete, when the
     live delete set does not match a supplied "roster-keyed" id set --
     the 315-point gap, minimised to a 1-point difference (same mechanism,
     smaller fixture).
  2. The matching case proceeds, and the pre-574 `stats` keys/values are
     byte-for-byte what `run()` returned before this task -- the guard is
     additive, not a behaviour change on the happy path.
  3. `export_stale()`/`run(export_path=...)` writes the SAME set it is
     about to delete (dense + sparse vectors + full payload), by
     construction, never a second computation.
  4. `restore_from_export()` round-trips that export byte-exactly, and
     refuses when the sample readback catches a corrupted landing that a
     bare `points_count` check would miss.
  5. `apply_pack_delta.main()` end-to-end: the same refuse/match behaviour
     through the CLI (`--verify-superseded-against`, `--export-superseded`),
     with the matching case's existing print lines unchanged.

    python3 -m pytest tests/test_upsert_supersede_rollback_guard.py -q
"""
from __future__ import annotations

import sys as _sys_1822
from pathlib import Path as _Path_1822

_sys_1822.path.insert(0, str(_Path_1822(__file__).resolve().parents[1] / "tests"))  # task 1822: the surface-delta fixtures live in tests/

import gzip
import json
import sys
from types import SimpleNamespace

import pytest

import cytir.core.common as AC
import cytir.index.supersede as US
from qdrant_client import models as qm

import cytir.index.pack_delta as APD


@pytest.fixture(autouse=True)
def _no_card_guard_read(monkeypatch):
    """task 820: `apply_pack_delta` now runs the drain's card check (an HTTP id retrieve) before
    any write; these fakes are not Qdrant. The guard's own tests: tests/test_aw_index_card_guard_writers.py."""
    monkeypatch.setattr(APD, "guard_card_overwrites", lambda *a, **k: {})
    monkeypatch.setattr(APD, "client_url", lambda client: "http://fake")


# ---------------------------------------------------------------------------
# a minimal, multi-collection, qdrant_client-shaped fake
# ---------------------------------------------------------------------------
class FakeInfo:
    def __init__(self, *, dim=8, points=0, url_indexed=True):
        dense = SimpleNamespace(size=dim, datatype=None)
        params = SimpleNamespace(vectors={"dense": dense})
        self.config = SimpleNamespace(params=params, quantization_config=None)
        self.points_count = points
        self.indexed_vectors_count = points
        self.status = "green"
        self.payload_schema = (
            {"url": SimpleNamespace(data_type="keyword")} if url_indexed else {})


class FakeQdrant:
    """One instance serves every `QdrantClient(...)` call a test needs --
    collections keyed by name, so the SAME fake plays both the pack source
    and the target destination (the trick `test_apply_pack_dim_guard.py`
    already uses). Enough of the real surface for `Superseder` and
    `apply_pack_delta.py`: `get_collection`, `count`, `scroll` (single page,
    ignores the filter -- test fixtures are small enough that this is
    exact, not an approximation), `retrieve`, `delete`, `upsert`,
    `create_payload_index`.
    """

    def __init__(self, collections: dict, *_, **__):
        # name -> {"points": {id: {"payload", "vector"}}, "dim", "url_indexed"}
        self.collections = collections
        self.delete_calls = 0
        self.upsert_calls = 0

    def get_collection(self, name):
        c = self.collections[name]
        return FakeInfo(dim=c.get("dim", 8), points=len(c["points"]),
                       url_indexed=c.get("url_indexed", True))

    def count(self, name, exact=True):
        return SimpleNamespace(count=len(self.collections[name]["points"]))

    def scroll(self, name, scroll_filter=None, limit=1000, offset=None,
              with_payload=True, with_vectors=True):
        if offset is not None:      # one page is exact for these fixtures
            return [], None
        pts = self.collections[name]["points"]
        out = [SimpleNamespace(
            id=pid,
            payload=(dict(rec["payload"]) if with_payload else None),
            vector=(dict(rec["vector"]) if (with_vectors and rec["vector"])
                   else None))
            for pid, rec in pts.items()]
        return out, None

    def retrieve(self, name, ids, with_payload=True, with_vectors=True):
        pts = self.collections[name]["points"]
        out = []
        for i in ids:
            i = str(i)
            if i in pts:
                rec = pts[i]
                out.append(SimpleNamespace(
                    id=i,
                    payload=(dict(rec["payload"]) if with_payload else None),
                    vector=(dict(rec["vector"]) if (with_vectors and rec["vector"])
                           else None)))
        return out

    def delete(self, name, points_selector, wait=True):
        self.delete_calls += 1
        pts = self.collections[name]["points"]
        for i in points_selector.points:
            pts.pop(str(i), None)

    def upsert(self, name, points, wait=False):
        self.upsert_calls += 1
        pts = self.collections[name]["points"]
        for p in points:
            pts[str(p.id)] = {"payload": p.payload, "vector": p.vector}

    def create_payload_index(self, name, field_name, field_schema, wait=True):
        self.collections[name]["url_indexed"] = True


def _rec(url, extra=None, dense=None, sparse=None):
    payload = {"url": url, "source_type": "content", "domain": "example.org"}
    if extra:
        payload.update(extra)
    vector = {}
    if dense is not None:
        vector["dense"] = dense
    if sparse is not None:
        vector["bm25"] = qm.SparseVector(indices=sparse[0], values=sparse[1])
    return {"payload": payload, "vector": vector or None}


URL = "https://example.org/a"


def _build_superseder(*, existing: dict, keep_ids, dry_run=False):
    """A page group with `existing` live points, where the incoming pack
    keeps only `keep_ids` -- everything else in the group is stale."""
    fake = FakeQdrant({"aw_e8": {"points": dict(existing), "dim": 8}})
    sup = US.Superseder(fake, "aw_e8", dry_run=dry_run)
    for pid in keep_ids:
        sup.observe(pid, URL, "content")
    return fake, sup


# ---------------------------------------------------------------------------
# 1 & 2. Superseder.run(expect_ids=...) -- refuse vs. match
# ---------------------------------------------------------------------------
def test_run_refuses_when_expect_ids_undercounts_the_live_delete_set():
    """The 572 mechanism, minimised: the live collection holds 4 stale
    points for the page (p1..p4); the incoming pack keeps none of them
    (p5 replaces them all). A roster-keyed "expect" set that only saw 3 of
    the 4 (p4 was staged by a lane the roster never scanned) must refuse
    BEFORE any delete -- exactly the 315-point gap, at unit scale."""
    existing = {f"p{i}": _rec(URL, dense=[float(i)] * 8) for i in (1, 2, 3, 4)}
    fake, sup = _build_superseder(existing=existing, keep_ids=["p5"])
    roster_expect = {"p1", "p2", "p3"}          # undercounts by 1 (p4 missing)

    with pytest.raises(SystemExit) as exc:
        sup.run(expect_ids=roster_expect)
    msg = str(exc.value)
    assert "4" in msg and "3" in msg and "1" in msg  # live=4, expect=3, diff=1

    # nothing was deleted -- the refusal happened before any write
    assert fake.delete_calls == 0
    assert set(fake.collections["aw_e8"]["points"]) == {"p1", "p2", "p3", "p4"}
    assert sup.stats["rollback_verified"] is False
    assert sup.stats["symmetric_difference"] == 1
    assert sup.stats["live_delete_count"] == 4
    assert sup.stats["expect_count"] == 3


def test_run_proceeds_when_expect_ids_matches_the_live_delete_set_exactly():
    """The corrected roster (all 4 ids) matches the live delete set exactly
    -- the apply proceeds, and the delete actually happens."""
    existing = {f"p{i}": _rec(URL, dense=[float(i)] * 8) for i in (1, 2, 3, 4)}
    fake, sup = _build_superseder(existing=existing, keep_ids=["p5"])
    roster_expect = {"p1", "p2", "p3", "p4"}

    stats = sup.run(expect_ids=roster_expect)

    assert fake.delete_calls >= 1
    assert set(fake.collections["aw_e8"]["points"]) == set()  # all 4 gone
    assert stats["rollback_verified"] is True
    assert stats["superseded"] == 4
    assert stats["live_delete_count"] == 4
    assert stats["expect_count"] == 4


def test_matching_case_stats_are_byte_identical_to_pre_574_shape():
    """The five stats keys `run()` returned before task 574 must hold the
    EXACT pre-574 values when the new guard is used and matches -- the
    guard only ADDS keys, never changes an existing one."""
    existing = {f"p{i}": _rec(URL, dense=[float(i)] * 8) for i in (1, 2, 3, 4)}
    fake, sup = _build_superseder(existing=existing, keep_ids=["p5"])
    stats = sup.run(expect_ids={"p1", "p2", "p3", "p4"})
    assert stats["groups"] == 1
    assert stats["existing_seen"] == 4
    assert stats["superseded"] == 4
    assert stats["gold_superseded"] == 0
    assert stats["unmatched_hits"] == 0

    # the OMITTED-guard call (no export_path/expect_ids at all) must be
    # identical too -- the true "byte-for-byte with today" baseline.
    existing2 = {f"q{i}": _rec(URL, dense=[float(i)] * 8) for i in (1, 2, 3, 4)}
    fake2, sup2 = _build_superseder(existing=existing2, keep_ids=["q5"])
    stats2 = sup2.run()
    assert stats2 == {"groups": 1, "existing_seen": 4, "superseded": 4,
                      "gold_superseded": 0, "unmatched_hits": 0}
    assert "rollback_verified" not in stats2
    assert "export_path" not in stats2


def test_dry_run_also_refuses_before_any_write():
    """A rehearsal (--dry-run) must catch the same mismatch, not just the
    real apply -- the whole point of a dry run is catching this BEFORE the
    write is committed for real."""
    existing = {f"p{i}": _rec(URL, dense=[float(i)] * 8) for i in (1, 2, 3, 4)}
    fake, sup = _build_superseder(existing=existing, keep_ids=["p5"],
                                  dry_run=True)
    with pytest.raises(SystemExit):
        sup.run(expect_ids={"p1", "p2", "p3"})
    assert fake.delete_calls == 0
    assert len(fake.collections["aw_e8"]["points"]) == 4  # untouched


# ---------------------------------------------------------------------------
# 3. export_stale() / run(export_path=...) -- the export IS the delete set
# ---------------------------------------------------------------------------
def test_export_stale_writes_exactly_the_live_delete_set_with_full_vectors(
        tmp_path):
    existing = {
        "p1": _rec(URL, dense=[0.1] * 8, sparse=([1, 5], [0.2, 0.7])),
        "p2": _rec(URL, dense=[0.2] * 8),
    }
    fake, sup = _build_superseder(existing=existing, keep_ids=["p3"])
    out = tmp_path / "rollback.jsonl.gz"

    stats = sup.run(export_path=str(out))

    assert stats["export_count"] == 2
    assert stats["superseded"] == 2
    rows = {}
    with gzip.open(out, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            rows[r["id"]] = r
    assert set(rows) == {"p1", "p2"}
    assert rows["p1"]["payload"]["url"] == URL
    assert rows["p1"]["vector"]["dense"] == [0.1] * 8
    assert rows["p1"]["vector"]["bm25"] == {
        "__sparse__": True, "indices": [1, 5], "values": [0.2, 0.7]}
    # the export happened BEFORE the delete -- same set, not a race
    assert fake.collections["aw_e8"]["points"] == {}


def test_export_path_alone_does_not_change_the_returned_stats_shape():
    """`export_path` with no `expect_ids` adds `export_path`/`export_count`
    only -- no `rollback_verified` key (nothing was verified, just written)."""
    existing = {"p1": _rec(URL, dense=[0.1] * 8)}
    fake, sup = _build_superseder(existing=existing, keep_ids=["p2"])
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        stats = sup.run(export_path=path)
        assert stats["export_count"] == 1
        assert "rollback_verified" not in stats
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 4. restore_from_export() -- points_count + sample readback
# ---------------------------------------------------------------------------
def test_restore_from_export_round_trips_byte_exactly(tmp_path):
    existing = {
        "p1": _rec(URL, dense=[0.1] * 8, sparse=([2, 9], [0.3, 0.4])),
        "p2": _rec(URL, dense=[0.2] * 8),
    }
    fake, sup = _build_superseder(existing=existing, keep_ids=["p3"])
    out = tmp_path / "rollback.jsonl.gz"
    sup.run(export_path=str(out))          # deletes p1, p2; export holds them
    assert fake.collections["aw_e8"]["points"] == {}

    result = US.restore_from_export(fake, "aw_e8", str(out), sample=10)

    assert result["restored"] == 2
    assert result["removed"] == 0
    assert result["sample_ok"] is True
    restored = fake.collections["aw_e8"]["points"]
    assert set(restored) == {"p1", "p2"}
    assert restored["p1"]["payload"]["url"] == URL
    assert restored["p1"]["vector"]["dense"] == [0.1] * 8
    assert restored["p1"]["vector"]["bm25"].indices == [2, 9]
    assert restored["p1"]["vector"]["bm25"].values == [0.3, 0.4]


def test_restore_from_export_catches_a_corrupted_sample_readback(tmp_path,
                                                                  monkeypatch):
    """A points_count match is not proof of content: if the restore lands
    the wrong payload on an id (simulated storage-layer fault), the sample
    readback must catch it and refuse -- the check a bare count cannot
    make (task 574)."""
    existing = {"p1": _rec(URL, dense=[0.1] * 8)}
    fake, sup = _build_superseder(existing=existing, keep_ids=["p2"])
    out = tmp_path / "rollback.jsonl.gz"
    sup.run(export_path=str(out))

    real_upsert = fake.upsert

    def corrupting_upsert(name, points, wait=False):
        real_upsert(name, points, wait=wait)
        # count is still right; payload for p1 is now wrong -- only a
        # content-level readback can see this
        if "p1" in fake.collections[name]["points"]:
            fake.collections[name]["points"]["p1"]["payload"] = {
                "url": "https://wrong.example/", "source_type": "content"}

    monkeypatch.setattr(fake, "upsert", corrupting_upsert)

    with pytest.raises(SystemExit, match="sample readback"):
        US.restore_from_export(fake, "aw_e8", str(out), sample=10)


def test_read_export_ids_reads_the_id_set_of_an_export_file(tmp_path):
    existing = {"p1": _rec(URL, dense=[0.1] * 8), "p2": _rec(URL, dense=[0.2] * 8)}
    fake, sup = _build_superseder(existing=existing, keep_ids=["p3"])
    out = tmp_path / "rollback.jsonl.gz"
    sup.run(export_path=str(out))
    assert US.read_export_ids(str(out)) == {"p1", "p2"}


# ---------------------------------------------------------------------------
# 5. apply_pack_delta.py end to end (CLI flags)
# ---------------------------------------------------------------------------
def _apply_registry():
    """pack aw_task574_pack (adds p5) vs. target aw_e8 (holds p1..p4, all
    stale once p5 lands) -- the same fixture as the unit tests, wired
    through the real CLI/main()."""
    return {
        "aw_task574_pack": {
            "points": {"p5": _rec(URL, dense=[0.5] * 8)},
            "dim": 8,
        },
        "aw_e8": {
            "points": {f"p{i}": _rec(URL, dense=[float(i)] * 8)
                      for i in (1, 2, 3, 4)},
            "dim": 8,
        },
    }


def test_apply_pack_delta_refuses_via_verify_superseded_against(
        tmp_path, monkeypatch, capsys):
    roster = tmp_path / "roster_export.jsonl.gz"
    with gzip.open(roster, "wt") as out:
        for i in (1, 2, 3):                     # undercounts p4, like 572
            out.write(json.dumps({"id": f"p{i}"}) + "\n")

    fake = FakeQdrant(_apply_registry())
    monkeypatch.setattr(APD, "qdrant_connect", lambda *a, **k: fake)
    # ADR-105 (task 581): the applier refuses a pack whose page is not
    # persisted in a registered store; this fixture gets a real store.
    from test_aw_index_surface_delta import install_persisted_index
    install_persisted_index(monkeypatch, tmp_path,
                            [{"url": URL, "point_id": "p", "text": "body"}])
    monkeypatch.setattr(sys, "argv", [
        "apply_pack_delta.py", "--pack", "aw_task574_pack",
        "--target", "aw_e8",
        "--verify-superseded-against", str(roster)])

    with pytest.raises(SystemExit) as exc:
        APD.main()
    assert "574" in str(exc.value) or "symmetric difference" in str(exc.value)
    # refused before the supersede delete ran
    assert fake.delete_calls == 0
    assert set(fake.collections["aw_e8"]["points"]) >= {"p1", "p2", "p3", "p4"}


def test_apply_pack_delta_matches_and_report_is_unchanged(
        tmp_path, monkeypatch, capsys):
    roster = tmp_path / "roster_export.jsonl.gz"
    with gzip.open(roster, "wt") as out:
        for i in (1, 2, 3, 4):
            out.write(json.dumps({"id": f"p{i}"}) + "\n")
    export_out = tmp_path / "rollback.jsonl.gz"

    fake = FakeQdrant(_apply_registry())
    monkeypatch.setattr(APD, "qdrant_connect", lambda *a, **k: fake)
    # ADR-105 (task 581): the applier refuses a pack whose page is not
    # persisted in a registered store; this fixture gets a real store.
    from test_aw_index_surface_delta import install_persisted_index
    install_persisted_index(monkeypatch, tmp_path,
                            [{"url": URL, "point_id": "p", "text": "body"}])
    monkeypatch.setattr(sys, "argv", [
        "apply_pack_delta.py", "--pack", "aw_task574_pack",
        "--target", "aw_e8",
        "--verify-superseded-against", str(roster),
        "--export-superseded", str(export_out)])

    APD.main()
    out = capsys.readouterr().out
    # the pre-574 report lines are unchanged in shape/content
    assert "pack aw_task574_pack=1 -> target aw_e8=4" in out
    assert "scrolled 1: +1 new, ~0 overwrote-existing" in out
    assert "OK" in out
    # and the delete actually ran: p1..p4 gone, p5 landed
    assert set(fake.collections["aw_e8"]["points"]) == {"p5"}
    assert export_out.exists()
    assert US.read_export_ids(str(export_out)) == {"p1", "p2", "p3", "p4"}


def test_apply_pack_delta_omitting_both_flags_is_untouched(tmp_path,
                                                            monkeypatch,
                                                            capsys):
    """No `--export-superseded` / `--verify-superseded-against` at all --
    the exact pre-574 behaviour, byte-for-byte."""
    fake = FakeQdrant(_apply_registry())
    monkeypatch.setattr(APD, "qdrant_connect", lambda *a, **k: fake)
    # ADR-105 (task 581): the applier refuses a pack whose page is not
    # persisted in a registered store; this fixture gets a real store.
    from test_aw_index_surface_delta import install_persisted_index
    install_persisted_index(monkeypatch, tmp_path,
                            [{"url": URL, "point_id": "p", "text": "body"}])
    monkeypatch.setattr(sys, "argv", [
        "apply_pack_delta.py", "--pack", "aw_task574_pack",
        "--target", "aw_e8"])
    APD.main()
    out = capsys.readouterr().out
    assert "pack aw_task574_pack=1 -> target aw_e8=4" in out
    assert "OK" in out
    assert set(fake.collections["aw_e8"]["points"]) == {"p5"}

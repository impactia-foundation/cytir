#!/usr/bin/env python3
"""`--export-overwritten`: the other half of the rollback (task 576).

`--export-superseded` (task 574) carries the rows a delta DELETES. It cannot
carry the rows a delta OVERWRITES, because `Superseder` protects every incoming
id -- an id that already existed in the target is never in the stale set. Task
576 measured 6 such rows on a 5,478-row tranche and 1,440 on a 533,724-row one,
and hit the consequence for real: reverting the small tranche by "delete the
pack's ids, restore the export" removes 6 rows that existed before the delta ran
and have no preserved copy anywhere.

These tests run against an in-memory fake shaped like the Qdrant client, so no
network and no collection. They assert:

  * the pre-image is what the target held BEFORE the upsert (payload and both
    vectors), not what the pack wrote;
  * a delta with no overwritten ids writes an empty file, not a missing one --
    "no pre-images" and "the flag did nothing" must be distinguishable;
  * omitting the flag reproduces the pre-576 output byte-for-byte;
  * the self-check counts pre-images back off the FILE and refuses a short
    export, so it can actually fail;
  * `restore_overwritten_from_export` round-trips the file, so the export is
    usable as a rollback rather than merely written -- and `restore_from_export`,
    the DELETION inverse, refuses it rather than half-applying it.
"""
from __future__ import annotations

import gzip
import json
import sys

import pytest

import cytir.index.pack_delta as APD  # noqa: E402
import cytir.index.supersede as US  # noqa: E402


@pytest.fixture(autouse=True)
def _no_card_guard_read(monkeypatch):
    """task 820: `apply_pack_delta` now runs the drain's card check (an HTTP id retrieve) before
    any write; these fakes are not Qdrant. The guard's own tests: tests/test_aw_index_card_guard_writers.py."""
    monkeypatch.setattr(APD, "guard_card_overwrites", lambda *a, **k: {})
    monkeypatch.setattr(APD, "client_url", lambda client: "http://fake")


# --------------------------------------------------------------------------
# the fake: a dict-backed collection pair with the client's method shapes
# --------------------------------------------------------------------------
class FakePoint:
    def __init__(self, pid, vector, payload):
        self.id = pid
        self.vector = vector
        self.payload = payload


class FakeCount:
    def __init__(self, n):
        self.count = n


class FakeInfo:
    def __init__(self, size):
        self.config = None
        self._size = size
        self.payload_schema = {"url": type("S", (), {"data_type": "keyword"})()}
        self.status = "green"


class FakeClient:
    """Only the surface `apply_pack_delta` and `Superseder` actually call."""

    def __init__(self, points: dict, size: int = 4):
        self.points = dict(points)
        self.size = size
        self.deleted: list = []
        self.upserted: list = []

    # -- reads ------------------------------------------------------------
    def get_collection(self, name):
        return FakeInfo(self.size)

    def count(self, name, exact=True):
        return FakeCount(len(self.points))

    def scroll(self, name, limit=100, offset=None, with_payload=True,
               with_vectors=True, scroll_filter=None):
        ids = sorted(self.points)
        start = ids.index(offset) if offset in ids else 0
        chunk = ids[start:start + limit]
        out = [FakePoint(i, self.points[i]["vector"], self.points[i]["payload"])
               for i in chunk]
        nxt = ids[start + limit] if start + limit < len(ids) else None
        return out, nxt

    def retrieve(self, name, ids, with_payload=True, with_vectors=True):
        out = []
        for i in ids:
            r = self.points.get(i)
            if r is None:
                continue
            out.append(FakePoint(
                i, r["vector"] if with_vectors else None,
                r["payload"] if with_payload else None))
        return out

    # -- writes -----------------------------------------------------------
    def upsert(self, name, points, wait=False):
        for p in points:
            self.points[p.id] = {"vector": p.vector, "payload": p.payload}
            self.upserted.append(p.id)

    def delete(self, name, points_selector=None, wait=False):
        for i in getattr(points_selector, "points", []) or []:
            self.points.pop(i, None)
            self.deleted.append(i)

    def create_payload_index(self, *a, **k):
        return None


def vec(x):
    return {"dense": [float(x)] * 4,
            "bm25": US.qm.SparseVector(indices=[int(x)], values=[1.0])
            if US.qm is not None else {"indices": [int(x)], "values": [1.0]}}


def payload(url, tag):
    return {"url": url, "source_type": "docs", "domain": "example.org",
            "tag": tag}


def build(tmp_path, monkeypatch, *, overlap=2, fresh=3, stale=1):
    """A target holding `overlap` ids the pack also carries, `stale` ids of the
    same page the pack does not, and a pack of `overlap + fresh` ids."""
    url = "https://example.org/a"
    tgt_points = {}
    for i in range(overlap):
        tgt_points[f"ov{i}"] = {"vector": vec(i), "payload": payload(url, "OLD")}
    for i in range(stale):
        tgt_points[f"st{i}"] = {"vector": vec(90 + i),
                                "payload": payload(url, "STALE")}
    pack_points = {}
    for i in range(overlap):
        pack_points[f"ov{i}"] = {"vector": vec(50 + i),
                                 "payload": payload(url, "NEW")}
    for i in range(fresh):
        pack_points[f"fr{i}"] = {"vector": vec(60 + i),
                                 "payload": payload(url, "NEW")}

    tgt = FakeClient(tgt_points)
    pack = FakeClient(pack_points)
    made = {"pack": pack, "target": tgt}

    def fake_client(url=None, timeout=None):
        return made.pop("pack") if "pack" in made else made["target"]

    monkeypatch.setattr(APD, "qdrant_connect", fake_client)
    monkeypatch.setattr(APD.SC, "shape_from_info", lambda info: {"dense_size": 4})
    monkeypatch.setattr(APD.SC, "assert_dim_matches",
                        lambda *a, **k: None)
    monkeypatch.setattr(US, "_gold_audit", lambda urls, log: 0)
    # ADR-105 (task 581): the applier refuses a pack whose pages are not
    # persisted in a registered store. Built for real over this fixture's one
    # page rather than stubbed -- a guard faked to let another guard's test
    # pass is a guard nobody is testing.
    from test_aw_index_surface_delta import install_persisted_index
    install_persisted_index(monkeypatch, tmp_path,
                            [{"url": url, "point_id": "a", "text": "body"}])
    return tgt, pack


def run(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pack-delta"] + argv)
    APD.main()


BASE = ["--pack", "P", "--pack-url", "http://p", "--target", "T",
        "--target-url", "http://t", "--batch", "2"]


@pytest.mark.skipif(US.qm is None, reason="qdrant_client not installed")
def test_preimage_is_what_the_target_held_before_the_upsert(tmp_path,
                                                            monkeypatch, capsys):
    tgt, _ = build(tmp_path, monkeypatch)
    out = tmp_path / "pre.jsonl.gz"
    run(BASE + ["--export-overwritten", str(out)], monkeypatch)

    rows = [json.loads(l) for l in gzip.open(out, "rt") if l.strip()]
    assert {r["id"] for r in rows} == {"ov0", "ov1"}
    for r in rows:
        # the OLD payload, not the pack's
        assert r["payload"]["tag"] == "OLD"
    # and the OLD vector: ov0 was written at vec(0), the pack carries vec(50)
    got = {r["id"]: r["vector"] for r in rows}
    assert got["ov0"]["dense"] == [0.0] * 4
    assert got["ov1"]["dense"] == [1.0] * 4
    # the target really was overwritten, so the pre-image is the only copy
    assert tgt.points["ov0"]["payload"]["tag"] == "NEW"
    assert "pre-images of 2 overwritten id(s)" in capsys.readouterr().out


@pytest.mark.skipif(US.qm is None, reason="qdrant_client not installed")
def test_no_overwrites_writes_an_empty_file_not_a_missing_one(tmp_path,
                                                              monkeypatch):
    build(tmp_path, monkeypatch, overlap=0, fresh=3, stale=1)
    out = tmp_path / "pre.jsonl.gz"
    run(BASE + ["--export-overwritten", str(out)], monkeypatch)
    assert out.exists(), "a missing file cannot be told from a skipped flag"
    assert [l for l in gzip.open(out, "rt") if l.strip()] == []


@pytest.mark.skipif(US.qm is None, reason="qdrant_client not installed")
def test_omitting_the_flag_reproduces_the_pre_576_output(tmp_path, monkeypatch,
                                                         capsys):
    tgt_a, _ = build(tmp_path, monkeypatch)
    run(BASE, monkeypatch)
    without = capsys.readouterr().out
    tgt_b, _ = build(tmp_path, monkeypatch)
    run(BASE + ["--export-overwritten", str(tmp_path / "p2.jsonl.gz")],
        monkeypatch)
    with_flag = capsys.readouterr().out

    assert "pre-images" not in without
    # the only difference is the one added line
    extra = [l for l in with_flag.splitlines() if l not in without.splitlines()]
    assert len(extra) == 1 and "pre-images of 2" in extra[0]
    assert tgt_a.points.keys() == tgt_b.points.keys()


@pytest.mark.skipif(US.qm is None, reason="qdrant_client not installed")
def test_the_selfcheck_reads_the_FILE_and_refuses_a_short_export(tmp_path,
                                                                 monkeypatch):
    """The check counts pre-images back off the artifact, not off a counter.

    A counter incremented next to a write proves the intention; this must fail
    when the bytes did not land, which is the only way the guard can fail at
    all (verification-doctrine 12).
    """
    build(tmp_path, monkeypatch)
    out = tmp_path / "pre.jsonl.gz"

    class DropsTheFirstRow:
        def __init__(self, fh):
            self.fh, self.n = fh, 0

        def write(self, s):
            self.n += 1
            return None if self.n == 1 else self.fh.write(s)

        def close(self):
            self.fh.close()

    real_open = gzip.open

    def patched_open(path, mode="rt", **kw):
        fh = real_open(path, mode, **kw)
        return (DropsTheFirstRow(fh)
                if "w" in mode and str(path).endswith("pre.jsonl.gz") else fh)

    monkeypatch.setattr(APD.gzip, "open", patched_open)
    with pytest.raises(SystemExit) as exc:
        run(BASE + ["--export-overwritten", str(out)], monkeypatch)
    assert "rollback is incomplete" in str(exc.value)
    assert "1 pre-image(s)" in str(exc.value)


@pytest.mark.skipif(US.qm is None, reason="qdrant_client not installed")
def test_the_preimage_file_round_trips_through_its_own_inverse(tmp_path,
                                                               monkeypatch):
    """`restore_overwritten_from_export` puts the pre-images back.

    `restore_from_export` is NOT the inverse for this file and is not used as
    one: it is built for rows a delta DELETED, so it verifies
    `before - removed + restored == after`, and re-upserting an id that still
    exists changes no count. Reusing it here would either fail honestly or, if
    someone relaxed its arithmetic, hide a restore that never landed.
    """
    tgt, _ = build(tmp_path, monkeypatch)
    out = tmp_path / "pre.jsonl.gz"
    run(BASE + ["--export-overwritten", str(out)], monkeypatch)
    assert tgt.points["ov0"]["payload"]["tag"] == "NEW"

    stats = US.restore_overwritten_from_export(tgt, "T", str(out),
                                               log=lambda *a: None)
    assert stats["restored"] == 2
    assert tgt.points["ov0"]["payload"]["tag"] == "OLD"
    assert tgt.points["ov0"]["vector"]["dense"] == [0.0] * 4
    assert tgt.points["ov1"]["payload"]["tag"] == "OLD"


@pytest.mark.skipif(US.qm is None, reason="qdrant_client not installed")
def test_restore_from_export_is_refused_for_a_preimage_file(tmp_path,
                                                            monkeypatch):
    """The deletion-inverse must not silently accept an overwrite file."""
    tgt, _ = build(tmp_path, monkeypatch)
    out = tmp_path / "pre.jsonl.gz"
    run(BASE + ["--export-overwritten", str(out)], monkeypatch)
    with pytest.raises(SystemExit) as exc:
        US.restore_from_export(tgt, "T", str(out), remove_ids=[],
                               log=lambda *a: None)
    assert "count MISMATCH" in str(exc.value)


def test_cli_registers_pack_delta_verb(capsys):
    """`cyt index pack-delta --help` reaches this module's parser (task 2062)."""
    from cytir import cli
    with pytest.raises(SystemExit) as exc:
        cli.main(["index", "pack-delta", "--help"])
    assert exc.value.code == 0
    assert "--export-overwritten" in capsys.readouterr().out

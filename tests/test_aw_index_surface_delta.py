"""Task 546 -- the join-rate guard and the two-surface detector.

Real-record tests against a sealed in-memory fake of the qdrant HTTP surface
(`api`/`points_count` are monkeypatched at the module level exactly as
imported -- `cytir.index.surface_delta` calls the bare names, so patching
them there intercepts every call the module makes without touching the real
`cytir.core.qdrant_http` module or any live collection). No network call is made by any
test in this file.

Fixture shape mirrors the assembly-binder tests and
the arm-classification script of task 538: a `shard-0000.vectors.manifest.json` +
`.vectors.f16` + `.row_ids.jsonl` per vectors dir, `shard-0000.texts.jsonl.gz`
per stage dir, `point_id` binding row order across both.

The context7 case (task 538 sec 1.5): a delta whose ids are minted from
`canonical_text`, so half of them address points the collection no longer
holds under those ids -- while the collection's real rows for the SAME urls
sit, unaddressed, under different ids. `test_half_joined_delta_is_refused`
and `test_half_joined_delta_reports_the_context7_pattern` fixture that
directly; `test_clean_delta_applies_unchanged` proves the existing (pre-546)
behaviour survives byte-for-byte when the join is clean and no
`--arm-a-vectors` is given.
"""
from __future__ import annotations

import gzip
import json
import sys

import numpy as np
import pytest

import cytir.index.surface_delta as mod

DIM = mod.DIM


# --------------------------------------------------------------------------
# fixture plumbing
# --------------------------------------------------------------------------
def write_shard(tmp_path, tag, rows, vectors):
    """rows: list of dicts with at least point_id/url/text. vectors: (n, DIM)
    float array, row i belongs to rows[i]."""
    stage = tmp_path / "stage"
    vecdir = tmp_path / "vectors"
    stage.mkdir(parents=True, exist_ok=True)
    vecdir.mkdir(parents=True, exist_ok=True)
    with gzip.open(stage / f"{tag}.texts.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with open(vecdir / f"{tag}.row_ids.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({"point_id": r["point_id"]}) + "\n")
    np.asarray(vectors, dtype="<f2").tofile(vecdir / f"{tag}.vectors.f16")
    (vecdir / f"{tag}.vectors.manifest.json").write_text(
        json.dumps({"n": len(rows)}))
    return str(stage), str(vecdir)


def install_persisted_index(monkeypatch, tmp_path, rows, layer="l1"):
    """Build a REAL persisted-inputs index over the fixture's own URLs.

    ADR-105 (task 581): every applier refuses rows whose page is not persisted
    in a registered store. A test fixture is a pack; it needs a store behind it
    like any other pack does, and the honest way to give it one is to write the
    pages into a store tree and index it -- not to stub the guard out. Returns
    the index directory; ``AW_PERSISTED_INDEX`` is pointed at it for the
    duration of the test.

    ``layer="l2"`` writes the pages as L2 passages instead, which is the
    fixture for "this page survives only as chunks" -- the ADR-093 route.
    """
    import cytir.corpus.persisted as PI
    corpus = tmp_path / "persisted-corpus"
    if layer == "l1":
        d = corpus / "l1" / "release=test" / "source=fixture"
        d.mkdir(parents=True, exist_ok=True)
        with gzip.open(d / "l1-markdown-00000.jsonl.gz", "wt",
                       encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({
                    "canonical_url": r["url"], "url": r["url"],
                    "markdown": r.get("text") or "body",
                    "markdown_sha256": f"sha-{r['point_id']}"}) + "\n")
    else:
        d = corpus / "l2" / "wave=fixture"
        d.mkdir(parents=True, exist_ok=True)
        with gzip.open(d / "passages-00000.jsonl.gz", "wt",
                       encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({"url": r["url"],
                                     "content_digest": "d" * 40}) + "\n")
    out = tmp_path / "persisted-index"
    PI.build_index(str(out), root=str(corpus), log=lambda *a, **k: None)
    monkeypatch.setenv("AW_PERSISTED_INDEX", str(out))
    return str(out)


def random_unit_matrix(n, seed):
    rng = np.random.RandomState(seed)
    m = rng.normal(size=(n, DIM)).astype("float32")
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return m


def rows_for(prefix, n):
    return [{"point_id": f"{prefix}-{i:04d}", "url": f"https://ex.test/{i}",
             "text": f"document body number {i} about topic {i}",
             "sparse_text": f"document body number {i} about topic {i}",
             "canonical_text": f"canon-{prefix}-{i}"} for i in range(n)]


class FakeCollection:
    """id -> {"payload": {...}, "vector": {"dense": [...], "bm25": {...}}}

    `dim` is the collection's OWN declared dense width (task 563's dim
    guard reads `GET /collections/<name>` before touching a single point) --
    defaulted from the first point's stored vector when omitted, so every
    caller from before task 563 that never passed it keeps reporting a
    collection that matches its own fixture data exactly.
    """

    def __init__(self, points, dim=None):
        self.points = points
        self.write_calls = 0
        if dim is None:
            first = next(iter(points.values()), None)
            dim = len(first["vector"]["dense"]) if first else 0
        self.dim = dim

    def count(self):
        return len(self.points)


def install_fake_qdrant(monkeypatch, coll: FakeCollection):
    def fake_api(base, path, payload=None, method=""):
        if path.startswith("/collections/") and "/points" not in path:
            return {"result": {"config": {"params": {"vectors": {
                "dense": {"size": coll.dim}}}}}}
        if path.endswith("/points/count"):
            return {"result": {"count": len(coll.points)}}
        if path.split("?", 1)[0].endswith("/points/vectors"):
            coll.write_calls += 1
            for u in payload["points"]:
                p = coll.points.get(u["id"])
                if p is not None:
                    p["vector"].update(u["vector"])
            return {"result": {}}
        if path.endswith("/points/scroll"):
            flt = payload["filter"]["must"][0]
            key, val = flt["key"], flt["match"]["value"]
            matches = [pid for pid, p in coll.points.items()
                      if p["payload"].get(key) == val]
            return {"result": {"points": [{"id": pid} for pid in matches],
                               "next_page_offset": None}}
        if path.endswith("/points"):
            ids = payload["ids"]
            with_vector = payload.get("with_vector")
            with_payload = payload.get("with_payload")
            out = []
            for pid in ids:
                p = coll.points.get(pid)
                if p is None:
                    continue
                row = {"id": pid}
                if with_payload:
                    row["payload"] = p["payload"]
                if with_vector:
                    keys = with_vector if isinstance(with_vector, list) \
                        else ["dense", "bm25"]
                    row["vector"] = {k: p["vector"][k] for k in keys
                                    if k in p["vector"]}
                out.append(row)
            return {"result": out}
        raise AssertionError(f"unexpected qdrant path: {path}")

    def fake_points_count(base, collection, exact=True):
        return len(coll.points)

    monkeypatch.setattr(mod, "api", fake_api)
    monkeypatch.setattr(mod, "points_count", fake_points_count)


def run_main(monkeypatch, argv):
    argv = list(argv)
    if "--qdrant" not in argv:         # no default target exists: the fake needs a name
        argv += ["--qdrant", "http://fake-qdrant.invalid"]
    if "--collection" not in argv:
        argv += ["--collection", "fake_coll"]
    return mod.main(argv)


# --------------------------------------------------------------------------
# 1. join-rate guard
# --------------------------------------------------------------------------
def test_clean_delta_applies_unchanged(tmp_path, monkeypatch):
    """100% id-join, no --arm-a-vectors: old behaviour, byte-for-byte."""
    rows = rows_for("clean", 6)
    vecs = random_unit_matrix(6, seed=1)
    stage, vectors = write_shard(tmp_path, "shard-0000", rows, vecs)
    # ADR-105 (task 581): a pack needs a store behind it. Built for real
    # over this fixture's own pages -- never stubbed, or the guard being
    # tested here would be passing behind a faked one.
    install_persisted_index(monkeypatch, tmp_path, rows)
    coll = FakeCollection({
        r["point_id"]: {"payload": {"url": r["url"]},
                        "vector": {"dense": [0.0] * 64}}
        for r in rows})
    install_fake_qdrant(monkeypatch, coll)

    report = tmp_path / "report.json"
    rc = run_main(monkeypatch, [
        "--stage", stage, "--vectors", vectors, "--report", str(report),
        "--mrl", "64", "--no-sparse", "--apply"])
    assert rc == 0
    doc = json.loads(report.read_text())
    assert doc["join_guard"]["id_join_rate"] == 1.0
    assert doc["join_guard"]["refused"] is False
    assert doc["two_surface_check"] == {
        "skipped": True, "reason": "--arm-a-vectors not given"}
    assert doc["conservation_ok"] is True
    assert doc["missing_in_collection"] == 0
    assert doc["rows"] == 6
    # the write actually happened
    for r in rows:
        assert coll.points[r["point_id"]]["vector"]["dense"] != [0.0] * 64


def test_half_joined_delta_is_refused(tmp_path, monkeypatch):
    """The context7 pattern (task 538 sec 1.5): 5 of 10 delta ids exist in
    the collection under those ids; the other 5 urls ARE in the collection,
    just under different (alias) ids the delta never mentions. The guard
    must refuse before any write, and the collection must be untouched."""
    rows = rows_for("c7", 10)
    vecs = random_unit_matrix(10, seed=2)
    stage, vectors = write_shard(tmp_path, "shard-0000", rows, vecs)
    # ADR-105 (task 581): a pack needs a store behind it. Built for real
    # over this fixture's own pages -- never stubbed, or the guard being
    # tested here would be passing behind a faked one.
    install_persisted_index(monkeypatch, tmp_path, rows)

    points = {}
    for i, r in enumerate(rows):
        if i < 5:
            points[r["point_id"]] = {"payload": {"url": r["url"]},
                                     "vector": {"dense": [0.0] * 64}}
        else:
            points[f"alias-{i}"] = {"payload": {"url": r["url"]},
                                    "vector": {"dense": [0.0] * 64}}
    coll = FakeCollection(points)
    install_fake_qdrant(monkeypatch, coll)

    report = tmp_path / "report.json"
    rc = run_main(monkeypatch, [
        "--stage", stage, "--vectors", vectors, "--report", str(report),
        "--mrl", "64", "--no-sparse", "--apply",
        "--join-url-sample", "10"])
    assert rc == 1
    doc = json.loads(report.read_text())
    g = doc["join_guard"]
    assert g["ids_total"] == 10
    assert g["ids_found"] == 5
    assert g["id_join_rate"] == pytest.approx(0.5)
    assert g["refused"] is True
    assert g["allow_partial_join_reason"] is None
    # reverse url join: 10 urls staged, 10 collection points across them
    # (5 direct + 5 alias), only 5 shared ids -> 5 collection-only, 5 staged-only
    assert g["url_join_staged_points"] == 10
    assert g["url_join_collection_points"] == 10
    assert g["url_join_overlap"] == 5
    assert g["url_join_collection_only"] == 5
    assert g["url_join_staged_only"] == 5
    assert doc["conservation_ok"] is False
    assert doc["two_surface_check"] == {"skipped": True,
                                        "reason": "refused by the join guard"}
    # no write of any kind was attempted
    assert coll.write_calls == 0
    for pid, p in coll.points.items():
        assert p["vector"]["dense"] == [0.0] * 64


def test_half_joined_delta_proceeds_with_allow_partial_join(tmp_path, monkeypatch):
    rows = rows_for("c7b", 10)
    vecs = random_unit_matrix(10, seed=3)
    stage, vectors = write_shard(tmp_path, "shard-0000", rows, vecs)
    # ADR-105 (task 581): a pack needs a store behind it. Built for real
    # over this fixture's own pages -- never stubbed, or the guard being
    # tested here would be passing behind a faked one.
    install_persisted_index(monkeypatch, tmp_path, rows)
    points = {}
    for i, r in enumerate(rows):
        if i < 5:
            points[r["point_id"]] = {"payload": {"url": r["url"]},
                                     "vector": {"dense": [0.0] * 64}}
        else:
            points[f"alias-{i}"] = {"payload": {"url": r["url"]},
                                    "vector": {"dense": [0.0] * 64}}
    coll = FakeCollection(points)
    install_fake_qdrant(monkeypatch, coll)

    report = tmp_path / "report.json"
    rc = run_main(monkeypatch, [
        "--stage", stage, "--vectors", vectors, "--report", str(report),
        "--mrl", "64", "--no-sparse", "--apply", "--join-url-sample", "10",
        "--allow-partial-join", "context7 split family, task 538"])
    assert rc == 0
    doc = json.loads(report.read_text())
    g = doc["join_guard"]
    assert g["id_join_rate"] == pytest.approx(0.5)
    assert g["refused"] is False
    assert g["allow_partial_join_reason"] == "context7 split family, task 538"
    assert doc["missing_in_collection"] == 5
    # only the 5 joined ids were written
    assert coll.write_calls > 0
    for i, r in enumerate(rows[:5]):
        assert coll.points[r["point_id"]]["vector"]["dense"] != [0.0] * 64
    for i in range(5, 10):
        assert coll.points[f"alias-{i}"]["vector"]["dense"] == [0.0] * 64


# --------------------------------------------------------------------------
# 2. two-surface detector
# --------------------------------------------------------------------------
def test_two_surface_detector_passes_when_write_lands(tmp_path, monkeypatch):
    rows = rows_for("ts", 5)
    arm_b_vecs = random_unit_matrix(5, seed=4)
    arm_a_vecs = random_unit_matrix(5, seed=5)  # distinct from arm B
    stage, vectors_b = write_shard(tmp_path / "b", "shard-0000", rows, arm_b_vecs)
    _, vectors_a = write_shard(tmp_path / "a", "shard-0000", rows, arm_a_vecs)
    # ADR-105 (task 581): a pack needs a store behind it. Built for real
    # over this fixture's own pages -- never stubbed, or the guard being
    # tested here would be passing behind a faked one.
    install_persisted_index(monkeypatch, tmp_path, rows)

    coll = FakeCollection({
        r["point_id"]: {"payload": {"url": r["url"]},
                        "vector": {"dense": (arm_a_vecs[i, :64] /
                                            np.linalg.norm(arm_a_vecs[i, :64])).tolist()}}
        for i, r in enumerate(rows)})
    install_fake_qdrant(monkeypatch, coll)

    report = tmp_path / "report.json"
    rc = run_main(monkeypatch, [
        "--stage", stage, "--vectors", vectors_b, "--report", str(report),
        "--mrl", "64", "--no-sparse", "--apply",
        "--arm-a-vectors", vectors_a])
    assert rc == 0
    doc = json.loads(report.read_text())
    ts = doc["two_surface_check"]
    assert ts["skipped"] is False
    assert ts["sampled"] == 5
    assert ts["matches_armB"] == 5
    assert ts["matches_armA"] == 0
    assert ts["neither"] == 0
    assert ts["ok"] is True
    assert doc["conservation_ok"] is True


def test_two_surface_detector_refuses_on_split_family(monkeypatch):
    """Direct unit test of the detector: some ids still hold arm A (a write
    that silently did not take), some hold arm B -- the split-family failure
    task 538 actually found. Must refuse regardless of points_count."""
    ids = [f"id-{i}" for i in range(6)]
    a = random_unit_matrix(6, seed=10)[:, :32]
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = random_unit_matrix(6, seed=11)[:, :32]
    b /= np.linalg.norm(b, axis=1, keepdims=True)

    def fake_load_full_arm(vec_dir, mrl):
        return {"A": (a, ids), "B": (b, ids)}[vec_dir]

    monkeypatch.setattr(mod, "load_full_arm", fake_load_full_arm)

    stored = {}
    for i, pid in enumerate(ids):
        if i < 3:
            stored[pid] = a[i]           # write did not take -- stale arm A
        elif i < 5:
            stored[pid] = b[i]           # correctly applied
        else:
            stored[pid] = np.ones(32, dtype="float32") / np.sqrt(32)  # neither

    def fake_api(base, path, payload=None, method=""):
        assert path.endswith("/points")
        out = []
        for pid in payload["ids"]:
            out.append({"id": pid, "vector": {"dense": stored[pid].tolist()}})
        return {"result": out}

    monkeypatch.setattr(mod, "api", fake_api)

    result = mod.two_surface_check("http://fake", "coll", ids, "A", "B", 32,
                                   sample_n=200, seed=1)
    assert result["sampled"] == 6
    assert result["found"] == 6
    assert result["matches_armA"] == 3
    assert result["matches_armB"] == 2
    assert result["neither"] == 1
    assert result["ok"] is False


def test_two_surface_detector_skipped_without_arm_a(tmp_path, monkeypatch):
    rows = rows_for("noa", 3)
    vecs = random_unit_matrix(3, seed=6)
    stage, vectors = write_shard(tmp_path, "shard-0000", rows, vecs)
    # ADR-105 (task 581): a pack needs a store behind it. Built for real
    # over this fixture's own pages -- never stubbed, or the guard being
    # tested here would be passing behind a faked one.
    install_persisted_index(monkeypatch, tmp_path, rows)
    coll = FakeCollection({
        r["point_id"]: {"payload": {"url": r["url"]},
                        "vector": {"dense": [0.0] * 64}}
        for r in rows})
    install_fake_qdrant(monkeypatch, coll)
    report = tmp_path / "report.json"
    rc = run_main(monkeypatch, [
        "--stage", stage, "--vectors", vectors, "--report", str(report),
        "--mrl", "64", "--no-sparse", "--apply"])
    assert rc == 0
    doc = json.loads(report.read_text())
    assert doc["two_surface_check"]["skipped"] is True


def test_dry_run_never_writes_and_skips_two_surface(tmp_path, monkeypatch):
    rows = rows_for("dry", 3)
    vecs = random_unit_matrix(3, seed=7)
    stage, vectors = write_shard(tmp_path, "shard-0000", rows, vecs)
    # ADR-105 (task 581): a pack needs a store behind it. Built for real
    # over this fixture's own pages -- never stubbed, or the guard being
    # tested here would be passing behind a faked one.
    install_persisted_index(monkeypatch, tmp_path, rows)
    coll = FakeCollection({
        r["point_id"]: {"payload": {"url": r["url"]},
                        "vector": {"dense": [0.0] * 64}}
        for r in rows})
    install_fake_qdrant(monkeypatch, coll)
    report = tmp_path / "report.json"
    rc = run_main(monkeypatch, [
        "--stage", stage, "--vectors", vectors, "--report", str(report),
        "--mrl", "64", "--no-sparse"])  # no --apply
    assert rc == 0
    doc = json.loads(report.read_text())
    assert doc["apply"] is False
    assert coll.write_calls == 0
    assert doc["two_surface_check"] == {"skipped": True,
                                        "reason": "not applied (dry run)"}


def test_sparse_path_still_runs_for_a_clean_apply(tmp_path, monkeypatch):
    """Confirms the pre-546 sparse/dense write path is untouched."""
    rows = rows_for("sp", 3)
    vecs = random_unit_matrix(3, seed=8)
    stage, vectors = write_shard(tmp_path, "shard-0000", rows, vecs)
    # ADR-105 (task 581): a pack needs a store behind it. Built for real
    # over this fixture's own pages -- never stubbed, or the guard being
    # tested here would be passing behind a faked one.
    install_persisted_index(monkeypatch, tmp_path, rows)
    coll = FakeCollection({
        r["point_id"]: {"payload": {"title": "T", "site_name": "S"},
                        "vector": {"dense": [0.0] * 64, "bm25": {"indices": [], "values": []}}}
        for r in rows})
    # url payload must be present for the reverse-join check to be meaningful
    for r in rows:
        coll.points[r["point_id"]]["payload"]["url"] = r["url"]
    install_fake_qdrant(monkeypatch, coll)
    report = tmp_path / "report.json"
    rc = run_main(monkeypatch, [
        "--stage", stage, "--vectors", vectors, "--report", str(report),
        "--mrl", "64", "--apply"])
    assert rc == 0
    doc = json.loads(report.read_text())
    assert doc["conservation_ok"] is True
    for r in rows:
        assert coll.points[r["point_id"]]["vector"]["bm25"]["indices"]



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
    r = subprocess.run([sys.executable, "-m", "cytir.index.surface_delta", "--stage", "x", "--vectors", "y",
         "--qdrant", "http://127.0.0.1:9", "--collection", "aw_e10_twin", "--report", str(tmp_path / "r.json"), "--apply"], cwd=app, env=dict(_os.environ, AW_TWIN_WRITE_HOLD=str(hold)),
        capture_output=True, text=True, timeout=120)
    assert r.returncode != 0 and "an aw_e10 build holds" in r.stderr, r.stderr[-800:]

#!/usr/bin/env python3
"""`superseded_generation` (ADR-114 am.1, ruling 26 item 21a): the queue reason for a PURE
tombstone pack that retires an old generation of rows whose new rows are already served.

Fixture `fixtures/loda-old-generation-rows.json`: REAL ids/urls/family of five loda urls of the
prepared pack `fr-loda-exact-old-generation-superseded-tombstones-excl-long-articles` (old and
new generation, read from the build-box twin 2026-10-05) and one real entity card. No text.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cytir.index import point_delete as PD  # noqa: E402
from cytir.index import promote_queue as Q  # noqa: E402
from cytir.distribute import pack  # noqa: E402

DOC = json.loads((HERE / "fixtures" / "loda-old-generation-rows.json").read_text())
ROWS = DOC["rows"]
BY_ID = {r["id"]: r for r in ROWS}
OLD = [r for r in ROWS if r["role"] == "old"]
NEW = [r for r in ROWS if r["role"] == "new"]
CARD = next(r for r in ROWS if r["role"] == "card")
SG = Q.SUPERSEDED_GENERATION


def tomb(row, reason=SG):
    return {"id": row["id"], "url": row["url"], "reason": reason}


class FakeTwin:
    """A twin holding `rows`: answers the id retrieve and the filtered url scroll the drain
    makes, nothing else (an unknown call fails the test)."""

    def __init__(self, rows):
        self.rows = {r["id"]: dict(r["payload"]) for r in rows}
        self.scrolls = 0

    def api(self, url, path, body=None, method=None):
        assert path.startswith("/collections/aw_e8/points"), path
        if path.endswith("/points/scroll"):
            self.scrolls += 1
            vals = set(body["filter"]["must"][0]["match"]["any"])
            return {"points": [{"id": i, "payload": dict(p)} for i, p in self.rows.items()
                               if p.get("url") in vals], "next_page_offset": None}
        keep = body["with_payload"]
        out = []
        for i in body["ids"]:
            if i in self.rows:
                pl = self.rows[i]
                out.append({"id": i, "payload": (dict(pl) if keep is True else
                            {k: v for k, v in pl.items() if keep and k in keep})})
        return out


@pytest.fixture
def twin(monkeypatch):
    t = FakeTwin(ROWS)
    monkeypatch.setattr(Q.PSP, "_api", t.api)
    monkeypatch.setattr(PD, "_api", t.api)
    return t


OLD_IDS = {r["id"] for r in OLD}
DENSE = {"dense": {"size": 2, "distance": "Cosine"}}
SPARSE = ["bm25"]


def validate(tombstones, **kw):
    kw.setdefault("export_ids", OLD_IDS)
    return Q.validate_tombstones(tombstones, pack_urls=set(), target_url="http://twin",
                                 collection="aw_e8", **kw)


# ------------------------------------------------------------- per-id rule
def test_every_old_row_with_a_surviving_new_sibling_passes(twin):
    report = validate([tomb(r) for r in OLD])
    assert report["count"] == len(OLD)
    assert report["by_reason"] == {SG: len(OLD)}
    assert twin.scrolls >= 1   # the sibling read ran (a filtered scroll on the twin)


def test_an_id_with_no_surviving_sibling_refuses_the_whole_pack_and_names_it(twin):
    # tombstone the new row of one url too: its old row now has no survivor
    url = OLD[0]["url"]
    new_here = [r for r in NEW if r["url"] == url]
    assert new_here
    tombs = [tomb(r) for r in OLD] + [tomb(new_here[0])]
    with pytest.raises(Q.QueueError) as exc:
        validate(tombs)
    msg = str(exc.value)
    assert OLD[0]["id"] in msg and "no surviving row" in msg
    other_url_old = next(r for r in OLD if r["url"] != url)
    assert other_url_old["id"] not in msg   # only the failing ids are named


def test_a_survivor_of_another_family_does_not_count(twin):
    url = OLD[0]["url"]
    for r in NEW:
        if r["url"] == url:
            twin.rows[r["id"]]["provenance"] = "legi"
    with pytest.raises(Q.QueueError, match="no surviving row"):
        validate([tomb(r) for r in OLD if r["url"] == url])


def test_a_card_at_the_url_is_not_a_survivor_and_a_card_row_is_never_tombstoned(twin):
    # a card at the old rows' url does not count as the new generation
    url = OLD[-1]["url"]
    for r in NEW:
        if r["url"] == url:
            twin.rows.pop(r["id"])
    twin.rows["card-1"] = {"url": url, "source_type": "entity_card", "provenance": "loda"}
    with pytest.raises(Q.QueueError, match="no surviving row"):
        validate([tomb(r) for r in OLD if r["url"] == url])
    # and the real card row itself is refused outright
    with pytest.raises(Q.QueueError, match="entity card"):
        validate([tomb(CARD)])


def test_the_tombstone_url_must_be_the_stored_rows_document(twin):
    bad = dict(tomb(OLD[0]), url="https://www.legifrance.gouv.fr/loda/article_lc/LEGIARTI000000000001")
    with pytest.raises(Q.QueueError, match="same-document key differs"):
        validate([bad])


def test_another_old_row_of_the_same_family_is_not_a_survivor(twin):
    # the url's only other same-family row is an OLD row (in the export): not the new generation
    url = OLD[0]["url"]
    for r in NEW:
        if r["url"] == url:
            twin.rows.pop(r["id"])
    olds = [r for r in OLD if r["url"] == url]
    assert len(olds) >= 2
    with pytest.raises(Q.QueueError, match="no surviving row") as exc:
        validate([tomb(olds[0])])      # olds[1] is untombstoned but is in the export
    assert olds[0]["id"] in str(exc.value)
    # the same read passes once the survivor is a row outside the export
    twin.rows["11111111-1111-5111-8111-111111111111"] = dict(NEW[0]["payload"], url=url)
    assert validate([tomb(olds[0])])["count"] == 1


def test_an_already_gone_id_is_dropped_like_any_tombstone_and_the_rest_checked(twin):
    ghost = {"id": "00000000-0000-0000-0000-00000000dead", "url": OLD[0]["url"], "reason": SG}
    report = validate([tomb(r) for r in OLD] + [ghost])
    assert report["missing_ids"] == [ghost["id"]]
    assert report["count"] == len(OLD)


# -------------------------------------------------------- pack shape + export
def test_the_pack_must_be_pure_and_of_one_reason():
    pts = [{"id": "p", "payload": {"url": "https://x"}}]
    with pytest.raises(Q.QueueError, match="carries no points"):
        Q.check_superseded_generation_pack({}, pts, [tomb(OLD[0])])
    with pytest.raises(Q.QueueError, match="only that reason"):
        Q.check_superseded_generation_pack({}, [], [tomb(OLD[0]), tomb(OLD[1], "takedown")])
    Q.check_superseded_generation_pack({}, [], [tomb(OLD[0])])


def write_export(tmp_path, rows, *, vector=True, mutate=None):
    """An export of `rows`; `mutate(row_dict)` doctors a line (for the negatives)."""
    path = tmp_path / "export.jsonl.gz"
    with gzip.open(path, "wt") as fh:
        for r in rows:
            line = {"id": r["id"], "payload": dict(r["payload"])}
            if vector:
                line["vector"] = {"dense": [0.1, 0.2], "bm25": {"indices": [1], "values": [1.0]}}
            if mutate:
                mutate(line)
            fh.write(json.dumps(line) + "\n")
    return path


def declared(path, rows):
    # the sha256 is of the RAW .gz bytes
    return {"export": {"path": str(path), "rows": rows,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}


def verify(manifest, tombs):
    return Q.verify_removed_rows_export(manifest, tombs, dense_config=DENSE, sparse_names=SPARSE)


def test_export_is_declared_exists_hashes_and_covers_every_tombstoned_id(tmp_path):
    path = write_export(tmp_path, OLD + NEW)
    got = verify(declared(path, len(OLD + NEW)), [tomb(r) for r in OLD])
    assert got["tombstoned_ids_covered"] == len(OLD)
    assert got["ids"] >= OLD_IDS
    with pytest.raises(Q.QueueError, match="declares its `export`"):
        verify({}, [tomb(OLD[0])])
    with pytest.raises(Q.QueueError, match="does not exist"):
        verify({"export": {"path": str(tmp_path / "gone.gz"), "sha256": "0" * 64}}, [tomb(OLD[0])])
    bad = declared(path, len(OLD + NEW))
    bad["export"]["sha256"] = "0" * 64
    with pytest.raises(Q.QueueError, match="sha256"):
        verify(bad, [tomb(OLD[0])])
    short = write_export(tmp_path, OLD[:-1])
    with pytest.raises(Q.QueueError, match="lacks 1 of"):
        verify(declared(short, len(OLD) - 1), [tomb(r) for r in OLD])
    novec = write_export(tmp_path, OLD, vector=False)
    with pytest.raises(Q.QueueError, match="id/payload/vector"):
        verify(declared(novec, len(OLD)), [tomb(OLD[0])])


def _doctor(victim, fn):
    def mutate(line):
        if line["id"] == victim:
            fn(line)
    return mutate


@pytest.mark.parametrize("name,fn,why", [
    ("dense missing", lambda l: l["vector"].pop("dense"), "dense vector 'dense' missing"),
    ("dense wrong size", lambda l: l["vector"].update(dense=[0.1, 0.2, 0.3]), "3 dims"),
    ("sparse missing", lambda l: l["vector"].pop("bm25"), "sparse vector 'bm25' missing"),
    ("sparse values short", lambda l: l["vector"].update(bm25={"indices": [1, 2], "values": [1.0]}),
     "2 indices vs 1 values"),
    ("sparse no arrays", lambda l: l["vector"].update(bm25={"indices": [1]}), "indices/values"),
    ("payload empty", lambda l: l.update(payload={}), "payload empty"),
    ("payload other url", lambda l: l["payload"].update(
        url="https://www.legifrance.gouv.fr/loda/article_lc/LEGIARTI000000000001"),
     "same-document key differs"),
])
def test_an_incomplete_export_row_refuses_the_pack_and_names_the_id(tmp_path, name, fn, why):
    victim = OLD[1]["id"]
    path = write_export(tmp_path, OLD, mutate=_doctor(victim, fn))
    with pytest.raises(Q.QueueError, match="not a complete restore row") as exc:
        verify(declared(path, len(OLD)), [tomb(r) for r in OLD])
    assert victim in str(exc.value) and why in str(exc.value)
    assert OLD[0]["id"] not in str(exc.value)


def test_the_export_url_is_compared_by_the_drains_same_document_key(tmp_path):
    # http vs https and a bare host fold to the same key the drain uses: not a refusal
    def respell(line):
        line["payload"]["url"] = line["payload"]["url"].replace("https://www.", "http://")
    path = write_export(tmp_path, OLD, mutate=respell)
    assert verify(declared(path, len(OLD)), [tomb(r) for r in OLD])["rows"] == len(OLD)


def test_dry_check_end_to_end_on_a_pack_directory(tmp_path, twin, monkeypatch):
    monkeypatch.setattr(Q, "fetch_collection_vector_config", lambda url, c: DENSE)
    monkeypatch.setattr(Q, "fetch_collection_sparse_names", lambda url, c: SPARSE)
    export = write_export(tmp_path, OLD)
    tf = tmp_path / "tombstones.jsonl"
    tf.write_text("".join(json.dumps(tomb(r)) + "\n" for r in OLD))
    pts = tmp_path / "points.jsonl"
    pts.write_text("")
    manifest = {"kind": "content_pack", "pack_id": "sg-test", "collection": "aw_e8",
                "pack_points": str(pts), "rows": 0, "tombstones": str(tf),
                "tombstones_count": len(OLD), **declared(export, len(OLD))}
    check = Q.content_pack_dry_check(manifest, "http://twin")
    assert check["points"] == 0
    assert check["tombstones"]["by_reason"] == {SG: len(OLD)}
    assert check["tombstones"]["export"]["rows"] == len(OLD)
    assert "ids" not in check["tombstones"]["export"]
    assert [t["id"] for t in check["_tombstones_data"]] == [r["id"] for r in OLD]
    # a pack with a point is refused before any read
    pts.write_text(json.dumps({"id": "p", "payload": {"url": "https://x"}, "vector": {}}) + "\n")
    manifest["rows"] = 1
    with pytest.raises(Q.QueueError, match="carries no points"):
        Q.content_pack_dry_check(manifest, "http://twin")


# ------------------------------------------- public side: not a takedown (pins)
def test_the_reason_is_in_both_vocabularies_but_never_in_the_takedown_set():
    assert SG in Q.TOMBSTONE_REASONS
    assert SG in pack.TOMBSTONE_REASONS
    # the queue also knows the two task-767 reasons the public vocabulary does not (ADR amendment: Tim)
    assert set(pack.TOMBSTONE_REASONS) <= set(Q.TOMBSTONE_REASONS)
    assert SG not in pack.TAKEDOWN_REASONS
    assert pack.TAKEDOWN_REASONS == {"takedown"}


def test_a_takedown_list_built_from_a_pack_never_holds_it_and_apply_deletes_it(tmp_path):
    pytest.importorskip("qdrant_client")
    from cytir.distribute import apply
    from cytir.core.qdrant import qdrant_client
    a, b = OLD[0]["id"], OLD[1]["id"]
    tombs = [pack.Tombstone(id=a, reason=SG, at="2026-10-05T00:00:00Z"),
             pack.Tombstone(id=b, reason="takedown", at="2026-10-05T00:00:00Z")]
    path = tmp_path / "tombstones.jsonl.gz"
    n, sha = pack.write_tombstones(tombs, str(path))
    got = list(pack.read_tombstones(str(path)))
    assert [t.reason for t in got] == [SG, "takedown"]       # reason survives the round trip
    takedown_ids = [t.id for t in got if t.reason in pack.TAKEDOWN_REASONS]
    assert takedown_ids == [b]                                 # the takedown list excludes it
    # an update pack applies it as an ordinary forced delete
    from qdrant_client.http import models as qm
    client = qdrant_client(":memory:")
    client.create_collection("c", vectors_config={"dense": qm.VectorParams(size=2, distance=qm.Distance.COSINE)})
    client.upsert("c", points=[qm.PointStruct(id=i, vector={"dense": [0.1, 0.2]}, payload={})
                               for i in (a, b)])
    manifest = {"tombstones": {"path": "tombstones.jsonl.gz", "count": n, "sha256": sha}}
    assert apply.apply_tombstones(client, "c", manifest, tmp_path) == 2
    assert client.count("c", exact=True).count == 0


def _pack_dir(tmp_path, reasons):
    tombs = [pack.Tombstone(id=OLD[0]["id"], reason=SG, at="2026-10-05T00:00:00Z")]
    path = tmp_path / "tombstones.jsonl.gz"
    n, sha = pack.write_tombstones(tombs, str(path))
    if reasons:   # append raw lines with a doctored reason (the writer refuses them)
        with gzip.open(path, "at") as fh:
            for rid, reason in reasons:
                fh.write(json.dumps({"id": rid, "reason": reason, "at": ""}) + "\n")
        n, sha = n + len(reasons), pack.sha256_file(str(path))
    return {"tombstones": {"path": "tombstones.jsonl.gz", "count": n, "sha256": sha}}


def test_apply_reads_every_tombstone_reason_before_any_write(tmp_path):
    from cytir.distribute import apply
    manifest = _pack_dir(tmp_path, [(OLD[1]["id"], "superseded_generation"),
                                    (OLD[2]["id"], "because_i_say_so")])
    with pytest.raises(apply.ApplyRefusal) as exc:
        apply.verify_tombstones(manifest, tmp_path)
    assert "tombstone 3" in str(exc.value) and "because_i_say_so" in str(exc.value)
    assert "\n" not in str(exc.value)                      # one clear line
    ok = _pack_dir(tmp_path, [(OLD[1]["id"], SG)])
    assert apply.verify_tombstones(ok, tmp_path) == 2


def test_apply_main_refuses_an_unknown_reason_with_no_write(tmp_path, monkeypatch):
    pytest.importorskip("qdrant_client")
    from cytir.distribute import apply

    class Boom(Exception):
        pass

    writes = []
    monkeypatch.setattr(apply, "upsert_shards", lambda *a, **k: writes.append("upsert") or 0)
    monkeypatch.setattr(apply, "verify_dump_signature", lambda *a, **k: None)
    monkeypatch.setattr(apply, "verify_shard_hashes", lambda *a, **k: None)
    monkeypatch.setattr(apply, "verify_idf_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(pack, "validate_manifest", lambda m: m)

    class Client:
        def collection_exists(self, c): return True
        def count(self, *a, **k): raise Boom("counted: the apply went on past the refusal")
        def close(self): pass

    import cytir.core.qdrant as cq
    monkeypatch.setattr(cq, "qdrant_client", lambda *a, **k: Client())
    manifest = dict(_pack_dir(tmp_path, [(OLD[2]["id"], "because_i_say_so")]),
                    schema=pack.DELTA_PACK_SCHEMA, base_manifest_sha256="b", sequence=1)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"applied_manifest_sha256": "b", "sequence": 0}))
    with pytest.raises(apply.ApplyRefusal, match="because_i_say_so"):
        apply.main(["--manifest", str(tmp_path), "--qdrant-url", "http://x", "--collection", "c",
                    "--allow-unsigned", "--state-file", str(state)])
    assert writes == []


def test_the_dump_format_spec_names_the_reason_and_keeps_it_out_of_takedown():
    spec = (REPO / "specs" / "dump-format.md").read_text()
    assert "superseded | aged_out | takedown | superseded_generation" in spec
    assert "It is not a takedown" in spec
    assert "before any write" in spec and "verify_tombstones" in spec

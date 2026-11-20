#!/usr/bin/env python3
"""`restore_rollback` (ADR-123, ruling 29 T12): the PURE, export-backed, operator-only tombstone
reason that undoes a LANDED restore pack, and the only export-backed reason that may delete a CARD.

Rows are REAL ids/urls of the loda fixture (`fixtures/loda-old-generation-rows.json`: old and new
generation rows and one real entity card, read from the build-box twin 2026-10-05). No text. The
fixture rows carry no `content_digest`; the tests add one where the check needs an identity.
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

from cytir.distribute import pack  # noqa: E402
from cytir.index import point_delete as PD  # noqa: E402
from cytir.index import promote_queue as Q  # noqa: E402

DOC = json.loads((HERE / "fixtures" / "loda-old-generation-rows.json").read_text())
ROWS = DOC["rows"]
CARD = next(r for r in ROWS if r["role"] == "card")
PAGES = [r for r in ROWS if r["role"] == "old"]
RR = Q.RESTORE_ROLLBACK
OTHERS = (Q.SUPERSEDED_GENERATION, Q.CANONICAL_DUPLICATE, Q.BINARY_CONTENT)
DENSE = {"dense": {"size": 2, "distance": "Cosine"}}
SPARSE = ["bm25"]


def tomb(row, reason=RR):
    return {"id": row["id"], "url": row["url"], "reason": reason}


def full_row(r, i):
    """The fixture row with a content_digest, so that its identity is not its url alone."""
    return {"id": r["id"], "url": r["url"], "payload": {**r["payload"], "content_digest": f"d{i}"}}


FULL_PAGES = [full_row(r, i) for i, r in enumerate(PAGES)]
KEYED_CARD = {"id": CARD["id"], "url": CARD["url"],
              "payload": {**CARD["payload"], "register_id": "CHE-1", "lei": "L1",
                          "content_digest": "dc"}}
TWIN_ROWS = [KEYED_CARD] + FULL_PAGES


class FakeTwin:
    def __init__(self, rows):
        self.rows = {r["id"]: dict(r["payload"]) for r in rows}

    def api(self, url, path, body=None, method=None):
        if path.endswith("/points/scroll"):
            vals = set(body["filter"]["must"][0]["match"]["any"])
            return {"points": [{"id": i, "payload": dict(p)} for i, p in self.rows.items()
                               if p.get("url") in vals], "next_page_offset": None}
        keep = body["with_payload"]
        return [{"id": i, "payload": (dict(self.rows[i]) if keep is True else
                                      {k: v for k, v in self.rows[i].items() if keep and k in keep})}
                for i in body["ids"] if i in self.rows]


@pytest.fixture
def twin(monkeypatch):
    t = FakeTwin(TWIN_ROWS)
    monkeypatch.setattr(Q.PSP, "_api", t.api)
    monkeypatch.setattr(PD, "_api", t.api)
    return t


def write_export(tmp_path, rows):
    path = tmp_path / "export.jsonl.gz"
    with gzip.open(path, "wt") as fh:
        for r in rows:
            line = {"id": r["id"], "payload": dict(r["payload"]),
                    "vector": {"dense": [0.1, 0.2], "bm25": {"indices": [1], "values": [1.0]}}}
            fh.write(json.dumps(line) + "\n")
    return path


def manifest_for(path, rows, **extra):
    return {"export": {"path": str(path), "rows": rows,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, **extra}


def verify(manifest, tombs):
    return Q.verify_removed_rows_export(manifest, tombs, dense_config=DENSE, sparse_names=SPARSE)


def validate(tombs, identities):
    return Q.validate_tombstones(tombs, pack_urls=set(), target_url="http://twin",
                                 collection="aw_e8", export_ids={t["id"] for t in tombs},
                                 export_identities=identities)


def make_queue(tmp_path, rows, *, status="LANDED", name="restore-pack-1"):
    """A queue with one restore pack `name` (manifest, points, receipt) and the directory the
    rollback pack lives in. Returns (queue dir, the rollback pack's directory)."""
    queue = tmp_path / "queue"
    pdir = queue / name
    pdir.mkdir(parents=True)
    pts = pdir / "points.jsonl"
    pts.write_text("".join(json.dumps({"id": r["id"], "payload": r["payload"], "vector": {}}) + "\n"
                           for r in rows))
    (pdir / "manifest.json").write_text(json.dumps({
        "pack_id": name, "pack_points": str(pts),
        "pack_points_sha256": hashlib.sha256(pts.read_bytes()).hexdigest()}))
    if status is not None:
        (pdir / "receipt.json").write_text(json.dumps({"status": status}))
    rb = queue / "rollback-1"
    rb.mkdir()
    (rb / "tombstones.jsonl").write_text("")
    return queue, rb


def rollback_manifest(rb):
    return {"pack_id": "rollback-1", "tombstones": str(rb / "tombstones.jsonl"),
            "rollback_of": "restore-pack-1"}


def evidence(tmp_path, rows, **kw):
    """(export identities, rollback manifest) for a restore pack that wrote `rows`."""
    queue, rb = make_queue(tmp_path, rows, **kw)
    path = write_export(tmp_path, rows)
    out = verify(manifest_for(path, len(rows)), [tomb(r) for r in rows])
    return out["identities"], rollback_manifest(rb)


# ----------------------------------------------------------------- accepted, with evidence
def test_a_rollback_of_a_card_is_accepted_with_its_export_and_its_restore_pack(tmp_path, twin):
    ident, man = evidence(tmp_path, [KEYED_CARD])
    assert Q.check_rolled_back_pack(man, [tomb(KEYED_CARD)], ident) == []
    report = validate([tomb(KEYED_CARD)], ident)
    assert report["by_reason"] == {RR: 1}
    assert [t["id"] for t in report["_tombstones"]] == [KEYED_CARD["id"]]


def test_a_rollback_of_page_rows_is_accepted_too(tmp_path, twin):
    ident, man = evidence(tmp_path, FULL_PAGES)
    tombs = [tomb(r) for r in FULL_PAGES]
    assert Q.check_rolled_back_pack(man, tombs, ident) == []
    assert validate(tombs, ident)["by_reason"] == {RR: len(FULL_PAGES)}


def test_identities_appear_only_for_a_rollback_pack(tmp_path):
    path = write_export(tmp_path, PAGES)
    out = verify(manifest_for(path, len(PAGES)), [tomb(r, Q.SUPERSEDED_GENERATION) for r in PAGES])
    assert "identities" not in out


def test_the_dry_check_returns_the_export_without_identities(tmp_path, twin, monkeypatch):
    monkeypatch.setattr(Q, "fetch_collection_vector_config", lambda url, c: DENSE)
    monkeypatch.setattr(Q, "fetch_collection_sparse_names", lambda url, c: SPARSE)
    queue, rb = make_queue(tmp_path, [KEYED_CARD])
    export = write_export(tmp_path, [KEYED_CARD])
    tf = rb / "tombstones.jsonl"
    tf.write_text(json.dumps(tomb(KEYED_CARD)) + "\n")
    pts = rb / "points.jsonl"
    pts.write_text("")
    manifest = {"kind": "content_pack", "pack_id": "rollback-1", "collection": "aw_e8",
                "pack_points": str(pts), "rows": 0, "tombstones": str(tf), "tombstones_count": 1,
                "rollback_of": "restore-pack-1", **manifest_for(export, 1)}
    check = Q.content_pack_dry_check(manifest, "http://twin")
    assert check["tombstones"]["by_reason"] == {RR: 1}
    exported = check["tombstones"]["export"]
    assert "identities" not in exported and "ids" not in exported
    # the same pack naming a pack that is not there is refused by the dry check itself
    manifest["rollback_of"] = "no-such-pack-anywhere"
    with pytest.raises(Q.QueueError, match="no pack directory"):
        Q.content_pack_dry_check(manifest, "http://twin")


# ------------------------------------------------------------------------- refused without
def test_refused_without_the_export_identities(twin):
    with pytest.raises(Q.QueueError, match="export_identities"):
        validate([tomb(KEYED_CARD)], None)


def test_refused_when_the_export_does_not_hold_the_id(tmp_path):
    path = write_export(tmp_path, PAGES)             # the card is not in it
    with pytest.raises(Q.QueueError, match="lacks 1 of 1 tombstoned id"):
        verify(manifest_for(path, len(PAGES)), [tomb(KEYED_CARD)])


def test_refused_when_the_export_sha_is_wrong(tmp_path):
    path = write_export(tmp_path, [KEYED_CARD])
    bad = manifest_for(path, 1)
    bad["export"]["sha256"] = "0" * 64
    with pytest.raises(Q.QueueError, match="sha256"):
        verify(bad, [tomb(KEYED_CARD)])
    with pytest.raises(Q.QueueError, match="declares its `export`"):
        verify({}, [tomb(KEYED_CARD)])


def test_refused_when_the_tombstone_names_another_document(tmp_path, twin):
    ident, _ = evidence(tmp_path, [KEYED_CARD])
    wrong = dict(tomb(KEYED_CARD), url="https://example.org/other")
    with pytest.raises(Q.QueueError, match="not the export row's document"):
        validate([wrong], ident)


def test_the_pack_must_be_pure_and_alone_and_name_what_it_undoes():
    t = [tomb(CARD)]
    with pytest.raises(Q.QueueError, match="rollback_of"):
        Q.check_superseded_generation_pack({}, [], t)
    Q.check_superseded_generation_pack({"rollback_of": "card-restore-slice-c-826"}, [], t)
    with pytest.raises(Q.QueueError, match="carries no points"):
        Q.check_superseded_generation_pack({"rollback_of": "x"}, [{"id": "p"}], t)
    with pytest.raises(Q.QueueError, match="holds only that reason"):
        Q.check_superseded_generation_pack({"rollback_of": "x"}, [],
                                           t + [tomb(PAGES[0], "takedown")])


# ------------------------------------- M1: rollback_of is resolved, not just non-empty
def test_an_unknown_rollback_of_is_refused(tmp_path):
    ident, man = evidence(tmp_path, [KEYED_CARD])
    man["rollback_of"] = "no-such-pack-anywhere"
    assert "no pack directory" in Q.check_rolled_back_pack(man, [tomb(KEYED_CARD)], ident)[0]


@pytest.mark.parametrize("name", ["../restore-pack-1", "a/b", "", ".", "rollback-1"])
def test_a_rollback_of_that_is_not_a_plain_other_pack_id_is_refused(tmp_path, name):
    ident, man = evidence(tmp_path, [KEYED_CARD])
    man["rollback_of"] = name
    assert Q.check_rolled_back_pack(man, [tomb(KEYED_CARD)], ident)


@pytest.mark.parametrize("status", ["REFUSED", "TWIN-ONLY", None])
def test_a_restore_pack_that_did_not_land_is_not_rolled_back(tmp_path, status):
    ident, man = evidence(tmp_path, [KEYED_CARD], status=status)
    assert "not LANDED" in Q.check_rolled_back_pack(man, [tomb(KEYED_CARD)], ident)[0]


def test_an_id_outside_the_restore_pack_is_refused(tmp_path):
    ident, man = evidence(tmp_path, FULL_PAGES)
    queue = pathlib.Path(man["tombstones"]).parent.parent
    pts = queue / "restore-pack-1" / "points.jsonl"
    pts.write_text("".join(json.dumps({"id": r["id"], "payload": r["payload"]}) + "\n"
                           for r in FULL_PAGES[1:]))
    mf = queue / "restore-pack-1" / "manifest.json"
    pm = json.loads(mf.read_text())
    pm["pack_points_sha256"] = hashlib.sha256(pts.read_bytes()).hexdigest()
    mf.write_text(json.dumps(pm))
    problems = Q.check_rolled_back_pack(man, [tomb(FULL_PAGES[0])], ident)
    assert "not a row of 'restore-pack-1'" in problems[0]


def test_an_export_taken_after_a_rewrite_is_not_the_restore_packs_row(tmp_path):
    ident, man = evidence(tmp_path, [KEYED_CARD])
    ident = {k: v[:-1] + ("a-newer-digest",) for k, v in ident.items()}    # an export of today's row
    problems = Q.check_rolled_back_pack(man, [tomb(KEYED_CARD)], ident)
    assert "is not 'restore-pack-1''s row" in problems[0]


def test_a_points_file_that_does_not_hash_is_refused(tmp_path):
    ident, man = evidence(tmp_path, [KEYED_CARD])
    queue = pathlib.Path(man["tombstones"]).parent.parent
    (queue / "restore-pack-1" / "points.jsonl").write_text("{}\n")
    assert "cannot be read" in Q.check_rolled_back_pack(man, [tomb(KEYED_CARD)], ident)[0]


# --------------------------------- M2: every identity key, and rows that prove nothing
@pytest.mark.parametrize("key", Q.ROLLBACK_IDENTITY_KEYS)
def test_the_stored_row_must_match_the_export_on_every_identity_key(tmp_path, twin, key):
    ident, _ = evidence(tmp_path, [KEYED_CARD])
    assert KEYED_CARD["payload"].get(key), f"the fixture row must carry {key}"
    twin.rows[KEYED_CARD["id"]][key] = "changed-since-the-restore"
    with pytest.raises(Q.QueueError, match="not the export's row any more"):
        validate([tomb(KEYED_CARD)], ident)


def test_the_identity_key_set_is_the_documented_one():
    assert Q.ROLLBACK_IDENTITY_KEYS == ("source_type", "entity_kind", "qid", "register_id", "lei",
                                        "content_digest")


def test_a_url_alone_identity_is_refused_on_either_side(tmp_path, twin):
    bare = PAGES[0]                                        # no content_digest, no entity key
    path = write_export(tmp_path, [bare])
    out = verify(manifest_for(path, 1), [tomb(bare)])
    twin.rows[bare["id"]] = dict(bare["payload"])
    with pytest.raises(Q.QueueError, match="proves nothing"):
        validate([tomb(bare)], out["identities"])
    # a stored row that lost its digest since the export is refused as well
    ident, _ = evidence(tmp_path, [FULL_PAGES[1]])
    twin.rows[FULL_PAGES[1]["id"]].pop("content_digest")
    with pytest.raises(Q.QueueError, match="proves nothing"):
        validate([tomb(FULL_PAGES[1])], ident)


def test_a_card_by_source_type_only_is_not_a_page(tmp_path, twin):
    row = {"id": FULL_PAGES[0]["id"], "url": FULL_PAGES[0]["url"],
           "payload": {**FULL_PAGES[0]["payload"], "source_type": "entity_card"}}
    twin.rows[row["id"]] = dict(row["payload"])
    ident, _ = evidence(tmp_path, [row])
    twin.rows[row["id"]]["source_type"] = "official"
    with pytest.raises(Q.QueueError, match="not the export's row any more"):
        validate([tomb(row)], ident)


# ------------------------------------------------- a card is deleted ONLY under this reason
@pytest.mark.parametrize("reason", OTHERS)
def test_the_other_export_backed_reasons_refuse_a_card(twin, reason):
    with pytest.raises(Q.QueueError, match="entity card"):
        Q.validate_tombstones([tomb(KEYED_CARD, reason)], pack_urls=set(), target_url="http://twin",
                              collection="aw_e8", export_ids={KEYED_CARD["id"]})


def test_the_composer_of_task_767_cannot_compose_it():
    from cytir.ops import index_text_tombstones as ITT
    with pytest.raises(SystemExit, match="REFUSED"):
        ITT.classify(RR, [], "http://twin", "aw_e8")


# ------------------------------------------------------------ never in the takedown export
def test_never_in_the_takedown_set_and_not_a_public_reason():
    assert RR in Q.TOMBSTONE_REASONS and RR in Q.EXPORT_BACKED_REASONS
    assert RR not in pack.TAKEDOWN_REASONS and pack.TAKEDOWN_REASONS == {"takedown"}
    assert RR not in pack.TOMBSTONE_REASONS          # operator-only: it is never a public reason
    with pytest.raises(ValueError):
        pack.Tombstone(id="a", reason=RR, at="2026-10-10T00:00:00Z")


def test_a_takedown_export_built_from_a_pack_never_holds_it(tmp_path):
    tombs = [pack.Tombstone(id="a", reason="takedown", at="2026-10-10T00:00:00Z"),
             pack.Tombstone(id="b", reason="superseded_generation", at="2026-10-10T00:00:00Z")]
    path = tmp_path / "tombstones.jsonl.gz"
    pack.write_tombstones(tombs, str(path))
    got = list(pack.read_tombstones(str(path)))
    assert [t.id for t in got if t.reason in pack.TAKEDOWN_REASONS] == ["a"]
    assert RR not in {t.reason for t in got}


def test_the_spec_names_the_vocabulary():
    spec = (REPO / "specs" / "dump-format.md").read_text()
    assert "restore_rollback" in spec and "canonical_duplicate" in spec and "binary_content" in spec
    assert "ADR-123" in spec

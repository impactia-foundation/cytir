"""Guards G1 and G2: twin and prod are compared by CONTENT, not by id (task 833).

THE DEFECT. A Tier-B point id is `uuid5(canonical_url + "#" + content_digest)`. The finalize of a
`CARRIED-BY-FLIP` pack compared the pack's digest with prod's digest for the ids present on prod:
"the same id" already IS "the same digest", so the check could not fail. 6,247 rows that the twin
had rewritten in place (consent, licence, title, url spelling, provenance, the dense vector) stayed
stale on prod `aw_e9`, every check green.

THE ROWS. `fixtures/mirror_guards/rows.v1.json`: six real rows read by id from the twin and from
its 2026-09-22 snapshot copy. `prod_only_old` = three in-place rewrites that prod still holds in
their snapshot form; `equal` = three rows that prod holds exactly as the twin. Prod's rows are
built from them by the one projection the writer uses (see the fixture's `about`).

What each test pins:
  (a) the OLD digest signature passes the three stale rows (the tautology, reproduced);
  (b) G1: the finalize refuses them (first ids + keys in the receipt) and finalizes the equal ones,
      with the same prod calls as before (one presence lookup, one retrieve, now with `dense`);
  (c) G2: a prod leg that leaves the in-place rows stale FAILS before DONE, and the per-pack list of
      in-place ids is written; the same pack with a correct write passes, twin == prod by content;
  (d) a replay (`AW_REPLAY=1`) sends no query to prod: G1 and G2 are refused before any request.
Plus the comparator's own edges on the same real rows (dense alone, open keys, float spelling,
the BM25 epoch stamp) and the in-place-first sample.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import urllib.parse
import urllib.request

import numpy as np
import pytest

from cytir.core.canon import canonical_url
from cytir.core import ids as IDS
from cytir.index import mirror_compare as MC
from cytir.index import payload_profile as PP
from cytir.index import prod_guard as PG
from cytir.index import promote_queue as PQ
from cytir.index import sparse_rule as SR

HERE = pathlib.Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "mirror_guards" / "rows.v1.json"
PROD = "http://127.0.0.1:%d" % PG.PROD_PORT
TWIN = "http://twin.invalid:16457"     # never resolved: urlopen is the FakeQdrant below


@pytest.fixture(scope="module")
def rows():
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def e9():
    return PP.Projector.for_collection("aw_e9")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("AW_PROD_READ_BUDGET", "unlimited")
    monkeypatch.delenv("AW_PROD_URL", raising=False)
    monkeypatch.delenv("AW_REPLAY", raising=False)
    monkeypatch.setattr(PG, "PROD_BATCH_SLEEP_S", 0)
    PG.reset()


def _prod_row(entry, e9, source):
    """What prod `aw_e9` stores for this id: the source row in aw_e9 form, no epoch stamp (the
    snapshot build predates it), dense as stored."""
    src = entry[source]
    return {"payload": e9.project(src["payload"]), "vector": copy.deepcopy(src["vector"])}


def _pre833_signature(payload: dict) -> str | None:
    """The finalize's signature before task 833, verbatim in effect: the digest when the row has
    one, else a hash of the payload without its volatile timestamps."""
    digest = payload.get("content_digest")
    if digest:
        return f"content_digest:{digest}"
    rest = {k: v for k, v in payload.items()
            if k not in {"last_seen", "fetched_at", "consent_checked_at", "ingested_at"}}
    if not any(rest.values()):
        return None
    return "fallback:" + hashlib.sha256(json.dumps(rest, sort_keys=True, default=str).encode()).hexdigest()


# ------------------------------------------------------------------------- a fake Qdrant
class FakeQdrant:
    """`urllib.request.urlopen` stand-in for two endpoints. It stores what a real Qdrant stores
    for these collections: the dense vector unit-normalised (Cosine) and in float16 (prod's
    datatype), the payload through a JSON round trip. Every request is logged."""

    def __init__(self):
        self.store: dict[tuple[str, str], dict[str, dict]] = {}
        self.aliases = {PROD: [{"alias_name": "aw_prod", "collection_name": "aw_e9"}], TWIN: []}
        self.log: list[tuple[str, str, str, dict]] = []

    @staticmethod
    def _stored_vector(vec: dict) -> dict:
        out = {}
        for name, v in (vec or {}).items():
            if name == "dense":
                a = np.asarray(v, dtype=np.float64)
                n = np.linalg.norm(a)
                a = a / n if n else a
                out[name] = [float(x) for x in a.astype(np.float16)]
            else:
                out[name] = v
        return out

    def put(self, base, collection, rows: dict):
        coll = self.store.setdefault((base, collection), {})
        for pid, r in rows.items():
            coll[pid] = {"payload": json.loads(json.dumps(r["payload"])),
                         "vector": self._stored_vector(r.get("vector"))}

    def prod_posts(self):
        return [e for e in self.log if e[0] == PROD and e[1] == "POST"]

    def __call__(self, req, timeout=None):
        if isinstance(req, str):
            full, method, body = req, "GET", None
        else:
            full, method = req.full_url, req.get_method()
            body = json.loads(req.data) if req.data else None
        p = urllib.parse.urlparse(full)
        base, path = f"{p.scheme}://{p.netloc}", p.path
        self.log.append((base, method, path, body or {}))
        if path == "/aliases":
            return _Resp({"status": "ok", "result": {"aliases": self.aliases.get(base, [])}})
        parts = path.strip("/").split("/")
        coll = self.store.setdefault((base, parts[1]), {})
        if method == "PUT" and parts[2:] == ["points"]:
            self.put(base, parts[1], {pt["id"]: pt for pt in body["points"]})
            return _Resp({"status": "ok", "result": {"status": "completed"}})
        if method == "POST" and parts[2:] == ["points"]:
            want_vec = body.get("with_vector")
            out = []
            for pid in body["ids"]:
                if pid not in coll:
                    continue
                r = {"id": pid}
                if body.get("with_payload"):
                    r["payload"] = copy.deepcopy(coll[pid]["payload"])
                if want_vec:
                    names = list(coll[pid]["vector"]) if want_vec is True else want_vec
                    r["vector"] = {n: coll[pid]["vector"][n] for n in names if n in coll[pid]["vector"]}
                out.append(r)
            return _Resp({"status": "ok", "result": out})
        if method == "POST" and parts[2:] == ["points", "delete"]:
            for pid in body["points"]:
                coll.pop(pid, None)
            return _Resp({"status": "ok", "result": {"status": "completed"}})
        raise AssertionError(f"unexpected request {method} {full}")


class _Resp:
    def __init__(self, doc):
        self._b = json.dumps(doc).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._b


@pytest.fixture()
def qd(monkeypatch):
    fake = FakeQdrant()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


# ------------------------------------------------------------------------- pack helpers
def _points(entries):
    """The pack rows: the twin's rows (the pack row of these ids, checked at capture)."""
    return [{"id": e["id"], "payload": copy.deepcopy(e["twin"]["payload"]),
             "vector": copy.deepcopy(e["twin"]["vector"])} for e in entries]


def _carried_pack(tmp_path, pack_id, points):
    queue = tmp_path / "queue"
    pack_dir = queue / pack_id
    pack_dir.mkdir(parents=True)
    pts = tmp_path / f"{pack_id}.points.jsonl"
    pts.write_text("".join(json.dumps(p) + "\n" for p in points))
    manifest = {"schema": PQ.MANIFEST_SCHEMA, "kind": "content_pack", "pack_id": pack_id,
                "family": "general_web", "collection": "aw_e8", "pack_points": str(pts),
                "rows": len(points)}
    (pack_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    sha = hashlib.sha256((pack_dir / "manifest.json").read_bytes()).hexdigest()
    (pack_dir / "READY").write_text(sha + "\n")
    art = tmp_path / "artifacts"
    PQ.write_content_pack_twin_state(art, pack_id, stage="twin-write-complete", manifest_sha=sha,
                                     pack_ids=[p["id"] for p in points], tomb_ids=[])
    PQ.mark_carried_by_flip(pack_dir, reason="test", task="833")
    return pack_dir, art


def _flip_world(qd, rows, e9, entries_old=(), entries_equal=()):
    """After the flip: the twin holds the NEW rows of every id; prod holds the snapshot row for the
    stale ids and the twin's row for the equal ids."""
    qd.put(TWIN, "aw_e8", {e["id"]: e["twin"] for e in (*entries_old, *entries_equal)})
    qd.put(PROD, "aw_e9", {e["id"]: _prod_row(e, e9, "copy") for e in entries_old})
    qd.put(PROD, "aw_e9", {e["id"]: _prod_row(e, e9, "twin") for e in entries_equal})


# ===================================================================== (a) the tautology
def test_a_the_old_digest_signature_passes_every_stale_row(rows, e9):
    cmp = MC.Comparator("aw_e9", allow_unstamped_rule_epochs=True)
    for e in rows["prod_only_old"]:
        twin = e["twin"]["payload"]
        # Tier B: the id is made from the canonical url and the digest
        assert IDS.mint_point_id(canonical_url(twin["url"]), twin["content_digest"]) == e["id"]
        prod = _prod_row(e, e9, "copy")
        assert _pre833_signature(e9.project(twin)) == _pre833_signature(prod["payload"]), \
            "the old check passes a row that prod holds in its pre-rewrite form"
        assert cmp.diff(e["twin"], prod), "and yet the rows differ"
    keys = {k for e in rows["prod_only_old"]
            for k in cmp.diff(e["twin"], _prod_row(e, e9, "copy"))}
    assert {"consent_state", "consent_basis", "provenance", "license", "dense", "title"} <= keys


# ================================================================== (b) G1: the finalize
def test_b_g1_finalize_refuses_the_stale_rows_and_names_them(tmp_path, qd, rows, e9):
    old, eq = rows["prod_only_old"], rows["equal"]
    _flip_world(qd, rows, e9, old, eq)
    pack_dir, art = _carried_pack(tmp_path, "l1-backfill-batch-00001", _points(old + eq))
    with pytest.raises(PQ.QueueError, match="3 of 6 present row.s. differ by content"):
        PQ.finalize_carried_by_flip(pack_dir, prod_url=PROD, task="833", artifact_root=art,
                                    twin_url=TWIN)
    assert (pack_dir / "CARRIED-BY-FLIP").exists() and not (pack_dir / "DONE").exists()
    receipt = json.loads((pack_dir / "receipt.json").read_text())
    assert receipt["status"] == "CARRIED-BY-FLIP"
    refused = receipt["finalize_refused"]["content"]
    assert refused["checked"] == 6 and refused["equal"] == 3 and refused["mismatched"] == 3
    assert sorted(x["id"] for x in refused["examples"]) == sorted(e["id"] for e in old)
    assert all("consent_state" in x["keys"] for x in refused["examples"])
    assert refused["reference"] == "twin aw_e8"
    # no extra prod call: one presence lookup and one retrieve, as before; the retrieve asks for dense
    posts = qd.prod_posts()
    assert len(posts) == 2
    assert posts[0][3]["with_vector"] is False
    assert posts[1][3]["with_payload"] is True and posts[1][3]["with_vector"] == ["dense"]


def test_b_g1_finalize_passes_the_equal_rows(tmp_path, qd, rows, e9):
    eq = rows["equal"]
    _flip_world(qd, rows, e9, (), eq)
    pack_dir, art = _carried_pack(tmp_path, "news-daily-20260924", _points(eq))
    receipt = PQ.finalize_carried_by_flip(pack_dir, prod_url=PROD, task="833",
                                          artifact_root=art, twin_url=TWIN)
    assert receipt["status"] == "DONE" and receipt["content_checked"] == 3
    assert receipt["content_compare"] == {"compared": "all payload keys + dense",
                                          "reference": "twin aw_e8", "equal": 3}
    assert (pack_dir / "DONE").exists() and len(qd.prod_posts()) == 2


def test_b_g1_without_a_twin_the_pack_rows_are_the_reference(tmp_path, qd, rows, e9):
    old = rows["prod_only_old"]
    _flip_world(qd, rows, e9, old, ())
    pack_dir, art = _carried_pack(tmp_path, "l1-backfill-batch-00001", _points(old))
    with pytest.raises(PQ.QueueError, match="from the pack rows"):
        PQ.finalize_carried_by_flip(pack_dir, prod_url=PROD, task="833", artifact_root=art)


# ================================================================== (c) G2: the prod leg
class _Lock:
    def __init__(self, *a, **k):
        pass

    def acquire(self):
        pass

    def release(self):
        pass


class _RB:
    from cytir.ops.write_lock import LOCKS
    assert_locks_free = staticmethod(lambda task: {})
    acquire = staticmethod(lambda name, task, purpose, *, pid=None: None)
    release = staticmethod(lambda name, task: None)


def _drain_world(monkeypatch, qd, rows, e9, *, skip_inplace_on_prod: bool):
    """Before the pack: the twin and prod hold the snapshot row of the three in-place ids (in each
    one's own form); the three equal ids are new to both. No BM25 rule on either side here (the
    stamp has its own test below and in the sparse-rule tests)."""
    monkeypatch.setattr(PQ, "prod_lock_host", lambda: "lockhost")
    monkeypatch.setattr(PQ, "_load_run_batch", lambda: _RB)
    monkeypatch.setattr(PQ.TFA, "ProductionWriterLock", _Lock)
    monkeypatch.setattr(PQ.LAYOUT, "fingerprint", lambda url, coll: {"points_count": 0})
    monkeypatch.setattr(PQ.UGS, "reverify_generation_export", lambda *a, **k: None)
    monkeypatch.setattr(SR.SparseRule, "for_collection", classmethod(lambda cls, c, **k: None))
    monkeypatch.setattr(SR, "declared_epoch", lambda c, **k: None)
    old = rows["prod_only_old"]
    qd.put(TWIN, "aw_e8", {e["id"]: e["copy"] for e in old})
    qd.put(PROD, "aw_e9", {e["id"]: _prod_row(e, e9, "copy") for e in old})
    if skip_inplace_on_prod:
        real = PQ.PSP.upsert

        def leaves_inplace_rows(url, collection, points, batch, **kw):
            if url == PROD:     # the defect: a prod writer that never rewrites a present id
                held = qd.store.get((PROD, collection), {})
                points = [p for p in points if p["id"] not in held]
            return real(url, collection, points, batch, **kw)
        monkeypatch.setattr(PQ.PSP, "upsert", leaves_inplace_rows)


def _apply(tmp_path, rows):
    pts = _points(rows["prod_only_old"] + rows["equal"])
    return PQ.live_apply_content_pack(
        pack_id="l1-backfill-batch-00001",
        manifest={"kind": "content_pack", "pack_id": "l1-backfill-batch-00001", "collection": "aw_e8"},
        twin_url=TWIN, prod_url=PROD, task="833", batch=50, pause_s=0, artifact_root=tmp_path,
        points=pts, tombstones=[], prod_collection="aw_e9", manifest_sha="cd" * 32)


def _ledger(tmp_path):
    return [json.loads(x) for x in
            (tmp_path / "l1-backfill-batch-00001-ledger.jsonl").read_text().splitlines()]


def test_c_g2_prod_leg_fails_when_the_in_place_rows_stay_stale(tmp_path, monkeypatch, qd, rows, e9):
    _drain_world(monkeypatch, qd, rows, e9, skip_inplace_on_prod=True)
    with pytest.raises(PQ.QueueError) as exc:
        _apply(tmp_path, rows)
    msg = str(exc.value)
    assert "3 of 6 sampled written row(s) are not in the declared 'normalized' form" in msg
    assert "3 of 6 sampled row(s) differ by content between twin 'aw_e8' and 'aw_e9'" in msg
    assert _ledger(tmp_path)[-1]["status"] == "FAILED"
    art = json.loads(PQ.inplace_ids_path(tmp_path, "l1-backfill-batch-00001").read_text())
    old_ids = sorted(e["id"] for e in rows["prod_only_old"])
    assert art["schema"] == PQ.INPLACE_IDS_SCHEMA and art["manifest_sha"] == "cd" * 32
    assert art["twin"]["ids"] == old_ids and art["prod"]["ids"] == old_ids
    assert art["prod"]["collection"] == "aw_e9" and art["twin"]["count"] == 3


def test_c_g2_the_same_pack_passes_when_prod_is_written(tmp_path, monkeypatch, qd, rows, e9):
    _drain_world(monkeypatch, qd, rows, e9, skip_inplace_on_prod=False)
    _apply(tmp_path, rows)
    proof = [x for x in _ledger(tmp_path) if x["status"] == "PROD-VERIFIED"][-1]["proof"]
    form = proof["payload_form"]
    assert form["checked"] == 6 and form["inplace_checked"] == 3 and form["inplace_total"] == 3
    assert form["twin_checked"] == 6 and form["mismatches"] == 0
    assert form["compared"] == "all payload keys + dense"
    assert form["inplace_ids_artifact"].endswith("l1-backfill-batch-00001-inplace-ids.json")
    # one retrieve per side for the readback (<= 500 ids), with the dense vector
    reads = [e for e in qd.log if e[1] == "POST" and e[3].get("with_vector") == ["dense"]]
    assert [(e[0], len(e[3]["ids"])) for e in reads] == [(PROD, 6), (TWIN, 6)]


def test_c_g2_a_retry_keeps_the_first_lookup_of_the_in_place_ids(tmp_path):
    first = PQ.record_inplace_ids(tmp_path, "p", role="prod", collection="aw_e9",
                                  manifest_sha="s1", ids={"b", "a"})
    again = PQ.record_inplace_ids(tmp_path, "p", role="prod", collection="aw_e9",
                                  manifest_sha="s1", ids={"a", "b", "c"})
    assert first == again == ["a", "b"]
    assert PQ.read_inplace_ids(tmp_path, "p", manifest_sha="s1") == (["a", "b"], "prod")
    assert PQ.read_inplace_ids(tmp_path, "p", manifest_sha="s2") == ([], "none")
    PQ.record_inplace_ids(tmp_path, "p", role="twin", collection="aw_e8", manifest_sha="s2", ids={"x"})
    assert PQ.read_inplace_ids(tmp_path, "p", manifest_sha="s2") == (["x"], "twin")


def test_c_g2_the_sample_takes_the_in_place_rows_first():
    pts = [{"id": "%04d" % i} for i in range(1000)]
    chosen, n = PQ._readback_sample(pts, {"0997", "0998", "0999"}, sample=200)
    assert n == 3 and [p["id"] for p in chosen[:3]] == ["0997", "0998", "0999"]
    assert len(chosen) == 203
    # an even spacing alone (the task-675 sample) holds none of them
    assert not {"0997", "0998", "0999"} & {p["id"] for p in pts[::5][:200]}
    chosen, n = PQ._readback_sample(pts, {p["id"] for p in pts[:800]}, sample=200)
    assert n == 500 and len(chosen) == 500 == PQ.VERIFY_FORM_MAX
    assert chosen[-1]["id"] > "0700", "spread over the whole in-place set, not its head"


# ===================================================================== (d) the replay
def test_d_a_replay_sends_no_query_to_prod(tmp_path, monkeypatch, qd, rows, e9):
    old = rows["prod_only_old"]
    _flip_world(qd, rows, e9, old, ())
    pack_dir, art = _carried_pack(tmp_path, "l1-backfill-batch-00001", _points(old))
    monkeypatch.setenv("AW_REPLAY", "1")
    with pytest.raises(PG.ProdReadRefused):
        PQ.finalize_carried_by_flip(pack_dir, prod_url=PROD, task="833", artifact_root=art,
                                    twin_url=TWIN)
    with pytest.raises(PG.ProdReadRefused):
        PQ.verify_written_form(PROD, "aw_e9", _points(old), inplace_ids=[e["id"] for e in old],
                               twin=(TWIN, "aw_e8"))
    assert qd.prod_posts() == []
    # the one prod request is the alias listing, a metadata GET the guard lets through
    assert [(e[1], e[2]) for e in qd.log if e[0] == PROD] == [("GET", "/aliases")]
    assert (pack_dir / "CARRIED-BY-FLIP").exists()


# ====================================================== the comparator on the same real rows
def test_a_re_embedded_vector_alone_is_a_mismatch(rows, e9):
    e = rows["prod_only_old"][0]          # its dense was re-embedded by the rewrite (title changed)
    twin = e["twin"]
    stale_vector = {"payload": e9.project(twin["payload"]), "vector": e["copy"]["vector"]}
    cmp = MC.Comparator("aw_e9", allow_unstamped_rule_epochs=True)
    assert cmp.diff(twin, stale_vector) == ["dense"]
    assert cmp.diff(twin, {"payload": e9.project(twin["payload"]), "vector": twin["vector"]}) == []


def test_the_epoch_stamp_is_required_after_a_write_and_tolerated_after_a_flip(rows, e9):
    e = rows["equal"][0]
    unstamped = _prod_row(e, e9, "twin")
    stamped = {"payload": e9.project(SR.stamp_epoch(e["twin"]["payload"], SR.declared_epoch("aw_e9"))),
               "vector": e["twin"]["vector"]}
    write = MC.Comparator("aw_e9")
    flip = MC.Comparator("aw_e9", allow_unstamped_rule_epochs=True)
    assert write.diff(e["twin"], stamped) == [] and flip.diff(e["twin"], stamped) == []
    assert write.diff(e["twin"], unstamped) == ["rule_epochs"]
    assert flip.diff(e["twin"], unstamped) == []
    other = copy.deepcopy(unstamped)
    other["payload"]["rule_epochs"] = e9.project(
        {"rule_epochs": dict(e["twin"]["payload"]["rule_epochs"], chunk=5)})["rule_epochs"]
    assert flip.diff(e["twin"], other) == ["rule_epochs"], "only the stamp is tolerated"


def test_open_keys_float_spelling_and_closed_keys(rows, e9):
    e = rows["equal"][0]
    cmp = MC.Comparator("aw_e9", allow_unstamped_rule_epochs=True)
    stored = _prod_row(e, e9, "twin")
    # an open key written raw (no code at write time) is the same value as its code today
    raw_open = copy.deepcopy(stored)
    raw_open["payload"]["consent_source"] = e["twin"]["payload"]["consent_source"]
    assert isinstance(stored["payload"]["consent_source"], int)
    assert cmp.diff(e["twin"], raw_open) == []
    # a CLOSED key stored raw in the normalized collection is a defect (task 675), not a spelling
    raw_closed = copy.deepcopy(stored)
    raw_closed["payload"]["provenance"] = e["twin"]["payload"]["provenance"]
    assert cmp.diff(e["twin"], raw_closed) == ["provenance"]
    # a float re-spelled by the store matches; a different value does not
    trust = stored["payload"]["trust_tier"]
    respelled = copy.deepcopy(stored)
    respelled["payload"]["trust_tier"] = trust * (1 + 1e-12)
    assert cmp.diff(e["twin"], respelled) == []
    respelled["payload"]["trust_tier"] = trust + 0.01
    assert cmp.diff(e["twin"], respelled) == ["trust_tier"]
    # a key one side lacks, a missing row, a missing vector: all mismatches
    lost = copy.deepcopy(stored)
    lost["payload"].pop("license")
    assert cmp.diff(e["twin"], lost) == ["license"]
    assert cmp.diff(e["twin"], None) == ["<absent-on-target>"]
    assert cmp.diff(None, stored) == ["<absent-on-reference>"]
    assert cmp.diff(e["twin"], dict(stored, vector={})) == ["dense"]


# ============================================ G1 and G2 on a real Qdrant, prod's collection shape
def test_g1_and_g2_on_a_real_qdrant_in_prods_and_the_twins_shapes(tmp_path, monkeypatch, rows):
    """The same rows through a throwaway Qdrant whose `aw_e8` / `aw_e9` are created from the sealed
    twin and prod serving profiles (float16 dense, Cosine, product quantization, on-disk payload).
    Proves the real store adds no false mismatch (normalised fp16 vectors, payload round trip) and
    that both guards still see the stale rows. Skips without a throwaway Qdrant; FAILS in the CI
    job (`AW_REQUIRE_QDRANT=1`)."""
    import sys
    sys.path.insert(0, str(HERE))
    import throwaway_qdrant  # noqa: E402  (tests/: the shared gate of every Qdrant-backed test)
    url = throwaway_qdrant.require_throwaway()
    throwaway_qdrant.assert_proven(url)
    from cytir.core import serving_profile as SP
    from cytir.core.qdrant import qdrant_client
    from qdrant_client import models as qm
    monkeypatch.setattr(SR.SparseRule, "for_collection", classmethod(lambda cls, c, **k: None))
    monkeypatch.setattr(SR, "declared_epoch", lambda c, **k: None)
    client = qdrant_client(url, timeout=120)
    old, eq = rows["prod_only_old"], rows["equal"]
    old_ids = [e["id"] for e in old]
    snapshot = [{"id": e["id"], **copy.deepcopy(e["copy"])} for e in old]
    pack = _points(old + eq)

    def drop():
        try:
            client.update_collection_aliases(change_aliases_operations=[
                qm.DeleteAliasOperation(delete_alias=qm.DeleteAlias(alias_name="aw_prod"))])
        except Exception:   # noqa: BLE001 -- no alias to drop is fine
            pass
        for name in ("aw_e8", "aw_e9"):
            if client.collection_exists(name):
                client.delete_collection(name)
    drop()
    try:
        SP.create_collection(client, "aw_e8", SP.TWIN_E8, log=lambda *_: None)
        SP.create_collection(client, "aw_e9", SP.PROD_E9, log=lambda *_: None)
        client.update_collection_aliases(change_aliases_operations=[
            qm.CreateAliasOperation(create_alias=qm.CreateAlias(alias_name="aw_prod",
                                                                collection_name="aw_e9"))])
        for coll in ("aw_e8", "aw_e9"):            # before the pack: the snapshot rows
            PQ.PSP.upsert(url, coll, snapshot, 50)
        # G2, the defect: the twin is written, prod keeps its in-place rows
        PQ.PSP.upsert(url, "aw_e8", pack, 50)
        PQ.PSP.upsert(url, "aw_e9", [p for p in pack if p["id"] not in old_ids], 50)
        with pytest.raises(PQ.QueueError, match="3 of 6 sampled row.s. differ by content between twin"):
            PQ.verify_written_form(url, "aw_e9", pack, inplace_ids=old_ids, twin=(url, "aw_e8"))
        # G1, the same state: refused, the three ids named
        pack_dir, art = _carried_pack(tmp_path / "a", "l1-backfill-batch-00001", pack)
        with pytest.raises(PQ.QueueError, match="3 of 6 present row.s. differ by content"):
            PQ.finalize_carried_by_flip(pack_dir, prod_url=url, task="833", artifact_root=art,
                                        twin_url=url)
        named = json.loads((pack_dir / "receipt.json").read_text())["finalize_refused"]["content"]
        assert sorted(x["id"] for x in named["examples"]) == sorted(old_ids)
        # prod written too: no false mismatch from the real store, and both guards pass
        PQ.PSP.upsert(url, "aw_e9", pack, 50)
        form = PQ.verify_written_form(url, "aw_e9", pack, inplace_ids=old_ids, twin=(url, "aw_e8"))
        assert (form["checked"], form["inplace_checked"], form["twin_checked"]) == (6, 3, 6)
        pack_dir, art = _carried_pack(tmp_path / "b", "l1-backfill-batch-00001", pack)
        done = PQ.finalize_carried_by_flip(pack_dir, prod_url=url, task="833", artifact_root=art,
                                           twin_url=url)
        assert done["status"] == "DONE" and done["content_compare"]["equal"] == 6
    finally:
        drop()

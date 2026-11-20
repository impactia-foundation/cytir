# Task 650 (migration S8): moved to tests/ (2026-10-04) -- where the public cut ships and re-runs it.
"""The CI release rehearsal (task 643 §6.2 S8's acceptance line: "the
rehearsal index answers the eval slice like its source — same golds at the
same h@k"; task 650 step 4).

No live Qdrant server and no real twin: a small, fully synthetic fixture
collection (`_build_source_collection`) stands in for "the twin", built
with `qdrant_client.QdrantClient(":memory:")` — a real client, a real
Qdrant query engine, just no network. `cyt dump` exports it (via the same
`cytir/distribute/dump.py` functions the CLI calls, not a re-implementation),
`cyt apply` applies it into a second, independent collection standing in for
"a self-hoster's own Qdrant", and this file then asks the two questions a
release rehearsal has to answer:

  1. Counts match, a signature verifies (when `minisign` is on the runner;
     skipped with a clear reason otherwise — this project's established
     convention for an environment-only gap, see requirements-ci.txt's own
     header), and tombstones from a follow-up delta pack actually delete.
  2. The REHEARSED collection answers a small, fixed "eval slice" (a few
     query vectors with deterministic, known nearest neighbours by
     construction) with the exact same top-k point ids, in the same order,
     as the SOURCE collection — not just "the right count of rows landed".
"""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest
from qdrant_client.http import models as qm

from cytir.core.ids import mint_point_id
from cytir.core.qdrant import qdrant_client
from cytir.distribute import apply, dump, pack

DIMS = 8


def _pid(tag: str) -> str:
    # `cytir.core.ids.mint_point_id` (Tier C: no anchor), not a raw
    # `uuid.uuid5(...)` here -- point-id minting has exactly one legal home
    # (task 646 S3's core ratchet) and a test
    # fixture is not exempt from that any more than production code is.
    return mint_point_id(f"https://example.org/{tag}", "")


def _basis_vector(i: int, n: int = DIMS) -> list[float]:
    """A one-hot vector -- cosine similarity cleanly separates these, so a
    query equal to one point's own vector has an UNAMBIGUOUS, exact nearest
    neighbour. Deterministic by construction, not measured after the fact."""
    v = [0.0] * n
    v[i % n] = 1.0
    return v


#: The fixture: 8 general points (one per basis direction) across two
#: source_types, plus 2 entity_card points (ADR-112 projection exercised
#: too) sharing a basis direction with a general point so a query can prove
#: cards rank alongside everything else, minus their sparse leg and names.
FIXTURE = [
    {"tag": f"page-{i}", "dims_i": i, "source_type": "web_page",
     "license": "CC0-1.0", "sparse": ([i, i + 100], [0.6, 0.4])}
    for i in range(6)
] + [
    {"tag": "card-0", "dims_i": 6, "source_type": "entity_card",
     "license": "unknown", "sparse": None,
     "dense_text": "Firma AG — Beratung. Registered signatories/officers "
                   "(commercial register): Hans Muster.",
     "extra_officer_name": "Hans Muster"},
    {"tag": "card-1", "dims_i": 7, "source_type": "entity_card",
     "license": "unknown", "sparse": None,
     "dense_text": "Beispiel GmbH — Handel. Registered "
                   "signatories/officers (commercial register): Anna Beispiel.",
     "extra_officer_name": "Anna Beispiel"},
]


def _build_source_collection(client):
    client.create_collection(
        "twin_fixture",
        vectors_config={"dense": qm.VectorParams(size=DIMS, distance=qm.Distance.COSINE)},
        sparse_vectors_config={"bm25": qm.SparseVectorParams()})
    points = []
    for row in FIXTURE:
        payload = {
            "url": f"https://example.org/{row['tag']}",
            "title": row["tag"], "lang": "en", "source_type": row["source_type"],
            "license": row["license"], "trust_tier": 0.7,
            "last_seen": "2026-09-24T00:00:00Z", "country": "CH",
        }
        if row["source_type"] == "entity_card":
            payload["dense_text"] = row["dense_text"]
            payload["snippet"] = row["dense_text"][:220]
            payload["sparse_text"] = row["extra_officer_name"]
            payload["facet_terms"] = [row["extra_officer_name"]]
            payload["register_id"] = f"CHE-{row['dims_i']}"
            payload["register_scheme"] = "zefix"
            payload["resolve_confidence"] = 0.9
        vector = {"dense": _basis_vector(row["dims_i"])}
        if row["sparse"]:
            vector["bm25"] = qm.SparseVector(indices=row["sparse"][0],
                                             values=row["sparse"][1])
        points.append(qm.PointStruct(id=_pid(row["tag"]), vector=vector, payload=payload))
    client.upsert("twin_fixture", points=points)
    return points


def _eval_slice_queries():
    """A handful of (query_vector, expected_top1_tag) pairs -- the "gold"
    the rehearsal must reproduce exactly, not approximately."""
    return [(_basis_vector(i), FIXTURE[i]["tag"]) for i in range(len(FIXTURE))]


def _top1_tags(client, collection, queries):
    got = []
    for vec, _expected in queries:
        r = client.query_points(collection, query=vec, using="dense", limit=1)
        got.append(r.points[0].payload["title"] if r.points else None)
    return got


@pytest.fixture()
def minisign_keys(tmp_path):
    """An ephemeral minisign keypair, generated fresh per test run -- never
    committed, never reused. Skips (not fails) the signed half of the
    rehearsal when `minisign` is not on the runner, matching this project's
    existing convention for an environment-only gap (requirements-ci.txt's
    header: heavy/optional tooling skips deterministically with a reason,
    never silently)."""
    if not shutil.which("minisign"):
        pytest.skip("minisign not installed on this runner -- signed-dump "
                    "half of the release rehearsal is skipped; unsigned "
                    "half still runs (see test_release_rehearsal_unsigned)")
    sec = tmp_path / "rehearsal.sec"
    pub = tmp_path / "rehearsal.pub"
    subprocess.run(["minisign", "-G", "-s", str(sec), "-p", str(pub),
                    "-W"], input="\n\n", text=True, capture_output=True, check=True)
    pubkey = pub.read_text().splitlines()[-1].strip()
    return {"sec": str(sec), "pub": pubkey}


def test_release_rehearsal_full_dump_signed(tmp_path, minisign_keys):
    src = qdrant_client(":memory:")
    _build_source_collection(src)
    queries = _eval_slice_queries()
    source_answers = _top1_tags(src, "twin_fixture", queries)

    out_dir = tmp_path / "dump1"
    rows = list(dump.scroll_rows(src, "twin_fixture"))
    shards, total, fps = dump.write_full_shards(iter(rows), out_dir, shard_rows=4)
    manifest = dump.build_manifest(
        dump_id="rehearsal-1", collection="twin_fixture", dims=DIMS,
        distance="cosine", shards=shards, total_points=total,
        any_row_has_sparse=True, fingerprints=fps, profile=None)
    manifest["signature"] = {"algo": "minisign",
                             "sig_file": "manifest.json.minisig"}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    sig_path = subprocess.run(
        ["minisign", "-S", "-s", minisign_keys["sec"], "-m",
         str(out_dir / "manifest.json")],
        capture_output=True, text=True)
    assert sig_path.returncode == 0, sig_path.stderr

    target = qdrant_client(":memory:")
    apply.verify_dump_signature(manifest, out_dir, pubkey=minisign_keys["pub"],
                                allow_unsigned=False)  # raises on failure
    apply.verify_shard_hashes(manifest, out_dir)
    apply.create_collection(target, "twin_fixture", manifest, legacy_layout=True)
    n = apply.upsert_shards(target, "twin_fixture", manifest, out_dir)

    assert total == len(FIXTURE)
    assert n == len(FIXTURE)
    assert target.count("twin_fixture", exact=True).count == len(FIXTURE)

    target_answers = _top1_tags(target, "twin_fixture", queries)
    assert target_answers == source_answers, (
        "rehearsed collection answers the eval slice DIFFERENTLY from its "
        "source -- same golds at the same h@k did not hold")

    card = target.retrieve("twin_fixture", ids=[_pid("card-0")], with_vectors=True)[0]
    assert "bm25" not in (card.vector or {})
    assert "Hans Muster" not in json.dumps(card.payload)


def test_release_rehearsal_unsigned_plus_delta_with_tombstone(tmp_path):
    """The unsigned half always runs (no minisign dependency) and adds the
    delta-pack leg: one id superseded, one tombstoned, one new -- proving
    "the tombstones applied", the other half of task 650 step 4's ask."""
    src = qdrant_client(":memory:")
    _build_source_collection(src)

    base_dir = tmp_path / "base"
    rows = list(dump.scroll_rows(src, "twin_fixture"))
    shards, total, fps = dump.write_full_shards(iter(rows), base_dir, shard_rows=4)
    base_manifest = dump.build_manifest(
        dump_id="rehearsal-base", collection="twin_fixture", dims=DIMS,
        distance="cosine", shards=shards, total_points=total,
        any_row_has_sparse=True, fingerprints=fps, profile=None)
    (base_dir / "manifest.json").write_text(json.dumps(base_manifest, indent=2))

    target = qdrant_client(":memory:")
    apply.create_collection(target, "twin_fixture", base_manifest, legacy_layout=True)
    apply.upsert_shards(target, "twin_fixture", base_manifest, base_dir)
    assert target.count("twin_fixture", exact=True).count == len(FIXTURE)

    # Mutate the SOURCE: page-0 changes, page-1 is gone (will be
    # tombstoned), one brand-new point appears.
    src.set_payload("twin_fixture", payload={"license": "MIT"},
                    points=[_pid("page-0")])
    src.delete("twin_fixture", points_selector=qm.PointIdsList(points=[_pid("page-1")]))
    src.upsert("twin_fixture", points=[qm.PointStruct(
        id=_pid("page-new"), vector={"dense": _basis_vector(7)},
        payload={"url": "https://example.org/page-new", "title": "page-new",
                 "lang": "en", "source_type": "web_page", "license": "CC0-1.0",
                 "trust_tier": 0.7, "last_seen": "2026-09-24T00:00:00Z"})])

    current_rows = list(dump.scroll_rows(src, "twin_fixture"))
    builder = dump.DeltaBuilder(base_manifest, base_dir)
    changed = list(builder.changed_rows(iter(current_rows)))
    tombstones = builder.tombstones({_pid("page-1"): "aged_out"})
    assert {t.id for t in tombstones} == {_pid("page-1")}

    delta_dir = tmp_path / "delta"
    d_shards, d_total = dump.write_shards(iter(changed), delta_dir, shard_rows=4)
    n_tomb, tomb_sha = pack.write_tombstones(
        tombstones, str(delta_dir / "tombstones.jsonl.gz"))
    delta_manifest = dump.build_manifest(
        dump_id="rehearsal-delta-1", collection="twin_fixture", dims=DIMS,
        distance="cosine", shards=d_shards, total_points=d_total,
        any_row_has_sparse=True, profile=None)
    delta_manifest["schema"] = pack.DELTA_PACK_SCHEMA
    delta_manifest["pack_id"] = "rehearsal-delta-1"
    delta_manifest["sequence"] = 1
    delta_manifest["base_dump_id"] = base_manifest["dump_id"]
    delta_manifest["base_manifest_sha256"] = pack.sha256_file(str(base_dir / "manifest.json"))
    delta_manifest["tombstones"] = {"path": "tombstones.jsonl.gz",
                                    "count": n_tomb, "sha256": tomb_sha}
    (delta_dir / "manifest.json").write_text(json.dumps(delta_manifest, indent=2))

    apply.verify_dump_signature(delta_manifest, delta_dir, pubkey=None,
                                allow_unsigned=True)
    apply.verify_shard_hashes(delta_manifest, delta_dir)
    n_upserted = apply.upsert_shards(target, "twin_fixture", delta_manifest, delta_dir)
    n_applied = apply.apply_tombstones(target, "twin_fixture", delta_manifest, delta_dir)

    assert n_applied == 1
    assert n_upserted == 2  # page-0 (changed) + page-new
    final_count = target.count("twin_fixture", exact=True).count
    assert final_count == len(FIXTURE) - 1 + 1  # -page-1, +page-new

    assert not target.retrieve("twin_fixture", ids=[_pid("page-1")])
    got0 = target.retrieve("twin_fixture", ids=[_pid("page-0")])[0]
    assert got0.payload["license"] == "MIT"
    assert target.retrieve("twin_fixture", ids=[_pid("page-new")])


# --------------------- the sealed real-twin fixture (task 687, S8 step A)
from cytir.distribute import rehearsal

_FIXTURE = rehearsal.DEFAULT_FIXTURE_DIR


def test_sealed_fixture_is_the_declared_projection_and_name_free():
    """What was committed from the twin names only projected keys and carries
    no officer-clause text (release-doctrine #1/#6)."""
    src, manifest = rehearsal._load_into_memory(_FIXTURE, "fx")
    allowed = set(pack.DEFAULT_PROJECTION_FIELDS) | set(pack.ENTITY_CARD_PROJECTION_FIELDS)
    n = 0
    for row in dump.scroll_rows(src, "fx"):
        payload = json.loads(row["payload"])
        assert not set(payload) - allowed
        assert rehearsal.OFFICER_MARKER not in row["payload"].lower()
        assert payload.get("license"), "every fixture row carries a license"
        n += 1
    assert n == manifest["counts"]["points"] >= 300
    assert sum(f.stat().st_size for f in _FIXTURE.rglob("*") if f.is_file()) < 2_000_000


def test_sealed_fixture_carries_consent_flags_and_32_hex_digests_task_768():
    src, _ = rehearsal._load_into_memory(_FIXTURE, "fx")
    rows = [json.loads(r["payload"]) for r in dump.scroll_rows(src, "fx")]
    assert all(k in p for p in rows for k in pack.CONSENT_FIELDS)
    assert any(p["train_ok"] is False for p in rows), "a train_ok=false row must be in the fixture"
    assert any(p["train_ok"] is True for p in rows) and any(p["train_ok"] is None for p in rows)
    digests = [p["content_digest"] for p in rows if "content_digest" in p]
    assert len(digests) >= 300
    assert all(len(d) == pack.DIGEST_HEX and pack._HEX.fullmatch(d) for d in digests)
    for p in rows:
        assert not set(p) & rehearsal.MASTER_ONLY


def test_sealed_fixture_rehearsal_signs_verifies_applies_and_answers_alike(tmp_path):
    if not shutil.which("minisign"):
        pytest.skip("minisign not on PATH (the CI step `python -m "
                    "cytir.distribute.rehearsal run` fails instead of skipping)")
    report = rehearsal.rehearse(_FIXTURE, tmp_path)
    assert report["signature"] == "verified"
    assert report["tamper_refusals"] == 3
    assert report["queries"] >= 30
    assert report["consent_flags"]["train_ok_false"] >= 1
    assert report["consent_flags"]["train_ok_false_survived"] == report["consent_flags"]["train_ok_false"]
    assert report["content_digest_rows"] >= 300
    # task 1283: the rehearsed release declares the layout prod serves, and apply accepted it
    assert report["apply_layout"]["layout"].startswith("profile prod-e9 "), report["apply_layout"]


def test_upgrade_stage_installs_n_minus_1_applies_delta_and_the_copy_is_n(tmp_path):
    """task 2308: the upgrade path on the sealed fixture, judged by the resulting state."""
    if not shutil.which("minisign"):
        pytest.skip("minisign not on PATH (the CI step fails instead of skipping)")
    up = rehearsal.rehearse(_FIXTURE, tmp_path)["upgrade"]
    assert up["delta_adds"] >= 20 and up["delta_replaces"] >= 20 and up["tombstones"] >= 20
    assert up["points_after"] == up["n_minus_1_points"] == 396
    assert set(up["tombstone_reasons"]) <= pack.TOMBSTONE_REASONS
    assert up["queries_equal_fresh_install"] >= 30 and up["refusals"] == 2


def test_upgrade_stage_fails_when_tombstones_do_not_reach_the_copy(tmp_path, monkeypatch):
    """The assertion can fail: an apply that drops the tombstones is caught."""
    if not shutil.which("minisign"):
        pytest.skip("minisign not on PATH (the CI step fails instead of skipping)")
    monkeypatch.setattr(apply, "apply_tombstones", lambda *a, **k: 0)
    with pytest.raises(rehearsal.RehearsalFailure, match="applied"):
        rehearsal.rehearse(_FIXTURE, tmp_path)


# ------------- exactness of the delta path on the REAL fixture rows (task 687 refactor)
def _fixture_rows():
    manifest = json.loads((_FIXTURE / "manifest.json").read_text())
    return [pack.row_to_record(r["id"], r["dense"], r["payload"],
                               (r["sparse_indices"], r["sparse_values"])
                               if r["sparse_indices"] is not None else None)
            for s in manifest["shards"]
            for r in pack.read_parquet_shard(str(_FIXTURE / s["path"]))]


def _old_4dp_rule(row):
    """The rule task 687 replaced, verbatim: payload + dense rounded to 4 dp."""
    import hashlib
    return hashlib.sha256(json.dumps(
        {"payload": row["payload"], "dense": [round(float(x), 4) for x in row["dense"]]},
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _with_dense(row, dense):
    out = dict(row)
    out["dense"] = dense
    return out


def test_unchanged_fixture_rows_are_a_zero_row_delta_on_every_transport(tmp_path):
    """Qdrant hands Python a float32 as REST text (a double that is not the
    float32) or as the exact double (gRPC, Parquet). The same fixture's real rows in
    either form, against a sidecar written from the other, are a 0-row delta."""
    import numpy as np
    exact = sorted(_fixture_rows(), key=lambda r: pack.point_id_key(r["id"]))
    rest = [_with_dense(r, [float(str(np.float32(x))) for x in r["dense"]]) for r in exact]
    assert any(a["dense"] != b["dense"] for a, b in zip(exact, rest))  # forms really differ
    shards, total, fps = dump.write_full_shards(iter(rest), tmp_path, shard_rows=1000)
    base = dump.build_manifest(
        dump_id="b", collection="fx", dims=len(exact[0]["dense"]), distance="cosine",
        shards=shards, total_points=total, any_row_has_sparse=True, fingerprints=fps, profile=None)
    for label, snapshot in (("rest", rest), ("exact", exact)):
        builder = dump.DeltaBuilder(base, tmp_path)
        assert list(builder.changed_rows(iter(snapshot))) == [], label
        assert builder.tombstones({}) == [], label
    # sentinel: the sealed vectors are float16-derived and carry no 4 dp tie, so
    # plant four REAL tie values scrolled from the twin (id 00005b2c-..., 00014439-...,
    # 00017227-..., 0001929f-...): the rule this replaces calls the same row changed.
    tie = [-0.06305, -0.09235, -0.14465, -0.16675] + exact[0]["dense"][4:]
    r_exact = _with_dense(exact[0], [float(np.float32(x)) for x in tie])
    r_rest = _with_dense(exact[0], [float(str(np.float32(x))) for x in tie])
    assert _old_4dp_rule(r_exact) != _old_4dp_rule(r_rest)
    assert pack.row_fingerprint(r_exact) == pack.row_fingerprint(r_rest)


def test_dump_apply_redump_of_the_fixture_is_the_release(tmp_path):
    """dump -> apply -> re-dump: payload text and sparse leg exact on all fixture
    rows, dense within the renormalisation bound, and the copy is a 0-row delta
    against its own dump."""
    src, fx_manifest = rehearsal._load_into_memory(_FIXTURE, "src")
    shards, total, fps = dump.write_full_shards(
        dump.scroll_rows(src, "src"), tmp_path / "rel", shard_rows=200)
    manifest = dump.build_manifest(
        dump_id="rel", collection="src", dims=fx_manifest["embedder"]["dims"],
        distance="cosine", shards=shards, total_points=total,
        any_row_has_sparse=True, fingerprints=fps, profile=None)
    target = qdrant_client(":memory:")
    apply.create_collection(target, "t", manifest, legacy_layout=True)
    apply.upsert_shards(target, "t", manifest, tmp_path / "rel")
    report = rehearsal.redump_check(src, "src", target, "t", tmp_path / "redump")
    assert report["rows"] == json.loads((_FIXTURE / "fixture.json").read_text())["points"] >= 300
    assert report["max_ulp"] <= rehearsal.RENORM_MAX_ULP


def _apply_pack(target, name, manifest, d):
    n = apply.upsert_shards(target, name, manifest, d)
    t = apply.apply_tombstones(target, name, manifest, d) if manifest.get("tombstones") else 0
    return n, t


def _state(client, name):
    return {r["id"]: (r["payload"], tuple(r["dense"]), r["sparse_indices"], r["sparse_values"])
            for r in dump.scroll_rows(client, name)}


def test_base_plus_delta_equals_the_full_dump_of_the_new_state(tmp_path):
    """The consumer's end state must not depend on the route: release N-1 then
    delta N is bit-identical to a fresh install of full release N, for adds,
    changes and tombstones, on the real fixture rows."""
    src, fx_manifest = rehearsal._load_into_memory(_FIXTURE, "src")
    dims = fx_manifest["embedder"]["dims"]

    def full(name, out):
        shards, total, fps = dump.write_full_shards(
            dump.scroll_rows(src, "src"), out, shard_rows=200)
        m = dump.build_manifest(dump_id=name, collection="src", dims=dims,
                                distance="cosine", shards=shards, total_points=total,
                                any_row_has_sparse=True, fingerprints=fps, profile=None)
        (out / "manifest.json").write_text(json.dumps(m))
        return m

    d0 = full("d0", tmp_path / "d0")
    ids = [r["id"] for r in dump.scroll_rows(src, "src")]
    changed, gone = ids[3:8], ids[100:103]
    src.set_payload("src", payload={"title": "task 687 changed"}, points=changed)
    src.delete("src", points_selector=qm.PointIdsList(points=gone))
    pts = src.retrieve("src", ids=ids[200:202], with_vectors=True, with_payload=True)
    new_ids = [mint_point_id(f"https://example.org/687-new-{i}", "") for i in range(2)]
    src.upsert("src", points=[qm.PointStruct(id=nid, vector=p.vector, payload=p.payload)
                              for nid, p in zip(new_ids, pts)])
    d1 = full("d1", tmp_path / "d1")

    builder = dump.DeltaBuilder(d0, tmp_path / "d0")
    rows = list(builder.changed_rows(dump.scroll_rows(src, "src")))
    tombs = builder.tombstones({g: "aged_out" for g in gone})
    assert {r["id"] for r in rows} == set(changed) | set(new_ids)
    assert {t.id for t in tombs} == set(gone)
    dd = tmp_path / "delta"
    shards, n = dump.write_shards(iter(rows), dd, shard_rows=200)
    n_t, sha = pack.write_tombstones(tombs, str(dd / "tombstones.jsonl.gz"))
    dm = dump.build_manifest(dump_id="p1", collection="src", dims=dims, distance="cosine",
                             shards=shards, total_points=n, any_row_has_sparse=True, profile=None)
    dm["tombstones"] = {"path": "tombstones.jsonl.gz", "count": n_t, "sha256": sha}

    x, y = qdrant_client(":memory:"), qdrant_client(":memory:")
    apply.create_collection(x, "c", d0, legacy_layout=True)
    _apply_pack(x, "c", d0, tmp_path / "d0")
    assert _apply_pack(x, "c", dm, dd) == (len(rows), len(gone))
    apply.create_collection(y, "c", d1, legacy_layout=True)
    _apply_pack(y, "c", d1, tmp_path / "d1")
    sx, sy = _state(x, "c"), _state(y, "c")
    assert sx.keys() == sy.keys()
    # payload text and sparse leg exact; dense within the local-mode bound
    # (qdrant-client's numpy normaliser is batch-shape dependent; a real
    # server is bit-stable: task 687 receipt, "Round trip")
    assert [i for i in sx if sx[i][0] != sy[i][0] or sx[i][2:] != sy[i][2:]] == []
    assert max(rehearsal._ulp_gap(list(sx[i][1]), list(sy[i][1])) for i in sx) \
        <= rehearsal.RENORM_MAX_ULP


def test_aw_rehearse_is_the_one_way_and_ci_calls_it(capsys):
    """`cyt rehearse run` is the rehearsal; the CI step runs that verb and no
    second command line (engineering-doctrine: one way to do a verb)."""
    import pathlib

    from cytir import cli

    ci_file = pathlib.Path(rehearsal.__file__).resolve().parents[3] / ".github" / "workflows" / "ci.yml"
    if ci_file.is_file():       # the exported public tree holds the app only, not the repo's CI
        ci = ci_file.read_text()
        assert "python -m cytir.cli rehearse run" in ci
        assert "cytir.distribute.rehearsal" not in ci
    assert "rehearse" in cli.build_parser().format_help()
    if not shutil.which("minisign"):
        pytest.skip("minisign not on PATH (the CI step fails instead of skipping)")
    assert cli.main(["rehearse", "run"]) == 0
    assert '"signature": "verified"' in capsys.readouterr().out


def test_apply_accepts_a_prod_e9_manifest_the_rehearsal_run_stamps_task_1283():
    from cytir.core import serving_profile as SP
    from cytir.core.qdrant import qdrant_client
    from cytir.distribute import apply, dump
    manifest = dump.build_manifest(
        dump_id="t", collection="c", dims=SP.PROD_E9.dense_dim, distance="cosine", shards=[],
        total_points=0, any_row_has_sparse=True, fingerprints=[], profile=SP.PROD_E9)
    assert manifest["profile"]["name"] == "prod-e9"
    receipt = apply.create_collection(qdrant_client(":memory:"), "t", manifest, log=lambda m: None)
    assert receipt["layout"].startswith("profile prod-e9 ")

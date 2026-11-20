"""`cyt index build` / `cyt index apply` (slice 8 step 2, specs/aw-index-serve-gate.md section 2).

THE test of the step: a collection `cyt index build` makes from sealed packs of two families
(`dila_juri`, four real CASS/JADE decisions in two stages; `loda_exact`, three real LEGI articles)
holds, point for point, what the live drain's twin-first transaction puts on PROD for the same
packs: ids, dense vectors, sparse vectors, projected payload. Because it is the same code:
compose -> plan -> project -> upsert, one transaction apart.

Needs a throwaway Qdrant (`AW_TEST_QDRANT_URL`, never a live port); skips without one. The vector
shards are deterministic stand-ins in the sealed shard format (marked `stand_in_for_tests`): the
write path never reads vector VALUES, and the embedder has its own tests.
"""
from __future__ import annotations

import gzip
import json
import pathlib
import shutil

import pytest

pytestmark = pytest.mark.usefixtures("no_sparse_rule")   # task 717: these test the drain/build plumbing, not the rule

import drain_transaction_harness as H  # noqa: E402
from loda_stage_support import loda_stage, vectors_for  # noqa: E402,F401
from cytir.index import promote_queue as Q                       # noqa: E402  (the alias of cytir.index.promote_queue)
from cytir.core import serving_profile as SP  # noqa: E402
from cytir.core.qdrant import qdrant_client  # noqa: E402
from cytir.index import build as B  # noqa: E402
from cytir.index import compose as COMPOSE  # noqa: E402
from cytir.index import promote_sealed_pack as PSP  # noqa: E402
from cytir.sources import family_stage as FS, run_family  # noqa: E402

TARGET = "aw_build_target"
PROFILE = SP.get("served")
DIM = PROFILE.dense_dim                  # 256: the served width


@pytest.fixture
def url():
    return H.qdrant_url()


# ------------------------------------------------------------------------ the sealed stages
def dila_stages(tmp: pathlib.Path) -> list[pathlib.Path]:
    """The four real decisions, cut into two stages of two by `family_stage` itself."""
    root = tmp / "dila-dump"
    d = root / "cass"
    d.mkdir(parents=True)
    for f in sorted(H.FIXTURE_DIR.glob("*.xml")):
        shutil.copy(f, d / f.name)
    out = tmp / "dila-run"
    run_family.run_family("dila_juri", out_dir=out, today="2026-10-03",
                          consent=lambda h, dd: pytest.fail("dump route: no host consent"),
                          options={"root": str(root)}, log=lambda m: None)
    FS.stage_family_pack(out / "pack", corpus_root=tmp / "corpus", family_id="dila_juri",
                         shard_rows=2, log=lambda m: None)
    stages = [d for _, d in FS.list_stages(tmp / "corpus", "dila_juri")]
    assert len(stages) == 2
    for s in stages:
        vectors_for(s)
    return stages


@pytest.fixture
def stages(tmp_path):
    d0, d1 = dila_stages(tmp_path)
    return {"dila0": d0, "dila1": d1, "loda": loda_stage(tmp_path)}


def pack(stage: pathlib.Path) -> B.StagePack:
    return B.load_stage_pack(str(stage))


# ----------------------------------------------------------------------- the two doors
def build_run(url, tmp, stage_list, *, name="run", collection=TARGET, mode="build", **kw):
    kw.setdefault("recreate", mode == "build")
    return B.run(mode=mode, packs=[pack(s) for s in stage_list], url=url, collection=collection,
                 profile_name="served", out_dir=tmp / name, as_of=H.AS_OF,
                 consent_cache_dir=str(H._empty_dir(tmp / "consent-cache")), log=lambda m: None,
                 **kw)


def reference_drain(url, tmp, stage_list, monkeypatch) -> list[dict]:
    """What PROD holds after the live drain's twin-first transaction applies the same packs: the
    two collections are made by the same creator from the same profile, then drained."""
    client = qdrant_client(url)
    for name in (H.TWIN, H.PROD):
        SP.create_collection(client, name, PROFILE, on_exists="recreate", log=lambda m: None)
    H.set_alias(url, H.ALIAS, H.PROD)
    queue = tmp / "ref-queue"
    ctx = COMPOSE.CONSENT_STAMP.load_context(
        as_of=H.AS_OF, cache_dir=str(H._empty_dir(tmp / "consent-cache")))
    for stage in stage_list:
        p = pack(stage)
        res = COMPOSE.compose_pack(
            pack_id=p.pack_id, family=p.family, chunks_dir=p.chunks_dir, vectors_dir=p.vectors_dir,
            collection=H.TWIN, mrl=DIM, queue_dir=str(queue), consent_ctx=ctx)
        mp = pathlib.Path(res["pack_dir"]) / "manifest.json"
        m = json.loads(mp.read_text())
        m["detect_language"] = False          # the per-pack override every drain-level test uses
        body = json.dumps(m, indent=1, sort_keys=True).encode()
        mp.write_bytes(body)
        (mp.parent / "READY").write_text(__import__("hashlib").sha256(body).hexdigest() + "\n")
    H.install_fakes(monkeypatch, Q, PSP)
    results = Q.drain(queue, twin_url=url, prod_url=url, task="696-ref", apply=True, batch=3,
                      pause_s=0, artifact_root=tmp / "ref-artifacts", refresh_index=False,
                      prod_alias=H.ALIAS)
    assert [r["status"] for r in results] == ["LANDED"] * len(stage_list), results
    points = H.dump_points(url, H.PROD)
    # the build's endpoint must look like a stranger's (it refuses one that HOLDS the prod alias or a
    # declared live collection), so the twin-first world is torn down once its points are read
    H.clear_live(url)
    return points


@pytest.fixture(autouse=True)
def _fresh_target(url):
    """Every test starts without the collections a previous test built, and on an endpoint that
    holds no live name (a stranger's)."""
    H.clear_live(url)
    for name in (TARGET, "aw_wrong_shape"):
        if name in {c["name"] for c in H.api(url, "/collections")["collections"]}:
            H.api(url, f"/collections/{name}", method="DELETE")
    yield
    H.clear_live(url)


@pytest.fixture(autouse=True)
def _build_uses_the_content_check_override(monkeypatch):
    """`detect_language: False` (the fasttext model file is not a CI dependency) is a per-pack
    MANIFEST override; the build composes its own manifests, so it is applied at the one place the
    drain reads it. Both the reference and the build go through the same `mechanical_content_status`."""
    real = Q.mechanical_content_status
    monkeypatch.setattr(Q, "mechanical_content_status",
                        lambda manifest, joined: real({**manifest, "detect_language": False}, joined))


# ------------------------------------------------------------------------- THE identity test
def test_a_built_collection_is_what_the_drain_puts_on_prod_point_for_point(
        tmp_path, url, stages, monkeypatch):
    order = [stages["dila0"], stages["dila1"], stages["loda"]]
    reference = reference_drain(url, tmp_path, order, monkeypatch)
    assert len(reference) == 7                               # 4 dila_juri + 3 loda_exact rows
    code, receipt = build_run(url, tmp_path, order)
    assert code == B.EXIT_OK, receipt
    built = H.dump_points(url, TARGET)
    assert len(built) == 7
    assert [p["id"] for p in built] == [p["id"] for p in reference]
    for a, b in zip(built, reference):
        assert a["vector"]["dense"] == b["vector"]["dense"], a["id"]
        assert a["vector"]["bm25"] == b["vector"]["bm25"], a["id"]
        assert a["payload"] == b["payload"], a["id"]
    assert json.dumps(built, sort_keys=True) == json.dumps(reference, sort_keys=True)
    # the receipt counts by stage: records -> admitted -> written, per pack, and in total
    packs = {p["pack_id"]: p for p in receipt["packs"]}
    assert {pid: (p["records"], p["admitted"], p["written"]) for pid, p in packs.items()} == {
        "dila-juri-v1-shard0": (2, 2, 2), "dila-juri-v1-shard1": (2, 2, 2),
        "loda-exact-v1-shard0": (3, 3, 3)}
    assert receipt["layout"]["points_count"] == 7
    assert packs["loda-exact-v1-shard0"]["id_scheme"] == COMPOSE.ID_SCHEME_LEGACY_TITLED
    assert receipt["profile"]["name"] == "served" and receipt["fingerprints"]["count"] == 7


def test_the_identity_comparison_can_fail(tmp_path, url, stages, monkeypatch):
    """Negative control: the same build with another consent stamp date differs from prod's rows in
    exactly the payload (`consent_checked_at`) and nothing else, so the test above is not vacuous."""
    order = [stages["dila0"], stages["dila1"], stages["loda"]]
    reference = reference_drain(url, tmp_path, order, monkeypatch)
    code, _ = B.run(mode="build", packs=[pack(s) for s in order], url=url, collection=TARGET,
                    profile_name="served", out_dir=tmp_path / "other", recreate=True,
                    as_of="2026-10-04T00:00:00Z", log=lambda m: None,
                    consent_cache_dir=str(H._empty_dir(tmp_path / "consent-cache")))
    assert code == B.EXIT_OK
    built = H.dump_points(url, TARGET)
    assert [p["id"] for p in built] == [p["id"] for p in reference]
    assert [p["vector"] for p in built] == [p["vector"] for p in reference]
    assert [p["payload"] for p in built] != [p["payload"] for p in reference]
    for a, b in zip(built, reference):
        assert {k for k in a["payload"] if a["payload"][k] != b["payload"].get(k)} == {"consent_checked_at"}


def test_apply_adds_a_delta_and_ends_where_the_drain_does(tmp_path, url, stages, monkeypatch):
    order = [stages["dila0"], stages["dila1"], stages["loda"]]
    reference = reference_drain(url, tmp_path, order, monkeypatch)
    code, first = build_run(url, tmp_path, [stages["dila0"], stages["loda"]], name="b1")
    assert code == B.EXIT_OK and first["layout"]["points_count"] == 5
    code, second = build_run(url, tmp_path, [stages["dila1"]], name="b2", mode="apply")
    assert code == B.EXIT_OK and second["mode"] == "apply" and second["collection_created"] is False
    assert second["layout"]["points_count"] == 7
    assert json.dumps(H.dump_points(url, TARGET), sort_keys=True) == json.dumps(reference, sort_keys=True)


def test_apply_checks_the_shape_and_refuses_a_missing_collection(tmp_path, url, stages):
    with pytest.raises(B.BuildRefusal, match="does not exist"):
        build_run(url, tmp_path, [stages["dila0"]], mode="apply", collection="aw_no_such_collection")
    # a collection that is not the served shape (128-d, no quantizer): refused, nothing written
    H.reset_collection(url, "aw_wrong_shape", dim=128)
    with pytest.raises(B.BuildRefusal, match="not the 'served' shape"):
        build_run(url, tmp_path, [stages["dila0"]], mode="apply", collection="aw_wrong_shape")
    assert H.dump_points(url, "aw_wrong_shape") == []


def test_a_rebuild_has_the_same_fingerprint_sidecar(tmp_path, url, stages):
    order = [stages["dila0"], stages["dila1"], stages["loda"]]
    _, a = build_run(url, tmp_path, order, name="a")
    _, b = build_run(url, tmp_path, order, name="b")                  # --recreate: from scratch
    lines = [gzip.open(tmp_path / n / "fingerprints.tsv.gz", "rt").read().splitlines() for n in "ab"]
    assert lines[0] == lines[1]
    assert a["fingerprints"]["sha256"] == b["fingerprints"]["sha256"]
    assert a["fingerprints"]["count"] == b["fingerprints"]["count"] == 7


def test_resume_skips_a_pack_that_already_landed(tmp_path, url, stages):
    order = [stages["dila0"], stages["loda"]]
    _, first = build_run(url, tmp_path, order, name="same-run")
    code, again = build_run(url, tmp_path, order, name="same-run", recreate=False, resume=True)
    assert code == B.EXIT_OK
    assert [p["status"] for p in again["packs"]] == ["DONE-EARLIER", "DONE-EARLIER"]
    assert again["layout"]["points_count"] == first["layout"]["points_count"] == 5


def test_a_build_keeps_its_receipts_in_its_own_state_dir_not_the_queue(tmp_path, url, stages):
    """The Single transaction never writes receipts or DONE next to the manifests it reads
    (review of #553, F1): they land in RUN_DIR/state."""
    _, receipt = build_run(url, tmp_path, [stages["dila0"]], name="r")
    pack_id = receipt["packs"][0]["pack_id"]
    assert sorted(p.name for p in (tmp_path / "r" / "queue" / pack_id).iterdir()) == [
        "READY", "manifest.json"]
    assert (tmp_path / "r" / "state" / pack_id / "DONE").exists()
    assert (tmp_path / "r" / "state" / pack_id / "receipt.json").exists()


@pytest.mark.parametrize("name", [H.TWIN, H.PROD])
def test_a_build_never_touches_a_declared_live_collection_even_with_allow_remote(
        tmp_path, url, stages, name):
    """`--allow-remote` lifts the build's own name check; the live-target refusal stays, and it runs
    BEFORE `--recreate` could delete the collection: the endpoint holds a declared live collection,
    so it is live, whatever the target is called."""
    H.reset_collection(url, name)
    for mode in ("build", "apply"):
        with pytest.raises(B.BuildRefusal, match="LIVE endpoint"):
            build_run(url, tmp_path, [stages["dila0"]], collection=name, mode=mode, allow_remote=True)
    assert H.api(url, f"/collections/{name}")["config"]["params"]["vectors"]["dense"]["size"] == 128
    assert H.dump_points(url, name) == []


# ----------------------------------------------------------------------------- the refusals
def refused(url, tmp, stage_list, match, **kw):
    before = {c["name"] for c in H.api(url, "/collections")["collections"]}
    with pytest.raises(B.BuildRefusal, match=match):
        build_run(url, tmp, stage_list, **kw)
    # nothing was created by a refused run
    assert {c["name"] for c in H.api(url, "/collections")["collections"]} <= before | {TARGET}


def test_refuses_an_existing_collection_without_resume_or_recreate(tmp_path, url, stages):
    build_run(url, tmp_path, [stages["dila0"]], name="first")
    refused(url, tmp_path, [stages["dila0"]], "already exists", name="second", recreate=False)


def test_refuses_an_unknown_or_private_profile(tmp_path, url, stages):
    for name in ("twin-e8", "nonsense"):          # task 1271: prod-e9 is now public (the --pack default)
        with pytest.raises(B.BuildRefusal, match="unknown profile"):
            B.run(mode="build", packs=[pack(stages["dila0"])], url=url, collection=TARGET,
                  profile_name=name, out_dir=tmp_path / "x", log=lambda m: None)


def test_refuses_a_served_name_or_a_remote_url_without_allow_remote(tmp_path, url, stages):
    for name in ("aw_prod", "aw_e9", "aw_v4"):
        refused(url, tmp_path, [stages["dila0"]], "never targets", collection=name)
    with pytest.raises(B.BuildRefusal, match="never targets"):
        B.check_target("http://qdrant.example.org:6333", "mine", allow_remote=False)
    B.check_target("http://qdrant.example.org:6333", "mine", allow_remote=True)      # the flag works


def test_refuses_vectors_of_another_width_or_an_unpinned_model(tmp_path, url, stages):
    s = stages["dila0"]
    man = next((s / "vectors").glob("*.vectors.manifest.json"))
    good = man.read_text()
    man.write_text(json.dumps({**json.loads(good), "dim": 768}))
    refused(url, tmp_path, [s], "768-dim")
    man.write_text(json.dumps({**json.loads(good), "embedder": {"model": "other/model",
                                                                "revision_pinned": "abc"}}))
    refused(url, tmp_path, [s], "pin mismatch")
    d = json.loads(good)
    d.pop("embedder")
    man.write_text(json.dumps(d))
    refused(url, tmp_path, [s], "no `embedder` record")


def test_refuses_an_id_that_does_not_reproduce_and_writes_nothing(tmp_path, url, stages):
    s = stages["dila0"]
    shard = s / "chunks" / "shard-0000.texts.jsonl.gz"
    with gzip.open(shard, "rt", encoding="utf-8") as fh:
        rows = [json.loads(x) for x in fh]
    rows[0]["point_id"] = "00000000-0000-5000-8000-000000000000"
    with gzip.open(shard, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    refused(url, tmp_path, [s], "identity check FAILED")
    assert TARGET not in {c["name"] for c in H.api(url, "/collections")["collections"]}


def test_refuses_a_stage_without_vectors_or_a_pack_id(tmp_path, stages):
    shutil.rmtree(stages["dila0"] / "vectors")
    with pytest.raises(B.BuildRefusal, match="the build never embeds"):
        pack(stages["dila0"])
    with pytest.raises(B.BuildRefusal, match="no PACK.json"):
        B.load_stage_pack(str(tmp_path))


def test_a_pack_the_mechanical_gate_refuses_exits_1_and_names_it(tmp_path, url, stages, monkeypatch):
    monkeypatch.setattr(Q, "mechanical_content_status", lambda manifest, joined: {
        "verdict": "REFUSE", "problems": {}, "other_failing_share": 1.0, "fail_share_max": 0.1,
        "word_floor_share": 0.0, "word_floor_fail_share_max": 0.1, "other_failing_rows": 2,
        "rows_scored": 2, "word_floor_dropped": 0})
    code, receipt = build_run(url, tmp_path, [stages["dila0"]])
    assert code == B.EXIT_GATE
    assert receipt["refused"]["dila-juri-v1-shard0"]["status"] == "PROVED-BLOCKED"


# ------------------------------------------------------------------------------------- CLI
def test_the_cli_verbs_exist_and_exit_codes_are_the_contract(tmp_path, url, stages, capsys):
    from cytir import cli
    argv = ["index", "build", "--pack", str(stages["dila0"]), "--qdrant", url,
            "--collection", TARGET, "--out", str(tmp_path / "cli"), "--as-of", H.AS_OF,
            "--consent-cache-dir", str(H._empty_dir(tmp_path / "cc")), "--recreate"]
    assert cli.main(argv) == 0
    assert (tmp_path / "cli" / "build-manifest.json").is_file()
    assert cli.main(["index", "build", "--pack", str(stages["dila0"]), "--qdrant", url,
                     "--collection", "aw_prod", "--out", str(tmp_path / "cli2")]) == 3
    assert cli.main(["index"]) == 2
    with pytest.raises(SystemExit) as e:
        cli.main(["index", "build"])                       # missing required flags: argparse, exit 2
    assert e.value.code == 2


# ------------------------------------------ a stranger's build never reaches the live box (R1-R3)
@pytest.fixture
def live_queue(tmp_path, monkeypatch):
    from cytir.core import paths as PATHS
    live = tmp_path / "the-live-queue"
    live.mkdir()
    monkeypatch.setenv("AW_QUEUE_DIR", str(live))
    PATHS.reset_settings_cache()
    yield live
    PATHS.reset_settings_cache()


def test_a_build_keeps_its_queue_state_and_artifacts_out_of_the_live_queue(
        tmp_path, url, stages, live_queue):
    """`--out` equal to, inside or above the live queue is refused BEFORE compose writes a manifest
    there; a RUN_DIR beside it is fine."""
    for out in (live_queue, live_queue / "run", live_queue.parent):
        with pytest.raises(B.BuildRefusal, match="live promotion queue"):
            B.run(mode="build", packs=[pack(stages["dila0"])], url=url, collection=TARGET,
                  profile_name="served", out_dir=out, recreate=True, as_of=H.AS_OF,
                  consent_cache_dir=str(H._empty_dir(tmp_path / "cc")), log=lambda m: None)
    assert list(live_queue.iterdir()) == []
    assert TARGET not in {c["name"] for c in H.api(url, "/collections")["collections"]}
    code, _ = build_run(url, tmp_path, [stages["dila0"]], name="beside")
    assert code == B.EXIT_OK


def test_a_build_refuses_a_live_endpoint_under_a_spelling_nobody_listed(tmp_path, url, stages):
    """The endpoint holds the twin, prod and the alias (it plays a live box): reached as
    `0.0.0.0:<port>`, the build is refused before it creates or deletes anything."""
    for name in (H.TWIN, H.PROD):
        H.reset_collection(url, name)
    H.set_alias(url, H.ALIAS, H.PROD)
    port = url.rsplit(":", 1)[1]
    for spelling in (f"http://0.0.0.0:{port}", f"http://localhost:{port}"):
        with pytest.raises(B.BuildRefusal, match="LIVE endpoint"):
            build_run(spelling, tmp_path, [stages["dila0"]], allow_remote=True)
    names = {c["name"] for c in H.api(url, "/collections")["collections"]}
    assert TARGET not in names and {H.TWIN, H.PROD} <= names

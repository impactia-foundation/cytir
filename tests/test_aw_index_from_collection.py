"""`cyt index build --from-collection` (cytir.index.from_collection, task 1256) on fake Qdrant only.

A fake client stands in for the creator's calls and a fake `_api` for the HTTP transport; nothing
here reaches a network. What is held: the collection is created as PROD_E9 unless a flag changes
one field; a prod or live target is refused before any write; a prod source is never scrolled; the
payload projection and the prune cut are the package's own and are applied to real-shaped rows;
the delta catch-up refuses each way a caught-up row would silently differ from the rows it joins.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import pathlib

import pytest

from cytir.core import serving_profile as SP
from cytir.index import build as B
from cytir.index import from_collection as FC
from cytir.index import payload_profile as PP
from cytir.index import prod_guard as PG
from cytir.index import promote_queue as PQ
from cytir.index import promote_sealed_pack as PSP
from cytir.index import twin_write_hold as TWH

#: No literal names a port a Qdrant listens on (tests/test_throwaway_qdrant_guard.py): the target is
#: local on the dead port 9 (check_target wants a local host) and the source an unresolvable host,
#: so a call that escaped the fakes would be refused, never answered.
TARGET_URL = "http://127.0.0.1:9"
SOURCE = "http://twin.invalid:6457:aw_twin_src"


def _row(i, terms):
    """A raw twin row: the payload keys the `normalized` profile touches, and two it keeps."""
    return {"id": i, "vector": {"dense": [0.01] * 256,
                                "bm25": {"indices": list(terms), "values": [1.0] * len(terms)}},
            "payload": {"url": f"https://www.fedlex.admin.ch/eli/cc/{i}/de", "lang": "de",
                        "stamps": {"x": 1}, "consent_state": "resolved", "provenance": "weko_docs",
                        "content_digest": "ab" * 32, "chunk_rule": "a-rule-no-table-holds"}}


ROWS = [_row(1, (1, 2, 9)), _row(2, (1, 3)), _row(3, (1, 3)), _row(4, (1, 7))]


class FakeClient:
    def __init__(self, exists=False):
        self.exists, self.created, self.indexes, self.patched, self.deleted = exists, None, [], [], 0

    def collection_exists(self, name):
        return self.exists

    def delete_collection(self, name):
        self.deleted += 1
        self.exists = False

    def create_collection(self, **kw):
        self.created, self.exists = kw, True

    def create_payload_index(self, name, field_name, field_schema):
        self.indexes.append((field_name, field_schema))

    def update_collection(self, name, hnsw_config):
        self.patched.append(hnsw_config.m)

    def readback(self) -> dict:
        """What a server would answer to `GET /collections/<name>` for the recorded create call and
        finalize PATCH: the sent fields as sent, the unsent ones at the pinned server defaults."""
        def dump(m):
            return m.model_dump(mode="json", exclude_none=True)
        d, kw = SP.SERVER_DEFAULTS, self.created
        hnsw = {k: d[k2] for k, k2 in (("m", "hnsw_m"), ("ef_construct", "ef_construct"),
                                       ("full_scan_threshold", "full_scan_threshold"),
                                       ("max_indexing_threads", "max_indexing_threads"))}
        hnsw.update(dump(kw["hnsw_config"]), m=self.patched[-1])
        opt = {k: d[k] for k in ("default_segment_number", "max_segment_size", "indexing_threshold",
                                 "max_optimization_threads", "memmap_threshold")}
        opt.update(dump(kw["optimizers_config"]))
        params = {"vectors": {n: dump(v) for n, v in kw["vectors_config"].items()},
                  "sparse_vectors": {n: dump(v) for n, v in kw["sparse_vectors_config"].items()},
                  "on_disk_payload": kw["on_disk_payload"]}
        params.update({k: d[k] for k in ("shard_number", "replication_factor", "write_consistency_factor")})
        schema = {f: {"data_type": dump(s)["type"], "params": dump(s)} for f, s in self.indexes}
        return {"config": {"params": params, "hnsw_config": hnsw, "optimizer_config": opt,
                           "quantization_config": dump(kw["quantization_config"])},
                "payload_schema": schema}


class FakeHttp:
    """The source (pages of ROWS) and the target (a dict of upserted points), at the `_api` seam."""

    def __init__(self, rows=ROWS, page=2):
        self.rows, self.page, self.target, self.calls = rows, page, {}, []
        self.crash_on_put = None        # the Nth PUT raises, as a dead connection would
        self.puts = 0

    def __call__(self, url, path, body=None, method=None, **_):
        self.calls.append((method or ("POST" if body is not None else "GET"), url, path))
        if path.endswith("/points/scroll"):
            start = body.get("offset") or 0
            nxt = start + body["limit"]
            return {"points": self.rows[start:nxt], "next_page_offset": nxt if nxt < len(self.rows) else None}
        if path.endswith("/points/count"):
            return {"count": len(self.target)}
        if path.startswith("/collections/aw_twin_src/points"):
            return [r for r in self.rows if str(r["id"]) in {str(x) for x in body["ids"]}]
        if "/points?wait=true" in path:
            self.puts += 1
            if self.puts == self.crash_on_put:
                raise ConnectionError("simulated crash mid-page")
            self.target.update({p["id"]: p for p in body["points"]})
            return {}
        return {"points_count": len(self.rows), "status": "green"}


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv(TWH.ENV, str(tmp_path / "no-hold.json"))
    monkeypatch.setattr(PQ, "live_queue_dir", lambda: None)
    monkeypatch.setattr(PQ, "assert_single_target_not_live", lambda single: None)
    monkeypatch.setattr(PQ, "single_target_lock", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(PG, "_fetch_aliases", lambda url: [])
    monkeypatch.setattr(B, "_settle", lambda url, coll, t: {"points_count": 4, "segments_count": 1})
    http, client = FakeHttp(), FakeClient()
    monkeypatch.setattr(PSP, "_api", http)
    monkeypatch.setattr(FC, "qdrant_client", lambda url: client)
    return argparse.Namespace(http=http, client=client, out=tmp_path / "run")


def _args(env, *extra):
    ap = B._parser("build", "cyt index build")
    return ap.parse_args(["--from-collection", SOURCE, "--qdrant", TARGET_URL,
                          "--collection", "aw_rebuilt", "--out", str(env.out), *extra])


def test_created_collection_is_prod_e9(env):
    code, receipt = FC.run(_args(env), log=lambda *_: None)
    assert code == B.EXIT_OK and receipt["points_exact"] == 4
    kw = env.client.created
    dense = kw["vectors_config"]["dense"]
    assert (dense.size, dense.datatype, dense.on_disk) == (256, SP.qm.Datatype.FLOAT16, True)
    assert kw["sparse_vectors_config"]["bm25"].index.datatype == SP.qm.Datatype.UINT8
    assert kw["hnsw_config"].m == 0 and kw["hnsw_config"].max_indexing_threads == 8   # bulk load
    assert env.client.patched == [8], "finalize PATCHes PROD_E9's m once the rows are in"
    opt = kw["optimizers_config"]
    assert (opt.default_segment_number, opt.max_segment_size, opt.indexing_threshold,
            opt.max_optimization_threads) == (30, 3_900_000, 10000, 4)
    assert kw["quantization_config"].product.compression == SP.qm.CompressionRatio.X8
    assert len(env.client.indexes) == 8 and all(s.on_disk for _, s in env.client.indexes)
    assert {f for f, _ in env.client.indexes} == {f for f, _, _ in SP.PROD_E9.payload_indexes}
    assert receipt["profile"] == SP.PROD_E9.block()
    # the WHOLE shape: equal to what PROD_E9 predicts and to aw_e9's sealed server read-back
    built = SP.shape_of(env.client.readback())
    assert SP.shape_diff(built, SP.PROD_E9.expected_shape()) == {}
    sealed_path = (pathlib.Path(__file__).resolve().parents[1] / "data" / "serving-profiles"
                   / "aw_e9.config.v1.json")
    if not sealed_path.is_file():    # the public cut ships the sealed configs once #1061 lands
        pytest.skip("sealed read-back not shipped in this tree")
    assert SP.shape_diff(built, SP.shape_of(json.loads(sealed_path.read_text()))) == {}


def test_an_override_changes_exactly_that_field(env):
    a = _args(env, "--hnsw-m", "16")
    prof = FC.profile_for(a)
    diff = {k for k, v in dataclasses.asdict(prof).items() if v != getattr(SP.PROD_E9, k)}
    assert diff == {"hnsw_m", "name"} and prof.hnsw_m == 16 and prof.name == "prod-e9+hnsw_m=16"
    a2 = _args(env, "--sparse-datatype", "float16")
    diff2 = {k for k, v in dataclasses.asdict(FC.profile_for(a2)).items() if v != getattr(SP.PROD_E9, k)}
    assert diff2 == {"sparse_index_datatype", "name"}
    assert FC.profile_for(_args(env)) is SP.PROD_E9, "no flag: the profile itself, not a copy"
    FC.run(a, log=lambda *_: None)
    assert env.client.patched == [16]
    assert env.client.created["sparse_vectors_config"]["bm25"].index.datatype == SP.qm.Datatype.UINT8


@pytest.mark.parametrize("url,coll,why", [
    (TARGET_URL, "aw_prod", "name the live boxes serve"),
    (TARGET_URL, "aw_e9", "name the live boxes serve"),
    ("http://box.invalid:6333", "mine", "non-local"),
    (f"http://127.0.0.1:{PG.PROD_PORT}", "mine", "is prod"),
])
def test_prod_or_live_target_is_refused_before_any_write(env, url, coll, why):
    a = _args(env)
    a.qdrant, a.collection = url, coll
    with pytest.raises(B.BuildRefusal, match=why):
        FC.run(a, log=lambda *_: None)
    assert env.client.created is None and env.http.target == {}
    assert not [c for c in env.http.calls if c[0] == "PUT"]


def test_a_prod_source_is_never_scrolled(env):
    a = _args(env, "--prune-df-lt", "2")
    a.from_collection = f"http://127.0.0.1:{PG.PROD_PORT}:aw_e9"
    with pytest.raises(B.BuildRefusal, match="REFUSED scroll on prod"):
        FC.run(a, log=lambda *_: None)
    assert not [c for c in env.http.calls if c[2].endswith("/points/scroll")]


def test_a_prod_source_is_refused_before_the_target_is_created(env):
    """Task 1289: the refusal is argument validation, so no create, delete or PUT reached the target."""
    a = _args(env, "--recreate")
    a.from_collection = f"http://127.0.0.1:{PG.PROD_PORT}:aw_e9"
    with pytest.raises(B.BuildRefusal, match="REFUSED scroll on prod"):
        FC.run(a, log=lambda *_: None)
    assert env.client.created is None and env.client.deleted == 0 and env.client.patched == []
    assert env.http.calls == [] and env.http.target == {}


def test_a_prod_source_exits_with_the_refusal_code_through_the_cli(env, capsys):
    argv = ["--from-collection", f"http://127.0.0.1:{PG.PROD_PORT}:aw_e9", "--qdrant", TARGET_URL,
            "--collection", "aw_rebuilt", "--out", str(env.out)]
    assert B.main(argv) == B.EXIT_REFUSED
    assert "REFUSED scroll on prod" in capsys.readouterr().err
    assert env.client.created is None and env.http.calls == []


def test_finalize_and_create_refuse_a_prod_target(env):
    prod = f"http://127.0.0.1:{PG.PROD_PORT}"
    with pytest.raises(B.BuildRefusal, match="is prod"):
        FC._finalize_target(env.client, prod, "aw_x", FC.SP.PROD_E9, log=lambda *_: None)
    assert env.client.patched == []
    single = argparse.Namespace(url=prod, collection="aw_x")
    for mode in ("refuse", "recreate"):
        with pytest.raises(B.BuildRefusal, match="is prod"):
            FC._create_target(env.client, single, "aw_x", FC.SP.PROD_E9, on_exists=mode, log=lambda *_: None)
    assert env.client.created is None and env.client.deleted == 0


def test_the_source_takes_reads_only(env):
    with pytest.raises(B.BuildRefusal, match="read only"):
        FC._source_api("http://twin.invalid:6457", "/collections/x/points/delete", {"points": [1]})


def test_slim_payload_and_prune_on_real_shaped_rows(env):
    code, receipt = FC.run(_args(env, "--prune-df-lt", "2", "--prune-df-gt", "0.9",
                                 "--payload-profile", "normalized"), log=lambda *_: None)
    assert code == B.EXIT_OK
    assert receipt["prune"]["rule"]["df_lt_abs"] == 2 and receipt["prune"]["terms"] == 4  # {1, 2, 9, 7}
    assert [env.http.target[i]["vector"].get("bm25", {}).get("indices") for i in (1, 2, 3, 4)] == \
        [None, [3], [3], None]
    assert receipt["lever"]["rows_no_sparse"] == 2 and receipt["lever"]["postings_dropped"] == 7
    sealed = PP.load_tables()
    pl = env.http.target[1]["payload"]
    assert "stamps" not in pl and len(pl["content_digest"]) == PP.DIGEST_PREFIX_HEX
    # the sealed table seeds the codes: a known value gets prod's integer, an unknown one is appended
    assert pl["consent_state"] == sealed["consent_state"]['"resolved"']
    assert pl["provenance"] == sealed["provenance"]['"weko_docs"']
    assert pl["chunk_rule"] == len(sealed["chunk_rule"])
    assert receipt["enum_tables_seed"]["from"] == B.PAYLOAD_PROFILE_OF["served"]
    assert receipt["payload_profile"] == "normalized"


def test_default_payload_profile_is_the_identity(env):
    code, receipt = FC.run(_args(env), log=lambda *_: None)
    assert receipt["payload_profile"] == "prod" and receipt["enum_tables_seed"] is None
    assert env.http.target[1]["payload"]["stamps"] == {"x": 1}
    assert receipt["idf_table"] is None and "bm25_idf.json" in receipt["idf_table_note"]


ROWS6 = [_row(i, (1, i)) for i in range(1, 7)]


def test_a_crash_mid_page_then_resume_lands_every_row(env):
    """Review 1268 B1: --scroll 3 --batch 2 over 6 rows, the 2nd PUT dies (rows 1-2 in, row 3 not).
    The state must still point at page 1, so the resume re-sends it and row 3 lands."""
    env.http.rows, env.http.crash_on_put = ROWS6, 2
    with pytest.raises(ConnectionError):
        FC.run(_args(env, "--scroll", "3", "--batch", "2"), log=lambda *_: None)
    assert sorted(env.http.target) == [1, 2]
    env.http.crash_on_put = None
    code, receipt = FC.run(_args(env, "--scroll", "3", "--batch", "2", "--resume"), log=lambda *_: None)
    assert code == B.EXIT_OK and sorted(env.http.target) == [1, 2, 3, 4, 5, 6]
    assert receipt["points_sent"] == receipt["points_exact"] == 6


def test_resume_without_a_state_refuses(env):
    env.client.exists = True
    with pytest.raises(B.BuildRefusal, match="--resume needs"):
        FC.run(_args(env, "--resume"), log=lambda *_: None)
    assert env.http.target == {} and env.client.created is None


def test_a_delta_crash_mid_page_then_rerun_lands_every_id(env, tmp_path):
    """Review 1268 B1, the delta half: ids 3-6, --scroll 4 --batch 2, the 2nd PUT dies."""
    env.http.rows, env.client.exists = ROWS6, True
    env.http.target = {1: {}, 2: {}}
    a = _delta(env, tmp_path, "3\n4\n5\n6\n", dict(TABLES, sent=2), "--scroll", "4", "--batch", "2")
    env.http.crash_on_put = env.http.puts + 2
    with pytest.raises(ConnectionError):
        FC.run(a, log=lambda *_: None)
    assert sorted(env.http.target) == [1, 2, 3, 4]
    env.http.crash_on_put = None
    code, receipt = FC.run(a, log=lambda *_: None)
    assert code == B.EXIT_OK and sorted(env.http.target) == [1, 2, 3, 4, 5, 6]
    assert receipt["points_sent"] == receipt["points_exact"] == 6


def test_existing_target_needs_resume_or_recreate(env):
    env.client.exists = True
    with pytest.raises(B.BuildRefusal, match="already exists"):
        FC.run(_args(env), log=lambda *_: None)
    FC.run(_args(env, "--recreate"), log=lambda *_: None)
    assert env.client.deleted == 1 and env.client.created is not None


# --------------------------------------------------------------- the delta catch-up (--only-ids)
def _delta(env, tmp_path, ids="1\n2\n", state=None, *extra):
    (tmp_path / "ids.txt").write_text(ids)
    env.out.mkdir(parents=True, exist_ok=True)
    if state is not None:
        (env.out / FC.STATE_FILE).write_text(json.dumps(state))
    return _args(env, "--only-ids", str(tmp_path / "ids.txt"), "--payload-profile", "normalized", *extra)


TABLES = {"enum_tables_forward": {"a": {"b": 0}}, "sent": 4}


@pytest.mark.parametrize("exists,ids,state,extra,why", [
    (True, "1\n", TABLES, ("--recreate",), "--recreate would delete it"),
    (False, "1\n", TABLES, (), "existing target"),
    (True, "1\n", {"sent": 5}, (), "enum_tables_forward"),
    (True, "1\n", TABLES, ("--prune-df-gt", "0.02"), "different vocabulary"),
    (True, "\n  \n", TABLES, (), "holds no ids"),
])
def test_delta_refusals(env, tmp_path, exists, ids, state, extra, why):
    env.client.exists = exists
    with pytest.raises(B.BuildRefusal, match=why):
        FC.run(_delta(env, tmp_path, ids, state, *extra), log=lambda *_: None)
    assert env.http.target == {}


def test_delta_runs_the_same_transform_and_counts_every_id(env, tmp_path):
    env.client.exists = True
    a = _delta(env, tmp_path, "3\n4\n", dict(TABLES, sent=2))
    env.http.target = {1: {}, 2: {}}            # the build sent 2 rows; the catch-up adds 2
    code, receipt = FC.run(a, log=lambda *_: None)
    assert code == B.EXIT_OK and receipt["points_exact"] == 4 and receipt["mode"] == "from-collection-delta"
    assert "stamps" not in env.http.target[3]["payload"], "the delta row is projected like the build's"
    assert (env.out / "apply-manifest.json").exists() and not (env.out / "build-manifest.json").exists()
    assert env.client.patched == [], "a delta does not re-finalize the target"
    with pytest.raises(B.BuildRefusal, match="1 of the 2 requested ids"):
        FC.run(_delta(env, tmp_path, "4\n404\n", TABLES), log=lambda *_: None)


def test_id_pages_walk_the_list_and_report_the_resume_position(env):
    pages = list(FC.id_pages("http://twin.invalid:6457", "aw_twin_src", ["1", "2", "3", "4", "9"], limit=2))
    assert [n for _, n in pages] == [2, 4, 5] and sum(len(p) for p, _ in pages) == 4
    resumed = list(FC.id_pages("http://twin.invalid:6457", "aw_twin_src", ["1", "2", "3"], limit=2, start=2))
    assert [[p["id"] for p in pts] for pts, _ in resumed] == [[3]]


def test_one_transform_two_inputs():
    """The catch-up grows no second definition of the levers: `id_pages` is an INPUT to the one
    loop `scroll_pages` feeds, and the projection is called in exactly one place."""
    import inspect
    src = inspect.getsource(FC)
    assert src.count("PP.project_payload(") == 1 and src.count("SR.prune_postings(") == 1
    body = inspect.getsource(FC.id_pages).split('"""', 2)[2]
    for lever in ("project_payload", "prune", "datatype", "bm25"):
        assert lever not in body, lever


# ------------------------------------------------------------------------------- the CLI door
def test_aw_index_build_from_collection_parses_and_routes(env, monkeypatch):
    from cytir import cli
    seen = []
    monkeypatch.setattr(FC, "run", lambda args: (seen.append(args), (0, {}))[1])
    rc = cli.main(["index", "build", "--from-collection", SOURCE, "--qdrant", TARGET_URL,
                   "--collection", "aw_rebuilt", "--out", str(env.out), "--hnsw-m", "12"])
    assert rc == 0 and seen[0].from_collection == SOURCE and seen[0].hnsw_m == 12
    with pytest.raises(SystemExit):
        B.main(["--from-collection", SOURCE, "--pack", "x", "--qdrant", TARGET_URL,
                "--collection", "c", "--out", str(env.out)])
    assert FC.parse_source(SOURCE) == ("http://twin.invalid:6457", "aw_twin_src")
    with pytest.raises(B.BuildRefusal, match="URL:COLLECTION"):
        FC.parse_source("aw_e8")

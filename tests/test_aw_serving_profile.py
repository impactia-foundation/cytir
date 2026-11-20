# Slice 8 step 1 (specs/aw-index-serve-gate.md): one ServingProfile, one creator.
# The `serving-profile` CI job runs this file by path. The Qdrant-service half runs where
# AW_TEST_QDRANT_URL is set (the `serving-profile` CI job); there AW_REQUIRE_QDRANT=1
# turns a missing service into a FAILURE, never a skip.
"""The serving profile is derived from, and proven equal to, what is live."""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import types
import urllib.request
import uuid
from urllib.parse import urlparse

import pytest
from qdrant_client import models as qm

from cytir.core.qdrant import qdrant_client

from cytir.core import serving_config as SC
from cytir.core import serving_profile as SP
from cytir.distribute import apply, dump, pack
from cytir.embed import collection_stage as CS
from tests.helpers.repo_files import repo_file_or_skip

_APP = pathlib.Path(__file__).resolve().parents[1]
_DATA = _APP / "data" / "serving-profiles"

# The shared gate of every Qdrant-backed test (tests/throwaway_qdrant.py): asks the endpoint what it
# HOLDS before the first write; the port list of the live boxes lives there as a third belt only.
sys.path.insert(0, str(_APP / "tests"))
import throwaway_qdrant as TQ  # noqa: E402


def _fixture(name: str) -> dict:
    return json.loads((_DATA / f"{name}.config.v1.json").read_text())


def _fixture_result(name: str) -> dict:
    fx = _fixture(name)
    return {"config": fx["config"], "payload_schema": fx["payload_schema"]}


def _diff(a: dict, b: dict, path="") -> dict:
    """Every leaf where two shape dicts differ: {path: (a, b)}."""
    out = {}
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k, "<absent>"), b.get(k, "<absent>")
        if isinstance(x, dict) and isinstance(y, dict):
            out.update(_diff(x, y, f"{path}{k}."))
        elif x != y:
            out[f"{path}{k}"] = (x, y)
    return out


# ------------------------------------------------------------ the sealed fixtures
def test_fixtures_are_whole_reads_of_the_live_collections():
    for name in ("aw_e9", "aw_e8"):
        fx = _fixture(name)
        assert fx["schema"] == "aw.qdrant-collection-config.v1"
        assert fx["source"]["qdrant_version"] == "1.18.3"
        assert fx["source"]["points_count_at_capture"] > 10_000_000
        assert {"params", "hnsw_config", "optimizer_config", "quantization_config"} <= set(fx["config"])


def test_prod_profile_equals_the_sealed_prod_fixture_field_by_field():
    live = SP.shape_of(_fixture_result("aw_e9"))
    assert _diff(SP.PROD_E9.expected_shape(), live) == {}


def test_twin_profile_equals_the_sealed_twin_fixture_field_by_field():
    live = SP.shape_of(_fixture_result("aw_e8"))
    assert _diff(SP.TWIN_E8.expected_shape(), live) == {}


def test_served_is_the_twin_shape_minus_its_runtime_thread_cap():
    live = SP.shape_of(_fixture_result("aw_e8"))
    assert _diff(SP.SERVED.expected_shape(), live) == {
        "optimizer.max_optimization_threads": (None, 3)}


def test_served_is_not_prod_and_the_differences_are_named():
    """The finding of this step: what `cyt embed` builds is the twin's shape, not prod's."""
    d = _diff(SP.SERVED.expected_shape(), SP.PROD_E9.expected_shape())
    assert d == {
        "hnsw.m": (16, 8),
        "hnsw.max_indexing_threads": (0, 8),
        "optimizer.default_segment_number": (0, 30),
        "optimizer.max_segment_size": (None, 3900000),
        "optimizer.max_optimization_threads": (None, 4),
        "sparse.bm25.index_datatype": ("float16", "uint8"),
        "payload_indexes.tier": ({"type": "keyword", "on_disk": False}, "<absent>"),
        "payload_indexes.lei": ({"type": "keyword", "on_disk": False}, "<absent>"),
        **{f"payload_indexes.{f}.on_disk": (False, True)
           for f in ("url", "domain", "lang", "country", "source_type", "trust_tier",
                     "published", "register_id")},
    }


def test_twin_and_prod_collection_configs_differ_by_design():
    assert _diff(SP.shape_of(_fixture_result("aw_e8")),
                 SP.shape_of(_fixture_result("aw_e9")))


# ------------------------------------------------------- creation kwargs, byte-equal
class _Fake:
    """Records the calls a creator makes; `collection_exists` is scriptable."""

    def __init__(self, exists=False):
        self.calls, self.exists = [], exists

    def collection_exists(self, n):
        return self.exists

    def delete_collection(self, n):
        self.calls.append(["delete", n])

    def create_collection(self, *a, **kw):
        self.calls.append(["create_collection", [_j(x) for x in a],
                           {k: _j(v) for k, v in kw.items()}])

    def create_payload_index(self, *a, **kw):
        self.calls.append(["create_payload_index", [_j(x) for x in a],
                           {k: _j(v) for k, v in kw.items()}])

    def get_collection(self, n):  # only the bm25_saturate shim reads the source back
        return types.SimpleNamespace(config=types.SimpleNamespace(params=types.SimpleNamespace(
            vectors={"dense": qm.VectorParams(size=256, distance=qm.Distance.COSINE)})))


def _j(v):
    if hasattr(v, "model_dump"):
        return v.model_dump(mode="json")
    if isinstance(v, dict):
        return {k: _j(x) for k, x in v.items()}
    if hasattr(v, "value"):
        return v.value
    return v


def _norm(calls):
    """`create_collection("x", ...)` and `create_collection(collection_name="x", ...)` are
    the same request; the old apply/bm25 creators passed the name positionally. A keyword
    sent as None (old apply: `sparse_vectors_config=None`) is the same as not sent."""
    out = []
    for c in calls:
        if c[0] == "create_collection":
            kw = {k: v for k, v in c[2].items() if v is not None}
            c = [c[0], [], {"collection_name": c[1][0], **kw} if c[1] else kw]
        out.append(c)
    return out


def _golden() -> dict:
    g = json.loads(repo_file_or_skip("data/serving-profiles/create-kwargs.old-code.v2.json").read_text())
    assert g["schema"] == "aw.create-kwargs-golden.v2"
    assert g["provenance"]["captured_from_commit"] == "2cbe863f8cbee5726be61a21809bbc02a7b80790"
    assert g["provenance"]["cases"] == len(g["cases"]) == 65
    return g


@pytest.fixture
def no_readback(monkeypatch):
    monkeypatch.setattr(SC, "assert_collection_matches", lambda *a, **k: {})


def _capture_create_kwargs() -> dict:
    """Every request kwarg the four creators send, through a recording fake. The SAME call
    list was run against the merge-base (2cbe863f) to make the golden file; here it runs
    against this tree. (apply.* has no profile block in these manifests: the legacy layout.)"""
    from cytir.core import bm25_saturate
    out = {}
    for name in SC.CONFIGS:
        for bulk in (True, False):
            for wpi in (True, False):
                f = _Fake()
                SC.create_collection(f, "x", SC.get(name), bulk=bulk, with_payload_indexes=wpi,
                                     log=lambda *a: None)
                out[f"sc.{name}.bulk={bulk}.idx={wpi}"] = f.calls
    for lay in CS.LAYOUT_NAMES:
        for dd in (None, 128, 256, 1024):
            try:
                L = CS.resolve_layout(lay, dense_dim=dd) if dd else CS.resolve_layout(lay)
            except Exception as e:  # noqa: BLE001
                out[f"cs.{lay}.dd={dd}"] = f"ERR {e!r}"
                continue
            f = _Fake()
            CS.create_collection(f, "x", L, recreate=False, log=lambda *a: None)
            out[f"cs.{lay}.dd={dd}"] = f.calls
            f = _Fake(exists=True)
            CS.create_collection(f, "x", L, recreate=True, log=lambda *a: None)
            out[f"cs.{lay}.dd={dd}.recreate-existing"] = f.calls
    for key, man in [
            ("apply.with-sparse-idf", {"embedder": {"dims": 256, "distance": "cosine"},
                                       "sparse": {"vector_name": "bm25", "modifier": "idf"}}),
            ("apply.no-sparse", {"embedder": {"dims": 1024, "distance": "cosine"}}),
            ("apply.sparse-no-modifier", {"embedder": {"dims": 1024, "distance": "cosine"},
                                          "sparse": {"vector_name": "bm25"}}),
            ("apply.dot", {"embedder": {"dims": 128, "distance": "dot"}}),
            ("apply.euclid", {"embedder": {"dims": 128, "distance": "euclid"}}),
            ("apply.Cosine-cap", {"embedder": {"dims": 128, "distance": "Cosine"}}),
            ("apply.nodist", {"embedder": {"dims": 128}})]:
        f = _Fake()
        apply.create_collection(f, "x", man, legacy_layout=True, log=lambda *a: None)
        out[key] = f.calls
        f = _Fake(exists=True)
        apply.create_collection(f, "x", man, recreate=True, legacy_layout=True,
                                log=lambda *a: None)
        out[key + ".recreate"] = f.calls
    for m in (0, 8, 16, 32):
        f = _Fake()
        bm25_saturate.ensure_dst(f, "src", "dst", m)
        out[f"bm25.m{m}"] = f.calls
    return out


def test_golden_the_65_cases_of_the_four_old_creators(no_readback):
    # the recorded `cs.config-b-128.*` cases are a fact about old code; the layout was removed
    # (task 1378), so only the cases the tree can still produce are compared
    cases = _golden()["cases"]
    # excluded: a different creator (config-b-128), not one of the four old creators;
    # the layout it recorded was removed (task 1378)
    excluded = sorted(k for k in cases if k.startswith("cs.config-b-128."))
    N = 8  # measured 2026-10-28: the golden file holds 8 recorded cs.config-b-128.* cases
    assert len(excluded) == N, excluded
    assert all(k.startswith("cs.config-b-128.") for k in excluded), excluded
    old = {k: cases[k] for k in cases if k not in excluded}
    assert len(old) == 65 - N, sorted(old)
    assert len(old) + len(excluded) == len(cases) == 65 - N + N
    new = _capture_create_kwargs()
    assert set(old) == set(new)
    bad = {k: (old[k], new[k]) for k in old
           if (_norm(old[k]) if isinstance(old[k], list) else old[k])
           != (_norm(new[k]) if isinstance(new[k], list) else new[k])}
    assert bad == {}


@pytest.mark.parametrize("dim", [128, 256])
def test_the_live_entry_stage_pack_b1_ensure_collection_equals_the_old_staging_creator(
        no_readback, dim):
    from cytir.index import stage_pack_b1 as spb
    f = _Fake()
    spb.ensure_collection(f, "x", False, dim)
    want = _golden()["cases"][f"cs.pack-staging.dd={dim}"]
    assert _norm(f.calls) == _norm(want)


def test_golden_collection_stage_recreate_and_keep(no_readback):
    f = _Fake(exists=True)
    assert CS.create_collection(f, "x", CS.resolve_layout("served"), recreate=False) is False
    assert f.calls == []
    f = _Fake(exists=True)
    CS.create_collection(f, "x", CS.resolve_layout("served"), recreate=True, log=lambda *a: None)
    assert f.calls[0] == ["delete", "x"]


def test_layout_profiles_have_the_shape_the_named_profiles_declare():
    assert (CS.layout_profile(CS.resolve_layout("served")).with_(name="served", note=SP.SERVED.note)
            == SP.SERVED)
    assert CS.layout_profile(CS.resolve_layout("pack-staging", dense_dim=256)).with_(
        note="") == SP.pack_staging(256).with_(note="")


# ------------------------------------------------------------- the creator's policy
def test_creator_refuses_an_existing_collection_by_default():
    with pytest.raises(SP.CollectionExistsError, match="already exists"):
        SP.create_collection(_Fake(exists=True), "x", SP.SERVED)


def test_creator_index_failure_policy():
    class Bad(_Fake):
        def create_payload_index(self, *a, **kw):
            raise RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        SP.create_collection(Bad(), "x", SP.SERVED, log=lambda *a: None)
    SP.create_collection(Bad(), "x", SP.SERVED, index_failure="warn", log=lambda *a: None)


def test_bulk_creates_at_m0_and_finalize_patches_the_profile_m():
    f = _Fake()
    SP.create_collection(f, "x", SP.PROD_E9, bulk=True, log=lambda *a: None)
    assert f.calls[0][2]["hnsw_config"]["m"] == 0

    class C:
        def update_collection(self, name, hnsw_config):
            self.m = hnsw_config.m
    c = C()
    SP.finalize_collection(c, "x", SP.PROD_E9, log=lambda *a: None)
    assert c.m == 8


# ----------------------------------------------------------------------- the manifest
def test_dump_manifest_carries_profile_name_and_digest():
    m = dump.build_manifest(dump_id="d", collection="c", dims=256, distance="cosine",
                            shards=[], total_points=0, any_row_has_sparse=True,
                            profile=SP.SERVED)
    assert m["profile"]["name"] == "served"
    assert m["profile"]["digest"] == SP.SERVED.digest()
    assert SP.ServingProfile.from_block(m["profile"]) == SP.SERVED


def test_dump_manifest_without_profile_is_an_explicit_choice():
    kw = dict(dump_id="d", collection="c", dims=3, distance="cosine", shards=[],
              total_points=0, any_row_has_sparse=False)
    with pytest.raises(TypeError, match="profile"):
        dump.build_manifest(**kw)           # forgetting the argument is an error
    assert "profile" not in dump.build_manifest(profile=None, **kw)   # legacy: said aloud


def test_dump_refuses_a_profile_of_another_width():
    with pytest.raises(dump.ProfileRefusal, match="256-d"):
        dump.build_manifest(dump_id="d", collection="c", dims=128, distance="cosine",
                            shards=[], total_points=0, any_row_has_sparse=False,
                            profile=SP.SERVED)


def _step2_profile_class():
    """The profile class of the NEXT format version: step 2 adds a defaulted field."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Step2Profile(SP.ServingProfile):
        embedder_pin: str = ""
    return Step2Profile


def test_a_step1_manifest_applies_with_code_that_has_one_more_defaulted_field():
    block = SP.SERVED.block()            # written by "step 1" (it has no embedder_pin)
    prof = _step2_profile_class().from_block(block)
    assert prof.embedder_pin == "" and prof.name == "served" and prof.dense_dim == 256
    # and through apply, with the real creator
    man = {"embedder": {"dims": 256, "distance": "cosine"}, "profile": block}
    c = qdrant_client(":memory:")
    assert apply.create_collection(c, "t", man, log=lambda m: None)["layout"].startswith(
        "profile served")


def test_a_newer_block_with_an_unknown_field_is_a_named_refusal_with_both_versions():
    P2 = _step2_profile_class()
    block = P2(**{**{f: getattr(SP.SERVED, f) for f in SP.ServingProfile.__dataclass_fields__},
                  "embedder_pin": "abc"}).block()
    with pytest.raises(SP.ProfileBlockRefusal) as e:
        SP.ServingProfile.from_block(block)            # step-1 code reads a step-2 block
    msg = str(e.value)
    assert "embedder_pin" in msg and "schema v1" in msg and "up to v1" in msg
    with pytest.raises(apply.ApplyRefusal, match="embedder_pin"):
        apply.create_collection(_Fake(), "x", {"embedder": {"dims": 256, "distance": "cosine"},
                                               "profile": block})


def test_a_block_of_a_newer_schema_version_is_refused(monkeypatch):
    block = SP.SERVED.block()
    block["profile"]["profile_schema_version"] = 2
    block["profile_schema_version"] = 2
    block["digest"] = SP.canonical_digest(block["profile"])
    with pytest.raises(SP.ProfileBlockRefusal, match=r"schema v2.*up to v1"):
        SP.ServingProfile.from_block(block)


def test_the_digest_is_over_the_block_as_stored_not_over_the_dataclass():
    block = SP.SERVED.block()
    assert block["digest"] == SP.canonical_digest(block["profile"])
    # a field a later class adds does not change what an earlier block hashes to
    assert _step2_profile_class().from_block(block).name == "served"


def test_a_tampered_profile_block_is_refused():
    block = SP.SERVED.block()
    block["profile"]["quantizer"] = "none"
    with pytest.raises(ValueError, match="digest"):
        SP.ServingProfile.from_block(block)
    man = {"embedder": {"dims": 256, "distance": "cosine"}, "profile": block}
    with pytest.raises(apply.ApplyRefusal, match="digest"):
        apply.create_collection(_Fake(), "x", man)


def test_a_block_that_claims_prod_e9_but_differs_is_refused_even_with_a_good_digest():
    fake = SP.SERVED.with_(name="prod-e9").block()     # honest digest, dishonest name
    with pytest.raises(SP.ProfileBlockRefusal, match=r"built-in.*hnsw_m"):
        SP.ServingProfile.from_block(fake)
    assert SP.ServingProfile.from_block(SP.PROD_E9.block()) == SP.PROD_E9
    ps = SP.pack_staging(128)
    assert SP.ServingProfile.from_block(ps.block()) == ps
    with pytest.raises(SP.ProfileBlockRefusal, match="built-in"):
        SP.ServingProfile.from_block(ps.with_(quantizer="none").block())
    man = {"embedder": {"dims": 256, "distance": "cosine"}, "profile": fake}
    with pytest.raises(apply.ApplyRefusal, match="built-in"):
        apply.create_collection(_Fake(), "x", man)


def test_apply_refuses_a_profile_of_another_width_than_the_shards():
    man = {"embedder": {"dims": 128, "distance": "cosine"}, "profile": SP.SERVED.block()}
    with pytest.raises(apply.ApplyRefusal, match="256-d"):
        apply.create_collection(_Fake(), "x", man)


def test_apply_with_a_profile_block_creates_that_profile():
    man = {"embedder": {"dims": 256, "distance": "cosine"}, "profile": SP.SERVED.block()}
    f, lines = _Fake(), []
    apply.create_collection(f, "x", man, log=lines.append)
    g = _Fake()
    SP.create_collection(g, "x", SP.SERVED, log=lambda *a: None)
    assert f.calls == g.calls
    assert f.calls[0][2]["quantization_config"]["product"]["compression"] == "x8"
    assert len([c for c in f.calls if c[0] == "create_payload_index"]) == 10
    assert any("profile served (digest " in l for l in lines)     # the layout is logged


LEGACY_MAN = {"embedder": {"dims": 256, "distance": "cosine"},
              "sparse": {"vector_name": "bm25", "modifier": "idf"}}


def test_apply_refuses_a_manifest_without_a_profile_unless_legacy_layout_is_named():
    f = _Fake()
    with pytest.raises(apply.ApplyRefusal, match=r"no serving profile.*--legacy-layout"):
        apply.create_collection(f, "x", LEGACY_MAN)
    assert f.calls == []                               # nothing was created
    lines = []
    r = apply.create_collection(f, "x", LEGACY_MAN, legacy_layout=True, log=lines.append)
    assert r["layout"].startswith("LEGACY layout (fp32, no quantizer, no payload indexes)")
    assert any("LEGACY layout" in l for l in lines)    # and it says so


def test_apply_main_refuses_a_legacy_manifest_without_the_flag(tmp_path):
    from cytir.distribute import pack as P
    m = dump.build_manifest(dump_id="d", collection="c", dims=3, distance="cosine",
                            shards=[], total_points=0, any_row_has_sparse=False, profile=None)
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    argv = ["--manifest", str(tmp_path), "--qdrant-url", ":memory:", "--collection", "x",
            "--allow-unsigned", "--state-file", str(tmp_path / "st.json")]
    with pytest.raises(apply.ApplyRefusal, match="--legacy-layout"):
        apply.main(argv)
    assert apply.main(argv + ["--legacy-layout"]) == 0


def test_the_receipt_of_a_local_client_says_not_verified_never_a_verified_shape():
    c = qdrant_client(":memory:")
    man = {"embedder": {"dims": 256, "distance": "cosine"}, "profile": SP.SERVED.block()}
    lines = []
    r = apply.create_collection(c, "t", man, log=lines.append)
    assert r["verified"] is False and "NOT verified" in r["reason"] + "NOT verified"
    assert "REQUESTED, not verified" in r["reason"]
    assert any("NOT VERIFIED" in l for l in lines)
    assert SP.is_server(c) is False
    with pytest.raises(apply.ApplyRefusal, match="already exists"):
        apply.create_collection(c, "t", man, log=lambda m: None)
    apply.create_collection(c, "t", man, recreate=True, log=lambda m: None)


def test_a_server_read_back_that_is_not_the_profile_refuses(monkeypatch):
    monkeypatch.setattr(SP, "is_server", lambda c: True)
    monkeypatch.setattr(SP, "read_shape", lambda c, n: SP.pack_staging(256).expected_shape())
    man = {"embedder": {"dims": 256, "distance": "cosine"}, "profile": SP.SERVED.block()}
    with pytest.raises(apply.ApplyRefusal, match=r"not profile 'served'.*quantization"):
        apply.create_collection(_Fake(), "x", man, log=lambda m: None)


def test_dump_refuses_a_wrong_width_before_exporting_a_single_shard(tmp_path):
    q = tmp_path / "q"
    c = qdrant_client(str(q))
    c.create_collection("c", vectors_config={"dense": qm.VectorParams(
        size=128, distance=qm.Distance.COSINE)})
    c.upsert("c", [qm.PointStruct(id=1, vector={"dense": [0.1] * 128}, payload={})])
    c.close()
    out = tmp_path / "out"
    argv = ["--collection", "c", "--qdrant-url", str(q), "--i-know-this-is-not-the-twin",
            "--out", str(out)]
    with pytest.raises(dump.ProfileRefusal, match=r"128-d vectors but profile 'prod-e9' is 256-d"):
        dump.main(argv)
    assert not list(out.glob("*.jsonl*")) and not (out / "manifest.json").exists()
    with pytest.raises(dump.ProfileRefusal, match="--dims 128"):
        dump.main(argv + ["--dims", "128"])


# ------------------------------------------------------------------- stale defaults
def test_serving_config_get_has_no_default_name(monkeypatch):
    monkeypatch.delenv("AW_SERVING_CONFIG", raising=False)
    with pytest.raises(ValueError, match="no default"):
        SC.get()
    monkeypatch.setenv("AW_SERVING_CONFIG", "256-pq8")
    assert SC.get().name == "256-pq8"
    assert not hasattr(SC, "DEFAULT_BUILD_CONFIG")


def test_dump_has_no_stale_128_fallback():
    assert not hasattr(dump, "DEFAULT_DENSE_DIMS")
    assert SP.get(dump.DEFAULT_PROFILE).dense_dim == 256


# ------------------------------------------------------ a real Qdrant (CI service container)
def _service_url():
    """The throwaway Qdrant, PROVEN by what it holds before any test writes (round 4, F1)."""
    return TQ.require_throwaway()


@pytest.fixture
def qd():
    url = _service_url()
    client = qdrant_client(url, timeout=60)
    made: list[str] = []

    def name(tag):
        n = f"awprof_{tag}_{uuid.uuid4().hex[:8]}"
        made.append(n)
        return n
    client.new_name = name
    client.url_ = url
    yield client
    for n in made:
        if client.collection_exists(n):
            client.delete_collection(n)


def _read(url, name) -> dict:
    with urllib.request.urlopen(f"{url}/collections/{name}", timeout=30) as r:
        return json.load(r)["result"]


def test_service_is_the_pinned_qdrant_version(qd):
    with urllib.request.urlopen(qd.url_ + "/", timeout=10) as r:
        assert json.load(r)["version"] == SC.QDRANT_IMAGE.rsplit(":v", 1)[1]


@pytest.mark.parametrize("profile,fixture,known", [
    (SP.PROD_E9, "aw_e9", {}),
    (SP.TWIN_E8, "aw_e8", {}),
    (SP.SERVED, "aw_e8", {"optimizer.max_optimization_threads": (None, 3)}),
])
def test_collection_created_from_profile_equals_the_sealed_live_config(qd, profile, fixture, known):
    name = qd.new_name(profile.name)
    assert SP.create_collection(qd, name, profile, verify=True, log=lambda *a: None) is True
    got = SP.shape_of(_read(qd.url_, name))
    want = SP.shape_of(_fixture_result(fixture))
    assert _diff(got, want) == known
    assert got == profile.expected_shape()


def test_bulk_then_finalize_reaches_the_profile_m(qd):
    name = qd.new_name("bulk")
    SP.create_collection(qd, name, SP.PROD_E9, bulk=True, log=lambda *a: None)
    assert SP.shape_of(_read(qd.url_, name))["hnsw"]["m"] == 0
    SP.finalize_collection(qd, name, SP.PROD_E9, log=lambda *a: None)
    assert SP.shape_of(_read(qd.url_, name))["hnsw"]["m"] == 8


def test_every_shim_creates_the_shape_its_profile_declares(qd):
    # serving_config (bulk, then finalize), collection_stage layouts, apply, bm25_saturate
    n1 = qd.new_name("sc")
    SC.create_collection(qd, n1, SC.get("256-pq8"), bulk=True, log=lambda *a: None)
    SC.finalize_collection(qd, n1, log=lambda *a: None)
    assert _diff(SP.shape_of(_read(qd.url_, n1)), SP.SERVED.expected_shape()) == {}
    for layout, kw in (("served", {}), ("pack-staging", {"dense_dim": 256})):
        n = qd.new_name("cs")
        lay = CS.resolve_layout(layout, **kw)
        CS.create_collection(qd, n, lay, recreate=False, log=lambda *a: None)
        assert _diff(SP.shape_of(_read(qd.url_, n)), CS.layout_profile(lay).expected_shape()) == {}
    n = qd.new_name("ap")
    apply.create_collection(qd, n, {"embedder": {"dims": 256, "distance": "cosine"},
                                    "sparse": {"vector_name": "bm25", "modifier": "idf"},
                                    "profile": SP.SERVED.block()}, log=lambda m: None)
    assert _diff(SP.shape_of(_read(qd.url_, n)), SP.SERVED.expected_shape()) == {}
    n = qd.new_name("legacy")
    apply.create_collection(qd, n, {"embedder": {"dims": 256, "distance": "cosine"},
                                    "sparse": {"vector_name": "bm25", "modifier": "idf"}},
                            legacy_layout=True, log=lambda m: None)
    legacy = SP.shape_of(_read(qd.url_, n))
    assert legacy["quantization"] is None and legacy["payload_indexes"] == {}
    assert legacy["vectors"]["dense"]["datatype"] == "float32"
    n = qd.new_name("sat")
    src = qd.new_name("satsrc")
    qd.create_collection(src, vectors_config={"dense": qm.VectorParams(
        size=256, distance=qm.Distance.COSINE)})
    from cytir.core import bm25_saturate
    bm25_saturate.ensure_dst(qd, src, n, 16)
    got = SP.shape_of(_read(qd.url_, n))
    assert got["quantization"] == {"kind": "scalar", "type": "int8", "always_ram": False}
    assert got["hnsw"]["m"] == 16 and got["vectors"]["dense"]["size"] == 256


def test_apply_on_a_server_reads_the_collection_back_and_says_verified(qd):
    for prof in (SP.SERVED, SP.PROD_E9):
        man = {"embedder": {"dims": 256, "distance": "cosine"}, "profile": prof.block()}
        r = apply.create_collection(qd, qd.new_name("rb"), man, log=lambda m: None)
        assert r["verified"] is True and r["profile"] == prof.name
    r = apply.create_collection(qd, qd.new_name("rbl"), LEGACY_MAN, legacy_layout=True,
                                log=lambda m: None)
    assert r["verified"] is True and r["layout"].startswith("LEGACY")


def test_a_collection_that_is_not_its_profile_is_caught_by_the_server_read_back(qd):
    n = qd.new_name("wrong")
    SP.create_collection(qd, n, SP.SERVED, log=lambda *a: None)
    with pytest.raises(SP.ShapeMismatch, match="prod-e9"):
        SP.verify_created(qd, n, SP.PROD_E9, log=lambda *a: None)
    assert SP.is_server(qd) is True


# The live staging layout, compared to SEALED reads of real staging collections that
# current main created on the build box:6398 (not to this PR's own `expected_shape()`).
@pytest.mark.parametrize("fixture,dim", [("aw_dnews256_swissinfo", 256),
                                         ("aw_dnews128_chmedia", 128)])
def test_pack_staging_profile_equals_a_sealed_read_of_a_live_staging_collection(fixture, dim):
    live = SP.shape_of(_fixture_result(fixture))
    assert _diff(SP.pack_staging(dim).expected_shape(), live) == {}
    assert live["hnsw"]["m"] == 0 and live["quantization"]["kind"] == "scalar"


@pytest.mark.parametrize("fixture,dim", [("aw_dnews256_swissinfo", 256),
                                         ("aw_dnews128_chmedia", 128)])
def test_the_live_staging_creator_reads_back_as_the_sealed_live_staging_collection(qd, fixture, dim):
    from cytir.index import stage_pack_b1 as spb
    n = qd.new_name("stg")
    spb.ensure_collection(qd, n, False, dim)
    assert _diff(SP.shape_of(_read(qd.url_, n)), SP.shape_of(_fixture_result(fixture))) == {}


# -------------------------------------------------------------------- the CI wiring
def test_ci_qdrant_is_the_pinned_release_on_free_local_ports():
    """The job runs the profile's pinned Qdrant as a process (the runner has no Docker
    access: PR 530's first run died in "Initialize containers")."""
    ci = repo_file_or_skip("../.github/workflows/ci.yml").read_text()
    job = ci[ci.index("\n  serving-profile:"):]
    version = SC.QDRANT_IMAGE.rsplit(":", 1)[1]
    assert f"QDRANT_VERSION: {version}" in job
    assert "QDRANT_SHA256: " in job
    assert "\n    services:" not in job   # no docker service
    # the start and the stop are ONE script, shared with the `index-serve-gate` job (step 5): the
    # job calls it, and the script is what pins the download, the host, the ports and the kill
    assert "bash ../.github/scripts/throwaway-qdrant.sh start" in job
    assert "bash ../.github/scripts/throwaway-qdrant.sh stop" in job
    script = repo_file_or_skip("../.github/scripts/throwaway-qdrant.sh").read_text()
    assert "sha256sum -c" in script
    assert "QDRANT__SERVICE__HOST=127.0.0.1" in script
    assert "bind(('127.0.0.1',0))" in script        # kernel-assigned ports, never a fixed one
    for live in ("6333", "6334", "6398", "6400", "6457"):
        assert f"HTTP_PORT={live}" not in job and f"HTTP_PORT={live}" not in script
    assert "AW_REQUIRE_QDRANT" in job and "if: always()" in job


def test_ci_job_runs_only_when_its_inputs_change_and_always_otherwise():
    ci = repo_file_or_skip("../.github/workflows/ci.yml").read_text()
    assert "schedule:" in ci[:ci.index("\njobs:")]            # the nightly run on main
    job = ci[ci.index("\n  serving-profile:"):]
    paths = re.search(r"\n      PROFILE_PATHS: '([^']+)'", job).group(1)
    rx = re.compile(paths)
    for hit in ("authoritative-web-app/cytir/core/serving_profile.py",
                "authoritative-web-app/cytir/embed/collection_stage.py",
                "authoritative-web-app/cytir/distribute/apply.py",
                "authoritative-web-app/data/serving-profiles/aw_e9.config.v1.json",
                "authoritative-web-app/tests/test_aw_serving_profile.py",
                # in the drain's import closure (round 4 of #553: tests/test_drain_ci_path_filter.py)
                "authoritative-web-app/cytir/core/serving_config.py",
                "authoritative-web-app/cytir/__init__.py", "authoritative-web-app/cytir/core/serving_config.py",
                ".github/workflows/ci.yml"):
        assert rx.search(hit), hit
    private_dir = "pipeline"
    for miss in ("authoritative-web-app/plan.md", "authoritative-web-app/cytir/news/x.py",
                 "authoritative-web-app/cytir/cli.py",
                 f"authoritative-web-app/{private_dir}/test_other.py"):
        assert not rx.search(miss), miss
    # task 1997 / #1508: the checkout runs BEFORE the gate (the gate script lives in the checkout),
    # so the gate step is sliced up to the next step after it, not up to the checkout
    gi = job.index("id: gate")
    assert job.index("- uses: actions/checkout@v4") < gi, "checkout must precede the path gate (#1508)"
    gate = job[gi:job.index("\n      - ", gi)]
    assert 'GITHUB_EVENT_NAME" != "pull_request"' in gate and "run=true" in gate
    assert "pull-requests: read" in job and "running the job" in gate   # lists files; fails open
    after = job[gi:]
    assert after.count("steps.gate.outputs.run == 'true'") >= 6


def test_the_source_cards_profile_builds_from_the_named_profile():
    """D4 (task 1378): the cards collection consumes `SP.SOURCE_CARDS`."""
    from cytir.corpus import source_cards
    p = source_cards._source_cards_profile()
    assert (p.name, p.dense_dim, p.quantizer) == ("source-cards", 128, "int8")
    assert p is SP.SOURCE_CARDS
    assert [f for f, _, _ in p.payload_indexes][:2] == ["disc_class", "jurisdiction"]


def test_source_cards_profile_kwargs_are_unchanged_by_task_1546():
    """Task 1546: the LIVE `source_cards` create request is byte-equal before and after the
    profile stopped deriving from the retired config-b-128 layout (literals recorded from origin/main)."""
    import json
    from dataclasses import asdict
    from cytir.corpus import source_cards
    p = source_cards._source_cards_profile()
    recorded = json.loads(
        '{"datatype_explicit": false, "dense_datatype": "float32", "dense_dim": 128, "dense_on_disk": true,'
        ' "distance": "cosine", "hnsw_m": 16, "hnsw_max_indexing_threads": null, "hnsw_on_disk": true,'
        ' "name": "source-cards", "note": "the out-of-pool source-authority card collection (ADR-021 / ADR-017)",'
        ' "on_disk_payload": true, "optimizers": [], "payload_indexes": [["disc_class", "keyword", false],'
        ' ["jurisdiction", "keyword", false], ["disposition", "keyword", false], ["domain", "keyword", false],'
        ' ["trust_tier", "float", false]], "quant_always_ram": false, "quantizer": "int8", "sparse_idf": true,'
        ' "sparse_index_datatype": null, "sparse_index_on_disk": null, "sparse_name": "bm25"}')
    assert json.loads(json.dumps({k: v for k, v in asdict(p).items() if k != "embedder_max_seq"},
                                 default=str)) == recorded
    assert p.digest() == "af429297dbf93a92f24e47c7b28e12b0bd1f161b2c6f692f26d02b7599cab986"
    assert (str(p.vectors_config()), str(p.sparse_vectors_config()), str(p.quantization_config())) == (
        "{'dense': VectorParams(size=128, distance=<Distance.COSINE: 'Cosine'>, hnsw_config=None,"
        " quantization_config=None, on_disk=True, datatype=None, multivector_config=None)}",
        "{'bm25': SparseVectorParams(index=None, modifier=<Modifier.IDF: 'idf'>)}",
        "scalar=ScalarQuantizationConfig(type=<ScalarType.INT8: 'int8'>, quantile=None, always_ram=False)")

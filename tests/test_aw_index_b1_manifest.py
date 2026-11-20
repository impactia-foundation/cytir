"""cytir.index.b1_manifest (task 2108): the collection config is unchanged, and the
manifest, state and ignition code moved without a change in behaviour. No test
touches a Qdrant server: a recording fake stands in for the client."""
from __future__ import annotations

import json

import pytest
from qdrant_client import models as qm

from cytir.index import b1_manifest as MF


class _Recorder:
    """Records every call a creator makes, in order."""

    def __init__(self, exists=False):
        self.calls = []
        self._exists = exists

    def collection_exists(self, name):
        self.calls.append(("collection_exists", name))
        return self._exists

    def delete_collection(self, name):
        self.calls.append(("delete_collection", name))

    def create_collection(self, **kw):
        self.calls.append(("create_collection", kw))

    def create_payload_index(self, name, field_name, field_schema):
        self.calls.append(("create_payload_index", name, field_name, field_schema))

    def update_collection(self, name, hnsw_config):
        self.calls.append(("update_collection", name, hnsw_config))


# The builder's own calls, copied from the code that was in the b1 builder script
# before task 2108: the "before" side of the proof.
_OLD_PAYLOAD_INDEXES = {
    "domain": qm.PayloadSchemaType.KEYWORD, "url": qm.PayloadSchemaType.KEYWORD,
    "source_type": qm.PayloadSchemaType.KEYWORD, "country": qm.PayloadSchemaType.KEYWORD,
    "tier": qm.PayloadSchemaType.KEYWORD, "lang": qm.PayloadSchemaType.KEYWORD,
    "trust_tier": qm.PayloadSchemaType.FLOAT, "register_id": qm.PayloadSchemaType.KEYWORD,
    "lei": qm.PayloadSchemaType.KEYWORD,
}


def _old_create(client, name):
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config={"dense": qm.VectorParams(
            size=128, distance=qm.Distance.COSINE, on_disk=True)},
        sparse_vectors_config={"bm25": qm.SparseVectorParams(
            modifier=qm.Modifier.IDF,
            index=qm.SparseIndexParams(on_disk=True, datatype=qm.Datatype.FLOAT16))},
        on_disk_payload=True,
        hnsw_config=qm.HnswConfigDiff(m=0, on_disk=True),
        quantization_config=qm.ScalarQuantization(scalar=qm.ScalarQuantizationConfig(
            type=qm.ScalarType.INT8, always_ram=False)),
    )
    for field, ftype in _OLD_PAYLOAD_INDEXES.items():
        client.create_payload_index(name, field_name=field, field_schema=ftype)


def _old_finalize(client, name):
    client.update_collection(name, hnsw_config=qm.HnswConfigDiff(m=16))


@pytest.mark.parametrize("exists", [False, True])
def test_create_collection_sends_the_same_qdrant_requests_as_before(exists):
    old, new = _Recorder(exists), _Recorder(exists)
    _old_create(old, "aw_x")
    MF.create_collection(new, "aw_x", log=lambda *a: None)
    assert new.calls == old.calls


def test_finalize_collection_sends_the_same_request_as_before():
    old, new = _Recorder(), _Recorder()
    _old_finalize(old, "aw_x")
    MF.finalize_collection(new, "aw_x", log=lambda *a: None)
    assert new.calls == old.calls


def test_create_collection_is_born_at_m0_and_finalize_sets_m16():
    rec = _Recorder()
    MF.create_collection(rec, "aw_x", log=lambda *a: None)
    MF.finalize_collection(rec, "aw_x", log=lambda *a: None)
    made = [c for c in rec.calls if c[0] == "create_collection"][0][1]
    assert made["hnsw_config"].m == 0 and made["vectors_config"]["dense"].size == 128
    assert [c for c in rec.calls if c[0] == "update_collection"][0][2].m == 16


def test_qc_connects_with_the_same_timeout_and_flag(monkeypatch):
    seen = {}
    monkeypatch.setattr(MF, "qdrant_connect",
                        lambda url, **kw: seen.update(url=url, **kw) or "client")
    assert MF.qc("http://h:1") == "client"
    assert seen == {"url": "http://h:1", "timeout": 1200, "check_compatibility": False}


def test_git_sha_override_and_default_shape(monkeypatch):
    monkeypatch.setattr(MF, "_GIT_SHA_OVERRIDE", "ab" * 20)
    assert MF.git_sha() == "ab" * 20
    monkeypatch.setattr(MF, "_GIT_SHA_OVERRIDE", "")
    sha = MF.git_sha()
    assert len(sha) == 40 and set(sha) <= set("0123456789abcdef")


def test_ignition_check_refuses_a_tree_that_is_not_a_git_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(MF.P, "REPO_ROOT", tmp_path)
    problems = MF.ignition_check()
    assert len(problems) == 1 and "not inside a git repository" in problems[0]


def test_assert_ignition_refuses_and_override_lets_it_through(tmp_path, monkeypatch):
    monkeypatch.setattr(MF.P, "REPO_ROOT", tmp_path)
    monkeypatch.delenv(MF.IGNITION_OVERRIDE_ENV, raising=False)
    with pytest.raises(SystemExit):
        MF.assert_ignition(log=lambda *a: None)
    monkeypatch.setenv(MF.IGNITION_OVERRIDE_ENV, "test")
    MF.assert_ignition(log=lambda *a: None)


def _manifest(tmp_path):
    return MF.write_manifest(
        str(tmp_path / "MANIFEST.json"), "aw_b7", None,
        [{"pack": "p", "source": "s", "version": "v", "count": 1,
          "policy": "POLICY.md@v1"}],
        {"points": 1, "adds": 1, "changes": 0, "deletes": 0},
        {"golds": 1, "pass": True, "report": "r"}, log=lambda *a: None)


def test_manifest_round_trips_and_validates(tmp_path, monkeypatch):
    monkeypatch.setattr(MF, "_GIT_SHA_OVERRIDE", "c" * 40)
    man = _manifest(tmp_path)
    assert json.loads((tmp_path / "MANIFEST.json").read_text()) == man
    assert MF.validate_manifest(man) == []


def test_validate_manifest_names_a_bad_build_id(tmp_path, monkeypatch):
    monkeypatch.setattr(MF, "_GIT_SHA_OVERRIDE", "c" * 40)
    man = _manifest(tmp_path)
    man["build_id"] = "nope"
    assert any("build_id" in e for e in MF.validate_manifest(man))


def test_state_files_round_trip_and_write_the_resume_command(tmp_path):
    assert MF.load_state(tmp_path)["phase"] == "kv"
    st = MF.load_state(tmp_path)
    st["phase"] = "base"
    st["base_processed"] = 5
    MF.save_state(tmp_path, st, "run again")
    assert MF.load_state(tmp_path)["phase"] == "base"
    md = (tmp_path / "STATE.md").read_text()
    assert "run again" in md and f"{MF.TWIN_BASE_COUNT:,}" in md


def test_scroll_collection_pages_until_the_offset_ends():
    class C:
        def __init__(self):
            self.pages = [([1, 2], "o"), ([3], None)]

        def scroll(self, coll, **kw):
            return self.pages.pop(0)

    assert list(MF.scroll_collection(C(), "c", None)) == [1, 2, 3]
    assert list(MF.scroll_collection(C(), "c", 2)) == [1, 2]


def test_module_has_no_host_or_path_literals():
    import pathlib
    text = pathlib.Path(MF.__file__).read_text()
    assert "localhost" not in text and "/home/" not in text and "sys.path" not in text


def test_real_jsonschema_runs_and_accepts_a_real_manifest(tmp_path, monkeypatch):
    """Task 2117: the strict draft-07 check must RUN (jsonschema < 4.20 raised
    TypeError on Python 3.14) and accept a good manifest."""
    import jsonschema
    monkeypatch.setattr(MF, "_GIT_SHA_OVERRIDE", "c" * 40)
    man = _manifest(tmp_path)
    jsonschema.validate(man, MF._MANIFEST_SCHEMA)
    assert MF.validate_manifest(man) == []


def test_strict_schema_refuses_what_the_hand_checks_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(MF, "_GIT_SHA_OVERRIDE", "c" * 40)
    man = _manifest(tmp_path)
    man["surprise"] = 1
    errs = MF.validate_manifest(man)
    assert any(e.startswith("jsonschema:") for e in errs)
    man2 = _manifest(tmp_path)
    man2["counts"]["adds"] = -1
    assert any(e.startswith("jsonschema:") for e in MF.validate_manifest(man2))


def test_strict_schema_keeps_the_legend_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(MF, "_GIT_SHA_OVERRIDE", "c" * 40)
    man = _manifest(tmp_path)
    man["rule_legend"]["not_a_family"] = "x@1"
    assert any(e.startswith("jsonschema:") for e in MF.validate_manifest(man))

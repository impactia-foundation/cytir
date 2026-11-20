"""The b1 builder's run modes after the move into `cytir.index.b1_rebuild` (task 2114).

Offline: no Qdrant, no network. The fakes record calls; nothing here opens a client."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from cytir.index import b1_rebuild as RB


def test_deferred_tripwire_fires_on_a_total_exclusion_and_is_quiet_on_a_sane_rate():
    with pytest.raises(SystemExit):
        RB.check_deferred_tripwire(100_000, 100_000)
    RB.check_deferred_tripwire(100_000, 2_000)
    RB.check_deferred_tripwire(100, 100)  # under the floor: no verdict


def test_account_exclusion_counts_each_class_once():
    st: dict = {}
    assert RB._account_exclusion(st, "defer_unknown") == "deferred"
    assert RB._account_exclusion(st, "undated_gov") == "undated_gov"
    assert RB._account_exclusion(st, "consent_declined") == "declined"
    assert RB._account_exclusion(st, "other") == "drop"
    assert (st["deferred_count"], st["staged_excluded_undated_gov"], st["consent_declined"],
            st["consent_excluded"], st["drops"]) == (1, 1, 1, 1, 1)


def test_consent_census_buckets_the_cache(tmp_path):
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({
        "a.ch": {"status": "ok", "search_ok": True},
        "b.ch": {"status": "ok", "search_ok": False},
        "c.ch": {"status": "error", "reason": "unreachable"},
        "d.ch": {"status": "ok", "search_ok": None},
    }))
    assert RB._consent_census(str(cache)) == {
        "search_ok_true": 1, "declined": 1, "unreachable_retryable": 1, "other": 1,
        "total_hosts": 4}
    assert RB._consent_census(str(tmp_path / "missing.json")) == {}


def test_load_delete_set_fails_loud_on_a_missing_or_empty_list(tmp_path):
    with pytest.raises(SystemExit):
        RB.load_delete_set(str(tmp_path / "none.txt"), print)
    empty = tmp_path / "empty.txt"
    empty.write_text("\n")
    with pytest.raises(SystemExit):
        RB.load_delete_set(str(empty), print)
    ids = tmp_path / "ids.txt"
    ids.write_text("a\nb\n\n")
    assert RB.load_delete_set(str(ids), lambda *_: None) == {"a", "b"}
    assert RB.load_delete_set("", lambda *_: None) == set()


class FakeBuild:
    def __init__(self):
        self.upserts = []

    def upsert(self, coll, points, wait):
        self.upserts.append((coll, points, wait))

    def count(self, coll, exact):
        return SimpleNamespace(count=7)


def test_flush_upserts_the_batch_after_the_dense_reembed_hook(monkeypatch):
    seen = []
    monkeypatch.setattr(RB, "_reembed_explicit_dense", lambda b: seen.append(list(b)) or 0)
    monkeypatch.setattr(RB, "qm", SimpleNamespace(PointStruct=lambda **kw: kw))  # a sibling test may leave a stub
    build = FakeBuild()
    point = {"id": "1", "vector": {"dense": [0.0]}, "payload": {"url": "u"}}
    RB._flush(build, "aw_b1_t", [point])
    assert seen == [[point]]
    assert len(build.upserts) == 1 and build.upserts[0][0] == "aw_b1_t" and build.upserts[0][2] is True


def test_reembed_leaves_a_batch_without_explicit_text_alone():
    point = {"id": "1", "vector": {"dense": [0.5]}}
    assert RB._reembed_explicit_dense([point]) == 0
    assert point["vector"]["dense"] == [0.5]


def test_run_full_manifest_writes_the_manifest_and_the_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(RB, "validate_manifest", lambda man: [])  # schema checks: the manifest module's tests
    out = tmp_path / "out"
    out.mkdir()
    st = {"base_added": 5, "deferred_added": 1, "deletes": 2,
          "staged_per_pack": {"aw_dzefix_cov": {"added": 1}}}
    args = SimpleNamespace(collection="aw_b1_t", consent_cache=str(tmp_path / "none.json"))
    man, errs = RB.run_full_manifest(FakeBuild(), st, str(out), args, lambda *_: None)
    assert st["phase"] == "done"
    assert st["counts"] == {"points": 7, "adds": 9, "changes": 0, "deletes": 2}
    assert (out / "aw_b1.MANIFEST.json").exists()
    assert man["exceptions"]["romansh_evicted"] == 2
    assert errs == []


def test_run_full_takes_corpus_dir_and_resume_prefix_by_keyword():
    import inspect
    sig = inspect.signature(RB.run_full)
    assert sig.parameters["corpus_dir"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["resume_prefix"].kind is inspect.Parameter.KEYWORD_ONLY

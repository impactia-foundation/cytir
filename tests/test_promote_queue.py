#!/usr/bin/env python3
"""Real-record guards for `promote_queue` (task 641).

Reuses the exact fixture `test_promote_sealed_pack.py` uses --
`tests/fixtures/promote/task629_sealed_pack/` -- because the whole point of a
promotion driver is that it handles the artefact shape this project really
produces, and a hand-built fixture would agree with a hand-built reader
while both were wrong.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import pathlib
import sys
import time

import pytest

from tests.helpers.repo_files import repo_glob_or_skip

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

@pytest.fixture(autouse=True)
def _operator_identity(monkeypatch):
    """Operator identity for each test, restored after it. A module-level setdefault leaked it into every later test."""
    monkeypatch.setenv("AW_OPERATOR_URL", "https://websear.ch")
    monkeypatch.setenv("AW_OPERATOR_CONTACT", "bender@impactia.org")

from cytir.index import promote_queue as Q  # noqa: E402
from cytir.index import promote_sealed_pack as PSP  # noqa: E402

FIXTURE = HERE / "fixtures" / "promote" / "task629_sealed_pack"
CHUNKS = str(FIXTURE / "chunks")
VECTORS = str(FIXTURE / "vectors")


def _nightly_script() -> str:
    """Text of the nightly wrapper in the deploy tree; SKIPS when the tree is absent (the public cut)."""
    return repo_glob_or_skip("deploy/systemd/*/promote_nightly.sh")[0].read_text()


@pytest.fixture(autouse=True)
def _operator_values(monkeypatch, tmp_path_factory):
    """The values the operator overlay supplies on a configured box; a stranger's suite runs with
    none, so this module pins neutral stand-ins (it asserts behaviour, not the box's real values)."""
    monkeypatch.setenv("AW_PROD_LOCK_HOST", "prod-lock-host")
    monkeypatch.setenv("AW_TWIN_SSH_HOST", "twin-ssh-host")
    monkeypatch.setenv("AW_WORK_ROOT", str(tmp_path_factory.getbasetemp() / "work-root"))

MANIFEST_BASE = {
    "schema": Q.MANIFEST_SCHEMA,
    "pack_id": "test-pack-001",
    "family": "technical_manufacturer_catalog",
    "pack_chunks": CHUNKS,
    "vectors": VECTORS,
    "mrl": 256,
    "collection": "aw_e8",
    "identity_check": "pack-declared",
    "identity_reason": "test: Pool A ids are not reproducible; module docstring",
    "family_payload": {"source_type": "docs", "trust_tier": 0.6},
    "title_from": ["source_metadata.title"],
}


def write_pack(dir_: pathlib.Path, manifest: dict, *, ready: bool = True) -> pathlib.Path:
    pack_dir = dir_ / manifest["pack_id"]
    pack_dir.mkdir(parents=True)
    manifest_path = pack_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1))
    if ready:
        sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        (pack_dir / "READY").write_text(sha + "\n")
    return pack_dir


@pytest.fixture(autouse=True)
def _default_prod_alias_resolution(monkeypatch):
    """Task 673 review round 3, finding 6/FLIP BLOCKER: `drain()` now
    unconditionally resolves prod's live collection through the `aw_prod`
    alias (`resolve_prod_alias`, `GET /aliases` via `PSP._api`) before
    touching any pack. Every test in this file predates that -- default
    the `/aliases` response to `aw_prod -> aw_e8` (matching
    MANIFEST_BASE's own declared collection, i.e. IDENTICAL to the
    pre-fix behaviour) so none of them need retrofitting, through the
    REAL `resolve_prod_alias` implementation."""
    def fake_api(url, path, body=None, method=None):
        if path == "/aliases":
            return {"aliases": [
                {"alias_name": "aw_prod", "collection_name": "aw_e8"}]}
        raise AssertionError(f"unstubbed PSP._api call: {path}")
    monkeypatch.setattr(PSP, "_api", fake_api)


def write_contract(dir_: pathlib.Path, profile: str) -> pathlib.Path:
    path = dir_ / "gate-contract.json"
    path.write_text(json.dumps({
        "schema": Q.GATE_CONTRACT_SCHEMA, "adr": "ADR-110", "pr": 302,
        "profile": profile,
    }))
    return path


# ---------------------------------------------------------------- manifest IO
# ------------------------------------------------------- atomic_write_*
def test_atomic_write_text_leaves_the_old_file_intact_on_a_mid_write_kill(
        tmp_path, monkeypatch):
    """Review finding 3, task 673: a plain `write_text` truncates the
    target in place, so a kill mid-write leaves broken JSON. The atomic
    writer must write to a temp file and rename over the target -- if the
    write itself dies, the ORIGINAL file (or no file, on first write) must
    be exactly what it was before, never a truncated new one."""
    path = tmp_path / "state.json"
    path.write_text("original content\n")

    real_fsync = os.fsync
    call_count = {"n": 0}

    def dying_fsync(fd):
        call_count["n"] += 1
        raise OSError("simulated kill mid-fsync")
    monkeypatch.setattr(Q.os, "fsync", dying_fsync)

    with pytest.raises(OSError, match="simulated kill mid-fsync"):
        Q.atomic_write_text(path, "new content\n")

    assert call_count["n"] == 1
    assert path.read_text() == "original content\n"  # untouched
    # no leftover temp file
    assert list(tmp_path.glob(".state.json.tmp*")) == []


def test_atomic_write_text_roundtrips_real_content(tmp_path):
    path = tmp_path / "nested" / "state.json"  # parent doesn't exist yet
    Q.atomic_write_text(path, '{"a": 1}\n')
    assert path.read_text() == '{"a": 1}\n'


def test_read_manifest_ok(tmp_path):
    pack_dir = write_pack(tmp_path, MANIFEST_BASE)
    manifest = Q.read_manifest(pack_dir)
    assert manifest["pack_id"] == "test-pack-001"


def test_ready_sha_mismatch_refuses(tmp_path):
    pack_dir = write_pack(tmp_path, MANIFEST_BASE)
    (pack_dir / "manifest.json").write_text(
        json.dumps({**MANIFEST_BASE, "family": "edited-after-ready"}, indent=1))
    with pytest.raises(Q.QueueError, match="edited after being marked ready"):
        Q.read_manifest(pack_dir)


def test_missing_ready_refuses(tmp_path):
    pack_dir = write_pack(tmp_path, MANIFEST_BASE, ready=False)
    with pytest.raises(Q.QueueError, match="not queued yet"):
        Q.read_manifest(pack_dir)


def test_wrong_schema_refuses(tmp_path):
    pack_dir = write_pack(tmp_path, {**MANIFEST_BASE, "schema": "not-the-schema"})
    with pytest.raises(Q.QueueError, match="schema"):
        Q.read_manifest(pack_dir)


def test_missing_required_field_refuses(tmp_path):
    bad = dict(MANIFEST_BASE)
    del bad["collection"]
    pack_dir = write_pack(tmp_path, bad)
    with pytest.raises(Q.QueueError, match="collection"):
        Q.read_manifest(pack_dir)


def test_read_manifest_accepts_a_genuine_rows_zero_content_pack(tmp_path):
    """Bug found live, task 660, 2026-09-25: `not manifest.get('rows')`
    treated a legitimate `rows: 0` (task 653's pure-tombstone/takedown
    pack -- zero new points, only deletions) the same as the field being
    absent, refusing at intake forever with 'manifest missing [rows]'.
    Present-but-zero must not be confused with missing."""
    manifest = {
        "schema": Q.MANIFEST_SCHEMA, "kind": "content_pack",
        "pack_id": "tombstone-only", "family": "l1-backfill",
        "pack_points": "/does/not/need/to/exist/for/this/check.jsonl.gz",
        "rows": 0, "collection": "aw_e8",
    }
    pack_dir = write_pack(tmp_path, manifest)
    read = Q.read_manifest(pack_dir)
    assert read["rows"] == 0


# ------------------------------------------------------------ smoke test
def test_real_path_smoke_test_passes_on_the_real_fixture():
    assert Q.assert_real_path(MANIFEST_BASE) == 1


def test_real_path_smoke_test_refuses_zero_passages(tmp_path):
    empty_chunks = tmp_path / "empty_chunks"
    empty_chunks.mkdir()
    (empty_chunks / "shard-0000.texts.jsonl.gz").write_bytes(
        __import__("gzip").compress(b""))
    manifest = {**MANIFEST_BASE, "pack_chunks": str(empty_chunks)}
    with pytest.raises(Q.QueueError, match="0 passages"):
        Q.assert_real_path(manifest)


def test_real_path_smoke_test_accepts_a_genuine_tombstone_only_content_pack(
        tmp_path):
    """The real l1-backfill-declined-takedown shape, task 653/ADR-103
    amendment 1: 0 new points (a declined publisher's pages carry no
    replacement row) but a real tombstones_count -- this is a deliberate
    pure-delete pack, not the silent-yield defect the smoke test guards
    against, and must pass."""
    empty_points = tmp_path / "empty.jsonl.gz"
    empty_points.write_bytes(__import__("gzip").compress(b""))
    manifest = {"kind": "content_pack", "pack_id": "tombstone-only",
               "pack_points": str(empty_points), "tombstones_count": 10}
    assert Q.assert_real_path(manifest) == 1


def test_real_path_smoke_test_refuses_a_content_pack_with_nothing_at_all(
        tmp_path):
    """0 points and 0 (or no) tombstones has no possible effect on the
    index -- still the genuine silent-yield defect, still refused."""
    empty_points = tmp_path / "empty.jsonl.gz"
    empty_points.write_bytes(__import__("gzip").compress(b""))
    manifest = {"kind": "content_pack", "pack_id": "truly-empty",
               "pack_points": str(empty_points)}
    with pytest.raises(Q.QueueError, match="0 points"):
        Q.assert_real_path(manifest)


# ------------------------------------------------------------- gate switch
def test_gate_profile_reads_the_tracked_field(tmp_path):
    contract = write_contract(tmp_path, "battery-required")
    gate = Q.gate_profile(contract)
    assert gate["profile"] == "battery-required"
    assert gate["adr"] == "ADR-110"
    assert gate["pr"] == 302


def test_gate_profile_mechanical_once_flipped(tmp_path):
    contract = write_contract(tmp_path, "mechanical")
    assert Q.gate_profile(contract)["profile"] == "mechanical"


def test_gate_profile_refuses_an_unknown_value(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": Q.GATE_CONTRACT_SCHEMA,
                                "profile": "whatever-i-feel-like"}))
    with pytest.raises(Q.QueueError, match="not one of"):
        Q.gate_profile(path)


def test_gate_profile_refuses_a_missing_file(tmp_path):
    with pytest.raises(Q.QueueError, match="no gate contract"):
        Q.gate_profile(tmp_path / "does-not-exist.json")


def test_the_real_shipped_contract_is_a_known_profile_citing_adr110():
    """The file this repo actually ships must always be one of the two
    named profiles, citing ADR-110/#302 -- never a stray third value from
    an unreviewed edit. FLIPPED to 'mechanical' 2026-09-23: Tim merged
    PR #302 at 06:09Z."""
    gate = Q.gate_profile(Q.GATE_CONTRACT_PATH)
    assert gate["profile"] in ("battery-required", "mechanical")
    assert gate["adr"] == "ADR-110"
    assert gate["pr"] == 302


# --------------------------------------------------- mechanical content check
def _real_joined(monkeypatch):
    """The real B01 fixture, joined exactly as plan() would (dry, no writes)."""
    _stub_plan_deps(monkeypatch)
    receipt = PSP.plan(chunks_dir=MANIFEST_BASE["pack_chunks"],
                      vectors_dir=MANIFEST_BASE["vectors"], qdrant="http://x",
                      collection="aw_e8", mrl=256,
                      **Q.plan_kwargs(MANIFEST_BASE))
    return receipt["_joined"]


def test_mechanical_content_rows_merges_dense_text_onto_the_payload(monkeypatch):
    joined = _real_joined(monkeypatch)
    rows = Q.mechanical_content_rows(joined)
    assert len(rows) == 3
    for row in rows:
        assert row["canonical_text"]
        assert row["point_id"] in joined


def _force_lang_gate_unavailable(monkeypatch):
    """Deterministic, not ambient (bit twice, 2026-09-23): another test
    module in the full suite sets AW_LANGGATE_OPTIONAL=1 without restoring
    it, which degrades fetch_process.detect_lang() to a URL-heuristic
    fallback (confident=False) instead of raising -- and since
    mechanical_content_check now scores a confident=False detection on
    non-empty text as a defect (orchestrator ruling, 2026-09-23), that
    leak turns a PASS fixture into a REFUSE depending on run order. Tests
    that don't care about language detection must force the SKIPPED path
    explicitly rather than rely on whatever venv/env happens to be live."""
    from cytir.acquire import fetch_process as FP  # noqa: PLC0415

    def _raise(*_a, **_k):
        raise FP.LangGateInactive("simulated: no lang-gate in this test")
    monkeypatch.setattr(FP, "detect_lang", _raise)


def test_mechanical_content_status_passes_on_the_real_clean_fixture(monkeypatch):
    _force_lang_gate_unavailable(monkeypatch)
    joined = _real_joined(monkeypatch)
    status = Q.mechanical_content_status(MANIFEST_BASE, joined)
    assert status["verdict"] == "PASS"
    assert status["failing_rows"] == 0


def test_mechanical_content_status_forwards_manifest_overrides(monkeypatch):
    joined = _real_joined(monkeypatch)
    manifest = {**MANIFEST_BASE, "word_floor": 10_000, "detect_language": False}
    status = Q.mechanical_content_status(manifest, joined)
    # every real row is far under a 10,000-word floor -> all 3 fail
    assert status["failing_rows"] == 3


# ------------------------------------------------------ persisted-index refresh
def _corpus_layout(root: pathlib.Path) -> pathlib.Path:
    for name in Q.PERSISTED_INDEX_ROOTS:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def test_persisted_index_stale_when_no_index_exists(tmp_path):
    corpus = _corpus_layout(tmp_path / "corpus")
    stale, reason = Q.persisted_index_is_stale(str(tmp_path / "no-such-index"),
                                               corpus_root=str(corpus))
    assert stale
    assert "does not exist" in reason


def test_persisted_index_fresh_when_nothing_changed_since_build(tmp_path):
    corpus = _corpus_layout(tmp_path / "corpus")
    (corpus / "l1" / "old.jsonl.gz").write_bytes(b"x")
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "INDEX.json").write_text("{}")
    # The index file itself is newer than the corpus file above (written
    # after it) -- exactly the "nothing changed since the build" case.
    stale, reason = Q.persisted_index_is_stale(str(index_dir),
                                               corpus_root=str(corpus))
    assert not stale
    assert "no file" in reason


def test_persisted_index_stale_when_a_store_file_is_newer(tmp_path):
    corpus = _corpus_layout(tmp_path / "corpus")
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    index_path = index_dir / "INDEX.json"
    index_path.write_text("{}")
    # A new L1 file lands AFTER the index was built -- the real LEGI/SE
    # shape (task 641, 2026-09-23): pages sealed after the last build.
    # mtimes forced 10s apart -- some filesystems round to whole seconds,
    # and this test must not depend on real wall-clock timing to pass.
    index_mtime = index_path.stat().st_mtime
    new_file = corpus / "l1" / "release=new" / "fresh.jsonl.gz"
    new_file.parent.mkdir(parents=True, exist_ok=True)
    new_file.write_bytes(b"x")
    os.utime(new_file, (index_mtime + 10, index_mtime + 10))
    stale, reason = Q.persisted_index_is_stale(str(index_dir),
                                               corpus_root=str(corpus))
    assert stale
    # `find -newer ... -quit` may report the containing directory (its own
    # mtime also moved when the file was created inside it) rather than the
    # leaf file -- either is correct evidence of staleness, so this checks
    # for the l1 root, not a specific leaf name.
    assert "l1" in reason


def test_refresh_persisted_index_merges_via_sqlite_and_reports_no_change(
        tmp_path, monkeypatch):
    """Task 664, 2026-09-26: refresh_persisted_index no longer stale-checks
    then full-rebuilds -- it always merges through the sqlite incremental
    updater, which itself reports whether anything actually changed."""
    index_dir = tmp_path / "index"
    calls = []

    def fake_update(d, **kw):
        calls.append(d)
        return {"files_scanned": 5, "files_changed": 0, "rows_indexed": 0,
               "seconds": 0.01}
    monkeypatch.setattr(Q.PI, "update_index_sqlite", fake_update)

    result = Q.refresh_persisted_index(str(index_dir))
    assert calls == [str(index_dir)]
    assert result["rebuilt"] is False
    assert "no store file changed" in result["reason"]


def test_refresh_persisted_index_reports_a_real_merge(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"

    def fake_update(d, **kw):
        return {"files_scanned": 5, "files_changed": 2, "rows_indexed": 4321,
               "seconds": 0.5}
    monkeypatch.setattr(Q.PI, "update_index_sqlite", fake_update)

    result = Q.refresh_persisted_index(str(index_dir))
    assert result["rebuilt"] is True
    assert "2 file(s) changed" in result["reason"]
    assert "4,321 row(s)" in result["reason"]


def test_refresh_persisted_index_full_wipes_the_sqlite_file_first(
        tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    db_path = pathlib.Path(Q.PI.sqlite_index_path(str(index_dir)))
    db_path.write_bytes(b"stale db bytes")
    reset_calls = []
    monkeypatch.setattr(Q.PI, "reset_index_cache",
                        lambda: reset_calls.append(1))
    monkeypatch.setattr(Q.PI, "update_index_sqlite",
                        lambda d, **kw: {"files_scanned": 0, "files_changed": 0,
                                        "rows_indexed": 0, "seconds": 0.0})

    Q.refresh_persisted_index(str(index_dir), full=True)
    assert not db_path.exists()
    assert reset_calls == [1]


def test_refresh_persisted_index_without_full_leaves_the_sqlite_file_alone(
        tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    db_path = pathlib.Path(Q.PI.sqlite_index_path(str(index_dir)))
    db_path.write_bytes(b"existing db bytes")
    monkeypatch.setattr(Q.PI, "update_index_sqlite",
                        lambda d, **kw: {"files_scanned": 0, "files_changed": 0,
                                        "rows_indexed": 0, "seconds": 0.0})

    Q.refresh_persisted_index(str(index_dir))
    assert db_path.read_bytes() == b"existing db bytes"


def test_drain_refreshes_the_index_by_default(tmp_path, monkeypatch):
    _stub_plan_deps(monkeypatch)
    calls = []
    monkeypatch.setattr(Q, "refresh_persisted_index",
                        lambda *a, **k: calls.append(1) or
                        {"rebuilt": False, "reason": "test stub"})
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    write_pack(queue_dir, MANIFEST_BASE)
    contract = write_contract(tmp_path, "battery-required")
    Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
           task="641-test", apply=True, batch=10, pause_s=0,
           artifact_root=tmp_path / "artifacts", contract_path=contract)
    assert calls == [1]


def test_drain_can_skip_the_refresh(tmp_path, monkeypatch):
    _stub_plan_deps(monkeypatch)
    calls = []
    monkeypatch.setattr(Q, "refresh_persisted_index", lambda *a, **k: calls.append(1))
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    write_pack(queue_dir, MANIFEST_BASE)
    contract = write_contract(tmp_path, "battery-required")
    Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
           task="641-test", apply=True, batch=10, pause_s=0,
           artifact_root=tmp_path / "artifacts", contract_path=contract,
           refresh_index=False)
    assert calls == []


# --------------------------------------------------------------- dry_plan
def _fixture_ids() -> list[str]:
    """The point ids the real fixture projects (identity `pack-declared`: the pack's own ids)."""
    return sorted(str(r["point_id"]) for r in PSP.read_records(CHUNKS))


def _stub_plan_deps(monkeypatch):
    monkeypatch.setattr(PSP, "target_width", lambda *_a, **_k: 256)
    monkeypatch.setattr(PSP, "already_present", lambda *_a, **_k: [])
    monkeypatch.setattr(PSP, "already_served_urls", lambda *_a, **_k: [])
    monkeypatch.setattr(PSP.PI, "assert_pages_persisted",
                        lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(PSP.PC, "assert_scoreable",
                        lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(PSP.LAYOUT, "fingerprint", lambda *_a, **_k: {})


def test_dry_plan_clears_on_the_real_fixture(monkeypatch):
    _stub_plan_deps(monkeypatch)
    receipt = Q.dry_plan(MANIFEST_BASE, "http://x")
    assert receipt["rows"]["projected"] == 3
    # Unlike promote_sealed_pack.main()'s own use of plan(), dry_plan() here
    # KEEPS _joined -- drain() needs it for the mechanical-content-v1 check.
    assert len(receipt["_joined"]) == 3
    assert "_matrix" in receipt and "_id_index" in receipt


def test_dry_plan_without_language_scope_filter_is_unchanged(monkeypatch):
    """No regression: the manifest field is OPT-IN (orchestrator ruling,
    2026-09-23) -- omitting it must not run a second detection pass or
    change plan()'s exclude_ids at all."""
    _stub_plan_deps(monkeypatch)
    calls = []
    monkeypatch.setattr(Q.MCC, "language_scope_filter",
                        lambda *a, **k: calls.append(1) or {"out_of_scope_ids": set()})
    receipt = Q.dry_plan(MANIFEST_BASE, "http://x")
    assert calls == []
    assert "language_scope_filter" not in receipt
    assert len(receipt["_joined"]) == 3


def test_dry_plan_language_scope_filter_excludes_before_payload_carry(monkeypatch):
    """Opted in: the real B01 fixture's 3 rows, one forced out-of-scope by a
    stubbed filter -- plan() must receive exactly that exclude set, and the
    dropped row's own url must be reported."""
    _stub_plan_deps(monkeypatch)
    raw = list(PSP.read_records(MANIFEST_BASE["pack_chunks"]))
    excluded_id = raw[0]["point_id"]

    def fake_filter(rows, **_k):
        rows = list(rows)
        assert len(rows) == 3  # ran on the RAW rows, not the projected ones
        return {"rows_checked": 3, "out_of_scope_ids": {excluded_id},
               "dropped_by_language": {"es": 1}, "gate_unavailable_rows": 0,
               "in_scope_langs": ["de", "en", "fr", "it"]}
    monkeypatch.setattr(Q.MCC, "language_scope_filter", fake_filter)

    manifest = {**MANIFEST_BASE, "language_scope_filter": True}
    receipt = Q.dry_plan(manifest, "http://x")
    assert excluded_id not in receipt["_joined"]
    assert len(receipt["_joined"]) == 2
    assert receipt["rows"]["excluded_out_of_scope_language"] == 1
    report = receipt["language_scope_filter"]
    assert report["dropped_by_language"] == {"es": 1}
    assert report["out_of_scope_ids_count"] == 1
    assert "out_of_scope_ids" not in report  # the raw id set stays internal


# --------------------------------------------------- lang_presence_waiver
# --------------------------------------------------- (Tim, 2026-09-24, task 652)
WAIVER = {"issuer": "Tim", "at": "2026-09-24T05:15Z",
         "scope": "cc-authority-frontier-w7-tranche1 only",
         "rule": "lang presence floor 99%"}


def test_dry_plan_without_lang_presence_waiver_is_unchanged(monkeypatch):
    """OPT-IN, same discipline as language_scope_filter: omitting the
    manifest field must not run lang_drop_detail or touch exclude_ids."""
    _stub_plan_deps(monkeypatch)
    calls = []
    monkeypatch.setattr(Q.MCC, "lang_drop_detail",
                        lambda *a, **k: calls.append(1) or [])
    receipt = Q.dry_plan(MANIFEST_BASE, "http://x")
    assert calls == []
    assert "waiver" not in receipt
    assert len(receipt["_joined"]) == 3


def test_dry_plan_lang_presence_waiver_excludes_and_records_detail(monkeypatch):
    """Opted in: a stubbed lang_drop_detail names one row -- plan() must
    receive it in exclude_ids (reason `no_confident_lang`) and the full
    per-row detail must ride in the receipt's `waiver` block, alongside
    Tim's own waiver fields verbatim."""
    _stub_plan_deps(monkeypatch)
    raw = list(PSP.read_records(MANIFEST_BASE["pack_chunks"]))
    dropped_id = raw[1]["point_id"]

    def fake_detail(rows, **_k):
        rows = list(rows)
        assert len(rows) == 3  # ran on the RAW rows
        return [{"point_id": dropped_id, "url": raw[1]["url"],
                "text_chars": 42, "detected": "en", "confident": False,
                "raw_label": "en", "raw_prob": 0.5, "gate_unavailable": False}]
    monkeypatch.setattr(Q.MCC, "lang_drop_detail", fake_detail)

    manifest = {**MANIFEST_BASE, "lang_presence_waiver": WAIVER}
    receipt = Q.dry_plan(manifest, "http://x")
    assert dropped_id not in receipt["_joined"]
    assert len(receipt["_joined"]) == 2
    waiver = receipt["waiver"]
    assert waiver["issuer"] == "Tim"
    assert waiver["at"] == "2026-09-24T05:15Z"
    assert waiver["dropped_count"] == 1
    assert waiver["dropped_rows"][0]["point_id"] == dropped_id
    dropped_rows = receipt["out_of_scope_language_dropped"]
    reasons = {d["point_id"]: d["reason"] for d in dropped_rows}
    assert reasons[dropped_id] == "no_confident_lang"


def test_dry_plan_out_of_scope_reason_wins_over_no_confident_lang(monkeypatch):
    """A row both filters name (out-of-scope AND lang-empty, the real CC
    tranche 1 shape for the 26 out-of-scope rows) keeps the MORE specific
    `out_of_scope_language` reason, not the waiver's generic one."""
    _stub_plan_deps(monkeypatch)
    raw = list(PSP.read_records(MANIFEST_BASE["pack_chunks"]))
    shared_id = raw[0]["point_id"]
    monkeypatch.setattr(Q.MCC, "language_scope_filter",
                        lambda rows, **k: {"rows_checked": 3,
                                          "out_of_scope_ids": {shared_id},
                                          "dropped_by_language": {"es": 1},
                                          "gate_unavailable_rows": 0,
                                          "in_scope_langs": ["de", "en", "fr", "it"]})
    monkeypatch.setattr(Q.MCC, "lang_drop_detail",
                        lambda rows, **k: [{"point_id": shared_id,
                                           "url": raw[0]["url"], "text_chars": 5,
                                           "detected": "es", "confident": True,
                                           "raw_label": "es", "raw_prob": 0.9,
                                           "gate_unavailable": False}])
    manifest = {**MANIFEST_BASE, "language_scope_filter": True,
               "lang_presence_waiver": WAIVER}
    receipt = Q.dry_plan(manifest, "http://x")
    reasons = {d["point_id"]: d["reason"]
              for d in receipt["out_of_scope_language_dropped"]}
    assert reasons[shared_id] == "out_of_scope_language"
    assert receipt["rows"]["excluded_out_of_scope_language"] == 1  # not 2


# ----------------------------------------------------------------- drain
def test_drain_proves_and_blocks_when_battery_required(tmp_path, monkeypatch):
    _stub_plan_deps(monkeypatch)
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    write_pack(queue_dir, MANIFEST_BASE)
    contract = write_contract(tmp_path, "battery-required")

    results = Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
                      task="641-test", apply=True, batch=10, pause_s=0,
                      artifact_root=tmp_path / "artifacts",
                      contract_path=contract, refresh_index=False)
    assert len(results) == 1
    rec = results[0]
    assert rec["status"] == "PROVED-BLOCKED"
    assert any("battery-required" in b for b in rec["blocked_on"])
    # battery-required blocks before the mechanical-content-v1 check ever
    # runs -- no point scoring rows for a pack about to be refused anyway.
    assert "mechanical_content" not in rec
    # A receipt was written beside the pack for a human to read later.
    assert (queue_dir / "test-pack-001" / "receipt.json").exists()
    # Never wrote anything: no DONE marker.
    assert not (queue_dir / "test-pack-001" / "DONE").exists()


def test_drain_without_apply_never_writes_even_if_mechanical_and_content_clean(
        tmp_path, monkeypatch):
    _stub_plan_deps(monkeypatch)
    _force_lang_gate_unavailable(monkeypatch)
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    write_pack(queue_dir, dict(MANIFEST_BASE))
    contract = write_contract(tmp_path, "mechanical")

    results = Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
                      task="641-test", apply=False, batch=10, pause_s=0,
                      artifact_root=tmp_path / "artifacts",
                      contract_path=contract, refresh_index=False)
    assert results[0]["status"] == "PROVED-DRY-RUN"
    assert results[0]["mechanical_content"]["verdict"] == "PASS"
    assert not (queue_dir / "test-pack-001" / "DONE").exists()


def _german_blind_drain(tmp_path, monkeypatch, guard_mode, *, none_text=False, guard_raises=None):
    """The real B01 rows made German, a writer with no splitter (the interpreter of task 710's
    drain), then one dry drain. Returns the receipt of the pack."""
    from cytir.core import paths
    _stub_plan_deps(monkeypatch)
    _force_lang_gate_unavailable(monkeypatch)
    real_plan = PSP.plan

    def german_plan(**kw):
        receipt = real_plan(**kw)
        for _d, _s, payload, _r in receipt["_joined"].values():
            payload["lang"] = "de"
        if none_text:       # a sealed row that was not predicted: sparse_text None (a real TypeError)
            first_id = next(iter(receipt["_joined"]))
            d, _s, payload, r = receipt["_joined"][first_id]
            receipt["_joined"][first_id] = (d, None, payload, r)
        return receipt
    monkeypatch.setattr(PSP, "plan", german_plan)
    decompound = importlib.import_module("cytir.core.decompound")

    def _no_splitter():
        raise decompound.DecompoundUnavailable("simulated: no splitter in this process")
    monkeypatch.setattr(decompound, "_splitter", None)       # a writer whose rule cannot load
    monkeypatch.setattr(decompound, "_build", _no_splitter)
    if guard_raises is not None:
        def boom(*a, **k):
            raise guard_raises
        monkeypatch.setattr(Q.GPG, "check_joined", boom)
    monkeypatch.setenv(paths.TEST_LOOKUP_ENV, "")
    monkeypatch.setenv("AW_GERMAN_PARTS_GUARD", guard_mode)
    paths.reset_settings_cache()
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    write_pack(queue_dir, dict(MANIFEST_BASE))
    contract = write_contract(tmp_path, "mechanical")
    kw = dict(twin_url="http://x", prod_url="http://y", task="710-test", apply=False, batch=10,
              pause_s=0, artifact_root=tmp_path / "artifacts", contract_path=contract,
              refresh_index=False)
    first = Q.drain(queue_dir, **kw)[0]
    second = Q.drain(queue_dir, **kw)[0]
    paths.reset_settings_cache()
    return first, second


def test_drain_german_parts_guard_warns_by_default_and_changes_no_verdict(tmp_path, monkeypatch):
    rec, _again = _german_blind_drain(tmp_path, monkeypatch, "warn")
    assert rec["status"] == "PROVED-DRY-RUN"                    # same verdict as without the guard
    assert rec["german_parts"]["verdict"] == "FAIL" and rec["german_parts"]["german_rows"] == 3


def test_drain_german_parts_guard_refuses_when_set_and_never_caches_the_refusal(tmp_path, monkeypatch):
    rec, again = _german_blind_drain(tmp_path, monkeypatch, "refuse")
    assert rec["status"] == "REFUSED" and rec["stage"] == "german_parts"
    assert rec["reason"].startswith("german_parts: only 0 of 3 German rows")
    assert "german_parts" not in Q._CACHEABLE_REFUSAL_STAGES
    assert again["status"] == "REFUSED" and again["stage"] == "german_parts"   # re-judged, not skipped


@pytest.mark.parametrize("guard_mode", ["warn", "refuse"])
def test_drain_sealed_row_with_no_sparse_text_is_not_predicted_and_stops_nothing(
        tmp_path, monkeypatch, guard_mode):
    rec, _again = _german_blind_drain(tmp_path, monkeypatch, guard_mode, none_text=True)
    assert rec["german_parts"].get("result") != "error" and rec["german_parts"]["no_text_rows"] == 1
    # the two remaining German rows are still judged: blind writer, so `refuse` refuses on THEM
    assert rec["status"] == ("REFUSED" if guard_mode == "refuse" else "PROVED-DRY-RUN")


@pytest.mark.parametrize("guard_mode", ["warn", "refuse"])
def test_drain_a_raising_guard_is_a_receipt_block_and_refuses_only_in_refuse(
        tmp_path, monkeypatch, capsys, guard_mode):
    rec, again = _german_blind_drain(tmp_path, monkeypatch, guard_mode,
                                     guard_raises=AttributeError("'list' object has no attribute 'get'"))
    assert rec["german_parts"] == {"schema": Q.GPG.SCHEMA, "result": "error",
                                   "error": "AttributeError: 'list' object has no attribute 'get'"}
    assert "[german_parts] ERROR" in capsys.readouterr().out
    if guard_mode == "warn":
        assert rec["status"] == "PROVED-DRY-RUN"          # the drain went on
    else:
        assert rec["status"] == "REFUSED" and rec["stage"] == "german_parts"
        assert rec["reason"].startswith("german_parts: guard error AttributeError")
        assert again["status"] == "REFUSED" and again["stage"] == "german_parts"   # not cached


def test_drain_blocks_on_mechanical_content_failure(tmp_path, monkeypatch):
    _stub_plan_deps(monkeypatch)
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    # An absurdly high word floor makes every real row fail -- 100% > 1%.
    write_pack(queue_dir, {**MANIFEST_BASE, "word_floor": 10_000,
                          "detect_language": False})
    contract = write_contract(tmp_path, "mechanical")

    results = Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
                      task="641-test", apply=True, batch=10, pause_s=0,
                      artifact_root=tmp_path / "artifacts",
                      contract_path=contract, refresh_index=False)
    rec = results[0]
    assert rec["status"] == "PROVED-BLOCKED"
    assert any("mechanical_content" in b for b in rec["blocked_on"])
    assert not (queue_dir / "test-pack-001" / "DONE").exists()
    # Task 664, 2026-09-26 ruling: 100% of rows here fail ONLY the word
    # floor (word_floor=10_000, nothing else set to fail), so this refuses
    # via the word-floor ceiling (10%), not the ordinary one (1%) -- the
    # message must name the ceiling that actually fired.
    (msg,) = [b for b in rec["blocked_on"] if "mechanical_content" in b]
    assert "word floor" in msg
    assert "non-word-floor check" not in msg
    # The split itself (a non-word-floor defect still names the ordinary
    # ceiling and never the word-floor one) is proved directly against
    # `mechanical_content_check.evaluate()` in
    # test_mechanical_content_check.py -- not re-proved here through the
    # full drain() path, which has its OWN earlier, stricter title gate
    # (payload_carry_contract, at plan() time) that a synthetic "no title"
    # fixture would trip before ever reaching this mechanical check.


def test_drain_carries_the_waiver_into_the_receipt(tmp_path, monkeypatch):
    """The receipt.json a human reads later must carry Tim's waiver and the
    full per-row drop detail -- not just live in dry_plan()'s in-memory
    result, which drain() otherwise only copies a couple of fields out of."""
    _stub_plan_deps(monkeypatch)
    _force_lang_gate_unavailable(monkeypatch)
    raw = list(PSP.read_records(MANIFEST_BASE["pack_chunks"]))
    dropped_id = raw[2]["point_id"]
    monkeypatch.setattr(Q.MCC, "lang_drop_detail",
                        lambda rows, **k: [{"point_id": dropped_id,
                                           "url": raw[2]["url"], "text_chars": 7,
                                           "detected": "en", "confident": False,
                                           "raw_label": "en", "raw_prob": 0.5,
                                           "gate_unavailable": False}])
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    write_pack(queue_dir, {**MANIFEST_BASE, "lang_presence_waiver": WAIVER})
    contract = write_contract(tmp_path, "mechanical")

    results = Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
                      task="652-test", apply=False, batch=10, pause_s=0,
                      artifact_root=tmp_path / "artifacts",
                      contract_path=contract, refresh_index=False,
                      dry_receipt_dir=tmp_path / "dry-receipts")
    rec = results[0]
    assert rec["status"] == "PROVED-DRY-RUN"
    assert rec["waiver"]["issuer"] == "Tim"
    assert rec["waiver"]["dropped_count"] == 1
    assert rec["waiver"]["dropped_rows"][0]["point_id"] == dropped_id
    # A dry run's receipt goes to --dry-receipt-dir, never into the queue (2026-10-03).
    assert not (queue_dir / "test-pack-001" / "receipt.json").exists()
    on_disk = json.loads((tmp_path / "dry-receipts" / "test-pack-001.receipt.json").read_text())
    assert on_disk["waiver"]["dropped_count"] == 1


def test_drain_skips_a_pack_already_marked_done(tmp_path, monkeypatch):
    _stub_plan_deps(monkeypatch)
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    pack_dir = write_pack(queue_dir, MANIFEST_BASE)
    (pack_dir / "DONE").write_text("2026-09-23T00:00:00Z\n")
    contract = write_contract(tmp_path, "mechanical")

    results = Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
                      task="641-test", apply=True, batch=10, pause_s=0,
                      artifact_root=tmp_path / "artifacts",
                      contract_path=contract, refresh_index=False)
    assert results == []


def test_drain_skips_a_pack_marked_superseded(tmp_path, monkeypatch):
    """A pack retired via `supersede_pack()` (task 669 -- l1-backfill-
    batch-00001, proven a near-total duplicate re-run) is skipped by the
    drain loop just like DONE, but the marker is `SUPERSEDED`, never
    `DONE` -- a superseded pack's rows never land anywhere, unlike DONE's
    "landed on both sides" meaning."""
    _stub_plan_deps(monkeypatch)
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    pack_dir = write_pack(queue_dir, MANIFEST_BASE)
    (pack_dir / "SUPERSEDED").write_text("2026-09-28T00:00:00Z\n")
    contract = write_contract(tmp_path, "mechanical")

    results = Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
                      task="641-test", apply=True, batch=10, pause_s=0,
                      artifact_root=tmp_path / "artifacts",
                      contract_path=contract, refresh_index=False)
    assert results == []


def test_drain_stops_before_a_new_pack_when_the_budget_is_nearly_exhausted(
        tmp_path, monkeypatch):
    """Task 673: the drain must stop STARTING a new pack once under
    `stop_margin_s` of its systemd `TimeoutStartSec` budget remains,
    rather than let a mid-pack SIGKILL happen the way the 2026-09-26/28
    weekend stall repeatedly did. With almost no budget left, NEITHER
    queued pack is even looked at -- they stay READY for the next run."""
    _stub_plan_deps(monkeypatch)
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "pack-a"})
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "pack-b"})
    contract = write_contract(tmp_path, "battery-required")

    results = Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
                      task="641-test", apply=True, batch=10, pause_s=0,
                      artifact_root=tmp_path / "artifacts",
                      contract_path=contract, refresh_index=False,
                      budget_seconds=60, stop_margin_s=5400)
    assert results == []


def test_drain_stops_before_a_new_pack_using_started_at_epoch_in_the_past(
        tmp_path, monkeypatch):
    """`started_at_epoch` (review finding 5), not this call's own start
    time, anchors the budget -- a wrapper script that spent most of its
    budget on git fetch/reset before ever calling drain() must still see
    the clock as already-mostly-spent, not reset to "now"."""
    _stub_plan_deps(monkeypatch)
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "pack-a"})
    contract = write_contract(tmp_path, "battery-required")

    results = Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
                      task="641-test", apply=True, batch=10, pause_s=0,
                      artifact_root=tmp_path / "artifacts",
                      contract_path=contract, refresh_index=False,
                      budget_seconds=21600,
                      started_at_epoch=time.time() - 21600 + 60,  # 60s left
                      stop_margin_s=5400)
    assert results == []


def test_drain_processes_packs_normally_with_ample_budget(tmp_path, monkeypatch):
    """No regression: a deadline far in the future must not change
    behaviour at all, matching every caller that passes
    `budget_seconds=None`."""
    _stub_plan_deps(monkeypatch)
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "pack-a"})
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "pack-b"})
    contract = write_contract(tmp_path, "battery-required")

    results = Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
                      task="641-test", apply=True, batch=10, pause_s=0,
                      artifact_root=tmp_path / "artifacts",
                      contract_path=contract, refresh_index=False,
                      budget_seconds=21600, stop_margin_s=5400)
    statuses = {r["pack_id"]: r["status"] for r in results}
    assert statuses == {"pack-a": "PROVED-BLOCKED", "pack-b": "PROVED-BLOCKED"}


def test_drain_stops_before_the_persisted_index_refresh_when_budget_is_low(
        tmp_path, monkeypatch):
    """Review finding 5: the persisted-index refresh itself is guarded,
    not just the per-pack loop -- it can take close to an hour (log
    2026-09-28 14:05Z), the actual root cause of the weekend stall."""
    _stub_plan_deps(monkeypatch)
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "pack-a"})
    contract = write_contract(tmp_path, "battery-required")

    def refuse_refresh(*_a, **_k):
        raise AssertionError("must not attempt the refresh with no budget for it")
    monkeypatch.setattr(Q, "refresh_persisted_index", refuse_refresh)

    results = Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
                      task="641-test", apply=True, batch=10, pause_s=0,
                      artifact_root=tmp_path / "artifacts",
                      contract_path=contract, refresh_index=True,
                      budget_seconds=60, stop_margin_s=5400)
    assert results == []


def test_supersede_pack_writes_receipt_and_marker(tmp_path):
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    pack_dir = write_pack(queue_dir, MANIFEST_BASE)

    receipt = Q.supersede_pack(pack_dir, reason="proven duplicate of batch-00000",
                               task="669")

    assert receipt["status"] == "SUPERSEDED"
    assert receipt["pack_id"] == "test-pack-001"
    assert receipt["reason"] == "proven duplicate of batch-00000"
    assert receipt["task"] == "669"
    assert (pack_dir / "SUPERSEDED").exists()
    on_disk = json.loads((pack_dir / "receipt.json").read_text())
    assert on_disk == receipt


def test_supersede_pack_refuses_a_pack_already_landed(tmp_path):
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    pack_dir = write_pack(queue_dir, MANIFEST_BASE)
    (pack_dir / "DONE").write_text("2026-09-20T00:00:00Z\n")

    with pytest.raises(Q.QueueError, match="already DONE"):
        Q.supersede_pack(pack_dir, reason="too late", task="669")
    assert not (pack_dir / "SUPERSEDED").exists()


def test_supersede_pack_refuses_a_missing_manifest(tmp_path):
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    pack_dir = queue_dir / "no-manifest"
    pack_dir.mkdir()

    with pytest.raises(Q.QueueError, match="no manifest.json"):
        Q.supersede_pack(pack_dir, reason="x", task="669")


def test_drain_refuses_a_bad_manifest_but_continues_the_queue(tmp_path, monkeypatch):
    _stub_plan_deps(monkeypatch)
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    (queue_dir / "broken-pack").mkdir()  # no manifest.json at all
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "good-pack"})
    contract = write_contract(tmp_path, "battery-required")

    results = Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
                      task="641-test", apply=True, batch=10, pause_s=0,
                      artifact_root=tmp_path / "artifacts",
                      contract_path=contract, refresh_index=False)
    statuses = {r["pack_id"]: r["status"] for r in results}
    assert statuses["broken-pack"] == "REFUSED"
    assert statuses["good-pack"] == "PROVED-BLOCKED"


def test_drain_survives_a_persisted_inputs_refusal_and_keeps_going(
        tmp_path, monkeypatch):
    """Regression (2026-09-23): a real LEGI dry-run raised
    `persisted_inputs.UnpersistedInputs` -- NOT a `PromotionError` -- and an
    earlier version of `drain()` only caught `PromotionError`, so it crashed
    the whole queue on the first pack instead of recording a refusal and
    moving on. `plan()` refuses through three separate exception
    hierarchies (PromotionError, PersistedInputsError, PayloadCarryError);
    all three must be caught per-pack."""
    monkeypatch.setattr(PSP, "target_width", lambda *_a, **_k: 256)
    monkeypatch.setattr(PSP, "already_present", lambda *_a, **_k: [])
    monkeypatch.setattr(PSP, "already_served_urls", lambda *_a, **_k: [])

    def refuse(*_a, **_k):
        raise PSP.PI.UnpersistedInputs(
            "130,233 of 222,490 pages unpersisted", {"verdict": "REFUSED"})
    monkeypatch.setattr(PSP.PI, "assert_pages_persisted", refuse)
    monkeypatch.setattr(PSP.PC, "assert_scoreable",
                        lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(PSP.LAYOUT, "fingerprint", lambda *_a, **_k: {})

    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "bad-pack"})
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "zzz-after-bad-pack"})
    contract = write_contract(tmp_path, "battery-required")

    results = Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
                      task="641-test", apply=True, batch=10, pause_s=0,
                      artifact_root=tmp_path / "artifacts",
                      contract_path=contract, refresh_index=False)
    statuses = {r["pack_id"]: r["status"] for r in results}
    assert statuses["bad-pack"] == "REFUSED"
    assert "UnpersistedInputs" in [r for r in results
                                   if r["pack_id"] == "bad-pack"][0]["reason"]
    # The SECOND pack still ran -- the whole queue did not crash.
    assert statuses["zzz-after-bad-pack"] == "REFUSED"


def test_drain_survives_a_projection_error_and_keeps_going(tmp_path, monkeypatch):
    """Regression (2026-09-23): CC tranche 1's real dry-run raised
    `aw_b1_projection.ProjectionError` ("unresolved license") uncaught --
    a FOURTH exception hierarchy plan() can raise, found live right after
    fixing the third. Same class of bug, same fix shape."""
    monkeypatch.setattr(PSP, "target_width", lambda *_a, **_k: 256)
    monkeypatch.setattr(PSP, "already_present", lambda *_a, **_k: [])
    monkeypatch.setattr(PSP, "already_served_urls", lambda *_a, **_k: [])
    monkeypatch.setattr(PSP.PI, "assert_pages_persisted",
                        lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(PSP.PC, "assert_scoreable",
                        lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(PSP.LAYOUT, "fingerprint", lambda *_a, **_k: {})

    def refuse(*_a, **_k):
        raise PSP.PROJ.ProjectionError("https://x: unresolved license")
    monkeypatch.setattr(PSP.PROJ, "project_record", refuse)

    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "unlicensed-pack"})
    contract = write_contract(tmp_path, "battery-required")

    results = Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
                      task="641-test", apply=True, batch=10, pause_s=0,
                      artifact_root=tmp_path / "artifacts",
                      contract_path=contract, refresh_index=False)
    assert results[0]["status"] == "REFUSED"
    assert "ProjectionError" in results[0]["reason"]


# ---------------------------------------------- task 660: HOL-blocking fix
def test_drain_write_time_refusal_before_any_upsert_is_refused_and_continues(
        tmp_path, monkeypatch):
    """The real incident, 2026-09-25: CC tranche 1's read-only `dry_plan()`
    passed, but `apply_side()`'s OWN re-derived `plan()` call -- run at
    write time, inside the locked twin transaction, BEFORE `PSP.upsert()`
    ever ran -- raised `PayloadCarryError` again. The old `drain()` caught
    this only via a blanket `except BaseException: ... raise`, which killed
    every later pack in the queue for four straight hourly runs. Simulated
    here with a counting `assert_scoreable` stub that passes on call #1
    (the dry-run) and fails on call #2 (apply_side's write-time plan()) --
    exactly the two-call shape the real bug had. No write must have been
    issued, and the SECOND pack (alphabetically after the bad one) must
    still fully land."""
    _stub_plan_deps(monkeypatch)
    monkeypatch.setattr(PSP.LAYOUT, "fingerprint",
                        lambda *_a, **_k: {"points_count": 3})

    calls: list[int] = []

    def scoreable(payloads, **_k):
        list(payloads)  # a generator; must be consumed like the real check
        calls.append(1)
        if len(calls) == 2:
            raise PSP.PC.PayloadCarryError(
                "payload carry FAILED on aw_e8 (sealed-pack promotion): "
                "lang 94.2% < 99%")
        return {"verdict": "PASS"}
    monkeypatch.setattr(PSP.PC, "assert_scoreable", scoreable)

    upserts: list[tuple[str, int]] = []
    monkeypatch.setattr(
        PSP, "upsert",
        lambda url, coll, points, batch, pause_s=0.0, parallel=1:
            upserts.append((url, len(points))) or len(points))

    monkeypatch.setattr(Q.TFA, "ProductionWriterLock", _FakeLock)
    _fake_run_batch(monkeypatch)

    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    # Both packs reach `mechanical_content_status()` (gate == "mechanical")
    # before either refusal fires -- `detect_language: False` keeps that
    # real-detector call out of this test's result entirely, same idiom as
    # every other drain()-level test in this file that reaches it
    # (`test_drain_blocks_on_mechanical_content_failure`,
    # `_force_lang_gate_unavailable`). Without it this test is exposed to
    # whatever langgate/env state a DIFFERENT test file left behind in the
    # full suite -- order-dependent, passes alone, fails in CI.
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "aaa-refuses-at-write",
                          "detect_language": False})
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "zzz-after-refusal",
                          "detect_language": False})
    contract = write_contract(tmp_path, "mechanical")

    results = Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
                      task="641-test", apply=True, batch=10, pause_s=0,
                      artifact_root=tmp_path / "artifacts",
                      contract_path=contract, refresh_index=False)

    by_id = {r["pack_id"]: r for r in results}
    assert by_id["aaa-refuses-at-write"]["status"] == "REFUSED"
    assert "PayloadCarryError" in by_id["aaa-refuses-at-write"]["reason"]
    # No write was ever issued for the refused pack: only the SECOND pack's
    # two legs (twin, prod) reached PSP.upsert() at all.
    assert upserts == [("http://x", 3), ("http://y", 3)], (
        "a pre-write refusal must never reach PSP.upsert()")
    # The pack keeps its READY -- it is not deleted or DONE-marked.
    assert not (queue_dir / "aaa-refuses-at-write" / "DONE").exists()
    assert (queue_dir / "aaa-refuses-at-write" / "READY").exists()
    # The SECOND pack still ran to completion: the whole drain did not
    # abort on the first pack's write-time refusal.
    assert by_id["zzz-after-refusal"]["status"] == "LANDED"
    assert (queue_dir / "zzz-after-refusal" / "DONE").exists()


def test_drain_stops_the_whole_queue_on_a_mid_write_failure(tmp_path, monkeypatch):
    """A refusal with no write issued is safe to skip past (previous test).
    A failure AFTER `PSP.upsert()` has actually run is a possible twin/prod
    divergence and must still stop the whole drain, non-zero (task 660:
    "exit non-zero only if a pack failed mid-write")."""
    _stub_plan_deps(monkeypatch)
    monkeypatch.setattr(PSP.LAYOUT, "fingerprint",
                        lambda *_a, **_k: {"points_count": 3})

    def fail_upsert(*_a, **_k):
        raise RuntimeError("qdrant PUT timed out mid-batch")
    monkeypatch.setattr(PSP, "upsert", fail_upsert)

    monkeypatch.setattr(Q.TFA, "ProductionWriterLock", _FakeLock)
    _fake_run_batch(monkeypatch)

    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    # Same order-dependence fix as the sibling test above: this pack also
    # reaches `mechanical_content_status()` under gate == "mechanical"
    # before the upsert failure fires.
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "aaa-mid-write-failure",
                          "detect_language": False})
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "zzz-never-reached",
                          "detect_language": False})
    contract = write_contract(tmp_path, "mechanical")

    with pytest.raises(RuntimeError, match="qdrant PUT timed out"):
        Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
               task="641-test", apply=True, batch=10, pause_s=0,
               artifact_root=tmp_path / "artifacts",
               contract_path=contract, refresh_index=False)

    receipt = json.loads(
        (queue_dir / "aaa-mid-write-failure" / "receipt.json").read_text())
    assert receipt["status"] == "ABORTED"
    assert "qdrant PUT timed out" in receipt["reason"]
    # The second pack was never even looked at -- the drain stopped dead,
    # exactly as it must for a possible partial-write divergence.
    assert not (queue_dir / "zzz-never-reached" / "receipt.json").exists()


# -------------------------------------------- task 664: partial-write resume
def test_drain_resumes_the_prod_leg_when_twin_landed_but_prod_did_not(
        tmp_path, monkeypatch):
    """The real incident, 2026-09-26: prod's qdrant restarted mid-write
    during a pack's PROD leg, after its TWIN leg had already landed and
    verified. The next drain's `dry_plan()` (always against twin) raised
    `PromotionError` before this fix -- a "plan"-stage refusal, cached
    forever, so the twin/prod gap could never close by itself. Now it must
    resume the PROD LEG ONLY, never re-write twin, and land the pack."""
    ids = frozenset({"id-1", "id-2", "id-3"})

    def fake_dry_plan(_manifest, _url):
        raise PSP.AlreadyLandedError("3 of 3 ids are already in aw_e8",
                                     ids=ids)
    monkeypatch.setattr(Q, "dry_plan", fake_dry_plan)
    # PROD is missing all three -- a genuine resume, not a no-op.
    monkeypatch.setattr(PSP, "already_present", lambda *_a, **_k: [])

    resume_calls: list[dict] = []

    def fake_resume(**kwargs):
        resume_calls.append(kwargs)
        return {"prod_receipt": {"written": 3}}
    monkeypatch.setattr(Q, "resume_prod_only", fake_resume)

    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "twin-landed-prod-did-not"})
    contract = write_contract(tmp_path, "mechanical")
    # This pack's OWN twin-write artifact exists (the original incident's
    # precondition: THIS pack_id's own apply wrote twin) -- drain() must
    # take the artifact path, not the no-artifact live-derivation path
    # (task 664 incident 2's fix, tested separately below).
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir(parents=True)
    (artifact_root / "twin-landed-prod-did-not-twin-written-ids.json").write_text(
        json.dumps(sorted(ids)))

    results = Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
                      task="641-test", apply=True, batch=10, pause_s=0,
                      artifact_root=artifact_root,
                      contract_path=contract, refresh_index=False)

    assert len(resume_calls) == 1
    assert resume_calls[0]["pack_id"] == "twin-landed-prod-did-not"
    assert results[0]["status"] == "LANDED"
    assert results[0]["resumed_prod_only"] is True
    assert results[0]["prod"] == {"written": 3}
    assert (queue_dir / "twin-landed-prod-did-not" / "DONE").exists()
    receipt = json.loads(
        (queue_dir / "twin-landed-prod-did-not" / "receipt.json").read_text())
    assert receipt["status"] == "LANDED"


def test_drain_resumes_via_live_twin_diff_when_no_per_pack_artifact_exists(
        tmp_path, monkeypatch):
    """Task 664, 2026-09-26, incident 2: `cc-authority-frontier-w7-
    tranche2`'s ids are already on twin because a DIFFERENT pack_id
    (tranche2-a..d's earlier split-pack landing) wrote them -- there is no
    `{pack_id}-twin-written-ids.json` for THIS pack_id, since it never
    itself wrote twin. Before this fix, resume_prod_only() raised
    QueueError (the exact message seen live) and the pack stayed refused
    forever (not cached, but never resolved either). It must instead
    derive the resumable set from a live twin-vs-prod membership diff and
    resume, marking the pack a landed duplicate."""
    ids = frozenset({"id-1", "id-2", "id-3"})

    def fake_dry_plan(_manifest, _url):
        raise PSP.AlreadyLandedError("3 of 3 ids are already in aw_e8",
                                     ids=ids)
    monkeypatch.setattr(Q, "dry_plan", fake_dry_plan)

    def fake_already_present(url, _collection, check_ids, *_a, **_k):
        if url == "http://y":  # prod -- missing everything
            return []
        if url == "http://x":  # twin -- confirms everything, live
            return list(check_ids)
        raise AssertionError(f"unexpected url {url!r}")
    monkeypatch.setattr(PSP, "already_present", fake_already_present)

    resume_calls: list[dict] = []

    def fake_resume(**kwargs):
        resume_calls.append(kwargs)
        return {"prod_receipt": {"written": 3}}
    monkeypatch.setattr(Q, "resume_prod_only", fake_resume)

    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "duplicate-of-split-packs"})
    contract = write_contract(tmp_path, "mechanical")

    results = Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
                      task="641-test", apply=True, batch=10, pause_s=0,
                      artifact_root=tmp_path / "artifacts",  # no artifact written
                      contract_path=contract, refresh_index=False)

    assert len(resume_calls) == 1
    assert sorted(resume_calls[0]["twin_ids_override"]) == sorted(ids)
    assert results[0]["status"] == "LANDED"
    assert "duplicate" in results[0]["note"]
    assert (queue_dir / "duplicate-of-split-packs" / "DONE").exists()


def test_drain_refuses_when_some_ids_are_on_neither_twin_nor_prod(
        tmp_path, monkeypatch):
    """No per-pack artifact AND some ids genuinely never landed anywhere --
    NOT a resume (writing them straight to prod would skip twin entirely).
    Must refuse clearly, never silently drop the ids or silently write
    unverified content to prod."""
    ids = frozenset({"id-1", "id-2"})

    def fake_dry_plan(_manifest, _url):
        raise PSP.AlreadyLandedError("2 of 2 ids are already in aw_e8",
                                     ids=ids)
    monkeypatch.setattr(Q, "dry_plan", fake_dry_plan)

    def fake_already_present(url, _collection, check_ids, *_a, **_k):
        if url == "http://y":  # prod -- missing everything
            return []
        if url == "http://x":  # twin -- confirms only id-1
            return [i for i in check_ids if i == "id-1"]
        raise AssertionError(f"unexpected url {url!r}")
    monkeypatch.setattr(PSP, "already_present", fake_already_present)

    resume_calls: list[dict] = []
    monkeypatch.setattr(Q, "resume_prod_only",
                        lambda **kw: resume_calls.append(kw))

    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "partially-orphaned"})
    contract = write_contract(tmp_path, "mechanical")

    results = Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
                      task="641-test", apply=True, batch=10, pause_s=0,
                      artifact_root=tmp_path / "artifacts",
                      contract_path=contract, refresh_index=False)

    assert resume_calls == [], "must not resume when some ids are on neither target"
    assert results[0]["status"] == "REFUSED"
    assert results[0]["stage"] == "live-apply-preflight"
    assert "id-2" in results[0]["reason"]
    assert not (queue_dir / "partially-orphaned" / "DONE").exists()


def test_drain_names_a_partial_overlap_with_an_earlier_pack(
        tmp_path, monkeypatch):
    """Task 690, 2026-10-02 (`eric-official-pdf-tranche-130000`): twin AND
    prod already hold ONE of the pack's ids (landed by its own smoke pack),
    the other ids are on neither. Still a refusal -- writing the new ids
    straight to prod would skip twin -- but the reason and the receipt must
    name the id that triggered the branch and say what to do. The old text
    named only the ids the refusal was NOT about."""
    ids = frozenset({"id-1", "id-2", "id-3"})

    def fake_dry_plan(_manifest, _url):
        raise PSP.AlreadyLandedError("1 of 3 ids are already in aw_e8",
                                     ids=ids, present=frozenset({"id-1"}))
    monkeypatch.setattr(Q, "dry_plan", fake_dry_plan)

    def fake_already_present(url, _collection, check_ids, *_a, **_k):
        if url in ("http://x", "http://y"):  # twin and prod: id-1 only
            return [i for i in check_ids if i == "id-1"]
        raise AssertionError(f"unexpected url {url!r}")
    monkeypatch.setattr(PSP, "already_present", fake_already_present)
    monkeypatch.setattr(Q, "resume_prod_only",
                        lambda **_k: pytest.fail("must not resume"))

    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "smoke-overlap"})
    contract = write_contract(tmp_path, "mechanical")

    results = Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
                      task="690-test", apply=True, batch=10, pause_s=0,
                      artifact_root=tmp_path / "artifacts",
                      contract_path=contract, refresh_index=False)

    rec = results[0]
    assert rec["status"] == "REFUSED"
    assert rec["stage"] == "live-apply-preflight"
    assert "2 of 2 id(s) missing from prod" in rec["reason"]
    assert "twin already holds 1 of 3" in rec["reason"]
    assert "'id-1'" in rec["reason"]
    assert "PARTIAL OVERLAP" in rec["reason"]
    assert rec["already_on_twin"] == {"count": 1, "also_on_prod": 1,
                                      "ids": ["id-1"]}
    assert not (queue_dir / "smoke-overlap" / "DONE").exists()


def test_overlap_note_does_not_claim_partial_overlap_when_prod_lacks_them():
    """Twin holds an id prod does not: that is a half-landed write, not an
    earlier pack's overlap. The note names the trigger but must not
    recommend excluding the id (it would never reach prod)."""
    exc = PSP.AlreadyLandedError("x", ids=frozenset({"a", "b"}),
                                 present=frozenset({"a"}))
    note = Q._already_landed_overlap_note(exc, ["a", "b"], ["b"])
    assert "twin already holds 1 of 2" in note and "0 of them on prod" in note
    assert "PARTIAL OVERLAP" not in note
    assert Q._already_landed_overlap_note(
        PSP.AlreadyLandedError("x", ids=frozenset({"a"})), ["a"], ["a"]) == ""


def test_drain_marks_done_when_both_targets_already_have_every_id(
        tmp_path, monkeypatch):
    """If PROD also already has every id (e.g. a receipt was lost or
    overwritten after a genuinely completed pack), there is nothing left to
    resume -- mark DONE rather than refuse forever with no fix available."""
    ids = frozenset({"id-1", "id-2"})

    def fake_dry_plan(_manifest, _url):
        raise PSP.AlreadyLandedError("2 of 2 ids are already in aw_e8",
                                     ids=ids)
    monkeypatch.setattr(Q, "dry_plan", fake_dry_plan)
    # PROD already has both -- nothing to resume.
    monkeypatch.setattr(PSP, "already_present",
                        lambda *_a, **_k: sorted(ids))

    resume_calls: list[dict] = []
    monkeypatch.setattr(Q, "resume_prod_only",
                        lambda **kw: resume_calls.append(kw))

    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "both-already-landed"})
    contract = write_contract(tmp_path, "mechanical")

    results = Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
                      task="641-test", apply=True, batch=10, pause_s=0,
                      artifact_root=tmp_path / "artifacts",
                      contract_path=contract, refresh_index=False)

    assert resume_calls == [], "nothing to resume -- must not attempt a write"
    assert results[0]["status"] == "LANDED"
    assert (queue_dir / "both-already-landed" / "DONE").exists()


def test_drain_retries_next_run_when_the_prod_membership_check_itself_fails(
        tmp_path, monkeypatch):
    """A prod connectivity hiccup WHILE deciding whether this is a resume
    must not be cached as a deterministic content refusal (stage="plan")
    -- that is the identical bug shape this whole fix exists for, on a
    different exception. It must retry next drain instead
    (stage="live-apply-preflight", never cached)."""
    ids = frozenset({"id-1"})

    def fake_dry_plan(_manifest, _url):
        raise PSP.AlreadyLandedError("1 of 1 ids are already in aw_e8", ids=ids)
    monkeypatch.setattr(Q, "dry_plan", fake_dry_plan)

    def fail_present(*_a, **_k):
        # `_api()` wraps a refused connection in `urllib.error.URLError`
        # (never a bare `ConnectionRefusedError`) -- match that real shape.
        raise Q.urllib.error.URLError(ConnectionRefusedError("prod qdrant is down"))
    monkeypatch.setattr(PSP, "already_present", fail_present)
    monkeypatch.setattr(Q, "resume_prod_only",
                        lambda **_k: pytest.fail("must not attempt to resume"))

    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "prod-unreachable"})
    contract = write_contract(tmp_path, "mechanical")

    results = Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
                      task="641-test", apply=True, batch=10, pause_s=0,
                      artifact_root=tmp_path / "artifacts",
                      contract_path=contract, refresh_index=False)

    assert results[0]["status"] == "REFUSED"
    assert results[0]["stage"] == "live-apply-preflight"
    assert not (queue_dir / "prod-unreachable" / "DONE").exists()


def test_drain_already_landed_check_survives_a_missing_prod_alias(
        tmp_path, monkeypatch):
    """#414 review follow-up 4: in the AlreadyLanded resume check,
    `get_prod_collection()` -> `resolve_prod_alias` raises `QueueError`
    when `GET /aliases` has no `aw_prod` (the mid-flip window). That used
    to escape the except tuple and crash the whole drain. It must be a
    retry-next-drain REFUSED (stage live-apply-preflight, uncached)."""
    ids = frozenset({"id-1"})

    def fake_dry_plan(_manifest, _url):
        raise PSP.AlreadyLandedError("1 of 1 ids are already in aw_e8", ids=ids)
    monkeypatch.setattr(Q, "dry_plan", fake_dry_plan)

    def fake_api(url, path, body=None, method=None):
        assert path == "/aliases"
        return {"aliases": [
            {"alias_name": "some_other_alias", "collection_name": "aw_x"}]}
    monkeypatch.setattr(PSP, "_api", fake_api)
    monkeypatch.setattr(Q, "resume_prod_only",
                        lambda **_k: pytest.fail("must not attempt to resume"))

    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "alias-missing"})
    contract = write_contract(tmp_path, "mechanical")

    results = Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
                      task="668-test", apply=True, batch=10, pause_s=0,
                      artifact_root=tmp_path / "artifacts",
                      contract_path=contract, refresh_index=False)

    assert results[0]["status"] == "REFUSED"
    assert results[0]["stage"] == "live-apply-preflight"
    assert "alias 'aw_prod' not found" in results[0]["reason"]
    assert not (queue_dir / "alias-missing" / "DONE").exists()


def test_dry_drain_records_a_prod_read_refusal_and_goes_on(tmp_path, monkeypatch):
    """Review of #660: a dry drain (no --apply) is a replay, and a replay never reads prod. The
    already-landed membership check then raises `ProdReadRefused`; it must be a retry-next-drain
    REFUSED (stage live-apply-preflight, uncached) for THAT pack, not an rc-1 stop of the run."""
    from cytir.index import prod_guard as PG
    ids = frozenset({"id-1"})

    def fake_dry_plan(_manifest, _url):
        raise PSP.AlreadyLandedError("1 of 1 ids are already in aw_e8", ids=ids)
    monkeypatch.setattr(Q, "dry_plan", fake_dry_plan)
    monkeypatch.setattr(PSP, "_api", lambda url, path, body=None, method=None: {
        "aliases": [{"alias_name": "aw_prod", "collection_name": "aw_e9"}]})

    def refused(*_a, **_k):
        raise PG.ProdReadRefused("REFUSED POST on prod: a replay takes the TWIN as both sides")
    monkeypatch.setattr(PSP, "already_present", refused)

    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "landed-a"})
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "landed-b"})
    contract = write_contract(tmp_path, "mechanical")

    results = Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
                      task="660-test", apply=False, batch=10, pause_s=0,
                      artifact_root=tmp_path / "artifacts",
                      contract_path=contract, refresh_index=False)

    assert [r["status"] for r in results] == ["REFUSED", "REFUSED"]     # both packs seen: no stop
    assert {r["stage"] for r in results} == {"live-apply-preflight"}   # a stage the cache retries
    assert "ProdReadRefused" in results[0]["reason"]


def test_resume_prod_only_writes_prod_alone_and_verifies_membership(
        tmp_path, monkeypatch):
    """`resume_prod_only()` itself: must call `apply_side` for "prod" only
    (never re-derive or re-write twin) and verify every one of the twin
    leg's own written ids is present on prod afterward."""
    _stub_plan_deps(monkeypatch)
    monkeypatch.setattr(PSP.LAYOUT, "fingerprint",
                        lambda *_a, **_k: {"points_count": 3})
    monkeypatch.setattr(Q.TFA, "ProductionWriterLock", _FakeLock)
    _fake_run_batch(monkeypatch)

    # Task 788: the twin-written ids are the fixture's own projected ids (a twin leg writes what the
    # pack projects); the resume refuses a twin-written id the pack does not project.
    twin_ids = _fixture_ids()
    present_calls: list[tuple[str, list[str]]] = []
    # Stateful: twin holds the pack, prod holds nothing until the resume writes it.
    held = {"http://x": set(twin_ids), "http://y": set()}

    def fake_already_present(url, _collection, ids, **_k):
        present_calls.append((url, list(ids)))
        return [i for i in ids if i in held[url]]
    monkeypatch.setattr(PSP, "already_present", fake_already_present)

    upserts: list[tuple[str, int]] = []

    def fake_upsert(url, coll, points, batch, pause_s=0.0, parallel=1):
        upserts.append((url, len(points)))
        held[url] |= {p["id"] for p in points}
        return len(points)
    monkeypatch.setattr(PSP, "upsert", fake_upsert)

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    (artifact_root / "resume-pack-twin-written-ids.json").write_text(
        json.dumps(twin_ids))

    state = Q.resume_prod_only(
        pack_id="resume-pack", manifest=MANIFEST_BASE, twin_url="http://x",
        prod_url="http://y", task="641-test", batch=10, pause_s=0,
        artifact_root=artifact_root, prod_collection="aw_e8")

    # Only PROD was ever upserted to -- twin is never re-written.
    assert upserts == [("http://y", 3)]
    assert state["prod_receipt"]["written"] == 3
    # The preflight checked twin membership, and verify_mirrors checked prod.
    assert ("http://x", twin_ids) in present_calls
    assert ("http://y", twin_ids) in present_calls


def test_resume_prod_only_uses_the_override_when_no_artifact_exists(
        tmp_path, monkeypatch):
    """Task 664, 2026-09-26, incident 2: no `{pack_id}-twin-written-ids.
    json` on disk at all (this pack_id never itself wrote twin) --
    `twin_ids_override` must be used in its place, with identical
    preflight/apply/verify behaviour to the artifact-file path."""
    _stub_plan_deps(monkeypatch)
    monkeypatch.setattr(PSP.LAYOUT, "fingerprint",
                        lambda *_a, **_k: {"points_count": 3})
    monkeypatch.setattr(Q.TFA, "ProductionWriterLock", _FakeLock)
    _fake_run_batch(monkeypatch)

    twin_ids = _fixture_ids()                       # task 788: see the test above
    held = {"http://x": set(twin_ids), "http://y": set()}

    def fake_already_present(url, _collection, ids, **_k):
        return [i for i in ids if i in held[url]]
    monkeypatch.setattr(PSP, "already_present", fake_already_present)

    upserts: list[tuple[str, int]] = []

    def fake_upsert(url, coll, points, batch, pause_s=0.0, parallel=1):
        upserts.append((url, len(points)))
        held[url] |= {p["id"] for p in points}
        return len(points)
    monkeypatch.setattr(PSP, "upsert", fake_upsert)

    artifact_root = tmp_path / "artifacts"  # deliberately never created/written

    state = Q.resume_prod_only(
        pack_id="resume-pack-no-artifact", manifest=MANIFEST_BASE,
        twin_url="http://x", prod_url="http://y", task="641-test",
        batch=10, pause_s=0, artifact_root=artifact_root,
        twin_ids_override=twin_ids, prod_collection="aw_e8")

    assert upserts == [("http://y", 3)]
    assert state["prod_receipt"]["written"] == 3
    assert not (artifact_root / "resume-pack-no-artifact-twin-written-ids.json").exists()


def test_resume_prod_only_still_refuses_with_no_artifact_and_no_override(
        tmp_path, monkeypatch):
    """No regression: the ORIGINAL guard stays in force when neither an
    artifact nor an override is given."""
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    with pytest.raises(Q.QueueError, match="does not exist"):
        Q.resume_prod_only(
            pack_id="resume-pack-neither", manifest=MANIFEST_BASE,
            twin_url="http://x", prod_url="http://y", task="641-test",
            batch=10, pause_s=0, artifact_root=artifact_root,
            prod_collection="aw_e8")


def test_drain_skips_a_previously_refused_pack_with_unchanged_manifest(
        tmp_path, monkeypatch):
    """Task 660: a REFUSED pack keeps its READY (it stays in the queue, not
    dropped) but must not re-run its full check every hourly drain forever
    -- only a genuinely new manifest (a restage, a fresh sha256) earns it
    another attempt."""
    monkeypatch.setattr(PSP, "target_width", lambda *_a, **_k: 256)
    monkeypatch.setattr(PSP, "already_present", lambda *_a, **_k: [])
    monkeypatch.setattr(PSP, "already_served_urls", lambda *_a, **_k: [])
    monkeypatch.setattr(PSP.PC, "assert_scoreable",
                        lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(PSP.LAYOUT, "fingerprint", lambda *_a, **_k: {})

    calls: list[int] = []

    def refuse(*_a, **_k):
        calls.append(1)
        raise PSP.PI.UnpersistedInputs(
            "nope, still unpersisted", {"verdict": "REFUSED"})
    monkeypatch.setattr(PSP.PI, "assert_pages_persisted", refuse)

    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "chronically-refused"})
    contract = write_contract(tmp_path, "battery-required")
    common = dict(twin_url="http://x", prod_url="http://y", task="641-test",
                 apply=True, batch=10, pause_s=0,
                 artifact_root=tmp_path / "artifacts",
                 contract_path=contract, refresh_index=False)

    first = Q.drain(queue_dir, **common)
    assert first[0]["status"] == "REFUSED"
    assert len(calls) == 1

    second = Q.drain(queue_dir, **common)
    assert second[0]["status"] == "SKIPPED-REFUSED-UNCHANGED"
    assert len(calls) == 1, "the expensive check must not re-run unchanged"

    # Reseal with a materially different manifest -> a new sha256 -> retried.
    pack_dir = queue_dir / "chronically-refused"
    new_manifest = {**MANIFEST_BASE, "pack_id": "chronically-refused",
                    "allow_supersede": True}
    manifest_path = pack_dir / "manifest.json"
    manifest_path.write_text(json.dumps(new_manifest, indent=1))
    sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (pack_dir / "READY").write_text(sha + "\n")

    third = Q.drain(queue_dir, **common)
    assert third[0]["status"] == "REFUSED"
    assert len(calls) == 2, "a genuinely new manifest sha must earn a retry"


def test_drain_always_retries_an_intake_refusal_even_with_unchanged_manifest(
        tmp_path, monkeypatch):
    """Bug found live, task 660, 2026-09-25: CC tranche 1 was parked (READY
    renamed away) while #348 was in flight, drew a 'REFUSED at intake' ('not
    queued yet') receipt from that hour's drain, and once READY was restored
    (same manifest, same sha -- no restage needed) the unchanged-manifest
    skip-cache would have skipped it FOREVER, since adding READY back never
    changes manifest.json's sha. An intake refusal is a cheap file-existence
    check, not the expensive pass the cache exists to protect -- it must
    always retry once the pack is genuinely re-queued."""
    _stub_plan_deps(monkeypatch)
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    # No READY yet -- this pack is not queued.
    pack_dir = write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "was-parked"},
                          ready=False)
    contract = write_contract(tmp_path, "battery-required")
    common = dict(twin_url="http://x", prod_url="http://y", task="641-test",
                 apply=True, batch=10, pause_s=0,
                 artifact_root=tmp_path / "artifacts",
                 contract_path=contract, refresh_index=False)

    first = Q.drain(queue_dir, **common)
    assert first[0]["status"] == "REFUSED"
    assert first[0]["stage"] == "intake"

    # Re-queue with the IDENTICAL manifest (no restage) -- matches restoring
    # a parked pack's READY, e.g. `cp PARKED-... READY`.
    sha = hashlib.sha256((pack_dir / "manifest.json").read_bytes()).hexdigest()
    (pack_dir / "READY").write_text(sha + "\n")

    second = Q.drain(queue_dir, **common)
    assert second[0]["status"] != "SKIPPED-REFUSED-UNCHANGED", (
        "an intake refusal must never be cached -- the pack is genuinely "
        "queued now and must be re-evaluated for real"
    )
    assert second[0]["status"] == "PROVED-BLOCKED"  # battery-required, this time it runs


# ---------------------------------------------- task 662: whitelist fix
def test_drain_never_caches_a_lock_acquire_refusal(tmp_path, monkeypatch):
    """Regression (task 662, 2026-09-25): a lock-acquire failure inside
    `live_apply()` -- `runb.assert_locks_free()` raising because a foreign
    lock is held, BEFORE any write -- is caught by `drain()`'s generic
    `except BaseException` and recorded as `status="REFUSED"` with no
    `stage` set. Task 660's cache only excluded `stage == "intake"` by
    name, so this refusal (stage=None) fell through that blacklist and got
    cached exactly like a genuine content refusal -- the pack would never
    retry again until someone noticed and resealed a byte-identical
    manifest just to mint a new sha256. The fix is a whitelist
    (`_CACHEABLE_REFUSAL_STAGES = {"plan", "content_pack_check"}`) and this
    call site now stamps `stage="live-apply-preflight"`, which is NOT in
    that whitelist: this pack must be re-evaluated (and its lock
    re-attempted) on every drain, exactly as it would be if the foreign
    lock clears in the meantime."""
    _stub_plan_deps(monkeypatch)
    monkeypatch.setattr(PSP.LAYOUT, "fingerprint",
                        lambda *_a, **_k: {"points_count": 3})
    monkeypatch.setattr(PSP, "upsert", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("must not write with a foreign lock held")))
    monkeypatch.setattr(Q.TFA, "ProductionWriterLock", _FakeLock)

    assert_locks_free_calls: list[str] = []

    class FakeRB:
        from cytir.ops.write_lock import LOCKS

        @staticmethod
        def assert_locks_free(task):
            assert_locks_free_calls.append(task)
            raise RuntimeError("cross-task lock held: task=626 owner=news-backlog")

        @staticmethod
        def acquire(name, task, purpose, *, pid=None):
            raise AssertionError("must never acquire past a failed preflight")

        @staticmethod
        def release(name, task):
            raise AssertionError("nothing was acquired to release")

    monkeypatch.setattr(Q, "_load_run_batch", lambda: FakeRB)

    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "lock-contended",
                          "detect_language": False})
    contract = write_contract(tmp_path, "mechanical")
    common = dict(twin_url="http://x", prod_url="http://y", task="641-test",
                 apply=True, batch=10, pause_s=0,
                 artifact_root=tmp_path / "artifacts",
                 contract_path=contract, refresh_index=False)

    first = Q.drain(queue_dir, **common)
    assert first[0]["status"] == "REFUSED"
    assert first[0]["stage"] == "live-apply-preflight"
    assert "cross-task lock held" in first[0]["reason"]
    assert len(assert_locks_free_calls) == 1

    # Manifest is UNCHANGED (same sha256) -- the old blacklist cache would
    # have skipped this pack forever from here on.
    second = Q.drain(queue_dir, **common)
    assert second[0]["status"] != "SKIPPED-REFUSED-UNCHANGED", (
        "a lock-acquire (transient/environmental) refusal must never be "
        "skip-cached -- only a deterministic content judgment may be")
    assert second[0]["status"] == "REFUSED"
    assert second[0]["stage"] == "live-apply-preflight"
    # The lock was genuinely retried, not answered from a cached receipt.
    assert len(assert_locks_free_calls) == 2, (
        "the retry must actually re-attempt the lock, not short-circuit")
    # The pack keeps its READY -- it is not deleted or DONE-marked.
    assert not (queue_dir / "lock-contended" / "DONE").exists()
    assert (queue_dir / "lock-contended" / "READY").exists()


def test_drain_empty_queue_is_a_noop(tmp_path):
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    assert Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
                   task="641-test", apply=True, batch=10, pause_s=0,
                   artifact_root=tmp_path / "artifacts") == []


def test_drain_only_runs_exactly_the_named_packs_in_the_given_order(tmp_path, monkeypatch):
    """Task 641 (2026-09-23): the coordinator asked for three specific news
    packs run ahead of, and in a different order than, everything else
    sitting in the queue (alphabetically: aaa-pack, mmm-pack, zzz-pack --
    `only` must run them zzz, aaa, mmm and touch nothing else)."""
    _stub_plan_deps(monkeypatch)
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "aaa-pack"})
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "mmm-pack"})
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "zzz-pack"})
    contract = write_contract(tmp_path, "battery-required")

    results = Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
                      task="641-test", apply=False, batch=10, pause_s=0,
                      artifact_root=tmp_path / "artifacts",
                      contract_path=contract, refresh_index=False,
                      only=["zzz-pack", "aaa-pack", "mmm-pack"])
    assert [r["pack_id"] for r in results] == ["zzz-pack", "aaa-pack", "mmm-pack"]


def test_drain_only_refuses_a_name_that_is_not_a_queued_pack(tmp_path):
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "real-pack"})
    contract = write_contract(tmp_path, "battery-required")

    with pytest.raises(Q.QueueError, match="no-such-pack"):
        Q.drain(queue_dir, twin_url="http://x", prod_url="http://y",
               task="641-test", apply=False, batch=10, pause_s=0,
               artifact_root=tmp_path / "artifacts", contract_path=contract,
               refresh_index=False, only=["real-pack", "no-such-pack"])


# ------------------------------------------------------ throttled upsert
def test_upsert_pause_is_forwarded_and_skipped_on_the_last_batch(monkeypatch):
    calls = []
    monkeypatch.setattr(PSP, "_api", lambda *a, **k: calls.append(a) or {})
    slept = []
    monkeypatch.setattr(PSP.time, "sleep", lambda s: slept.append(s))
    points = [{"id": str(i), "vector": {}, "payload": {}} for i in range(5)]
    written = PSP.upsert("http://x", "aw_e8", points, batch=2, pause_s=0.5)
    assert written == 5
    # 3 batches (2,2,1); a pause between the first two, none after the last.
    assert slept == [0.5, 0.5]


def test_upsert_parallel_writes_every_point_exactly_once(monkeypatch):
    """Task 641 (2026-09-23), SE throughput: parallel>1 must still write
    every point exactly once, batched the same way -- concurrency changes
    WHEN the batches fire, never WHAT they contain."""
    import threading
    calls = []
    lock = threading.Lock()

    def fake_api(url, path, body=None, method=None):
        with lock:
            calls.append([p["id"] for p in body["points"]])
        return {}
    monkeypatch.setattr(PSP, "_api", fake_api)
    points = [{"id": str(i), "vector": {}, "payload": {}} for i in range(23)]
    written = PSP.upsert("http://x", "aw_e8", points, batch=5, parallel=4)
    assert written == 23
    all_ids = sorted(int(i) for chunk in calls for i in chunk)
    assert all_ids == list(range(23))
    assert len(calls) == 5  # 5,5,5,5,3


def test_upsert_parallel_ignores_pause_s(monkeypatch):
    monkeypatch.setattr(PSP, "_api", lambda *a, **k: {})
    slept = []
    monkeypatch.setattr(PSP.time, "sleep", lambda s: slept.append(s))
    points = [{"id": str(i), "vector": {}, "payload": {}} for i in range(10)]
    PSP.upsert("http://x", "aw_e8", points, batch=2, pause_s=1.0, parallel=3)
    assert slept == []  # concurrency IS the throttle; pause_s does not apply


def test_upsert_default_pause_is_zero_unchanged_behaviour(monkeypatch):
    monkeypatch.setattr(PSP, "_api", lambda *a, **k: {})
    slept = []
    monkeypatch.setattr(PSP.time, "sleep", lambda s: slept.append(s))
    points = [{"id": str(i), "vector": {}, "payload": {}} for i in range(4)]
    PSP.upsert("http://x", "aw_e8", points, batch=2)
    assert slept == []


# ------------------------------------------- locks, live apply, rehearsal
class _FakeLock:
    """Stands in for `twin_first_apply.ProductionWriterLock` -- no real ssh
    to the prod host in a unit test."""

    def __init__(self, *_a, **_k):
        self.calls: list[str] = []

    def acquire(self) -> None:
        self.calls.append("acquire")

    def release(self) -> None:
        self.calls.append("release")


def _fake_run_batch(monkeypatch, *, foreign: str | None = None):
    """Stands in for the task-629 batch runner's lock helpers --
    records every acquire/release without touching the build box over ssh."""
    calls: list[tuple] = []

    class FakeRB:
        from cytir.ops.write_lock import LOCKS

        @staticmethod
        def assert_locks_free(task):
            if foreign:
                raise RuntimeError(f"cross-task lock held: {foreign}")
            return {}

        @staticmethod
        def acquire(name, task, purpose, *, pid=None):
            calls.append(("acquire", name))

        @staticmethod
        def release(name, task):
            calls.append(("release", name))

    monkeypatch.setattr(Q, "_load_run_batch", lambda: FakeRB)
    return calls


# ------------------------------- round 4 finding 3: prod_collection required
def test_live_apply_requires_prod_collection(tmp_path):
    """Round 4 finding 3: no silent `manifest["collection"]` fallback --
    a caller that forgot to resolve the live alias must fail loudly at
    the call boundary, not write the wrong, possibly-deleted collection."""
    with pytest.raises(TypeError, match="prod_collection"):
        Q.live_apply(pack_id="p1", manifest=MANIFEST_BASE,
                    twin_url="http://twin", prod_url="http://prod",
                    task="641-test", batch=10, pause_s=0,
                    artifact_root=tmp_path / "artifacts")


def test_resume_prod_only_requires_prod_collection(tmp_path):
    with pytest.raises(TypeError, match="prod_collection"):
        Q.resume_prod_only(
            pack_id="resume-pack", manifest=MANIFEST_BASE, twin_url="http://x",
            prod_url="http://y", task="641-test", batch=10, pause_s=0,
            artifact_root=tmp_path / "artifacts")


def test_live_apply_writes_twin_then_prod_under_both_locks(tmp_path, monkeypatch):
    _stub_plan_deps(monkeypatch)
    upserts = []
    monkeypatch.setattr(
        PSP, "upsert",
        lambda url, coll, points, batch, pause_s=0.0, parallel=1:
            upserts.append((url, len(points))) or len(points))
    monkeypatch.setattr(PSP.LAYOUT, "fingerprint",
                        lambda *_a, **_k: {"points_count": 3})
    monkeypatch.setattr(Q.TFA, "ProductionWriterLock", _FakeLock)
    lock_calls = _fake_run_batch(monkeypatch)

    state = Q.live_apply(pack_id="p1", manifest=MANIFEST_BASE,
                         twin_url="http://twin", prod_url="http://prod",
                         task="641-test", batch=10, pause_s=0,
                         artifact_root=tmp_path / "artifacts",
                         prod_collection="aw_e8")

    assert state["twin_receipt"]["written"] == 3
    assert state["prod_receipt"]["written"] == 3
    assert [u for u, _n in upserts] == ["http://twin", "http://prod"], (
        "twin must be written and verified before prod is ever touched")
    assert [c for c in lock_calls if c[0] == "acquire"] == [
        ("acquire", "TWIN-WRITE-LOCK"), ("acquire", "PROD-WRITE-LOCK")]
    assert [c for c in lock_calls if c[0] == "release"] == [
        ("release", "PROD-WRITE-LOCK"), ("release", "TWIN-WRITE-LOCK")], (
        "locks release in reverse acquisition order")


def test_live_apply_twin_parallel_reaches_twin_only_never_prod(tmp_path, monkeypatch):
    """Coordinator's ruling, 2026-09-23: parallel upserts on the TWIN leg
    only, prod always stays sequential+throttled."""
    _stub_plan_deps(monkeypatch)
    seen = []
    monkeypatch.setattr(
        PSP, "upsert",
        lambda url, coll, points, batch, pause_s=0.0, parallel=1:
            seen.append((url, pause_s, parallel)) or len(points))
    monkeypatch.setattr(PSP.LAYOUT, "fingerprint",
                        lambda *_a, **_k: {"points_count": 3})
    monkeypatch.setattr(Q.TFA, "ProductionWriterLock", _FakeLock)
    _fake_run_batch(monkeypatch)

    Q.live_apply(pack_id="p1", manifest=MANIFEST_BASE,
                twin_url="http://twin", prod_url="http://prod",
                task="641-test", batch=10, pause_s=0.3,
                artifact_root=tmp_path / "artifacts", twin_parallel=8,
                prod_collection="aw_e8")

    twin_call = next(c for c in seen if c[0] == "http://twin")
    prod_call = next(c for c in seen if c[0] == "http://prod")
    assert twin_call == ("http://twin", 0, 8)
    assert prod_call == ("http://prod", 0.3, 1)


def test_live_apply_refuses_a_foreign_lock_before_writing_anything(
        tmp_path, monkeypatch):
    _stub_plan_deps(monkeypatch)
    monkeypatch.setattr(PSP, "upsert", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("must not write with a foreign lock held")))
    _fake_run_batch(monkeypatch, foreign="task=626 owner=news-backlog")
    with pytest.raises(RuntimeError, match="cross-task lock held"):
        Q.live_apply(pack_id="p1", manifest=MANIFEST_BASE,
                    twin_url="http://twin", prod_url="http://prod",
                    task="641-test", batch=10, pause_s=0,
                    artifact_root=tmp_path / "artifacts",
                    prod_collection="aw_e8")


def test_rehearse_writes_twin_only_then_rolls_back(tmp_path, monkeypatch):
    _stub_plan_deps(monkeypatch)
    monkeypatch.setattr(PSP, "upsert",
                        lambda url, coll, points, batch, pause_s=0.0, parallel=1: len(points))
    monkeypatch.setattr(PSP.LAYOUT, "fingerprint",
                        lambda *_a, **_k: {"points_count": 100})
    lock_calls = _fake_run_batch(monkeypatch)
    monkeypatch.setattr(Q.RB, "assert_all_present",
                        lambda *_a, **_k: {"status": "PASS"})
    deleted: list[list[str]] = []
    monkeypatch.setattr(Q.RB, "delete_ids",
                        lambda url, coll, ids: deleted.append(list(ids)))
    monkeypatch.setattr(Q.RB, "assert_rolled_back",
                        lambda *_a, **_k: {"status": "PASS"})

    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    pack_dir = write_pack(queue_dir, MANIFEST_BASE)

    result = Q.rehearse(pack_dir, twin_url="http://twin", task="641-test",
                        batch=10, pause_s=0, artifact_root=tmp_path / "artifacts")

    assert result["status"] == "REHEARSED-AND-ROLLED-BACK"
    assert deleted and len(deleted[0]) == 3
    assert lock_calls == [("acquire", "TWIN-WRITE-LOCK"),
                          ("release", "TWIN-WRITE-LOCK")], (
        "rehearsal must never touch PROD-WRITE-LOCK")


# ---------------------------------------------- task 664: prod-url drift
# Real incident, 2026-09-26: a real --apply run's PROD leg ABORTED
# mid-write on ConnectionRefusedError. The CLI's --prod-url default
# (16333) had no listener on the build box at all -- the real forward is
# aw-prod-tunnel.service's build-box:6334 -> prod-host:6333 (created when
# prod flipped to aw_e8; the many other build scripts still
# defaulting to 16333 are a stale pre-flip reference, not fixed here).
# Twin was 200,616 points ahead of prod at the time -- exactly the silent
# divergence corpus-doctrine's twin-mirror rule exists to catch.

def test_cli_prod_url_default_points_at_the_real_tunnel_not_the_stale_port():
    import inspect
    source = inspect.getsource(Q.main)
    assert '"--prod-url", default="http://127.0.0.1:6334"' in source
    assert '"--prod-url", default="http://127.0.0.1:16333"' not in source


def test_promote_nightly_wrapper_states_the_prod_url_explicitly():
    """Defense in depth: the nightly wrapper must not depend on
    promote_queue.py's own default staying correct -- it says the real
    tunnel port itself."""
    script = _nightly_script()
    assert "--prod-url http://127.0.0.1:6334" in script
    assert "--prod-url http://127.0.0.1:16333" not in script


# ------------------------------------------------------- throughput (task 689)
def test_dry_plan_looks_up_at_the_twin_width(monkeypatch):
    seen = {}

    def fake_plan(**kw):
        seen.update(kw)
        raise RuntimeError("stop")
    monkeypatch.setattr(PSP, "plan", fake_plan)
    with pytest.raises(RuntimeError):
        Q.dry_plan(MANIFEST_BASE, "http://x")
    assert seen["lookup_workers"] == Q.LOOKUP_WORKERS["twin"]


@pytest.mark.parametrize("role", ["twin", "prod"])
def test_apply_side_looks_up_at_its_own_role_width(monkeypatch, tmp_path, role):
    seen = {}

    def fake_plan(**kw):
        seen.update(kw)
        raise RuntimeError("stop")
    monkeypatch.setattr(PSP, "plan", fake_plan)
    with pytest.raises(RuntimeError):
        Q.apply_side(role, "http://x", MANIFEST_BASE, "aw_e8", batch=500,
                     pause_s=0, ids_dir=tmp_path)
    assert seen["lookup_workers"] == Q.LOOKUP_WORKERS[role]
    # prod is a 30 GB box that also serves search: never wider than the twin
    assert Q.LOOKUP_WORKERS["prod"] <= Q.LOOKUP_WORKERS["twin"]


def test_run_passes_repeats_only_while_a_pass_lands_something():
    landed = iter([3, 1, 0, 9])
    n = []

    def one_pass():
        n.append(1)
        return next(landed)
    assert Q.run_passes(one_pass, repeat_while_landing=True) == 0
    assert len(n) == 3, "stops on the first pass that lands nothing"


def test_run_passes_without_the_flag_is_one_pass():
    n = []
    assert Q.run_passes(lambda: n.append(1) or 5, repeat_while_landing=False) == 0
    assert len(n) == 1


def test_promote_nightly_wrapper_throttle_and_repeat_are_tracked():
    script = _nightly_script()
    assert "--batch 1000 --pause-s 0.3" in script
    assert "--repeat-while-landing" in script
    assert "--budget-seconds" in script   # every pass still obeys the unit's clock


def test_main_accepts_repeat_while_landing_through_argparse(monkeypatch, tmp_path, capsys):
    """Task 689 review blocker: the flag was read but never declared, so every
    firing would have exited 2. This goes THROUGH argparse."""
    calls = []
    monkeypatch.setattr(Q, "drain", lambda *a, **k: calls.append(k) or [])
    rc = Q.main(["--queue-dir", str(tmp_path), "--repeat-while-landing",
                 "--no-refresh-persisted-index", "--apply"])
    assert rc == 0
    assert len(calls) == 1, "an empty pass ends the repeat"
    assert "[drain] pass 1" in capsys.readouterr().out


def test_main_without_the_flag_still_parses(monkeypatch, tmp_path):
    monkeypatch.setattr(Q, "drain", lambda *a, **k: [])
    assert Q.main(["--queue-dir", str(tmp_path), "--no-refresh-persisted-index"]) == 0


def test_wrapper_command_line_parses_against_the_real_parser(monkeypatch, tmp_path):
    """The exact flags promote_nightly.sh passes (read from the script, not
    retyped), argparse only, drain stubbed."""
    import shlex
    script = _nightly_script()
    tail = script[script.index("exec python3 -m cytir.index.promote_queue"):]
    words = shlex.split(tail.replace("\\\n", " "))
    argv = [w for w in words[4:] if not w.startswith("${")]
    argv = [str(tmp_path) if a.endswith("/work/promotion-queue") else a
            for a in argv] + ["--budget-seconds", "21600",
                              "--started-at-epoch", "1790832482",
                              "--no-refresh-persisted-index"]
    seen = {}
    monkeypatch.setattr(Q, "drain", lambda *a, **k: seen.update(k) or [])
    assert Q.main(argv) == 0
    assert seen["batch"] == 1000 and seen["pause_s"] == 0.3
    assert seen["twin_parallel"] == 8


# ---- single_apply check 5 against a receipt built by the real projection (task 605 review, S1/S2)
def _single_with_real_rows(monkeypatch, tmp_path, exclude_ids, manifest):
    """`single_apply` with only the transport stubbed. The receipt's `rows` block is built by
    `project_rows` over the real fixture pack, the same call and the same keys `plan` reports."""
    chunks = str(HERE / "fixtures" / "promote" / "task629_sealed_pack" / "chunks")
    recs = list(PSP.read_records(chunks))
    ids = [str(r["point_id"]) for r in recs]
    joined, skipped, modes, _vk, undated = PSP.project_rows(
        iter(recs), consent=None, attribution=None, license_version=None,
        identity_check="pack-declared", identity_reason="test: fixture ids are not reproducible",
        exclude_ids=frozenset(exclude_ids(ids)) or None)
    rows = {"projected": len(joined), "skipped_by_projection": skipped,
            "excluded_undated": len(undated),
            "excluded_out_of_scope_language": modes.get("excluded_out_of_scope_language", 0)}

    class _Locks:
        def acquire(self, *a, **k):
            pass
        release = acquire

    monkeypatch.setattr(Q, "_load_run_batch", lambda: _Locks())
    monkeypatch.setattr(Q, "assert_single_target_not_live", lambda s: None)
    monkeypatch.setattr(Q.LAYOUT, "fingerprint", lambda url, coll: {"points_count": 0})
    monkeypatch.setattr(Q, "_single_persisted_check", lambda: None)
    w = len(joined)
    monkeypatch.setattr(Q, "apply_side", lambda *a, **k: (
        {"written": w, "layout_after": {"points_count": w}, "rows": rows}, tmp_path / "ids.json"))
    out = Q.single_apply(pack_id="p", manifest={**manifest, "rows_staged": len(ids)},
                         single=Q.Single(url="http://127.0.0.1:1", collection="c"), task="t",
                         batch=1, artifact_root=tmp_path / "art")
    return out, rows, len(ids), w


def test_single_check5_part_c_is_skipped_when_an_exclusion_file_dropped_rows(monkeypatch, tmp_path):
    # the exclusion-file drop is reported as `excluded_out_of_scope_language`, so part (c) of
    # check 5 does not bind, whatever the manifest declares (same before and after the two names)
    out, rows, staged, written = _single_with_real_rows(
        monkeypatch, tmp_path, lambda ids: ids[:1], {"rows": 0, "rows_to_write_max": 999})
    assert rows["excluded_out_of_scope_language"] == 1 and written == staged - 1
    assert out["check5"]["manifest_rows"] == 999 and out["check5"]["written"] == written


def test_single_check5_part_c_still_pins_a_pack_with_no_drop(monkeypatch, tmp_path):
    with pytest.raises(Q.QueueError, match="declares"):
        _single_with_real_rows(monkeypatch, tmp_path, lambda ids: [],
                               {"rows": 0, "rows_to_write_max": 999})


def test_rows_to_write_max_is_an_upper_bound_a_language_scope_drop_is_not_refused(monkeypatch, tmp_path):
    # compose-time bound = the whole stage (no exclusion file); the drain drops one row for
    # language scope, so written < rows_to_write_max, and nothing this PR touches refuses it
    staged = len(list(PSP.read_records(str(HERE / "fixtures" / "promote" / "task629_sealed_pack" / "chunks"))))
    out, rows, _s, written = _single_with_real_rows(
        monkeypatch, tmp_path, lambda ids: ids[-1:], {"rows": staged, "rows_to_write_max": staged})
    assert written < out["check5"]["manifest_rows"] == staged


def test_2933_the_writer_lock_pair_and_its_order_are_pinned():
    """The drain takes the locks in this order and gives them back in the reverse. The test fakes read this
    tuple (`from cytir.ops.write_lock import LOCKS`), so a change here is a change of the protocol, said once."""
    from cytir.ops import write_lock
    assert write_lock.LOCKS == ("TWIN-WRITE-LOCK", "PROD-WRITE-LOCK")
    assert write_lock.LOCKS[0].startswith("TWIN") and write_lock.LOCKS[-1].startswith("PROD")   # twin first, prod last


# ---------------------------------------------------------------- task 2933: pins from the census of drain()
def _drain_plain(tmp_path, **kw):
    return Q.drain(tmp_path / "queue", twin_url="http://x", prod_url="http://y", task="2933-test", apply=True,
                   batch=10, pause_s=0, artifact_root=tmp_path / "artifacts",
                   contract_path=write_contract(tmp_path, "mechanical"), refresh_index=False, **kw)


def _one_pack_queue(tmp_path, monkeypatch):
    _stub_plan_deps(monkeypatch)
    (tmp_path / "queue").mkdir()
    return write_pack(tmp_path / "queue", MANIFEST_BASE)


def test_2933_a_pack_marked_carried_by_flip_is_passed_over(tmp_path, monkeypatch):
    pack = _one_pack_queue(tmp_path, monkeypatch)
    (pack / "CARRIED-BY-FLIP").write_text("flip\n")
    assert _drain_plain(tmp_path) == []


def test_2933_the_pass_over_line_counts_each_retired_marker(tmp_path, monkeypatch, capsys):
    pack = _one_pack_queue(tmp_path, monkeypatch)
    (pack / "DONE").write_text("x\n")
    assert _drain_plain(tmp_path) == []
    assert "1 DONE" in capsys.readouterr().out


def test_2933_a_drain_without_urls_and_without_a_transaction_is_refused(tmp_path):
    (tmp_path / "queue").mkdir()
    with pytest.raises(Q.QueueError, match="twin_url and prod_url are required"):
        Q.drain(tmp_path / "queue", task="2933-test", apply=True, batch=10, pause_s=0,
                artifact_root=tmp_path / "artifacts", refresh_index=False)


def test_2933_the_skip_cache_only_skips_a_refused_receipt(tmp_path, monkeypatch):
    """A receipt that says anything but REFUSED, at the same manifest sha, never skips the pack."""
    import hashlib
    pack = _one_pack_queue(tmp_path, monkeypatch)
    sha = hashlib.sha256((pack / "manifest.json").read_bytes()).hexdigest()
    (pack / "receipt.json").write_text(json.dumps({"status": "PROVED-DRY-RUN", "stage": "plan", "manifest_sha": sha,
                                                    "transaction_key": None}))
    res = _drain_plain(tmp_path)
    assert res and res[0]["status"] != "SKIPPED-REFUSED-UNCHANGED"


def test_2938_a_refusal_another_transaction_wrote_never_skips_the_twin_first_drain(tmp_path, monkeypatch):
    """M6: the skip cache is keyed by (manifest sha, transaction key). A cacheable refusal whose receipt carries
    a Single transaction's key (the pack was refused by a drain onto one named target) does NOT make the
    twin-first drain skip the pack: the check runs again and writes its own receipt, without a key."""
    monkeypatch.setattr(PSP, "target_width", lambda *_a, **_k: 256)
    monkeypatch.setattr(PSP, "already_present", lambda *_a, **_k: [])
    monkeypatch.setattr(PSP, "already_served_urls", lambda *_a, **_k: [])
    monkeypatch.setattr(PSP.PC, "assert_scoreable", lambda *_a, **_k: {"verdict": "PASS"})
    monkeypatch.setattr(PSP.LAYOUT, "fingerprint", lambda *_a, **_k: {})
    calls: list[int] = []

    def refuse(*_a, **_k):
        calls.append(1)
        raise PSP.PI.UnpersistedInputs("nope, still unpersisted", {"verdict": "REFUSED"})
    monkeypatch.setattr(PSP.PI, "assert_pages_persisted", refuse)
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "keyed-refusal"})
    contract = write_contract(tmp_path, "battery-required")
    common = dict(twin_url="http://x", prod_url="http://y", task="2938-test", apply=True, batch=10,
                  pause_s=0, artifact_root=tmp_path / "artifacts", contract_path=contract,
                  refresh_index=False)
    assert Q.drain(queue_dir, **common)[0]["status"] == "REFUSED"
    receipt_path = queue_dir / "keyed-refusal" / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["stage"] in Q._CACHEABLE_REFUSAL_STAGES and "transaction_key" not in receipt
    assert Q.drain(queue_dir, **common)[0]["status"] == "SKIPPED-REFUSED-UNCHANGED" and len(calls) == 1
    receipt_path.write_text(json.dumps({**receipt, "transaction_key": "single:http://elsewhere:aw_other"}))
    again = Q.drain(queue_dir, **common)
    assert again[0]["status"] == "REFUSED" and len(calls) == 2, "a foreign transaction's receipt skipped the pack"
    assert "transaction_key" not in json.loads(receipt_path.read_text())


def test_2938_a_pack_with_zero_passages_is_refused_at_intake_by_the_drain_before_any_plan(tmp_path, monkeypatch):
    """I2: the real-path smoke test runs inside `drain()` at intake. A sealed pack with 0 passages is a REFUSED
    receipt at stage `intake` (receipt on disk, no DONE, READY kept), the plan never runs, the next pack still
    gets its turn. (The unit tests above call `assert_real_path` directly; none of them proves the drain calls it.)"""
    empty_chunks = tmp_path / "empty_chunks"
    empty_chunks.mkdir()
    (empty_chunks / "shard-0000.texts.jsonl.gz").write_bytes(__import__("gzip").compress(b""))
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "aa-empty", "pack_chunks": str(empty_chunks)})
    write_pack(queue_dir, {**MANIFEST_BASE, "pack_id": "zz-next"})
    planned = []
    monkeypatch.setattr(Q, "dry_plan", lambda m, *a, **k: planned.append(m["pack_id"]) or (_ for _ in ()).throw(
        PSP.PromotionError("stop here")))
    res = Q.drain(queue_dir, twin_url="http://x", prod_url="http://y", task="2938-test", apply=True, batch=10,
                  pause_s=0, artifact_root=tmp_path / "artifacts", contract_path=write_contract(tmp_path, "mechanical"),
                  refresh_index=False)
    first = {r["pack_id"]: r for r in res}["aa-empty"]
    assert first["status"] == "REFUSED" and first["stage"] == "intake" and "0 passages" in first["reason"]
    saved = json.loads((queue_dir / "aa-empty" / "receipt.json").read_text())
    assert saved["status"] == "REFUSED" and saved["stage"] == "intake"
    assert (queue_dir / "aa-empty" / "READY").exists() and not (queue_dir / "aa-empty" / "DONE").exists()
    assert "aa-empty" not in planned and "zz-next" in planned


_PROD_URL, _TGT_URL, _TWIN_URL = "http://prod:2", "http://t:3", "http://twin:1"      # never connected to: the drain is stubbed

# ---- task 2938: `main()` -- flags, exit codes, refusals (the real argparse, real verbs, `drain` at its boundary) ---

def _spy_drain(monkeypatch, results=None):
    seen = []
    monkeypatch.setattr(Q, "drain", lambda *a, **k: seen.append(k) or list(results or []))
    return seen


def test_2938_main_hands_every_flag_to_the_drain_and_defaults_are_the_documented_ones(monkeypatch, tmp_path, capsys):
    seen = _spy_drain(monkeypatch)
    assert Q.main(["--queue-dir", str(tmp_path)]) == 0
    d = seen[0]
    assert d["apply"] is False and d["refresh_index"] is True and d["only"] is None
    assert d["artifact_root"] == tmp_path / "_artifacts" and d["batch"] == Q.DEFAULT_BATCH
    assert d["pause_s"] == Q.DEFAULT_PAUSE_S and d["budget_seconds"] is None and d["stop_margin_s"] == 5400.0
    assert d["prod_alias"] == "aw_prod" and d["prod_url"] == "http://127.0.0.1:6334" and d["twin_parallel"] == 1
    assert d["full_persisted_index"] is False and d["dry_receipt_dir"] is None and "transaction" not in d
    seen.clear()
    monkeypatch.setattr(Q, "sparse_rule_startup_refusal", lambda *a, **k: None)   # --apply asks prod: see the test below
    assert Q.main(["--queue-dir", str(tmp_path), "--twin-url", _TWIN_URL, "--prod-url", _PROD_URL,
                   "--task", "T9", "--batch", "7", "--pause-s", "0.5", "--artifact-root", str(tmp_path / "art"),
                   "--gate-contract", str(tmp_path / "gate.json"), "--apply", "--budget-seconds", "99",
                   "--started-at-epoch", "12345", "--stop-margin-s", "11", "--no-refresh-persisted-index",
                   "--full-persisted-index", "--only", "b-pack", "--only", "a-pack", "--twin-parallel", "4",
                   "--prod-alias", "aw_alias", "--dry-receipt-dir", str(tmp_path / "dry")]) == 0
    d = seen[0]
    assert (d["twin_url"], d["prod_url"], d["task"], d["batch"], d["pause_s"]) == (
        _TWIN_URL, _PROD_URL, "T9", 7, 0.5)
    assert d["artifact_root"] == tmp_path / "art" and d["contract_path"] == tmp_path / "gate.json"
    assert d["apply"] is True and d["refresh_index"] is False and d["full_persisted_index"] is True
    assert (d["budget_seconds"], d["started_at_epoch"], d["stop_margin_s"]) == (99.0, 12345.0, 11.0)
    assert d["only"] == ["b-pack", "a-pack"] and d["twin_parallel"] == 4 and d["prod_alias"] == "aw_alias"
    assert d["dry_receipt_dir"] == str(tmp_path / "dry") and "transaction" not in d
    out = capsys.readouterr().out
    assert "[drain] 0 pack(s): 0 landed" in out
    # task 696 slice 4: both location lines are printed first, before any drain
    assert Q.PATHS.location_line("queue_dir", str(tmp_path), "queue_dir") in out and Q.PATHS.lock_location_line() in out


def test_2938_main_with_a_target_builds_the_single_transaction_and_no_twin_or_prod_url(monkeypatch, tmp_path):
    seen = _spy_drain(monkeypatch)
    monkeypatch.setattr(Q, "single_startup_refusal", lambda **k: None)
    assert Q.main(["--queue-dir", str(tmp_path), "--target-url", _TGT_URL, "--target-collection", "aw_c",
                   "--target-parallel", "3", "--target-payload-profile-of", "aw_e9",
                   "--target-state-dir", str(tmp_path / "st"), "--prod-alias", "aw_alias"]) == 0
    d = seen[0]
    assert "twin_url" not in d and "prod_url" not in d
    tx = d["transaction"]
    assert isinstance(tx, Q.Single) and (tx.url, tx.collection, tx.parallel) == (_TGT_URL, "aw_c", 3)
    assert tx.payload_profile_of == "aw_e9" and tx.state_dir == tmp_path / "st" and tx.prod_alias == "aw_alias"


@pytest.mark.parametrize("argv, message", [
    (["--target-url", _TGT_URL], "needs an explicit --queue-dir"),
    (["--target-collection", "aw_c"], "needs an explicit --queue-dir"),
    (["--queue-dir", "@Q", "--target-url", _TGT_URL], "--target-url and --target-collection go together"),
    (["--queue-dir", "@Q", "--target-collection", "aw_c"], "--target-url and --target-collection go together"),
    (["--queue-dir", "@Q", "--target-payload-profile-of", "aw_e9"], "--target-payload-profile-of needs --target-url"),
    (["--queue-dir", "@Q", "--full-persisted-index"], "full-persisted-index"),
    (["--queue-dir", "@Q/missing"], "no such queue dir"),
])
def test_2938_main_refusals_are_system_exits_that_name_the_cause_and_start_no_drain(
        monkeypatch, tmp_path, capsys, argv, message):
    seen = _spy_drain(monkeypatch)
    monkeypatch.setattr(Q, "single_startup_refusal", lambda **k: None)
    argv = [a.replace("@Q", str(tmp_path)) for a in argv]
    with pytest.raises(SystemExit) as ei:
        Q.main(argv)
    assert str(ei.value).startswith("[REFUSED]") and message in str(ei.value), ei.value
    assert seen == []


def test_2938_a_full_index_rebuild_is_refused_without_apply_and_allowed_with_it(monkeypatch, tmp_path):
    seen = _spy_drain(monkeypatch)
    with pytest.raises(SystemExit):
        Q.main(["--queue-dir", str(tmp_path), "--full-persisted-index"])
    assert seen == []
    assert Q.main(["--queue-dir", str(tmp_path), "--full-persisted-index", "--apply"]) == 0
    assert seen[0]["full_persisted_index"] is True


def test_2938_a_single_never_runs_on_the_live_promotion_queue(monkeypatch, tmp_path):
    seen = _spy_drain(monkeypatch)
    monkeypatch.setattr(Q, "single_startup_refusal", lambda **k: None)
    monkeypatch.setattr(Q, "live_queue_dir", lambda: tmp_path.resolve())
    with pytest.raises(SystemExit, match="refuses the live promotion queue"):
        Q.main(["--queue-dir", str(tmp_path), "--target-url", _TGT_URL, "--target-collection", "aw_c"])
    assert seen == []


def test_2938_the_prod_alias_and_rule_check_is_for_an_apply_twin_first_drain_only(monkeypatch, tmp_path, capsys):
    """sparse_rule_startup_refusal: an apply drain whose prod alias cannot be resolved exits 2 BEFORE any drain; a
    dry run and a Single transaction never ask prod, and exit 0 although the alias cannot be resolved."""
    seen = _spy_drain(monkeypatch)
    monkeypatch.setattr(Q, "single_startup_refusal", lambda **k: None)

    def unreachable(*a, **k):
        raise ConnectionError("prod tunnel is down")
    monkeypatch.setattr(Q, "resolve_prod_alias", unreachable)
    assert Q.main(["--queue-dir", str(tmp_path), "--apply"]) == 2
    assert seen == [] and "cannot resolve alias" in capsys.readouterr().err
    assert Q.main(["--queue-dir", str(tmp_path)]) == 0 and len(seen) == 1                       # dry run
    assert Q.main(["--queue-dir", str(tmp_path), "--apply", "--target-url", _TGT_URL,
                   "--target-collection", "aw_c"]) == 0 and len(seen) == 2                       # Single


def test_2938_a_drain_that_raises_leaves_main_as_an_exception_never_as_exit_0(monkeypatch, tmp_path):
    """`drain()` re-raises a mid-write ABORTED: `main()` lets it out (the process exits non-zero with a traceback);
    it is not turned into a return code. The pass loop does not run another pass after it."""
    calls = []

    def aborted(*a, **k):
        calls.append(1)
        raise RuntimeError("ABORTED mid-write")
    monkeypatch.setattr(Q, "drain", aborted)
    with pytest.raises(RuntimeError, match="ABORTED mid-write"):
        Q.main(["--queue-dir", str(tmp_path), "--repeat-while-landing"])
    assert calls == [1]


def test_2938_the_pass_summary_counts_only_landed_packs_as_progress(monkeypatch, tmp_path, capsys):
    """The pass loop goes on only after a pass that LANDED a pack: blocked, refused, skipped and budget results
    are not progress (a loop that took them for progress would never end)."""
    passes = [[{"status": "LANDED"}, {"status": "PROVED-BLOCKED"}, {"status": "REFUSED"}],
              [{"status": "PROVED-BLOCKED"}, {"status": "REFUSED"}, {"status": "SKIPPED-REFUSED-UNCHANGED"},
               {"status": "SKIPPED-BUDGET"}, {"status": "BUDGET-EXCEEDS-TIMEOUT"}]]
    seen = []
    monkeypatch.setattr(Q, "drain", lambda *a, **k: seen.append(1) or passes[len(seen) - 1])
    assert Q.main(["--queue-dir", str(tmp_path), "--repeat-while-landing", "--apply"]) == 0
    assert len(seen) == 2
    out = capsys.readouterr().out
    assert "[drain] 3 pack(s): 1 landed, 1 blocked, 1 refused, 0 skipped (refused, unchanged), 0 skipped (budget)" in out
    assert "[drain] 5 pack(s): 0 landed, 1 blocked, 1 refused, 1 skipped (refused, unchanged), 1 skipped (budget), 1 exceed" in out


def _queue_with_pack(tmp_path):
    queue = tmp_path / "queue"
    queue.mkdir()
    pack = write_pack(queue, {**MANIFEST_BASE, "pack_id": "verb-pack"})
    return queue, pack


def test_2938_the_supersede_verb_retires_the_pack_without_a_drain(monkeypatch, tmp_path, capsys):
    queue, pack = _queue_with_pack(tmp_path)
    monkeypatch.setattr(Q, "drain", lambda *a, **k: pytest.fail("a verb must not drain"))
    with pytest.raises(SystemExit, match="--supersede requires --reason"):
        Q.main(["--queue-dir", str(queue), "--supersede", "verb-pack"])
    with pytest.raises(SystemExit, match="no such pack dir"):
        Q.main(["--queue-dir", str(queue), "--supersede", "nope", "--reason", "r"])
    assert not (pack / "SUPERSEDED").exists() and not (pack / "receipt.json").exists()
    assert Q.main(["--queue-dir", str(queue), "--supersede", "verb-pack", "--reason", "a duplicate",
                   "--task", "T7"]) == 0
    got = json.loads((pack / "receipt.json").read_text())
    assert (got["status"], got["reason"], got["task"]) == ("SUPERSEDED", "a duplicate", "T7")
    assert (pack / "SUPERSEDED").exists() and not (pack / "CARRIED-BY-FLIP").exists() and not (pack / "DONE").exists()


def test_2938_the_mark_carried_by_flip_verb_writes_its_own_marker_not_the_supersede_one(monkeypatch, tmp_path):
    queue, pack = _queue_with_pack(tmp_path)
    monkeypatch.setattr(Q, "drain", lambda *a, **k: pytest.fail("a verb must not drain"))
    with pytest.raises(SystemExit, match="--mark-carried-by-flip requires --reason"):
        Q.main(["--queue-dir", str(queue), "--mark-carried-by-flip", "verb-pack"])
    with pytest.raises(SystemExit, match="no such pack dir"):
        Q.main(["--queue-dir", str(queue), "--mark-carried-by-flip", "nope", "--reason", "r"])
    assert not (pack / "CARRIED-BY-FLIP").exists()
    assert Q.main(["--queue-dir", str(queue), "--mark-carried-by-flip", "verb-pack", "--reason", "flip follows",
                   "--task", "T8"]) == 0
    got = json.loads((pack / "receipt.json").read_text())
    assert (got["status"], got["reason"], got["task"]) == ("CARRIED-BY-FLIP", "flip follows", "T8")
    assert (pack / "CARRIED-BY-FLIP").exists() and not (pack / "SUPERSEDED").exists() and not (pack / "DONE").exists()


def test_2938_the_finalize_verb_asks_the_pack_marker_and_is_given_prod_alias_twin_and_artifact_root(
        monkeypatch, tmp_path):
    queue, pack = _queue_with_pack(tmp_path)
    monkeypatch.setattr(Q, "drain", lambda *a, **k: pytest.fail("a verb must not drain"))
    with pytest.raises(SystemExit, match="no such pack dir"):
        Q.main(["--queue-dir", str(queue), "--finalize-carried-by-flip", "nope"])
    with pytest.raises(Q.QueueError, match="not CARRIED-BY-FLIP -- nothing to finalize"):
        Q.main(["--queue-dir", str(queue), "--finalize-carried-by-flip", "verb-pack"])
    (pack / "CARRIED-BY-FLIP").write_text("t\n")
    seen = []
    monkeypatch.setattr(Q, "finalize_carried_by_flip", lambda p, **k: seen.append((p.name, k)) or {"status": "DONE"})
    assert Q.main(["--queue-dir", str(queue), "--finalize-carried-by-flip", "verb-pack", "--prod-url", _PROD_URL,
                   "--twin-url", _TWIN_URL, "--prod-alias", "aw_alias", "--task", "T6",
                   "--artifact-root", str(tmp_path / "art")]) == 0
    assert seen == [("verb-pack", {"prod_url": _PROD_URL, "task": "T6", "artifact_root": tmp_path / "art",
                                   "prod_alias": "aw_alias", "twin_url": _TWIN_URL})]


def test_2938_the_rehearse_flag_runs_the_twin_only_rehearsal_and_exits_by_its_verdict(monkeypatch, tmp_path):
    queue, pack = _queue_with_pack(tmp_path)
    monkeypatch.setattr(Q, "drain", lambda *a, **k: pytest.fail("a rehearsal must not drain"))
    with pytest.raises(SystemExit, match="no such pack dir"):
        Q.main(["--queue-dir", str(queue), "--rehearse", "nope"])
    seen = []
    verdict = ["REHEARSED-AND-ROLLED-BACK"]
    monkeypatch.setattr(Q, "rehearse", lambda p, **k: seen.append((p.name, k)) or {"status": verdict[0]})
    argv = ["--queue-dir", str(queue), "--rehearse", "verb-pack", "--twin-url", _TWIN_URL, "--task", "T5",
            "--batch", "9", "--pause-s", "0.25", "--artifact-root", str(tmp_path / "art")]
    assert Q.main(argv) == 0
    verdict[0] = "REFUSED"
    assert Q.main(argv) == 1
    assert seen[0] == ("verb-pack", {"twin_url": _TWIN_URL, "task": "T5", "batch": 9, "pause_s": 0.25,
                                     "artifact_root": tmp_path / "art"}) and len(seen) == 2

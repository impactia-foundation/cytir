"""Tests for cytir/ops/growth_pending_compose.py (task 664, 2026-09-26).

The real incident this guards against: CC tranche 2/3's GPU embed finished
cleanly, SE tranche 8's chunk stage finished cleanly, and both sat idle
~12h because composing them into a promotion-queue pack was something only
an awake agent knew how to do. These tests build the exact real directory
shapes (CC's tranches/t<N>_gpu/<part>/{chunks,vectors}, SE's
production/stages/<label>-413a-portable-v3/) and prove the discovery
functions find them, skip what's already enqueued, and never touch a
directory that doesn't match the real shape.
"""
from __future__ import annotations

import gzip
import json
import re as re_module
from pathlib import Path

import pytest

import cytir.ops.growth_pending_compose as GPC


def _touch(path: Path, content: bytes = b"") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _write_gz_chunk_shard(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


# --------------------------------------------------------------- CC discovery

def test_discover_cc_pending_finds_lettered_parts(tmp_path):
    root = tmp_path / "task634"
    _touch(root / "tranches/t2_gpu/a/chunks/shard-0000.texts.jsonl.gz")
    _touch(root / "tranches/t2_gpu/a/vectors/shard-0000.vectors.f16")
    _touch(root / "tranches/t2_gpu/b/chunks/shard-0000.texts.jsonl.gz")

    jobs = GPC.discover_cc_pending(str(root))
    by_id = {j.pack_id: j for j in jobs}
    assert set(by_id) == {"cc-authority-frontier-w7-tranche2-a",
                          "cc-authority-frontier-w7-tranche2-b"}
    assert by_id["cc-authority-frontier-w7-tranche2-a"].family == \
        "cc_authority_frontier_w7"
    assert by_id["cc-authority-frontier-w7-tranche2-a"].chunks_dir == \
        str(root / "tranches/t2_gpu/a/chunks")
    assert by_id["cc-authority-frontier-w7-tranche2-a"].vectors_dir == \
        str(root / "tranches/t2_gpu/a/vectors")


def test_discover_cc_pending_all_part_has_no_letter_suffix(tmp_path):
    root = tmp_path / "task634"
    _touch(root / "tranches/t3_gpu/all/chunks/shard-0000.texts.jsonl.gz")

    jobs = GPC.discover_cc_pending(str(root))
    assert [j.pack_id for j in jobs] == ["cc-authority-frontier-w7-tranche3"]


def test_discover_cc_pending_skips_a_part_with_no_chunk_shards_yet(tmp_path):
    """A part dir can exist (mkdir'd by the embed driver) before any shard
    has actually landed -- must not be discovered as pending work."""
    root = tmp_path / "task634"
    (root / "tranches/t2_gpu/c/chunks").mkdir(parents=True)
    jobs = GPC.discover_cc_pending(str(root))
    assert jobs == []


def test_discover_cc_pending_ignores_non_tranche_directories(tmp_path):
    root = tmp_path / "task634"
    _touch(root / "tranches/t2_out/gated/passages_ok.jsonl")
    _touch(root / "tranches/other_stuff/chunks/shard-0000.texts.jsonl.gz")
    jobs = GPC.discover_cc_pending(str(root))
    assert jobs == []


# ---------------------------------------- task 664, incident 2: duplicates
# cc-authority-frontier-w7-tranche2 (the unlettered "all" restage) turned
# out to duplicate the already-landed lettered a-d split -- composing it
# recreated a twin/prod resume gap that needed a per-pack artifact that
# never existed. discover_cc_pending must recognize the duplicate BEFORE
# composing, given a twin_url to check against.

def test_stage_already_landed_fraction_is_zero_with_no_twin_url(tmp_path):
    shard = tmp_path / "chunks/shard-0000.texts.jsonl.gz"
    _write_gz_chunk_shard(shard, [{"point_id": "id-1"}])
    assert GPC._stage_already_landed_fraction(
        str(shard.parent), twin_url=None, collection="aw_e8") == 0.0


def test_stage_already_landed_fraction_reads_real_ids_against_twin(
        tmp_path, monkeypatch):
    chunks_dir = tmp_path / "chunks"
    _write_gz_chunk_shard(chunks_dir / "shard-0000.texts.jsonl.gz",
                          [{"point_id": "id-1"}, {"point_id": "id-2"},
                           {"point_id": "id-3"}, {"point_id": "id-4"}])
    monkeypatch.setattr(GPC.AIC.PSP, "already_present",
                        lambda url, coll, ids, **kw: ids[:3])

    fraction = GPC._stage_already_landed_fraction(
        str(chunks_dir), twin_url="http://x", collection="aw_e8")
    assert fraction == 0.75


def test_stage_already_landed_fraction_is_zero_when_the_twin_check_fails(
        tmp_path, monkeypatch):
    """Never blocks discovery -- an unreachable twin self-heals next tick."""
    chunks_dir = tmp_path / "chunks"
    _write_gz_chunk_shard(chunks_dir / "shard-0000.texts.jsonl.gz",
                          [{"point_id": "id-1"}])

    def boom(*_a, **_k):
        raise ConnectionError("twin unreachable")
    monkeypatch.setattr(GPC.AIC.PSP, "already_present", boom)

    assert GPC._stage_already_landed_fraction(
        str(chunks_dir), twin_url="http://x", collection="aw_e8") == 0.0


def test_discover_cc_pending_skips_a_duplicate_stage(tmp_path, monkeypatch):
    """The regression test: a re-staged 'all' pack whose ids are ~100% on
    twin already (under presumably a different, earlier pack_id) must not
    be discovered at all -- never composed, never enqueued, never handed
    to drain() to untangle."""
    root = tmp_path / "task634"
    chunks_dir = root / "tranches/t2_gpu/all/chunks"
    _write_gz_chunk_shard(chunks_dir / "shard-0000.texts.jsonl.gz",
                          [{"point_id": f"id-{i}"} for i in range(100)])
    monkeypatch.setattr(GPC.AIC.PSP, "already_present",
                        lambda url, coll, ids, **kw: ids)  # 100% present

    logged = []
    jobs = GPC.discover_cc_pending(str(root), twin_url="http://x",
                                   log=logged.append)
    assert jobs == []
    assert any("SKIPPED" in m and "duplicate" in m for m in logged)


def test_discover_cc_pending_keeps_a_stage_that_is_mostly_but_not_all_new(
        tmp_path, monkeypatch):
    """A stage with genuinely new rows (fraction below the duplicate
    threshold) must still be discovered -- this guard is for near-total
    overlap, not any overlap at all."""
    root = tmp_path / "task634"
    chunks_dir = root / "tranches/t4_gpu/all/chunks"
    _write_gz_chunk_shard(chunks_dir / "shard-0000.texts.jsonl.gz",
                          [{"point_id": f"id-{i}"} for i in range(100)])
    # Only half already on twin -- well under the 0.99 duplicate threshold.
    monkeypatch.setattr(GPC.AIC.PSP, "already_present",
                        lambda url, coll, ids, **kw: ids[:50])

    jobs = GPC.discover_cc_pending(str(root), twin_url="http://x")
    assert [j.pack_id for j in jobs] == ["cc-authority-frontier-w7-tranche4"]


def test_discover_cc_pending_missing_root_is_empty_not_an_error(tmp_path):
    assert GPC.discover_cc_pending(str(tmp_path / "does-not-exist")) == []


# --------------------------------------------------------------- SE discovery

def test_discover_se_pending_finds_a_staged_tranche(tmp_path):
    root = tmp_path / "n1-stackexchange-v1"
    label = "stackoverflow-qa-score10-tranche-070000-079999-v1"
    stage = root / "production/stages" / f"{label}-413a-portable-v3"
    _touch(stage / "STAGE-RECEIPT.json", b"{}")
    _touch(stage / "shard-0000.texts.jsonl.gz")
    vectors = (root / "vectors/release=aw-task631-so-tranche8-v1"
              / f"source={label}/vectors")
    _touch(vectors / "shard-0000.vectors.f16")
    # A finished embed carries its manifest; a dir with shards but no manifest
    # is an embed still in progress (task 680: `_find_vectors_dir` requires it).
    _touch(vectors / "shard-0000.vectors.manifest.json", b"{}")

    jobs = GPC.discover_se_pending(str(root))
    assert len(jobs) == 1
    job = jobs[0]
    assert job.pack_id == label
    assert job.family == "stackoverflow_qa"
    assert job.chunks_dir == str(stage)
    assert job.vectors_dir == str(vectors)
    # Task 664, 2026-09-26: `stackoverflow_qa` rows carry no top-level
    # `license` and no host-table entry resolves one -- every tranche this
    # discovered without a licence lift path refused at promotion with
    # "unresolved license". `source_metadata.accepted_answer.license` is
    # the SAME path `--license-from` already uses for the sibling family.
    assert job.license_from == "source_metadata.accepted_answer.license"


def test_discover_se_pending_sets_no_license_from_for_an_unmapped_family(
        tmp_path):
    """No regression: a family this module has no licence-lift convention
    for (or a caller that overrides `family=`) gets `license_from=None`,
    not a guess."""
    root = tmp_path / "n1-stackexchange-v1"
    label = "some-other-se-family-tranche-1"
    stage = root / "production/stages" / f"{label}-413a-portable-v3"
    _touch(stage / "STAGE-RECEIPT.json", b"{}")

    jobs = GPC.discover_se_pending(str(root), family="stackexchange_other_qa")
    assert len(jobs) == 1
    assert jobs[0].license_from is None


def test_discover_se_pending_without_vectors_yet_still_returns_the_job(tmp_path):
    """A stage that finished chunking but has not been embedded yet is
    STILL pending work -- embed_compose_enqueue embeds it first. The
    vectors_dir path is a not-yet-existing placeholder in this case."""
    root = tmp_path / "n1-stackexchange-v1"
    label = "stackoverflow-qa-score10-tranche-080000-089999-v1"
    stage = root / "production/stages" / f"{label}-413a-portable-v3"
    _touch(stage / "STAGE-RECEIPT.json", b"{}")

    jobs = GPC.discover_se_pending(str(root))
    assert len(jobs) == 1
    assert jobs[0].pack_id == label
    assert not Path(jobs[0].vectors_dir).exists()


def test_discover_se_pending_skips_an_unfinished_stage(tmp_path):
    """No STAGE-RECEIPT.json means chunking itself hasn't finished -- not
    this driver's job to touch a half-written stage."""
    root = tmp_path / "n1-stackexchange-v1"
    label = "stackoverflow-qa-score10-tranche-090000-099999-v1"
    _touch(root / "production/stages" / f"{label}-413a-portable-v3" /
          "shard-0000.texts.jsonl.gz")
    jobs = GPC.discover_se_pending(str(root))
    assert jobs == []


def test_discover_se_pending_ignores_non_413a_stage_dirs(tmp_path):
    root = tmp_path / "n1-stackexchange-v1"
    _touch(root / "production/stages/some-other-thing/STAGE-RECEIPT.json", b"{}")
    jobs = GPC.discover_se_pending(str(root))
    assert jobs == []


# ------------------------------------ SE: the twin decides, not the ledger (680)
# Real incident (task 680, 2026-09-30): the admission ledger named Stack
# Overflow tranches 1-7 as "admitted through an OLDER mechanism", so discovery
# skipped them -- but aw_e8 held 0 of 1,400 sampled ids and 0 of 700 sampled
# urls of those tranches (91,959 embedded points, unserved for weeks). The
# ledger records that vectors EXIST; only the collection records SERVING.

def _stage(root: Path, label: str, ids: list[str]) -> Path:
    stage = root / "production/stages" / f"{label}-413a-portable-v3"
    _touch(stage / "STAGE-RECEIPT.json", b"{}")
    _write_gz_chunk_shard(stage / "shard-0000.texts.jsonl.gz",
                          [{"point_id": i} for i in ids])
    return stage


def test_discover_se_pending_lists_an_unserved_tranche_whatever_a_ledger_says(
        tmp_path, monkeypatch):
    root = tmp_path / "n1-stackexchange-v1"
    label = "stackoverflow-qa-score10-tranche-060000-069999-v1"
    _stage(root, label, [f"id-{i}" for i in range(50)])
    monkeypatch.setattr(GPC.AIC.PSP, "already_present",
                        lambda url, coll, ids, **kw: [])  # 0% served
    jobs = GPC.discover_se_pending(str(root), twin_url="http://x")
    assert [j.pack_id for j in jobs] == [label]


def test_discover_se_pending_skips_a_tranche_the_twin_already_serves(
        tmp_path, monkeypatch):
    root = tmp_path / "n1-stackexchange-v1"
    served = "stackoverflow-qa-score10-tranche-070000-079999-v1"
    fresh = "stackoverflow-qa-score10-tranche-080000-089999-v1"
    _stage(root, served, [f"a-{i}" for i in range(50)])
    _stage(root, fresh, [f"b-{i}" for i in range(50)])
    monkeypatch.setattr(GPC.AIC.PSP, "already_present",
                        lambda url, coll, ids, **kw:
                        ids if ids[0].startswith("a-") else [])
    logged = []
    jobs = GPC.discover_se_pending(str(root), twin_url="http://x",
                                   log=logged.append)
    assert [j.pack_id for j in jobs] == [fresh]
    assert any(served in m and "SKIPPED" in m for m in logged)


def test_vectors_complete_needs_every_chunk_shard_to_have_a_vector_manifest(
        tmp_path):
    chunks, vectors = tmp_path / "chunks", tmp_path / "vectors"
    for tag in ("shard-0000", "shard-0001"):
        _touch(chunks / f"{tag}.texts.jsonl.gz")
    _touch(vectors / "shard-0000.vectors.manifest.json", b"{}")
    assert GPC.vectors_complete(str(chunks), str(vectors)) is False
    _touch(vectors / "shard-0001.vectors.manifest.json", b"{}")
    assert GPC.vectors_complete(str(chunks), str(vectors)) is True


def test_vectors_complete_is_false_for_an_empty_stage_or_a_missing_dir(tmp_path):
    assert GPC.vectors_complete(str(tmp_path / "c"), str(tmp_path / "v")) is False


def test_run_skips_the_embed_for_a_job_whose_vectors_are_complete(
        tmp_path, monkeypatch):
    """Task 680: embed_stage() refuses while the GPU is busy and would reopen a
    finished embed on an idle one; a complete vector dir goes straight to
    compose."""
    root = tmp_path / "n1-stackexchange-nonso-v1"
    _nonso_unit(root, "math.stackexchange.com-part19", ["p-1"])
    vroot = tmp_path / "vectors"
    _touch(vroot / ("release=r/source=stackexchange-nonso-qa-math."
                    "stackexchange.com-part19-v1/vectors/"
                    "shard-0000.vectors.manifest.json"), b"{}")
    monkeypatch.setattr(GPC.AIC.PSP, "already_present",
                        lambda url, coll, ids, **kw: [])
    seen = {}

    def fake(*, skip_embed, **kwargs):
        seen["skip_embed"] = skip_embed
        return {"ready_sha": "x"}
    monkeypatch.setattr(GPC.AIC, "embed_compose_enqueue", fake)
    GPC.run(cc_root=None, se_root=None, nonso_root=str(root),
            vectors_roots=[str(vroot)], queue_dir=str(tmp_path / "queue"),
            qdrant_url="http://x", log=lambda m: None)
    assert seen == {"skip_embed": True}


def test_se_label_aliases_maps_the_stage_label_to_the_pilot_vector_label():
    assert GPC._se_label_aliases(
        "stackoverflow-qa-score10-tranche-000000-009999-v1") == [
        "stackoverflow-qa-score10-tranche-000000-009999-v1",
        "stackoverflow-score10-production-tranche-000000-009999-v1"]


def test_find_vectors_dir_finds_both_on_disk_shapes(tmp_path):
    vroot = tmp_path / "vectors"
    # shape 1: release=*/source=<label>/vectors/ (every pack from task 631 on)
    a = vroot / "release=r1/source=label-a/vectors"
    _touch(a / "shard-0000.vectors.manifest.json", b"{}")
    # shape 2: shards directly in source=<label>/ (the e3-rebuild SO pilots)
    b = vroot / "release=r2/source=label-b"
    _touch(b / "shard-0000.vectors.manifest.json", b"{}")
    assert GPC._find_vectors_dir(vroot, ["label-a"]) == str(a)
    assert GPC._find_vectors_dir(vroot, ["label-b"]) == str(b)
    assert GPC._find_vectors_dir(vroot, ["nope", "label-b"]) == str(b)
    assert GPC._find_vectors_dir(vroot, ["nope"]) is None


def test_find_vectors_dir_ignores_a_dir_with_no_vector_manifest(tmp_path):
    vroot = tmp_path / "vectors"
    _touch(vroot / "release=r1/source=empty/vectors/readme.txt", b"x")
    assert GPC._find_vectors_dir(vroot, ["empty"]) is None


def test_discover_se_pending_finds_vectors_in_the_corpus_vectors_root(tmp_path):
    root = tmp_path / "n1-stackexchange-v1"
    corpus_vectors = tmp_path / "corpus/vectors"
    label = "stackoverflow-qa-score10-tranche-000000-009999-v1"
    _stage(root, label, ["id-1"])
    pilot = corpus_vectors / ("release=aw-e3-rebuild/source="
                              "stackoverflow-score10-production-tranche-"
                              "000000-009999-v1")
    _touch(pilot / "shard-0000.vectors.manifest.json", b"{}")
    jobs = GPC.discover_se_pending(str(root),
                                   vectors_roots=[str(corpus_vectors)])
    assert [j.vectors_dir for j in jobs] == [str(pilot)]


# ------------------------------------------------ non-SO SE discovery (680)

def _nonso_unit(root: Path, unit: str, ids: list[str]) -> Path:
    stage = root / unit / "stage-413a-portable-v3"
    _touch(stage / "STAGE-RECEIPT.json", b"{}")
    _write_gz_chunk_shard(stage / "shard-0000.texts.jsonl.gz",
                          [{"point_id": i} for i in ids])
    return stage


def test_discover_nonso_pending_finds_unqueued_parts_with_the_served_manifest_fields(
        tmp_path, monkeypatch):
    root = tmp_path / "n1-stackexchange-nonso-v1"
    stage = _nonso_unit(root, "math.stackexchange.com-part19", ["p-1", "p-2"])
    vroot = tmp_path / "vectors"
    vec = vroot / ("release=aw-task631-se-nonso-v1/source=stackexchange-nonso-"
                   "qa-math.stackexchange.com-part19-v1/vectors")
    _touch(vec / "shard-0000.vectors.manifest.json", b"{}")
    monkeypatch.setattr(GPC.AIC.PSP, "already_present",
                        lambda url, coll, ids, **kw: [])

    jobs = GPC.discover_nonso_pending(str(root), vectors_roots=[str(vroot)],
                                      twin_url="http://x")
    assert len(jobs) == 1
    job = jobs[0]
    assert job.pack_id == "stackexchange-nonso-qa-math.stackexchange.com-part19-v1"
    assert job.family == "stackexchange_nonso_qa_p0"
    assert job.chunks_dir == str(stage)
    assert job.vectors_dir == str(vec)
    # The same three fields the landed math-part18 manifest carries.
    assert job.license_from == "source_metadata.accepted_answer.license"
    assert job.family_payload == {"source_type": "community", "trust_tier": 0.55}
    assert job.title_from == "source_metadata.title"


def test_discover_nonso_pending_skips_served_and_unstaged_units(
        tmp_path, monkeypatch):
    root = tmp_path / "n1-stackexchange-nonso-v1"
    _nonso_unit(root, "unix.stackexchange.com", ["s-1"])
    (root / "askubuntu.com/pack").mkdir(parents=True)   # parent dir: no stage
    monkeypatch.setattr(GPC.AIC.PSP, "already_present",
                        lambda url, coll, ids, **kw: ids)  # all served
    assert GPC.discover_nonso_pending(str(root), twin_url="http://x") == []


def test_discover_nonso_pending_missing_root_is_empty_not_an_error(tmp_path):
    assert GPC.discover_nonso_pending(str(tmp_path / "nope")) == []


def test_pending_jobs_does_not_probe_the_twin_for_an_already_enqueued_stage(
        tmp_path, monkeypatch):
    """55+ stages x ~20k ids every 15 minutes is real CPU on the box the ERIC
    chain runs on: a stage whose manifest already sits in the queue is settled
    without a single twin call."""
    root = tmp_path / "n1-stackexchange-nonso-v1"
    _nonso_unit(root, "math.stackexchange.com-part1", ["a-1"])
    queue = tmp_path / "queue"
    _touch(queue / "stackexchange-nonso-qa-math.stackexchange.com-part1-v1"
           / "manifest.json", b"{}")

    def boom(*_a, **_k):
        raise AssertionError("twin probed for an already-enqueued stage")
    monkeypatch.setattr(GPC.AIC.PSP, "already_present", boom)
    assert GPC.pending_jobs(cc_root=None, se_root=None, nonso_root=str(root),
                            queue_dir=str(queue), twin_url="http://x") == []


def test_pending_jobs_includes_the_nonso_root(tmp_path, monkeypatch):
    root = tmp_path / "n1-stackexchange-nonso-v1"
    _nonso_unit(root, "math.stackexchange.com-part20", ["q-1"])
    monkeypatch.setattr(GPC.AIC.PSP, "already_present",
                        lambda url, coll, ids, **kw: [])
    jobs = GPC.pending_jobs(cc_root=None, se_root=None,
                            nonso_root=str(root),
                            queue_dir=str(tmp_path / "queue"),
                            twin_url="http://x")
    assert [j.pack_id for j in jobs] == [
        "stackexchange-nonso-qa-math.stackexchange.com-part20-v1"]


# ------------------------------------------------------------------ filter

def test_already_enqueued_checks_manifest_presence(tmp_path):
    queue = tmp_path / "queue"
    (queue / "pack-a").mkdir(parents=True)
    (queue / "pack-a" / "manifest.json").write_text("{}")
    (queue / "pack-b").mkdir(parents=True)

    assert GPC.already_enqueued(str(queue), "pack-a") is True
    assert GPC.already_enqueued(str(queue), "pack-b") is False
    assert GPC.already_enqueued(str(queue), "pack-never-seen") is False


def test_pending_jobs_excludes_already_enqueued_packs(tmp_path):
    cc_root = tmp_path / "task634"
    _touch(cc_root / "tranches/t2_gpu/a/chunks/shard-0000.texts.jsonl.gz")
    _touch(cc_root / "tranches/t2_gpu/b/chunks/shard-0000.texts.jsonl.gz")
    queue = tmp_path / "queue"
    (queue / "cc-authority-frontier-w7-tranche2-a").mkdir(parents=True)
    (queue / "cc-authority-frontier-w7-tranche2-a" / "manifest.json").write_text("{}")

    jobs = GPC.pending_jobs(cc_root=str(cc_root), se_root=None,
                            queue_dir=str(queue))
    assert [j.pack_id for j in jobs] == ["cc-authority-frontier-w7-tranche2-b"]


def test_pending_jobs_with_no_roots_given_is_empty(tmp_path):
    assert GPC.pending_jobs(cc_root=None, se_root=None,
                            queue_dir=str(tmp_path / "queue")) == []


# ------------------------------------------------- task 664: stale refusals
# Real incident, 2026-09-26: CC tranche2/4/5 composed under a manifest
# missing `family_license`, REFUSED at `plan()` ("unresolved license"),
# then #381 fixed the composer -- but `already_enqueued()` alone never
# revisited them (they had a manifest.json), so a human had to delete the
# stale pack dir by hand to force a recompose. `_refusal_is_stale` +
# `pending_jobs`'s OR-clause close that gap.

def _write_refused_receipt(queue: Path, pack_id: str, *, stage: str,
                          checked_at: str, manifest: bytes = b"{}") -> None:
    pack_dir = queue / pack_id
    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "manifest.json").write_bytes(manifest)
    (pack_dir / "receipt.json").write_text(json.dumps({
        "status": "REFUSED", "stage": stage, "checked_at": checked_at,
        "reason": "unresolved license"}))


def test_refusal_is_stale_true_for_a_cacheable_refusal_before_the_code_change(
        tmp_path):
    queue = tmp_path / "queue"
    _write_refused_receipt(queue, "pack-a", stage="plan",
                          checked_at="2026-09-26T12:20:29+00:00")
    assert GPC._refusal_is_stale(
        str(queue), "pack-a",
        code_changed_at="2026-09-26T12:32:13+00:00") is True


def test_refusal_is_stale_false_when_the_refusal_is_already_newer(tmp_path):
    """The code hasn't changed since this refusal -- re-running plan()
    would reproduce the SAME judgment, exactly what promote_queue.py's own
    cache exists to avoid."""
    queue = tmp_path / "queue"
    _write_refused_receipt(queue, "pack-a", stage="plan",
                          checked_at="2026-09-26T13:00:00+00:00")
    assert GPC._refusal_is_stale(
        str(queue), "pack-a",
        code_changed_at="2026-09-26T12:32:13+00:00") is False


def test_refusal_is_stale_false_for_a_transient_stage(tmp_path):
    """A lock-contention/intake failure isn't a content judgment --
    promote_queue.py already retries those every drain on its own; forcing
    a recompose here would just waste GPU/compose work for no reason."""
    queue = tmp_path / "queue"
    _write_refused_receipt(queue, "pack-a", stage="intake",
                          checked_at="2026-09-26T12:20:29+00:00")
    assert GPC._refusal_is_stale(
        str(queue), "pack-a",
        code_changed_at="2026-09-26T12:32:13+00:00") is False


def test_refusal_is_stale_false_when_status_is_not_refused(tmp_path):
    queue = tmp_path / "queue"
    pack_dir = queue / "pack-a"
    pack_dir.mkdir(parents=True)
    (pack_dir / "manifest.json").write_bytes(b"{}")
    (pack_dir / "receipt.json").write_text(json.dumps(
        {"status": "LANDED", "checked_at": "2026-09-26T12:20:29+00:00"}))
    assert GPC._refusal_is_stale(
        str(queue), "pack-a",
        code_changed_at="2026-09-26T12:32:13+00:00") is False


def test_refusal_is_stale_false_with_no_receipt_at_all(tmp_path):
    queue = tmp_path / "queue"
    (queue / "pack-a").mkdir(parents=True)
    (queue / "pack-a" / "manifest.json").write_bytes(b"{}")
    assert GPC._refusal_is_stale(
        str(queue), "pack-a",
        code_changed_at="2026-09-26T12:32:13+00:00") is False


def test_refusal_is_stale_false_when_code_changed_at_is_unknown(tmp_path):
    """git unavailable / the call failed -- never force a recompose on a
    guess."""
    queue = tmp_path / "queue"
    _write_refused_receipt(queue, "pack-a", stage="plan",
                          checked_at="2026-09-26T12:20:29+00:00")
    assert GPC._refusal_is_stale(str(queue), "pack-a",
                                 code_changed_at=None) is False


def _inside_git_work_tree() -> bool:
    import subprocess
    try:
        return subprocess.run(
            ["git", "-C", str(GPC._REPO_ROOT), "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=10).stdout.strip() == "true"
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _inside_git_work_tree(),
                    reason="the exported public tree carries no git history")
def test_compose_code_last_changed_reads_the_real_repo_git_log():
    """Real repo, not a fixture: `cytir/index/compose.py` is a tracked file
    with real history, so this must return a real ISO timestamp, not
    None."""
    ts = GPC._compose_code_last_changed()
    assert ts is not None
    assert re_module.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", ts)


def test_compose_code_last_changed_returns_none_outside_a_git_repo(tmp_path):
    assert GPC._compose_code_last_changed(repo_root=tmp_path) is None


def test_pending_jobs_rediscovers_a_pack_with_a_stale_cacheable_refusal(
        tmp_path, monkeypatch):
    cc_root = tmp_path / "task634"
    _touch(cc_root / "tranches/t2_gpu/all/chunks/shard-0000.texts.jsonl.gz")
    queue = tmp_path / "queue"
    _write_refused_receipt(queue, "cc-authority-frontier-w7-tranche2",
                          stage="plan",
                          checked_at="2026-09-26T12:20:29+00:00")
    monkeypatch.setattr(GPC, "_compose_code_last_changed",
                        lambda *a, **k: "2026-09-26T12:32:13+00:00")

    jobs = GPC.pending_jobs(cc_root=str(cc_root), se_root=None,
                            queue_dir=str(queue))
    assert [j.pack_id for j in jobs] == ["cc-authority-frontier-w7-tranche2"]


def test_pending_jobs_still_skips_a_refusal_newer_than_the_code(
        tmp_path, monkeypatch):
    cc_root = tmp_path / "task634"
    _touch(cc_root / "tranches/t2_gpu/all/chunks/shard-0000.texts.jsonl.gz")
    queue = tmp_path / "queue"
    _write_refused_receipt(queue, "cc-authority-frontier-w7-tranche2",
                          stage="plan",
                          checked_at="2026-09-26T13:00:00+00:00")
    monkeypatch.setattr(GPC, "_compose_code_last_changed",
                        lambda *a, **k: "2026-09-26T12:32:13+00:00")

    jobs = GPC.pending_jobs(cc_root=str(cc_root), se_root=None,
                            queue_dir=str(queue))
    assert jobs == []


# --------------------------------------------------------------------- run

def test_run_calls_embed_compose_enqueue_per_job_and_survives_one_failure(
        tmp_path, monkeypatch):
    cc_root = tmp_path / "task634"
    _touch(cc_root / "tranches/t2_gpu/a/chunks/shard-0000.texts.jsonl.gz")
    _touch(cc_root / "tranches/t2_gpu/b/chunks/shard-0000.texts.jsonl.gz")
    queue = tmp_path / "queue"

    calls = []

    def fake_embed_compose_enqueue(*, pack_id, family, chunks_dir,
                                   vectors_dir, collection, mrl, qdrant_url,
                                   queue_dir, ceiling, log, license_from=None,
                                   family_license=None, family_payload=None,
                                   family_payload_basis=None,
                                   title_from=(), skip_embed=False,
                                   exclude_ids_file=None,
                                   exclude_ids_reason=None,
                                   served_targets=None):
        calls.append(pack_id)
        if pack_id.endswith("-a"):
            raise GPC.AIC.ComposeError("simulated identity mismatch")
        return {"ready_sha": "deadbeef"}

    monkeypatch.setattr(GPC.AIC, "embed_compose_enqueue",
                        fake_embed_compose_enqueue)

    results = GPC.run(cc_root=str(cc_root), se_root=None,
                      queue_dir=str(queue), log=lambda m: None)
    assert sorted(calls) == ["cc-authority-frontier-w7-tranche2-a",
                             "cc-authority-frontier-w7-tranche2-b"]
    by_id = {r["pack_id"]: r for r in results}
    assert by_id["cc-authority-frontier-w7-tranche2-a"]["status"] == "failed"
    assert by_id["cc-authority-frontier-w7-tranche2-b"]["status"] == "enqueued"
    assert by_id["cc-authority-frontier-w7-tranche2-b"]["ready_sha"] == "deadbeef"


def test_run_passes_the_family_ceiling_through(tmp_path, monkeypatch):
    se_root = tmp_path / "n1-stackexchange-v1"
    label = "stackoverflow-qa-score10-tranche-100000-109999-v1"
    _touch(se_root / "production/stages" / f"{label}-413a-portable-v3" /
          "STAGE-RECEIPT.json", b"{}")

    seen_ceiling = {}

    def fake_embed_compose_enqueue(*, ceiling, **kwargs):
        seen_ceiling["value"] = ceiling
        return {"ready_sha": "x"}

    monkeypatch.setattr(GPC.AIC, "embed_compose_enqueue",
                        fake_embed_compose_enqueue)
    GPC.run(cc_root=None, se_root=str(se_root),
           queue_dir=str(tmp_path / "queue"),
           ceilings={"stackoverflow_qa": 999}, log=lambda m: None)
    assert seen_ceiling["value"] == 999


def test_run_passes_license_from_through_to_embed_compose_enqueue(
        tmp_path, monkeypatch):
    """Task 664, 2026-09-26: the driver must forward what `discover_se_
    pending` resolved, all the way to the one door that writes it into the
    manifest -- a break anywhere in that chain reproduces the "unresolved
    license" refusal."""
    se_root = tmp_path / "n1-stackexchange-v1"
    label = "stackoverflow-qa-score10-tranche-100000-109999-v1"
    _touch(se_root / "production/stages" / f"{label}-413a-portable-v3" /
          "STAGE-RECEIPT.json", b"{}")

    seen = {}

    def fake_embed_compose_enqueue(*, license_from, **kwargs):
        seen["license_from"] = license_from
        return {"ready_sha": "x"}

    monkeypatch.setattr(GPC.AIC, "embed_compose_enqueue",
                        fake_embed_compose_enqueue)
    GPC.run(cc_root=None, se_root=str(se_root),
           queue_dir=str(tmp_path / "queue"), log=lambda m: None)
    assert seen["license_from"] == "source_metadata.accepted_answer.license"


def test_run_passes_cc_family_license_through_to_embed_compose_enqueue(
        tmp_path, monkeypatch):
    """Real incident, task 664, 2026-09-26: the drain after the anchor fix
    refused all 3 CC packs with `ProjectionError: unresolved license` --
    `discover_cc_pending` never forwarded a `family_license` at all. CC
    rows carry no top-level `license` and span many hosts, so the basis is
    family-level (ADR-111's opt-out+minimum-record reading) -- the SAME
    value the family's original, pre-unattended landing already declared."""
    cc_root = tmp_path / "task634"
    _touch(cc_root / "tranches/t2_gpu/all/chunks/shard-0000.texts.jsonl.gz")

    seen = {}

    def fake_embed_compose_enqueue(*, family_license, **kwargs):
        seen["family_license"] = family_license
        return {"ready_sha": "x"}

    monkeypatch.setattr(GPC.AIC, "embed_compose_enqueue",
                        fake_embed_compose_enqueue)
    GPC.run(cc_root=str(cc_root), se_root=None,
           queue_dir=str(tmp_path / "queue"), log=lambda m: None)
    assert seen["family_license"] == "ADR-111-opt-out-minimum-record"


def test_run_passes_family_payload_through_to_embed_compose_enqueue(
        tmp_path, monkeypatch):
    """Task 664, 2026-09-26 (SE tranche 8): stackoverflow_qa rows carry no
    top-level source_type/trust_tier/title, so every pack this driver
    composed without these refused with PayloadCarryError. The coordinator's
    ruling (derive from the family's own served precedent, cite it, never
    invent a number) must reach the manifest the same way license_from
    does."""
    se_root = tmp_path / "n1-stackexchange-v1"
    label = "stackoverflow-qa-score10-tranche-100000-109999-v1"
    _touch(se_root / "production/stages" / f"{label}-413a-portable-v3" /
          "STAGE-RECEIPT.json", b"{}")

    seen = {}

    def fake_embed_compose_enqueue(*, family_payload, family_payload_basis,
                                   title_from, **kwargs):
        seen["family_payload"] = family_payload
        seen["family_payload_basis"] = family_payload_basis
        seen["title_from"] = title_from
        return {"ready_sha": "x"}

    monkeypatch.setattr(GPC.AIC, "embed_compose_enqueue",
                        fake_embed_compose_enqueue)
    GPC.run(cc_root=None, se_root=str(se_root),
           queue_dir=str(tmp_path / "queue"), log=lambda m: None)
    assert seen["family_payload"] == {"source_type": "community", "trust_tier": 0.55}
    assert "family precedent" in seen["family_payload_basis"]
    assert seen["title_from"] == ["source_metadata.title"]


def test_run_passes_no_family_payload_for_a_family_with_no_mapping(
        tmp_path, monkeypatch):
    """No regression: a CC job (or any family not in _FAMILY_PAYLOAD) must
    forward None/empty, never a guess borrowed from a different family."""
    cc_root = tmp_path / "task634"
    _touch(cc_root / "tranches/t2_gpu/all/chunks/shard-0000.texts.jsonl.gz")

    seen = {}

    def fake_embed_compose_enqueue(*, family_payload, family_payload_basis,
                                   title_from, **kwargs):
        seen["family_payload"] = family_payload
        seen["family_payload_basis"] = family_payload_basis
        seen["title_from"] = title_from
        return {"ready_sha": "x"}

    monkeypatch.setattr(GPC.AIC, "embed_compose_enqueue",
                        fake_embed_compose_enqueue)
    GPC.run(cc_root=str(cc_root), se_root=None,
           queue_dir=str(tmp_path / "queue"), log=lambda m: None)
    assert seen["family_payload"] is None
    assert seen["family_payload_basis"] is None
    assert seen["title_from"] == ()


# ---------------------------------------------- declared sets + queue cap (680)

def _set_corpus(tmp_path: Path, shards: int = 4, rows: int = 3) -> Path:
    """A corpus root holding one source pack of `shards` paired shards and a
    sha-pinnable exclusion file."""
    root = tmp_path / "corpus"
    for i in range(shards):
        tag = f"shard-{i:04d}"
        _write_gz_chunk_shard(root / f"stage/{tag}.texts.jsonl.gz".replace(
            "stage/", "stage/"), [{"point_id": f"{tag}-{j}", "url": f"u{i}{j}"}
                                   for j in range(rows)])
        vec = root / "vec"
        _touch(vec / f"{tag}.vectors.f16", b"\0" * rows * 2048)
        _touch(vec / f"{tag}.vectors.manifest.json", json.dumps({"n": rows}).encode())
        _touch(vec / f"{tag}.row_ids.jsonl", b"")
    excl = root / "trim" / "trimmed.jsonl.gz"
    _write_gz_chunk_shard(excl, [{"point_id": "shard-0000-0"}])
    return root


def _sets_file(tmp_path: Path, root: Path, *, status: str) -> Path:
    from cytir.index import exclude_file as XF
    doc = {"schema": "aw.growth-pending-sets.v1", "sets": [{
        "set_id": "demo", "family": "demo_family", "status": status,
        "hold": {"reason": "because Tim said so", "unhold_when": "Tim merges X"},
        "pack_id_prefix": "demo-", "shards_per_pack": 2,
        "subpack_root": "subpacks", "title_from": "source_metadata.title",
        "family_license": "CC0-1.0", "family_payload": {"source_type": "reference"},
        "exclude_ids": {"path": "trim/trimmed.jsonl.gz",
                        "sha256": XF.sha256_file(root / "trim/trimmed.jsonl.gz"),
                        "reason": "below_floor"},
        "sources": [{"label": "a", "chunks": "stage", "vectors": "vec"}]}]}
    f = tmp_path / "sets.json"
    f.write_text(json.dumps(doc))
    return f


def test_a_held_set_is_logged_with_its_unhold_condition_and_never_composed(tmp_path):
    root = _set_corpus(tmp_path)
    sets = _sets_file(tmp_path, root, status="held")
    logged = []
    jobs = GPC.discover_sets_pending(str(sets), str(root), log=logged.append)
    assert jobs == []
    assert any("HELD" in m and "because Tim said so" in m and "Tim merges X" in m
               for m in logged)
    assert not (root / "subpacks").exists()         # nothing materialized either


def test_an_active_set_yields_subpack_jobs_with_the_exclusion_pinned(
        tmp_path, monkeypatch):
    root = _set_corpus(tmp_path, shards=4)
    sets = _sets_file(tmp_path, root, status="active")
    monkeypatch.setattr(GPC.AIC.PSP, "already_present",
                        lambda url, coll, ids, **kw: [])
    jobs = GPC.discover_sets_pending(str(sets), str(root), twin_url="http://x")
    assert [j.pack_id for j in jobs] == ["demo-a-s0000", "demo-a-s0001"]
    j = jobs[0]
    assert j.family == "demo_family" and j.family_license == "CC0-1.0"
    assert j.family_payload == {"source_type": "reference"}
    assert j.exclude_ids_file == str(root / "trim/trimmed.jsonl.gz")
    assert j.exclude_ids_reason == "below_floor"
    assert j.chunks_dir == str(root / "subpacks/demo-a-s0000/chunks")
    assert sorted(p.name for p in Path(j.chunks_dir).iterdir()) == [
        "shard-0000.texts.jsonl.gz", "shard-0001.texts.jsonl.gz"]


def test_an_enqueued_subpack_is_settled_before_anything_is_materialized(
        tmp_path, monkeypatch):
    root = _set_corpus(tmp_path, shards=2)
    sets = _sets_file(tmp_path, root, status="active")
    queue = tmp_path / "queue"
    _touch(queue / "demo-a-s0000" / "manifest.json", b"{}")
    monkeypatch.setattr(GPC.AIC.PSP, "already_present",
                        lambda url, coll, ids, **kw: [])
    jobs = GPC.pending_jobs(cc_root=None, se_root=None, sets_file=str(sets),
                            corpus_root=str(root), queue_dir=str(queue),
                            twin_url="http://x")
    assert jobs == []
    assert not (root / "subpacks/demo-a-s0000").exists()


def test_queue_depth_rows_counts_only_ready_and_not_terminal_packs(tmp_path):
    q = tmp_path / "queue"
    for name, rows, markers in (("a", 100, ["READY"]),
                                ("b", 200, ["READY", "DONE"]),
                                ("c", 400, ["READY", "SUPERSEDED"]),
                                ("d", 800, []),                 # not READY yet
                                ("e", 1600, ["READY"])):
        _touch(q / name / "manifest.json", json.dumps({"rows": rows}).encode())
        for m in markers:
            _touch(q / name / m, b"x")
    assert GPC.queue_depth_rows(str(q)) == 1700
    assert GPC.queue_depth_rows(str(tmp_path / "nope")) == 0


def test_run_defers_the_rest_once_the_queue_holds_the_cap(tmp_path, monkeypatch):
    root = tmp_path / "n1-stackexchange-nonso-v1"
    for u in ("math.stackexchange.com-part1", "math.stackexchange.com-part2"):
        _nonso_unit(root, u, [f"{u}-1"])
    queue = tmp_path / "queue"
    _touch(queue / "other" / "manifest.json", json.dumps({"rows": 500}).encode())
    _touch(queue / "other" / "READY", b"x")
    monkeypatch.setattr(GPC.AIC.PSP, "already_present",
                        lambda url, coll, ids, **kw: [])
    calls = []
    monkeypatch.setattr(GPC.AIC, "embed_compose_enqueue",
                        lambda **kw: calls.append(kw["pack_id"]) or {"ready_sha": "x"})
    logged = []
    results = GPC.run(cc_root=None, se_root=None, nonso_root=str(root),
                      queue_dir=str(queue), qdrant_url="http://x",
                      max_queue_rows=500, log=logged.append)
    assert calls == []
    assert [r["status"] for r in results] == ["deferred", "deferred"]
    assert any("cap 500" in m for m in logged)


def test_run_passes_the_exclusion_file_through_to_the_compose_door(
        tmp_path, monkeypatch):
    root = _set_corpus(tmp_path, shards=2)
    sets = _sets_file(tmp_path, root, status="active")
    monkeypatch.setattr(GPC.AIC.PSP, "already_present",
                        lambda url, coll, ids, **kw: [])
    seen = {}

    def fake(*, exclude_ids_file, exclude_ids_reason, skip_embed, **kwargs):
        seen.update(file=exclude_ids_file, reason=exclude_ids_reason,
                    skip_embed=skip_embed)
        return {"ready_sha": "x"}
    monkeypatch.setattr(GPC.AIC, "embed_compose_enqueue", fake)
    GPC.run(cc_root=None, se_root=None, sets_file=str(sets),
            corpus_root=str(root), queue_dir=str(tmp_path / "queue"),
            qdrant_url="http://x", log=lambda m: None)
    assert seen == {"file": str(root / "trim/trimmed.jsonl.gz"),
                    "reason": "below_floor", "skip_embed": True}


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data"
         / "growth-pending-sets.v1.json").exists(),
    reason="growth-pending-sets.v1.json is operator data, not in the public cut")
def test_the_declared_openalex_set_is_well_formed_and_active():
    """The repo's own sets file: schema, the ten packs the ADR-102 trim receipt
    accounts for (6,404,873 sealed points) and the declared licence + payload.
    Activated by the POLICY PR Tim merged (2026-09-30); the lifted hold stays
    on record."""
    f = (Path(__file__).resolve().parents[1] / "data"
         / "growth-pending-sets.v1.json")
    sets = GPC._load_sets(str(f))
    oa = next(s for s in sets if s["set_id"] == "openalex_academic_dump")
    assert oa["status"] == "active"
    assert oa["lifted_hold"]["reason"] and oa["lifted_hold"]["unhold_when"]
    assert oa["family_license"] == "CC0-1.0"
    assert oa["family_payload"] == {"source_type": "reference",
                                    "trust_tier": 0.70}
    assert oa["family_payload_basis"]
    assert len(oa["sources"]) == 10
    assert sum(s["sealed_points"] for s in oa["sources"]) == 6_404_873
    assert len(oa["exclude_ids"]["sha256"]) == 64


def test_run_reports_nothing_pending_cleanly(tmp_path):
    results = GPC.run(cc_root=None, se_root=None,
                      queue_dir=str(tmp_path / "queue"), log=lambda m: None)
    assert results == []


# ----------------------------------------------------------------------- CLI

def test_cli_ceiling_flag_parses_family_equals_number(monkeypatch):
    parsed = GPC.build_parser().parse_args(
        ["--ceiling", "eric_official_pdf=4500000",
         "--ceiling", "stackoverflow_qa=999"])
    assert parsed.ceiling == ["eric_official_pdf=4500000",
                             "stackoverflow_qa=999"]


def test_queue_depth_rows_counts_rows_to_write_and_reads_a_legacy_pack_by_rows(tmp_path):
    q = tmp_path / "q"
    for name, man in (("legacy", {"rows": 100}),
                      ("new", {"rows": 900, "rows_staged": 900, "rows_to_write_max": 40})):
        _touch(q / name / "manifest.json", json.dumps(man).encode())
        _touch(q / name / "READY", b"x")
    assert GPC.queue_depth_rows(str(q)) == 100 + 40


def test_queue_depth_rows_ignores_a_cached_refusal_but_counts_a_resealed_one(tmp_path):
    import hashlib
    q = tmp_path / "queue"
    for name, rows in (("live", 100), ("refused", 5000)):
        _touch(q / name / "manifest.json", json.dumps({"rows": rows}).encode())
        _touch(q / name / "READY", b"x")
    sha = hashlib.sha256((q / "refused" / "manifest.json").read_bytes()).hexdigest()
    receipt = {"status": "REFUSED", "stage": "plan", "manifest_sha": sha}
    _touch(q / "refused" / "receipt.json", json.dumps(receipt).encode())
    assert GPC.queue_depth_rows(str(q)) == 100
    # a transient-stage refusal retries, so it is still inventory
    receipt["stage"] = "intake"
    _touch(q / "refused" / "receipt.json", json.dumps(receipt).encode())
    assert GPC.queue_depth_rows(str(q)) == 5100
    # a changed manifest sha is owed again
    receipt.update(stage="plan", manifest_sha="0" * 64)
    _touch(q / "refused" / "receipt.json", json.dumps(receipt).encode())
    assert GPC.queue_depth_rows(str(q)) == 5100


def test_reseal_retires_a_refused_receipt_so_compose_does_not_recompose_it(tmp_path):
    import hashlib
    from cytir.index import reseal as RS
    q = tmp_path / "queue"
    pack = q / "p1"
    manifest = json.dumps({"rows": 5}).encode()
    _touch(pack / "manifest.json", manifest)
    _touch(pack / "READY", hashlib.sha256(manifest).hexdigest().encode())
    _touch(pack / "receipt.json", json.dumps(
        {"status": "REFUSED", "stage": "plan",
         "checked_at": "2026-09-30T15:42:53Z"}).encode())
    RS.reseal_pack(str(q), "p1", reason="L1 filed", l1_releases=["r"])
    assert not (pack / "receipt.json").exists()
    assert (pack / "receipt.refused-2026-09-30T154253Z.json").is_file()
    # settled for compose: a manifest exists and there is no stale refusal
    assert GPC._refusal_is_stale(str(q), "p1",
                                 code_changed_at="2099-01-01T00:00:00+00:00") is False
    assert GPC.already_enqueued(str(q), "p1")


# ------------------------------------------------ OpenLegalData (task 680)

def _old_shard(root: Path, n: int, *, vectors: bool, l1: bool, packed: bool = True):
    stage = root / "stages/de-openlegaldata-v1" / f"chunks-shard{n}"
    _write_gz_chunk_shard(stage / "chunks/shard-0000.texts.jsonl.gz",
                          [{"point_id": f"p-{n}-1", "text": "t"}])
    if packed:
        _touch(stage / "PACK.json", b"{}")
    if vectors:
        v = (root / "vectors/release=aw-task680-openlegaldata-v1"
             / f"source=de-openlegaldata-shard{n}")
        _touch(v / "shard-0000.vectors.manifest.json", b"{}")
    if l1:
        _touch(root / "l1" / f"release=openlegaldata-v1-shard{n}"
               / "_register_manifest.json", b"{}")
    return stage


def test_discover_openlegaldata_pending_yields_a_vectored_filed_shard_with_the_shard0_fields(
        tmp_path):
    root = tmp_path / "corpus"
    stage = _old_shard(root, 7, vectors=True, l1=True)
    jobs = GPC.discover_openlegaldata_pending(str(root))
    assert len(jobs) == 1
    job = jobs[0]
    assert job.pack_id == "de-openlegaldata-shard7"
    assert job.family == "de_openlegaldata_v1"
    assert job.chunks_dir == str(stage / "chunks")
    assert job.vectors_dir.endswith(
        "vectors/release=aw-task680-openlegaldata-v1/source=de-openlegaldata-shard7")
    # copied from the landed de-openlegaldata-shard0-part1 manifest
    assert job.family_payload == {"source_type": "official", "trust_tier": 0.9}
    assert job.title_from == "source_metadata.title"


def test_discover_openlegaldata_pending_never_embeds_an_unembedded_stage(tmp_path):
    root = tmp_path / "corpus"
    _old_shard(root, 7, vectors=False, l1=True)
    msgs: list[str] = []
    assert GPC.discover_openlegaldata_pending(str(root), log=msgs.append) == []
    assert any("AWAITING VECTORS" in m for m in msgs)


def test_discover_openlegaldata_pending_holds_a_shard_with_no_l1(tmp_path):
    """The OpenAlex gap: ADR-105 require=l1 refuses an unfiled pack at the drain."""
    root = tmp_path / "corpus"
    _old_shard(root, 7, vectors=True, l1=False)
    msgs: list[str] = []
    assert GPC.discover_openlegaldata_pending(str(root), log=msgs.append) == []
    assert any("AWAITING L1" in m and "openlegaldata-v1-shard7" in m for m in msgs)


def test_discover_openlegaldata_pending_skips_an_unsealed_stage_and_orders_numerically(
        tmp_path):
    root = tmp_path / "corpus"
    _old_shard(root, 10, vectors=True, l1=True)
    _old_shard(root, 2, vectors=True, l1=True)
    _old_shard(root, 3, vectors=True, l1=True, packed=False)   # chunker not finished
    assert [j.pack_id for j in GPC.discover_openlegaldata_pending(str(root))] == [
        "de-openlegaldata-shard2", "de-openlegaldata-shard10"]


def test_discover_openlegaldata_pending_missing_tree_is_empty(tmp_path):
    assert GPC.discover_openlegaldata_pending(str(tmp_path / "nope")) == []


def test_discover_openlegaldata_pending_skips_a_shard_the_twin_already_serves(
        tmp_path, monkeypatch):
    root = tmp_path / "corpus"
    _old_shard(root, 7, vectors=True, l1=True)
    monkeypatch.setattr(GPC.AIC.PSP, "already_present",
                        lambda url, coll, ids, **kw: list(ids))
    assert GPC.discover_openlegaldata_pending(str(root), twin_url="http://x") == []


def test_pending_jobs_includes_openlegaldata_only_when_asked_and_skips_enqueued(tmp_path):
    root = tmp_path / "corpus"
    _old_shard(root, 7, vectors=True, l1=True)
    _old_shard(root, 8, vectors=True, l1=True)
    queue = tmp_path / "queue"
    _touch(queue / "de-openlegaldata-shard8" / "manifest.json", b"{}")
    kw = dict(cc_root=None, se_root=None, queue_dir=str(queue),
              corpus_root=str(root))
    assert GPC.pending_jobs(**kw) == []
    assert [j.pack_id for j in GPC.pending_jobs(openlegaldata=True, **kw)] == [
        "de-openlegaldata-shard7"]


def test_run_passes_the_served_targets_and_reports_all_served(tmp_path, monkeypatch):
    """Ruling 23.5: `--prod-url` turns the compose-time check on for every job; a pack that
    composes to nothing is `all-served`, not `failed`, and is not enqueued."""
    cc_root = tmp_path / "cc"
    _touch(cc_root / "tranches/t2_gpu/a/chunks/shard-0000.texts.jsonl.gz")
    seen = {}

    def fake(*, pack_id, served_targets=None, **kw):
        seen[pack_id] = served_targets
        return {"all_served": True, "ready_sha": None}

    monkeypatch.setattr(GPC.AIC, "embed_compose_enqueue", fake)
    res = GPC.run(cc_root=str(cc_root), se_root=None, queue_dir=str(tmp_path / "q"),
                  qdrant_url="http://twin", prod_url="http://prod", log=lambda m: None)
    assert [r["status"] for r in res] == ["all-served"]
    (targets,) = seen.values()
    assert targets["prod"] == ("http://prod", "aw_prod") and targets["twin"][0] == "http://twin"

"""ADR-103 (task 505 §2.3 / task 511 §0): the carried-declined-row stamp.

Census v34 measured, per point, which carried rows sit on a declined or
unreadable origin -- but recorded the fact rather than reaching the payload
with it (task 505 §7's named gap). This is the binder's other half: it reads
the SAME artifacts the census already pinned (by path and sha256, both
recorded in `consent_corrections.carried_declined_rows_adr103`), refuses on
any drift or unmapped state, and returns the point-level stamp
`_stage_partition` writes onto the payload.

No hand list anywhere in this file: every fixture below is shaped exactly
like the real census/flags/bucket-assignment artifacts (task 511 verified
their real sha256/row counts against the GPU box before writing this test), and
every assertion is either a refusal direction or the derived stamp -- never a
count copied from the production numbers.

Import pattern matches `test_assembly_binder_pack_route.py`: `assembly_binder`
touches numpy/qdrant_client at module scope; the functions under test touch
neither, so those two are stubbed here if the runtime lacks them.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

for _name in ("numpy", "qdrant_client"):
    if _name not in sys.modules:
        try:
            __import__(_name)
        except ImportError:
            sys.modules[_name] = types.ModuleType(_name)
            sys.modules[f"{_name}.models"] = types.ModuleType(f"{_name}.models")
            setattr(sys.modules[_name], "models", sys.modules[f"{_name}.models"])
            # Marks this as OUR stand-in, not the real package -- so a test
            # that genuinely needs qdrant_client (not just an importable
            # module symbol) can tell the difference and skip rather than
            # fail against a stub too thin for its own module-level code
            # (serving_config.py reads qm.PayloadSchemaType at import time).
            sys.modules[_name]._AW_TEST_STUB = True

from cytir.index import assemble_inputs as AI  # noqa: E402
from cytir.index import assemble_plan as AP  # noqa: E402
from cytir.index import assemble_stage as AS  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl_gz(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _card_flag_row(pid, url, lane, cls, state):
    return {"point_id": pid, "url": url, "lane": lane, "class": cls,
            "consent_state": state, "non_refreshable": True}


def _bucket_row(pid, member, covered_by, url="https://example.test/x"):
    return {"point_id": pid, "census_bucket": member.split()[0],
            "census_member": member, "covered": True, "covered_by": covered_by,
            "domain": "example.test", "frontier_lane": "lookup",
            "frontier_url": url, "nonmine_owner": None,
            "provenance": "sitemap-ingest", "route_hint": "x",
            "source_type": "reference"}


def _build_fixture(tmp_path, *, card_rows, extra_bucket_rows=(),
                   card_set="task467_entity_card_1024_reembed",
                   news_set="task500_news_1024_in_place",
                   card_set_declared=None, news_set_declared=10,
                   bad_sha=False):
    """Lay out a census + flags-file + bucket-assignment tree shaped exactly
    like the real one, at absolute tmp paths (matching production, where the
    census's `uri` fields are absolute the GPU box paths, not repo-relative)."""
    flags_path = tmp_path / "flags" / "task505-carried-consent-flags.v1.jsonl.gz"
    _write_jsonl_gz(flags_path, card_rows)
    bucket_path = tmp_path / "census" / "remainder-by-census-bucket.v1.jsonl.gz"
    _write_jsonl_gz(bucket_path, list(extra_bucket_rows))

    census = {
        "basis": {
            "bucket_assignment": {
                "uri": str(bucket_path),
                "sha256": ("BAD" if bad_sha else _sha256(bucket_path)),
            },
        },
        "consent_corrections": {
            "carried_declined_rows_adr103": {
                "per_row_flags": {
                    "uri": str(flags_path),
                    "sha256": _sha256(flags_path),
                    "rows": len(card_rows),
                },
                "sets": {
                    card_set: (card_set_declared if card_set_declared is not None
                              else len(card_rows)),
                    news_set: news_set_declared,
                },
            },
        },
    }
    census_path = tmp_path / "census.json"
    census_path.write_text(json.dumps(census))
    return census_path


def _happy_fixture(tmp_path):
    card_rows = [
        _card_flag_row("11111111-0000-0000-0000-000000000001",
                       "http://a.test", "direct",
                       "origin: robots.txt 401/403 (fail-closed)",
                       "unreadable"),
        _card_flag_row("11111111-0000-0000-0000-000000000002",
                       "http://b.test", "direct",
                       "origin: User-agent:* Disallow: / (genuine)",
                       "declined"),
        _card_flag_row("11111111-0000-0000-0000-000000000003",
                       "http://c.test", "cc",
                       "cc: CCBot disallowed (binding for CC-derived copy)",
                       "not_applicable"),
        _card_flag_row("11111111-0000-0000-0000-000000000004",
                       "http://d.test", "cc",
                       "cc: no consent result found", "unresolved"),
    ]
    news_rows = [
        _bucket_row("22222222-0000-0000-0000-00000000000%d" % i,
                    "E cc-consent-declined", ["task500_news_1024_in_place"])
        for i in range(1, 3)
    ] + [
        # a decoy: same set, different member -- must NOT be picked up as a
        # "declined" row (the real data has E direct-oversized / D
        # frontier-non-mine siblings on this exact set).
        _bucket_row("22222222-0000-0000-0000-000000000099",
                    "E direct-oversized", ["task500_news_1024_in_place"]),
        # a decoy: right member, WRONG set -- must not leak into either.
        _bucket_row("22222222-0000-0000-0000-000000000098",
                    "E direct-consent-declined", ["some_other_set"]),
    ]
    census_path = _build_fixture(tmp_path, card_rows=card_rows,
                                 extra_bucket_rows=news_rows,
                                 news_set_declared=2)
    return census_path, card_rows, news_rows


def test_happy_path_derives_both_sources_and_folds_unresolved(tmp_path):
    census_path, card_rows, news_rows = _happy_fixture(tmp_path)
    flags, stats = AI.load_adr103_carried_flags(census_path)

    assert stats["total"] == 6  # 4 card rows + 2 news rows
    assert stats["by_set"] == {"task467_entity_card_1024_reembed": 4,
                               "task500_news_1024_in_place": 2}
    assert stats["unresolved_folded_into_unreadable"] == 1

    # the three ruled states pass through unchanged
    assert flags["11111111-0000-0000-0000-000000000001"] == {
        "consent_state": "unreadable", "non_refreshable": True}
    assert flags["11111111-0000-0000-0000-000000000002"] == {
        "consent_state": "declined", "non_refreshable": True}
    assert flags["11111111-0000-0000-0000-000000000003"] == {
        "consent_state": "not_applicable", "non_refreshable": True}
    # `unresolved` folds to `unreadable`, not a fifth invented value
    assert flags["11111111-0000-0000-0000-000000000004"] == {
        "consent_state": "unreadable", "non_refreshable": True}

    # the two page-derived news rows: declined, per ADR-103's own text
    assert flags["22222222-0000-0000-0000-000000000001"] == {
        "consent_state": "declined", "non_refreshable": True}
    assert flags["22222222-0000-0000-0000-000000000002"] == {
        "consent_state": "declined", "non_refreshable": True}

    # the two decoys must not appear
    assert "22222222-0000-0000-0000-000000000099" not in flags
    assert "22222222-0000-0000-0000-000000000098" not in flags


def test_refuses_on_flags_file_sha_drift(tmp_path):
    census_path, _, _ = _happy_fixture(tmp_path)
    census = json.loads(census_path.read_text())
    census["consent_corrections"]["carried_declined_rows_adr103"][
        "per_row_flags"]["sha256"] = "0" * 64
    census_path.write_text(json.dumps(census))
    with pytest.raises(SystemExit, match="sha256"):
        AI.load_adr103_carried_flags(census_path)


def test_refuses_on_bucket_assignment_sha_drift(tmp_path):
    _, card_rows, news_rows = _happy_fixture(tmp_path)
    # rebuild the census with a deliberately wrong bucket-assignment hash,
    # over the same underlying files
    census_path = _build_fixture(
        tmp_path, card_rows=card_rows, extra_bucket_rows=news_rows,
        news_set_declared=2, bad_sha=True)
    with pytest.raises(SystemExit, match="sha256"):
        AI.load_adr103_carried_flags(census_path)


def test_refuses_on_unmapped_consent_state(tmp_path):
    bad_row = _card_flag_row("33333333-0000-0000-0000-000000000001",
                             "http://e.test", "direct", "mystery class",
                             "something_new")
    census_path = _build_fixture(tmp_path, card_rows=[bad_row],
                                 news_set_declared=0)
    with pytest.raises(SystemExit, match="unmapped ADR-103 consent_state"):
        AI.load_adr103_carried_flags(census_path)


def test_refuses_when_derived_set_total_disagrees_with_census(tmp_path):
    # the census says the news set has 10 rows; the bucket-assignment fixture
    # here only supplies 1 -- must refuse, not silently short-load.
    card_rows = [_card_flag_row("44444444-0000-0000-0000-000000000001",
                                "http://f.test", "direct",
                                "origin: robots.txt 401/403 (fail-closed)",
                                "unreadable")]
    one_news_row = [_bucket_row("55555555-0000-0000-0000-000000000001",
                                "E cc-consent-declined",
                                ["task500_news_1024_in_place"])]
    census_path = _build_fixture(tmp_path, card_rows=card_rows,
                                 extra_bucket_rows=one_news_row,
                                 news_set_declared=10)
    with pytest.raises(SystemExit, match="census declares"):
        AI.load_adr103_carried_flags(census_path)


def test_refuses_when_no_adr103_block_in_census(tmp_path):
    census_path = tmp_path / "census.json"
    census_path.write_text(json.dumps({"schema": "x"}))
    with pytest.raises(SystemExit, match="carried_declined_rows_adr103"):
        AI.load_adr103_carried_flags(census_path)


def test_adr103_ruled_states_matches_card_recipe_closed_set():
    """One truth: ADR-103's 3 states are `card_recipe_adr060.CONSENT_STATES`
    minus "ok" -- not a re-typed list that can silently drift from it."""
    from cytir.corpus.card_recipe import CONSENT_STATES
    assert AI._adr103_ruled_states() == tuple(
        s for s in CONSENT_STATES if s != "ok")


def test_stage_partition_stamps_only_flagged_rows_others_byte_identical(
        tmp_path, monkeypatch):
    """The load-bearing property of task 511 §0: staging with ADR-103 flags
    changes ONLY the payload of rows named in the flags dict. Everything
    else -- including a row's key order and every other field -- must be
    byte-identical to a run with no flags at all. Runs the real
    `_stage_partition` against a two-row synthetic shard (no the GPU box, no
    qdrant, no real vectors) rather than asserting this from the diff of the
    stamping block alone.
    """
    np = pytest.importorskip("numpy")
    if getattr(sys.modules.get("qdrant_client"), "_AW_TEST_STUB", False):
        pytest.skip("real qdrant_client not installed -- `_stage_partition` "
                    "dead-imports serving_config, which reads qm.* at "
                    "module-load time; runs for real on the GPU box (task 511)")

    # -- a minimal fake `project_record` + gate, standing in for build_aw_b1 --
    class FakeGate:
        def __init__(self, *a, **k):
            pass

    # the stage looks `ConsentGate` up in its own namespace (tasks 2077, 2105)
    monkeypatch.setattr(AS, "ConsentGate", FakeGate)

    class FakeProj:

        @staticmethod
        def project_record(src, origin, corpus_row, published_map, conf_map,
                           gate, stats, metadata_registry=None):
            # task 515: the binder now passes an optional serving-metadata
            # registry. `None` here is the no-registry path, which must stay
            # byte-identical -- that is what this test asserts.
            assert metadata_registry is None
            payload = dict(src.payload)
            payload.setdefault("url", payload.get("url"))
            return {"id": src.id, "payload": payload,
                    "vector": {"dense": src.vector["dense"]}}, {}

    # the stage looks `project_record` up in its own namespace (tasks 2103, 2105)
    monkeypatch.setattr(AS, "project_record", FakeProj.project_record)

    pid_flagged = "66666666-0000-0000-0000-000000000001"
    pid_plain = "66666666-0000-0000-0000-000000000002"
    records_path = tmp_path / "shard.records.jsonl.gz"
    _write_jsonl_gz(records_path, [
        {"url": "http://flagged.test", "text": "hello world",
         "title": "Flagged", "trust_tier": 1},
        {"url": "http://plain.test", "text": "goodbye world",
         "title": "Plain", "trust_tier": 1},
    ])
    row_ids_path = tmp_path / "shard.row_ids.jsonl"
    row_ids_path.write_text(
        json.dumps({"point_id": pid_flagged}) + "\n" +
        json.dumps({"point_id": pid_plain}) + "\n")
    vecs = np.zeros((2, AP.FULL_DIM), dtype="<f2")
    vectors_path = tmp_path / "shard.vectors.f16"
    vecs.tofile(vectors_path)

    partition = {"set_id": "test_set", "shard": "shard-0000",
                "payload_route": "stage", "vectors": str(vectors_path),
                "row_ids": str(row_ids_path), "records": str(records_path),
                "rows": 2}

    class Args:
        consent_mode = "trust"
        grace_days = 30
        consent_cache = str(tmp_path / "cache.json")

    flags = {pid_flagged: {"consent_state": "declined",
                           "non_refreshable": True}}

    out_flagged = tmp_path / "out_flagged"
    out_flagged.mkdir()
    AS._stage_partition(partition, out_flagged, set(), Args(),
                        adr103_flags=flags)

    out_plain = tmp_path / "out_plain"
    out_plain.mkdir()
    AS._stage_partition(partition, out_plain, set(), Args(),
                        adr103_flags={})

    def _load_payloads(outdir):
        by_id = {}
        with gzip.open(outdir / "test_set.shard-0000.payload.jsonl.gz",
                       "rt") as fh:
            for line in fh:
                row = json.loads(line)
                by_id[row["id"]] = row
        return by_id

    flagged_rows = _load_payloads(out_flagged)
    plain_rows = _load_payloads(out_plain)

    # the flagged row gained exactly the two ADR-103 keys
    fp = flagged_rows[pid_flagged]["payload"]
    pp = plain_rows[pid_flagged]["payload"]
    assert fp["consent_state"] == "declined"
    assert fp["non_refreshable"] is True
    assert "consent_state" not in pp
    assert "non_refreshable" not in pp
    for k in pp:
        assert fp[k] == pp[k]
    assert set(fp) - set(pp) == {"consent_state", "non_refreshable"}

    # the unflagged row's payload is byte-identical between the two runs
    assert flagged_rows[pid_plain] == plain_rows[pid_plain]

    receipt = json.loads(
        (out_flagged / "test_set.shard-0000.receipt.json").read_text())
    assert receipt["adr103_stamped"] == 1
    assert receipt["adr103_conflicts"] == 0
    assert receipt["adr103_states"] == {"declined": 1}

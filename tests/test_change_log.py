"""The drain's change-log writer (task 2314): every op, a refused pack, the off switch, the verifier."""
from __future__ import annotations

import json

import pytest

from tests.helpers.repo_files import repo_file_or_skip

from cytir.index import change_log as CL

RECEIPT = {"pack_id": "p1", "status": "LANDED", "pack_sha": "ab" * 32,
           "prod": {"written": 3, "tombstones_requested": 1}}
PLAN = {"upserts": ["i1", ("i2", "replace"), "i3"], "op": "add",
        "tombstones": [{"id": "t1", "reason": "superseded"}]}


def test_every_op_is_written_in_landing_order(tmp_path):
    path = CL.record_landing(RECEIPT, PLAN, tmp_path, landed_at="2026-11-05T02:14:09Z")
    assert path.name == "20261105T021409Z-p1.jsonl"
    header, body = CL.read_segment(path)
    assert header == {"schema": CL.SCHEMA, "pack_id": "p1", "pack_sha": "ab" * 32,
                      "landed_at": "2026-11-05T02:14:09Z", "n": 4, "seq": 1}
    assert body == [{"id": "i1", "op": "add"}, {"id": "i2", "op": "replace"}, {"id": "i3", "op": "add"},
                    {"id": "t1", "op": "tombstone", "reason": "superseded", "queue_reason": "superseded"}]
    assert CL.verify(path, RECEIPT) == {"add": 2, "replace": 1, "upsert": 0, "tombstone": 1, "rebase": 0}


def test_rebase_line_is_last(tmp_path):
    path = CL.record_landing({"pack_id": "flip1", "status": "LANDED"}, {"rebase": "flip"}, tmp_path)
    _, body = CL.read_segment(path)
    assert body == [{"op": "rebase", "why": "flip"}]
    assert CL.verify(path, {"pack_id": "flip1"})["rebase"] == 1


@pytest.mark.parametrize("status", ["REFUSED", "ABORTED", "PROVED-DRY-RUN", None])
def test_a_pack_that_did_not_land_writes_nothing(tmp_path, status):
    assert CL.record_landing(dict(RECEIPT, status=status), PLAN, tmp_path) is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("off", [None, ""])
def test_unset_dir_writes_no_file(tmp_path, off, monkeypatch):
    assert CL.record_landing(RECEIPT, PLAN, off) is None
    monkeypatch.delenv("AW_CHANGE_LOG_DIR", raising=False)
    from cytir.core import paths
    paths.reset_settings_cache()
    assert CL.configured_dir() is None


def test_an_empty_plan_writes_nothing(tmp_path):
    assert CL.record_landing(RECEIPT, {"upserts": []}, tmp_path) is None


def test_a_second_call_replaces_the_same_file(tmp_path):
    first = CL.record_landing(RECEIPT, PLAN, tmp_path, landed_at="2026-11-05T02:14:09Z")
    again = CL.record_landing(RECEIPT, PLAN, tmp_path, landed_at="2026-11-06T09:00:00Z")
    assert again == first and len(list(tmp_path.glob("*.jsonl"))) == 1
    assert CL.read_segment(first)[0]["landed_at"] == "2026-11-05T02:14:09Z"


def test_a_tombstone_without_a_reason_is_refused_loudly(tmp_path):
    with pytest.raises(CL.ChangeLogError):
        CL.record_landing(RECEIPT, {"tombstones": [{"id": "t1", "reason": None}]}, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_verify_refuses_a_count_that_disagrees_and_the_cli_exits_1(tmp_path, capsys):
    path = CL.record_landing(RECEIPT, PLAN, tmp_path)
    bad = dict(RECEIPT, prod={"written": 2, "tombstones_requested": 1})
    with pytest.raises(CL.ChangeLogError, match="upsert lines 3 != receipt written 2"):
        CL.verify(path, bad)
    good_file, bad_file = tmp_path / "good.json", tmp_path / "bad.json"
    good_file.write_text(json.dumps(RECEIPT))
    bad_file.write_text(json.dumps(bad))
    assert CL.main(["verify", str(path), str(good_file)]) == 0
    assert CL.main(["verify", str(path), str(bad_file)]) == 1
    assert "MISMATCH" in capsys.readouterr().err


def test_a_truncated_segment_is_refused(tmp_path):
    path = CL.record_landing(RECEIPT, PLAN, tmp_path)
    path.write_text("".join(path.read_text().splitlines(True)[:-1]))
    with pytest.raises(CL.ChangeLogError, match="header n=4"):
        CL.read_segment(path)


def test_a_pack_id_that_is_a_suffix_of_another_keeps_its_own_segment(tmp_path):
    long = CL.record_landing(dict(RECEIPT, pack_id="l1-backfill-batch-00001"), PLAN, tmp_path,
                             landed_at="2026-11-05T02:14:09Z")
    short = CL.record_landing(dict(RECEIPT, pack_id="batch-00001"), PLAN, tmp_path,
                              landed_at="2026-11-05T03:00:00Z")
    assert long != short and long.exists() and short.exists()
    assert CL.read_segment(long)[0]["pack_id"] == "l1-backfill-batch-00001"
    assert CL.read_segment(short)[0]["pack_id"] == "batch-00001"


def test_superseded_rows_are_tombstones_and_verify_counts_them(tmp_path):
    receipt = dict(RECEIPT, prod={"written": 2, "superseded": 2})
    plan = {"upserts": ["a", "b"], "op": "add",
            "tombstones": [{"id": "o1", "reason": "superseded"}, {"id": "o2", "reason": "superseded"}]}
    path = CL.record_landing(receipt, plan, tmp_path)
    assert CL.verify(path, receipt)["tombstone"] == 2
    with pytest.raises(CL.ChangeLogError, match="superseded 3"):
        CL.verify(path, dict(receipt, prod={"written": 2, "superseded": 3}))


def test_the_segment_write_refuses_in_a_dry_run(tmp_path):
    from cytir.index import dry_run as DRY
    with pytest.raises(DRY.DryRunWriteRefused):      # the guard re-raises on exit when a write was refused
        with DRY.guard(True):
            CL.record_landing(RECEIPT, PLAN, tmp_path)
    assert list(tmp_path.glob("*.jsonl")) == []


def test_a_dry_run_leaves_no_change_log_directory_behind(tmp_path):
    from cytir.index import dry_run as DRY
    sub = tmp_path / "sub"
    with pytest.raises(DRY.DryRunWriteRefused):
        with DRY.guard(True):
            CL.record_landing(RECEIPT, PLAN, sub)
    assert not sub.exists()


def test_a_real_write_creates_the_change_log_directory_and_the_segment(tmp_path):
    sub = tmp_path / "sub"
    path = CL.record_landing(RECEIPT, PLAN, sub, landed_at="2026-11-04T10:00:00Z")
    assert sub.is_dir() and path is not None and path.parent == sub and path.exists()


def test_a_flip_without_a_pack_writes_one_rebase_segment_whose_sha_is_the_receipt_sha(tmp_path):
    state = b'{"schema": "aw.flip-state.v1", "new_collection": "aw_e11"}\n'
    path = CL.record_rebase("flip-aw_e11", state, tmp_path, landed_at="2026-11-05T02:14:09Z")
    header, body = CL.read_segment(path)
    assert body == [{"op": "rebase", "why": "flip"}]
    assert header["pack_sha"] == CL.receipt_sha(state) and len(header["pack_sha"]) == 64
    assert CL.receipt_sha({"b": 1, "a": 2}) == CL.receipt_sha({"a": 2, "b": 1})
    assert CL.record_rebase("flip-aw_e11", state, None) is None


def test_the_rebase_verb_reads_stdin_and_is_off_without_a_dir(tmp_path, monkeypatch):
    import io
    import sys
    monkeypatch.setattr(sys, "stdin", type("S", (), {"buffer": io.BytesIO(b'{"a": 1}')})())
    assert CL.main(["rebase", "--pack-id", "flip-x", "--flip-receipt", "-", "--out", str(tmp_path)]) == 0
    seg = CL.segment_for(tmp_path, "flip-x")
    assert seg is not None and CL.read_segment(seg)[0]["pack_sha"] == CL.receipt_sha(b'{"a": 1}')
    assert CL.segment_for(tmp_path / "nope", "flip-x") is None


def test_a_resumed_landing_writes_its_ids_keeps_an_existing_segment_and_refuses_unknown_deletes(
        tmp_path, monkeypatch):
    from cytir.core import paths
    from cytir.index import promote_queue as Q
    rec = {"pack_id": "p9", "status": "LANDED"}
    monkeypatch.setenv("AW_CHANGE_LOG_DIR", "")
    paths.reset_settings_cache()
    Q._log_resumed_landing(rec, {"a", "b"}, manifest={}, manifest_sha="s" * 64)      # off: nothing
    monkeypatch.setenv("AW_CHANGE_LOG_DIR", str(tmp_path))
    paths.reset_settings_cache()
    try:
        Q._log_resumed_landing(rec, {"b", "a"}, manifest={}, manifest_sha="s" * 64)
        seg = CL.segment_for(tmp_path, "p9")
        assert [x["id"] for x in CL.read_segment(seg)[1]] == ["a", "b"]
        before = seg.read_text()
        Q._log_resumed_landing(rec, {"zzz"}, manifest={}, manifest_sha="s" * 64)      # same sha: kept
        assert seg.read_text() == before
        Q._log_resumed_landing(rec, {"zzz"}, manifest={}, manifest_sha="t" * 64, op="upsert")  # a reseal
        header, body = CL.read_segment(CL.segment_for(tmp_path, "p9"))
        assert header["pack_sha"] == "t" * 64 and body == [{"id": "zzz", "op": "upsert"}]
    finally:
        paths.reset_settings_cache()


def _queue(tmp_path, done=(), undone=()):
    q = tmp_path / "queue"
    for name, at in done:
        (q / name).mkdir(parents=True)
        (q / name / "DONE").write_text(at + "\n")
    for name in undone:
        (q / name).mkdir(parents=True)
    return q


def test_audit_lists_done_without_segment_and_segment_without_done(tmp_path):
    q = _queue(tmp_path, done=[("a", "2026-11-05T03:00:00Z"), ("b", "2026-11-05T04:00:00Z"),
                               ("old", "2026-10-01T00:00:00Z")], undone=["c"])
    log = tmp_path / "log"
    CL.record_landing({"pack_id": "a", "status": "LANDED"}, {"upserts": ["x"]}, log,
                      landed_at="2026-11-05T02:59:00Z")
    CL.record_landing({"pack_id": "c", "status": "LANDED"}, {"upserts": ["y"]}, log,
                      landed_at="2026-11-05T05:00:00Z")
    CL.record_rebase("flip-aw_e11", b"r", log, landed_at="2026-11-05T06:00:00Z")
    rep = CL.audit(q, log)
    assert rep["since"] == "2026-11-05T02:59:00Z"
    assert rep["done_without_segment"] == ["b"]          # "old" landed before the log was switched on
    assert rep["segment_without_done"] == ["c"]          # the flip's rebase segment is not a queue pack
    assert CL.main(["audit", str(q), str(log)]) == 1


def test_audit_passes_when_every_done_pack_has_its_segment_and_an_empty_log_checks_nothing(tmp_path):
    q = _queue(tmp_path, done=[("a", "2026-11-05T03:00:00Z")])
    log = tmp_path / "log"
    log.mkdir()
    assert CL.audit(q, log)["done_checked"] == 0 and CL.main(["audit", str(q), str(log)]) == 0
    CL.record_landing({"pack_id": "a", "status": "LANDED"}, {"upserts": ["x"]}, log,
                      landed_at="2026-11-05T02:59:00Z")
    assert CL.main(["audit", str(q), str(log)]) == 0
    assert CL.main(["audit", str(q), str(log), "--since", "2026-11-05T00:00:00Z"]) == 0


def test_the_weekly_audit_units_are_well_formed():
    import configparser
    import pathlib
    import shutil
    import subprocess
    d = repo_file_or_skip(pathlib.Path("deploy") / ("heim" + "dall") / "aw-change-log-audit.service").parent
    parsed = {}
    for name in ("aw-change-log-audit.service", "aw-change-log-audit.timer"):
        cp = configparser.ConfigParser(strict=False, interpolation=None)
        cp.optionxform = str
        cp.read_string((d / name).read_text())
        parsed[name] = cp
    svc, tmr = parsed["aw-change-log-audit.service"], parsed["aw-change-log-audit.timer"]
    assert svc["Service"]["Type"] == "oneshot"
    assert "cytir.index.change_log audit" in svc["Service"]["ExecStart"]
    assert tmr["Timer"]["OnCalendar"] == "Sun *-*-* 06:30:00 Europe/Zurich"
    assert tmr["Timer"]["Unit"] == "aw-change-log-audit.service"
    assert tmr["Install"]["WantedBy"] == "timers.target"
    if shutil.which("systemd-analyze"):
        r = subprocess.run(["systemd-analyze", "calendar", tmr["Timer"]["OnCalendar"]],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def test_audit_with_an_unset_or_missing_dir_exits_2_and_checks_nothing(tmp_path, monkeypatch, capsys):
    from cytir.core import paths
    q = tmp_path / "q"
    q.mkdir()
    monkeypatch.setenv("AW_CHANGE_LOG_DIR", "")
    paths.reset_settings_cache()
    try:
        assert CL.main(["audit", str(q)]) == 2                       # log dir unset
        assert CL.main(["audit", str(q), str(tmp_path / "nope")]) == 2   # log dir missing
        assert CL.main(["audit", str(tmp_path / "nope"), str(q)]) == 2   # queue dir missing
        assert CL.main(["audit", "", ""]) == 2
    finally:
        paths.reset_settings_cache()
    assert "AUDIT NOT RUN" in capsys.readouterr().err


def test_audit_defaults_both_dirs_from_the_settings(tmp_path, monkeypatch):
    from cytir.core import paths
    q, log = tmp_path / "queue", tmp_path / "log"
    q.mkdir()
    log.mkdir()
    monkeypatch.setenv("AW_QUEUE_DIR", str(q))
    monkeypatch.setenv("AW_CHANGE_LOG_DIR", str(log))
    paths.reset_settings_cache()
    try:
        assert CL.main(["audit"]) == 0
    finally:
        paths.reset_settings_cache()


@pytest.mark.parametrize("raw", [b"", b"   \n", b"not json", b"[]", b"{}"])
def test_the_rebase_verb_refuses_an_empty_receipt(tmp_path, monkeypatch, raw):
    import io
    import sys
    monkeypatch.setattr(sys, "stdin", type("S", (), {"buffer": io.BytesIO(raw)})())
    assert CL.main(["rebase", "--pack-id", "flip-x", "--flip-receipt", "-", "--out", str(tmp_path)]) == 2
    assert list(tmp_path.iterdir()) == []


def test_the_already_held_exits_read_the_supersede_state(tmp_path):
    """Finding 4: applied -> the ids prod deleted; no state -> []; deleting -> refuse (log on or off)."""
    from cytir.index import promote_queue as Q
    from cytir.index import url_group_supersede as UGS
    man = {"pack_id": "pk", "supersede": "url-group"}
    sha = "a" * 64
    assert Q._supersede_tombstones_on_resume({"pack_id": "pk"}, sha, tmp_path) == []   # not a supersede pack
    assert Q._supersede_tombstones_on_resume(man, sha, tmp_path) == []                 # no state: no delete ran
    plan = {"pack_id": "pk", "manifest_sha": sha, "ids": ["o1", "o2"], "urls": {}, "export": None,
            "prod": {"to_delete": ["o2", "o1"]}, "summary": {}, "skipped_pack_ids": []}
    UGS.write_state(tmp_path, plan, stage="export-verified")
    assert Q._supersede_tombstones_on_resume(man, sha, tmp_path) == []
    UGS.write_state(tmp_path, plan, stage="applied")
    assert Q._supersede_tombstones_on_resume(man, sha, tmp_path) == [
        {"id": "o1", "reason": "superseded"}, {"id": "o2", "reason": "superseded"}]
    UGS.write_state(tmp_path, plan, stage="deleting")
    with pytest.raises(CL.ChangeLogError, match="settles this"):
        Q._supersede_tombstones_on_resume(man, sha, tmp_path)
    with pytest.raises(CL.ChangeLogError):      # a reseal of a pack that began deleting
        Q._supersede_tombstones_on_resume(man, "b" * 64, tmp_path)


def test_the_reason_table_is_total_over_both_vocabularies_and_lands_in_the_public_one():
    from cytir.distribute import pack
    from cytir.index import promote_queue as Q
    assert set(CL.QUEUE_TO_PUBLIC_REASON) == set(Q.TOMBSTONE_REASONS) | set(pack.TOMBSTONE_REASONS)
    assert set(CL.QUEUE_TO_PUBLIC_REASON.values()) <= pack.TOMBSTONE_REASONS
    assert all(CL.public_reason(r) == r for r in pack.TOMBSTONE_REASONS)
    assert [r for r, v in CL.QUEUE_TO_PUBLIC_REASON.items() if v == "takedown"] == ["takedown"]


def test_a_queue_only_reason_is_written_as_its_public_reason_and_an_unmapped_one_is_refused(tmp_path):
    plan = {"tombstones": [{"id": "t1", "reason": "canonical_duplicate"},
                           {"id": "t2", "reason": "binary_content"}]}
    path = CL.record_landing(RECEIPT, plan, tmp_path)
    assert [x["reason"] for x in CL.read_segment(path)[1]] == ["superseded_generation"] * 2
    with pytest.raises(CL.ChangeLogError):
        CL.record_landing(dict(RECEIPT, pack_id="p2"), {"tombstones": [{"id": "t", "reason": "bogus"}]}, tmp_path)
    assert CL.segment_for(tmp_path, "p2") is None


def test_t4_no_fp_is_written_even_when_the_plan_carries_a_stray_fps_key(tmp_path):
    path = CL.record_landing(RECEIPT, {"upserts": ["i1", ("i2", "replace")],
                                       "fps": {"i1": "0123456789abcdef"}}, tmp_path)
    body = CL.read_segment(path)[1]
    assert body == [{"id": "i1", "op": "upsert"}, {"id": "i2", "op": "replace"}]
    assert not hasattr(CL, "_FP")


def _log_dir(monkeypatch, tmp_path):
    from cytir.core import paths
    monkeypatch.setenv("AW_CHANGE_LOG_DIR", str(tmp_path))
    paths.reset_settings_cache()
    return paths


def test_t2_every_queue_reason_lands_with_public_reason_and_queue_reason_on_both_log_paths(
        tmp_path, monkeypatch):
    from cytir.index import promote_queue as Q
    paths = _log_dir(monkeypatch, tmp_path / "log")
    try:
        for n, raw in enumerate(sorted(Q.TOMBSTONE_REASONS)):
            rec = {"pack_id": f"r{n}", "status": "LANDED"}
            Q._log_resumed_landing(rec, {"a"}, manifest={}, manifest_sha="s" * 64,
                                   tombstones=[{"id": "t", "reason": raw}])
            line = CL.read_segment(CL.segment_for(tmp_path / "log", f"r{n}"))[1][-1]
            assert line == {"id": "t", "op": "tombstone", "reason": CL.QUEUE_TO_PUBLIC_REASON[raw],
                            "queue_reason": raw}
        assert "canonical_duplicate" in Q.TOMBSTONE_REASONS
        assert CL.QUEUE_TO_PUBLIC_REASON["canonical_duplicate"] == "superseded_generation"
        # the sealed path: its url-group supersede deletes are always the queue reason `superseded`
        art = tmp_path / "art"
        art.mkdir()
        CL.written_ids_path(art, "sealed", "prod").write_text(json.dumps(["n1"]))
        Q._log_sealed_landing({"pack_id": "sealed", "status": "LANDED"}, manifest_sha="s" * 64,
                              artifact_root=art, superseded_ids=["old1"])
        line = CL.read_segment(CL.segment_for(tmp_path / "log", "sealed"))[1][-1]
        assert line == {"id": "old1", "op": "tombstone", "reason": "superseded",
                        "queue_reason": "superseded"}
    finally:
        paths.reset_settings_cache()


def test_t3_read_segment_refuses_a_fossil_tombstone_with_a_raw_queue_reason(tmp_path):
    path = tmp_path / "20261105T000000Z-old.jsonl"
    head = {"schema": CL.SCHEMA, "pack_id": "old", "pack_sha": "x", "landed_at": "2026-11-05T00:00:00Z", "n": 1}
    path.write_text(json.dumps(head) + "\n" +
                    json.dumps({"id": "t", "op": "tombstone", "reason": "canonical_duplicate"}) + "\n")
    with pytest.raises(CL.ChangeLogError, match="not public"):
        CL.read_segment(path)
    with pytest.raises(CL.ChangeLogError, match="not public"):
        CL.verify(path, {"pack_id": "old"})


def test_t1_an_unmapped_tombstone_reason_is_refused_at_plan_time_with_no_write_started(monkeypatch):
    """Task 2497 finding 1: `validate_tombstones` refuses a reason the change log cannot map, before
    any write. The only calls it makes are the id lookups; no delete or upsert is reachable from it."""
    from cytir.index import promote_queue as Q
    monkeypatch.setattr(Q.PSP, "already_present", lambda url, coll, ids: list(ids))
    started = []
    monkeypatch.setattr(Q, "WriteAttempt", lambda *a, **k: started.append("write_attempt"), raising=False)
    args = dict(pack_urls=set(), target_url="http://x", collection="c")
    assert Q.validate_tombstones([{"id": "a", "reason": "takedown"}], **args)["count"] == 1
    monkeypatch.setattr(CL, "QUEUE_TO_PUBLIC_REASON",
                        {k: v for k, v in CL.QUEUE_TO_PUBLIC_REASON.items() if k != "takedown"})
    with pytest.raises(Q.QueueError, match="no public change-log mapping"):
        Q.validate_tombstones([{"id": "a", "reason": "takedown"}], **args)
    assert started == []


def test_change_log_does_not_import_the_pack_writer_module():
    """Task 2532: the drain imports the change log, so the log may only reach the stdlib-only vocabulary leaf."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(CL))
    froms = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    mods = {n.module for n in froms}
    names = {a.name for n in froms for a in n.names}
    assert "cytir.distribute.pack" not in mods
    assert not ("cytir.distribute" in mods and "pack" in names)
    assert "cytir.distribute.tombstone_vocab" in mods


def test_seq_is_monotonic_survives_reland_and_refuses_a_bad_counter(tmp_path):
    a = CL.record_landing(RECEIPT, PLAN, tmp_path, landed_at="2026-11-05T02:14:09Z")
    b = CL.record_landing(dict(RECEIPT, pack_id="p2"), PLAN, tmp_path, landed_at="2026-11-05T02:14:09Z")
    assert [CL.read_segment(x)[0]["seq"] for x in (a, b)] == [1, 2]
    CL.record_landing(RECEIPT, PLAN, tmp_path)                 # crash re-land keeps its seq
    assert CL.read_segment(a)[0]["seq"] == 1
    assert CL.latest_cut(tmp_path) == {"stamp": "20261105T021409Z", "seq": 2}
    (tmp_path / CL.SEQ_FILE).write_text("junk")
    with pytest.raises(CL.ChangeLogError, match="counter unreadable"):
        CL.record_landing(dict(RECEIPT, pack_id="p3"), PLAN, tmp_path)


def _land3(tmp_path):
    return [CL.record_landing(dict(RECEIPT, pack_id=f"p{i}"), PLAN, tmp_path,
                              landed_at=f"2026-11-05T02:14:0{i}Z") for i in (1, 2, 3)]


def test_assert_segments_complete_counter_ahead_raises_equal_passes(tmp_path):
    """Task 2588: counter == last seq read passes; counter ahead (newest segment missing) raises."""
    segs = _land3(tmp_path)
    assert CL.assert_segments_complete(tmp_path) == 3
    segs[2].unlink()                                  # the newest segment is not synced / deleted
    with pytest.raises(CL.MissingSegmentError, match="is at seq 3 but the newest segment read holds seq 2"):
        CL.assert_segments_complete(tmp_path)
    assert issubclass(CL.MissingSegmentError, CL.ChangeLogError)


def test_assert_segments_complete_gap_in_the_middle_raises(tmp_path):
    segs = _land3(tmp_path)
    segs[1].unlink()
    with pytest.raises(CL.MissingSegmentError, match="not contiguous at seq 3: expected 2"):
        CL.assert_segments_complete(tmp_path)
    assert CL.assert_segments_complete(tmp_path, after_seq=2) == 3     # the gap is below the cut


def test_assert_segments_complete_race_segment_lands_after_counter_read(tmp_path, monkeypatch):
    """Review 2589: a segment landing between the `.seq` read and the listing must not refuse."""
    _land3(tmp_path)
    real = CL._list_segment_seqs

    def lister(d, after_stamp=None):
        out = real(d, after_stamp)                    # the real listing is done; the segment lands only now
        CL.record_landing(dict(RECEIPT, pack_id="p4"), PLAN, tmp_path, landed_at="2026-11-05T02:14:09Z")
        return out
    monkeypatch.setattr(CL, "_list_segment_seqs", lister)
    # counter read first (3) and listing (3) both predate the landing; a swapped order reads counter 4
    assert CL.assert_segments_complete(tmp_path) == 3


def test_assert_segments_complete_missing_counter_refuses_empty_log_passes(tmp_path):
    _land3(tmp_path)
    (tmp_path / CL.SEQ_FILE).unlink()
    with pytest.raises(CL.MissingSegmentError, match="is missing but"):
        CL.assert_segments_complete(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert CL.assert_segments_complete(empty) == 0


def test_assert_segments_complete_idle_log_with_compacted_cut_passes(tmp_path):
    for x in _land3(tmp_path):
        x.unlink()
    assert CL.assert_segments_complete(tmp_path, after_seq=3) == 0   # counter 3 == cut, nothing newer


def test_assert_segments_complete_skips_pre_cut_files_like_the_delta_builder(tmp_path):
    """Task 2592: a pre-cut segment with an unreadable header is not opened when the cut stamp is
    given, so it cannot refuse a delta that never reads it; without the stamp it still fails loud."""
    segs = _land3(tmp_path)
    segs[0].write_text("{not json\n")
    stamp = CL.segment_stamp(segs[1].name)
    assert CL.assert_segments_complete(tmp_path, after_seq=1, after_stamp=stamp) == 3
    with pytest.raises(CL.ChangeLogError, match="header unreadable"):
        CL.assert_segments_complete(tmp_path, after_seq=1)

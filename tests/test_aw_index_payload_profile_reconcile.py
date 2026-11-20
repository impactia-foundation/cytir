"""Payload-profile reconcile (task 675, moved by task 1241): compare a prod row with the twin row,
list the rows the drain landed, re-project them, and `--apply` payload-only.

Real rows (`fixtures/payload_profile_reconcile/rows-fixture.jsonl`: rows read from the live stores on
2026-09-29), no live calls: the Qdrant client is a fake.
"""
from __future__ import annotations

import gzip
import json
import pathlib

import pytest

from cytir.index import payload_profile as PP
from cytir.index import payload_profile_reconcile as REC
from cytir.core.published import normalize_payload_published as NP

FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures" / "payload_profile_reconcile" / "rows-fixture.jsonl"


def _rows(case=None):
    rows = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
    return [r for r in rows if case is None or r["case"] == case]


@pytest.fixture(scope="module")
def e9():
    return PP.Projector.for_collection("aw_e9")


def test_compare_row_agrees_on_every_real_pair(e9):
    pairs = [dict(r, twin=NP(r["twin"]), prod=NP(r["prod"])) for r in _rows("pair")]
    assert len(pairs) >= 100
    for r in pairs:
        got = REC.compare_row(r["prod"], r["twin"], e9)
        assert got["forward_equal"] and got["decode_equal"], r["id"]


def _queue_with_landed_pack(tmp_path, rows, *, pack_id="news-daily-x"):
    queue = tmp_path / "queue"
    art = queue / "_artifacts"
    art.mkdir(parents=True)
    pack_dir = queue / pack_id
    pack_dir.mkdir()
    points = tmp_path / "pts.jsonl.gz"
    with gzip.open(points, "wt") as fh:
        for r in rows:
            fh.write(json.dumps({"id": r["id"], "payload": r["prod"], "vector": {}}) + "\n")
    (pack_dir / "manifest.json").write_text(json.dumps(
        {"kind": "content_pack", "pack_id": pack_id, "pack_points": str(points),
         "collection": "aw_e8"}))
    (pack_dir / "DONE").write_text("x\n")
    (pack_dir / "receipt.json").write_text(json.dumps(
        {"status": "LANDED", "checked_at": "2026-09-29T12:07:06Z"}))
    (art / f"{pack_id}-content-pack-twin-state.json").write_text(json.dumps(
        {"stage": "twin-write-complete", "pack_ids": [r["id"] for r in rows]}))
    return queue, art


def test_reconcile_dry_run_counts_the_rows_it_would_rewrite(tmp_path):
    rows = _rows("news-daily-20260928") + _rows("hn-discussion-daily-20260928")
    queue, art = _queue_with_landed_pack(tmp_path, rows)
    old = queue / "older"
    old.mkdir()
    (old / "DONE").write_text("x")
    (old / "receipt.json").write_text(json.dumps(
        {"status": "LANDED", "checked_at": "2026-09-29T09:00:00Z"}))
    report = REC.plan_reconcile(queue, art, "2026-09-29T11:39:09Z", "aw_e9")
    assert report["mode"] == "dry-run" and report["roster"] == ["news-daily-x"]
    assert report["totals"]["written_ids"] == len(rows) == report["totals"]["rows_to_rewrite"]
    assert report["totals"]["rows_refused"] == 0
    assert report["packs"][0]["changed_keys"]["stamps"] == sum("stamps" in r["prod"] for r in rows)


def test_reconcile_reports_a_refused_row_and_apply_refuses_to_start(tmp_path):
    rows = [dict(r) for r in _rows("news-daily-20260928")[:2]]
    rows[1] = dict(rows[1], prod=dict(rows[1]["prod"], provenance="never-seen-lane"))
    queue, art = _queue_with_landed_pack(tmp_path, rows)
    report = REC.plan_reconcile(queue, art, "2026-09-29T11:39:09Z", "aw_e9")
    assert report["totals"]["rows_refused"] == 1
    assert report["packs"][0]["refused_keys"] == {"provenance": 1}
    with pytest.raises(PP.PayloadProfileError, match="closed key cannot encode"):
        REC.apply_reconcile(report, art, "http://prod", "aw_e9", pause_s=0,
                            api=lambda *a, **k: {"points_count": 1})


def test_reconcile_apply_is_payload_only_verified_and_count_neutral(tmp_path, monkeypatch):
    rows = _rows("news-daily-20260928")[:4] + _rows("batch2")[:2]
    queue, art = _queue_with_landed_pack(tmp_path, rows)
    report = REC.plan_reconcile(queue, art, "2026-09-29T11:39:09Z", "aw_e9")
    store = {r["id"]: dict(r["prod"]) for r in rows}
    calls = []

    def api(url, path, body=None, method=None):
        calls.append(path)
        if path.endswith("/points/batch?wait=true"):
            for op in body["operations"]:
                (spec,) = op.values()
                assert list(op) == ["overwrite_payload"], "payload-only: no vector op"
                for pid in spec["points"]:
                    store[pid] = spec["payload"]
            return {}
        return {"points_count": len(store)}
    monkeypatch.setattr(REC.PD, "fetch_payload_fields",
                        lambda url, coll, ids, fields, batch=1000:
                        {i: store[i] for i in ids if i in store})
    out = REC.apply_reconcile(report, art, "http://prod", "aw_e9", pause_s=0, batch=3, api=api)
    e9 = PP.Projector.for_collection("aw_e9")
    assert out["rewritten"] == 6 and out["points_count"] == 6
    assert all(PP.form_of(v, "normalized", e9.open_keys) == "projected" for v in store.values())
    assert all(e9.project(v) == v for v in store.values())
    # a second run finds nothing left to rewrite
    assert REC.apply_reconcile(report, art, "http://prod", "aw_e9", pause_s=0, api=api)["rewritten"] == 0


def test_the_raw_profile_has_nothing_to_reconcile(tmp_path):
    with pytest.raises(PP.PayloadProfileError, match="stores the raw form"):
        REC.apply_reconcile({"totals": {}, "packs": []}, tmp_path, "http://twin", "aw_e8",
                            api=lambda *a, **k: {"points_count": 0})


# ---- task 1276: the `published-from-pages` refusals, through main() with nothing reachable ----

_PFP = ["published-from-pages", "--twin-url", "http://twin.invalid", "--twin-collection",
        "fake_twin_1276", "--prod-url", "http://prod.invalid"]


def _no_store(monkeypatch):
    """Any plan or store read fails the test: the refusals must come before them."""
    def boom(*a, **k):
        raise AssertionError("a refusal must come before any store access")
    monkeypatch.setattr(REC, "plan_published", boom)
    monkeypatch.setattr(REC, "verify_published", boom)
    monkeypatch.setattr(REC, "verify_rows", boom)


def test_published_from_pages_without_work_is_refused(monkeypatch, capsys):
    _no_store(monkeypatch)
    with pytest.raises(SystemExit) as e:
        REC.main(_PFP)
    assert str(e.value) == "REFUSED: published-from-pages needs --work (the news work dir)"
    # --apply does not get around it either
    with pytest.raises(SystemExit) as e:
        REC.main([*_PFP, "--apply"])
    assert "needs --work" in str(e.value)


def test_published_from_pages_apply_names_each_missing_guard_input(monkeypatch, tmp_path):
    from cytir.core.paths import SettingRefused
    _no_store(monkeypatch)

    def refused():
        raise SettingRefused("not configured")
    monkeypatch.setattr("cytir.core.paths.operator_settings", refused)
    with pytest.raises(SystemExit) as e:
        REC.main([*_PFP, "--work", str(tmp_path), "--apply"])
    msg = str(e.value)
    assert msg.startswith("REFUSED: --apply needs ")
    for flag in ("--guard-ssh-host", "--guard-container", "--guard-search-url"):
        assert flag in msg
    # one flag given: only the other two are named
    with pytest.raises(SystemExit) as e:
        REC.main([*_PFP, "--work", str(tmp_path), "--apply", "--guard-container", "qd"])
    msg = str(e.value)
    assert "--guard-container" not in msg and "--guard-ssh-host" in msg and "--guard-search-url" in msg


def test_published_leg_request_body_is_the_one_set_payload_body():
    """task 2786: the `published` apply goes through `point_delete.set_payload_batch`; the request
    body is byte-identical to the hand-built one it replaced."""
    import json
    store = {"id-1": {"published": "2026-01-01"}, "id-2": {"published": "2026-02-02"}}
    rows = [{"id": "id-1"}, {"id": "id-2"}]
    sent = []

    def api(url, path, body=None, method=None):
        if path.endswith("/points/batch?wait=true"):
            sent.append((path, method, json.dumps(body, sort_keys=True)))
            return {}
        return {"points_count": 2}
    fake_fetch = lambda url, coll, ids, fields, batch=1000: {i: dict(store[i]) for i in ids}
    import unittest.mock as um
    with um.patch.object(REC.PD, "fetch_payload_fields", fake_fetch), \
            um.patch("cytir.core.published.normalize_published", lambda v: "NORM-" + v):
        try:
            REC.apply_published_leg(rows, "http://x", "c", pause_s=0, api=api)
        except PP.PayloadProfileError:
            pass                                    # the read-back (unchanged fake store) refuses; irrelevant here
    old_body = {"operations": [{"set_payload": {"payload": {"published": "NORM-" + store[i]["published"]},
                                                "points": [i]}} for i in ("id-1", "id-2")]}
    assert sent == [("/collections/c/points/batch?wait=true", None, json.dumps(old_body, sort_keys=True))]

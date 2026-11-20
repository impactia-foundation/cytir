"""The aw_e10 writer contract, staged (task 847, slice 3 of the aw_e10 builder lane).

What is pinned here:
* the STAGED declaration of `aw_e10` (`data/collection-payload-profiles.v1.json`): profile `rebuild_e10`,
  the frozen aw_e9 BM25 rule, the tracked enum tables, `id_rule: I1`, not serving;
* "off until the flip": a live writer is refused a staged collection, the drain refuses at start when
  the prod alias resolves to one, and `live_target` does not count it as a live name (it is the BUILD's
  target); a staged collection can never be `serving`;
* "on at the flip" (the same declaration with `state: live`): the projection drops `tier`, `build_id`,
  `point_id` and `stamps`, and the drain's upsert refuses a batch whose ids are not the I1 ids of their
  own rows, BEFORE any PUT (`cytir.index.id_rule`);
* the out-of-queue apply ledgers (task 834: the news catch-up runner) resolve to ids, after T0 only;
* the enum-delta data PR (`rebuild_e10 tables-pr`, deleted builder verb, task 1312): appends exactly the build's codes and moves every
  declaration that pinned the old table; the build adopts codes main assigned during the window, and
  refuses a code main gave another number;
* `rebuild_e10 map-ids` (deleted builder verb, task 1312): old twin ids -> aw_e10 ids through the build's current id map.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import pathlib
import uuid
from datetime import datetime, timezone

import pytest

from cytir.core.canon import canonical_url
from cytir.core.ids import mint_point_id
from cytir.index import id_rule as IR
from cytir.index import live_target as LT
from cytir.index import out_of_queue as OOQ
from cytir.index import payload_profile as PP
from cytir.index import prod_guard as PG
from cytir.index import promote_queue as PQ
from cytir.index import promote_sealed_pack as PSP

TABLES = PP.TABLES_PATH
PROFILES = PP.PROFILES_PATH


def _declared() -> dict:
    return json.loads(PROFILES.read_text(encoding="utf-8"))


def _value(key: str, code: int = 0):
    """The raw value the tracked table codes `code` under `key` (tables hold canonical JSON)."""
    doc = json.loads(TABLES.read_text(encoding="utf-8"))
    return json.loads(doc["tables"][key][str(code)])


def _payload(url: str, *, tier: str = "B", text: str = "Some text of a page.", **kw) -> dict:
    digest = hashlib.sha256(text.encode()).hexdigest()
    pl = {"url": url, "tier": tier, "title": "A title", "lang": "en", "content_digest": digest,
          "build_id": _value("build_id"), "rule_epochs": _value("rule_epochs"),
          "provenance": _value("provenance"), "prov_id": _value("prov_id"),
          "chunk_rule": _value("chunk_rule"), "consent_basis": _value("consent_basis"),
          "consent_state": _value("consent_state"), "consent_source": _value("consent_source"),
          "stamps": {"prov_id": "x"}, "source_type": "reference"}
    comp = {"A": kw.get("anchor"), "B": digest, "C": ""}[tier]
    pl["point_id"] = mint_point_id(canonical_url(url), comp)
    pl.update(kw)
    return pl


def _live_copy(tmp_path: pathlib.Path, *, sparse_rule_null: bool = True) -> pathlib.Path:
    """The declarations with aw_e10 switched ON (state live), as the flip commit would have it."""
    doc = _declared()
    e = doc["collections"]["aw_e10"]
    e["state"] = "live"
    if sparse_rule_null:      # the term set (231 MB) is not in CI; the rule's own tests pin it
        e["sparse_rule"] = None
    p = tmp_path / "profiles.json"
    p.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return p


# ------------------------------------------------------------------------------ the declaration
def test_aw_e10_is_declared_live_serving_with_the_rebuild_profile_and_the_frozen_aw_e9_rule():
    d = PP.load_profiles()
    e10, e9 = d["aw_e10"], d["aw_e9"]
    assert e10["profile"] == "rebuild_e10" and not PP.is_staged(e10) and e10.get("serving") is True
    assert e10["id_rule"] == IR.ID_RULE_I1
    assert e10["open_keys"] == ["consent_source"]
    assert e10["tables_sha256"] == e9["tables_sha256"] == PP.file_sha256(TABLES)
    pinned = ("id", "epoch", "rule_epochs_key", "df_gt_frac", "df_lt_abs", "df_n", "terms",
              "drop_terms_sha256")
    assert {k: e10["sparse_rule"][k] for k in pinned} == {k: e9["sparse_rule"][k] for k in pinned}
    # after the F7 data commit (task 964): aw_e9 stopped serving, aw_e10_twin declared as its twin
    assert d["aw_e8"]["sparse_rule"] is None and d["aw_e9"]["serving"] is False
    tw = d["aw_e10_twin"]
    assert tw["profile"] == e10["profile"] and tw["id_rule"] == e10["id_rule"]
    assert tw["serving"] is False and not PP.is_staged(tw)
    assert PP.staged_collections(d) == frozenset()


def test_a_staged_collection_cannot_serve_and_an_unknown_state_is_refused(tmp_path):
    doc = _declared()
    doc["collections"]["aw_e10"].update(state="staged", serving=True)
    p = tmp_path / "p.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(PP.PayloadProfileError, match="staged.*serving"):
        PP.load_profiles(p)
    doc["collections"]["aw_e10"].update(serving=False, state="building")
    p.write_text(json.dumps(doc))
    with pytest.raises(PP.PayloadProfileError, match="state"):
        PP.load_profiles(p)


def _pre_flip(tmp_path, monkeypatch) -> None:
    """The declarations BEFORE the F7 data commit (task 964): aw_e10 staged, aw_e9 serving, no aw_e10_twin."""
    doc = _declared()
    doc["collections"]["aw_e10"].update(state="staged", serving=False)
    doc["collections"]["aw_e9"]["serving"] = True
    doc["collections"].pop("aw_e10_twin", None)
    p = tmp_path / "pre-flip.json"
    p.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    monkeypatch.setattr(PP, "PROFILES_PATH", p)
    PG.reset()


def test_prod_guard_does_not_count_the_staged_collection_as_prod(tmp_path, monkeypatch):
    _pre_flip(tmp_path, monkeypatch)
    assert not PG.is_prod(None, "aw_e10")
    assert PG.is_prod(None, "aw_e9")


# --------------------------------------------------------------------------- OFF until the flip
def test_a_live_writer_is_refused_a_staged_collection(tmp_path, monkeypatch):
    _pre_flip(tmp_path, monkeypatch)
    with pytest.raises(PP.StagedCollectionError, match="staged"):
        PP.Projector.for_collection("aw_e10")
    # a reader of the staged contract (this test, a dry check) may ask explicitly
    pj = PP.Projector.for_collection("aw_e10", allow_staged=True)
    assert pj.profile == "rebuild_e10"


def test_the_staged_collection_is_the_builds_target_not_a_live_name(tmp_path, monkeypatch):
    _pre_flip(tmp_path, monkeypatch)
    assert "aw_e10" not in LT.live_collection_names()
    assert {"aw_e8", "aw_e9"} <= LT.live_collection_names()
    monkeypatch.setattr(PP, "PROFILES_PATH", _live_copy(tmp_path))
    assert "aw_e10" in LT.live_collection_names()            # after the flip commit: live


def test_the_drain_refuses_at_start_when_prod_resolves_to_a_staged_collection(tmp_path, monkeypatch, capsys):
    _pre_flip(tmp_path, monkeypatch)
    monkeypatch.setattr(PQ, "resolve_prod_alias", lambda url, alias="aw_prod": "aw_e10")
    assert PQ.sparse_rule_startup_refusal("http://127.0.0.1:1", "aw_prod") == 2
    assert "staged" in capsys.readouterr().err


# ---------------------------------------------------------------------------- ON at the flip
def test_with_state_live_the_projection_drops_the_three_keys_and_codes_the_enums(tmp_path):
    pj = PP.Projector.for_collection("aw_e10", profiles_path=_live_copy(tmp_path))
    out = pj.project(_payload("https://example.org/a"))
    for gone in ("tier", "build_id", "point_id", "stamps"):
        assert gone not in out
    assert isinstance(out["provenance"], int) and isinstance(out["rule_epochs"], int)
    assert len(out["content_digest"]) == 32




def test_check_points_names_every_point_not_keyed_by_its_i1_id():
    good = _payload("https://example.org/g")
    gen0 = _payload("https://example.org/h")
    pts = [{"id": good["point_id"], "payload": good},
           {"id": str(uuid.UUID(int=0xA0)), "payload": gen0}]
    bad = IR.check_points(pts)
    assert [b["id"] for b in bad] == [pts[1]["id"]] and bad[0]["expected"] == gen0["point_id"]


def test_the_guard_is_off_for_every_collection_that_declares_no_id_rule(tmp_path):
    pts = [{"id": str(uuid.uuid4()), "payload": _payload("https://example.org/x")}]
    assert IR.refuse_unless_i1("aw_e9", pts) == 0 and IR.refuse_unless_i1("aw_e8", pts) == 0
    assert IR.refuse_unless_i1("a_strangers_target", pts) == 0
    doc = _declared()
    doc["collections"]["aw_e10"]["id_rule"] = "I2"
    p = tmp_path / "p.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(PP.PayloadProfileError, match="id_rule"):
        IR.declared_id_rule("aw_e10", profiles_path=p)


def test_the_drain_upsert_refuses_a_non_i1_batch_before_any_put(tmp_path, monkeypatch):
    monkeypatch.setattr(PP, "PROFILES_PATH", _live_copy(tmp_path))
    calls = []
    monkeypatch.setattr(PSP, "_api", lambda url, path, body=None, method=None, **kw:
                        calls.append((method, path, body)) or {})
    good = _payload("https://example.org/ok")
    gen0 = _payload("https://example.org/old")
    vec = {"dense": [0.1] * 4}
    batch = [{"id": good["point_id"], "payload": good, "vector": vec},
             {"id": str(uuid.UUID(int=0xA1)), "payload": gen0, "vector": vec}]
    with pytest.raises(IR.IdRuleRefused, match="I1"):
        PSP.upsert("http://127.0.0.1:1", "aw_e10", copy.deepcopy(batch), batch=10)
    assert calls == []                                       # nothing reached the collection
    assert PSP.upsert("http://127.0.0.1:1", "aw_e10", copy.deepcopy(batch[:1]), batch=10) == 1
    (method, path, body), = calls
    stored = body["points"][0]["payload"]
    assert method == "PUT" and "point_id" not in stored and "tier" not in stored
    # the same batch to today's prod (no id_rule) is not checked by the guard
    calls.clear()
    assert IR.refuse_unless_i1("aw_e9", batch) == 0


# ------------------------------------------------------------------ the out-of-queue ledgers (834)
def _oob_dir(tmp_path: pathlib.Path, ids: list[str], tomb: list[str], ts: str) -> tuple[pathlib.Path, str]:
    d = tmp_path / "news_work"
    (d / "pack-x").mkdir(parents=True)
    blob = gzip.compress("".join(json.dumps({"id": i, "payload": {}}) + "\n" for i in ids).encode())
    sha = hashlib.sha256(blob).hexdigest()
    (d / "pack-x" / f"news-points.{sha}.jsonl.gz").write_bytes(blob)
    tdoc = json.dumps({"schema": "aw.explicit-tombstone-pointers.v1", "artifact_sha256": sha,
                       "aged_out": [{"predecessor_id": t, "reason": "news_window_aged_out"} for t in tomb],
                       "aged_out_dropped_as_still_current": ["not-a-tombstone"]}).encode()
    tsha = hashlib.sha256(tdoc).hexdigest()
    (d / "pack-x" / f"tombstone-pointers.{tsha}.json").write_bytes(tdoc)
    (d / "run-x-ledger.jsonl").write_text(json.dumps(
        {"status": "RESTORE-PLAN", "artifact_sha256": sha, "artifact_ids": len(ids),
         "tombstone_pointers_sha256": tsha, "ts": ts}) + "\n"
        + json.dumps({"status": "RESTORE-VERIFIED", "ts": ts}) + "\n")
    return d, sha


def test_out_of_queue_ledgers_name_their_pack_and_tombstone_ids_after_t0_only(tmp_path):
    ids = [str(uuid.UUID(int=k)) for k in range(1, 4)]
    tomb = [str(uuid.UUID(int=99))]
    d, _ = _oob_dir(tmp_path, ids, tomb, ts="2026-10-12T10:00:00Z")
    since = datetime(2026, 10, 12, 9, 0, tzinfo=timezone.utc).timestamp()
    os.utime(d / "run-x-ledger.jsonl", (since + 3 * 3600, since + 3 * 3600))   # written after T0
    got, summary = OOQ.ledger_receipts([str(d)], since)
    assert {k: v.split(":")[0] for k, v in got.items()} == {
        **{i: "oob_written" for i in ids}, tomb[0]: "oob_tombstoned"}
    assert summary["rows_since"] == 1 and summary["ids_unique"] == 4
    # the same ledger row BEFORE T0: its write was read by the build, it accounts for nothing now
    later = since + 2 * 3600
    assert OOQ.ledger_receipts([str(d)], later)[0] == {}


def test_an_out_of_queue_artifact_that_is_missing_or_changed_refuses(tmp_path):
    d, sha = _oob_dir(tmp_path, [str(uuid.UUID(int=5))], [], ts="2026-10-12T10:00:00Z")
    pts = d / "pack-x" / f"news-points.{sha}.jsonl.gz"
    pts.write_bytes(gzip.compress(b'{"id": "x"}\n'))
    with pytest.raises(OOQ.OutOfQueueError, match="sha256"):
        OOQ.ledger_receipts([str(d)], 0)
    pts.unlink()
    with pytest.raises(OOQ.OutOfQueueError, match="names pack artifact"):
        OOQ.ledger_receipts([str(d)], 0)




# --------------------------------------------------------------------------- the enum-delta PR










# ---------------------------------------------------------------------------------------- map-ids














# ------------------------------------------- task 863: the slice-4 follow-ups of review 854 (S1-S3)
#: The ledgers below are "written" after T0 (2026-10-12 09:00Z): their mtime is set to 12:00Z.
WRITTEN_AFTER_T0 = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc).timestamp()


def _ledger(d: pathlib.Path, name: str, rows: list[dict]) -> pathlib.Path:
    p = d / name
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    os.utime(p, (WRITTEN_AFTER_T0, WRITTEN_AFTER_T0))
    return p


def test_s1_only_an_applied_artifact_counts_a_plan_row_alone_never(tmp_path):
    """Review 854 S1 (a): the real ledgers name artifacts on PLAN rows, also of dry runs and of runs
    that ended FAILED before writing. Only a plan row followed by a twin-write outcome counts; the
    rest is reported apart (`plan_only_not_counted`)."""
    ids = [str(uuid.UUID(int=k)) for k in range(11, 14)]
    d, sha = _oob_dir(tmp_path, ids, [], ts="2026-10-12T10:00:00Z")
    since = datetime(2026, 10, 12, 9, 0, tzinfo=timezone.utc).timestamp()
    plan = {"status": "CONTENT-PACK-PLAN", "artifact_sha256": sha, "artifact_ids": len(ids),
            "ts": "2026-10-12T10:00:00Z"}
    (d / "run-x-ledger.jsonl").unlink()
    _ledger(d, "preflight-dry-ledger.jsonl", [plan, {"status": "PREFLIGHT-OK", "ts": "2026-10-12T10:00:05Z"}])
    _ledger(d, "apply-armed-ledger.jsonl", [plan, {"status": "PREFLIGHT-OK", "ts": "2026-10-12T10:01:00Z"},
                                            {"status": "PRESERVATION-VERIFIED", "ts": "2026-10-12T10:02:00Z"},
                                            {"status": "FAILED", "error": "ApplyAbort", "ts": "2026-10-12T10:03:00Z"}])
    got, summary = OOQ.ledger_receipts([str(d)], since)
    assert got == {} and summary["rows_since"] == 0
    assert len(summary["plan_only_not_counted"]) == 2
    # the same plan followed by its twin verification counts -- even when the PROD leg failed after it
    _ledger(d, "apply-armed-ledger.jsonl", [plan, {"status": "TWIN-VERIFIED", "ts": "2026-10-12T10:02:00Z"},
                                            {"status": "FAILED", "error": "prod leg", "ts": "2026-10-12T10:03:00Z"}])
    got, summary = OOQ.ledger_receipts([str(d)], since)
    assert set(got) == set(ids) and summary["rows_since"] == 1
    # two plans in one ledger: the outcome belongs to the plan it follows, never to an earlier one
    other = dict(plan, artifact_sha256="f" * 64)
    _ledger(d, "apply-armed-ledger.jsonl", [plan, other, {"status": "TWIN-VERIFIED", "ts": "2026-10-12T10:02:00Z"}])
    applied, plan_only = OOQ.ledger_applies(str(d / "apply-armed-ledger.jsonl"))
    assert [a["sha"] for a in applied] == ["f" * 64] and [a["sha"] for a in plan_only] == [sha]


def test_s1_an_applied_artifact_is_dated_by_its_own_rows_never_by_the_file_mtime(tmp_path):
    """Review 854 S1 (b): real RESTORE-PLAN rows can carry no `ts`; dating them by the file's mtime
    re-dates every old row when one unrelated line is appended. An undated apply is not counted
    (reported apart); the outcome row's time dates an apply."""
    ids = [str(uuid.UUID(int=21))]
    d, sha = _oob_dir(tmp_path, ids, [], ts="2026-10-12T10:00:00Z")
    since = datetime(2026, 10, 12, 9, 0, tzinfo=timezone.utc).timestamp()
    led = _ledger(d, "run-x-ledger.jsonl", [{"status": "RESTORE-PLAN", "artifact_sha256": sha,
                                            "artifact_ids": 1}, {"status": "RESTORE-VERIFIED"}])
    old = since - 30 * 86400
    os.utime(led, (old, old))
    assert OOQ.ledger_receipts([str(d)], since)[0] == {}                   # the file predates T0
    with open(led, "a") as fh:                                               # an unrelated append
        fh.write(json.dumps({"status": "NOTE"}) + "\n")
    os.utime(led, (WRITTEN_AFTER_T0, WRITTEN_AFTER_T0))
    got, summary = OOQ.ledger_receipts([str(d)], since)
    assert got == {} and len(summary["undated_applied_not_counted"]) == 1
    # the outcome row's own time dates the apply (the plan row has none)
    _ledger(d, "run-x-ledger.jsonl", [{"status": "RESTORE-PLAN", "artifact_sha256": sha, "artifact_ids": 1},
                                      {"status": "RESTORE-VERIFIED", "ts": "2026-10-12T11:00:00Z"}])
    assert set(OOQ.ledger_receipts([str(d)], since)[0]) == set(ids)
    _ledger(d, "run-x-ledger.jsonl", [{"status": "RESTORE-PLAN", "artifact_sha256": sha, "artifact_ids": 1},
                                      {"status": "RESTORE-VERIFIED", "ts": "2026-10-12T08:00:00Z"}])
    assert OOQ.ledger_receipts([str(d)], since)[0] == {}                     # applied before T0







"""The drain's `transaction=` parameter (slice 8 step 2, specs/aw-index-serve-gate.md §2).

1. CHARACTERIZATION (written and its golden recorded on the code BEFORE the change): one sealed real
   pack through the default twin-first path, against two throwaway Qdrant collections plus the
   `aw_prod` alias, compared byte for byte (timestamps and durations normalised) with the golden
   `fixtures/drain_transaction/twin_first.golden.json`: every pack receipt, the DONE/READY markers,
   the per-pack ledger and written-id artifacts, the lock protocol the drain ran, and every point
   (id, vectors, sparse vectors, payload) both collections hold afterwards.
2. `Single` (below): the same code, one target, no twin leg.

Needs a throwaway Qdrant (`AW_TEST_QDRANT_URL`, never a live port); skips without one. Regenerate the
golden ONLY for a deliberate, reviewed change of what the drain writes: `AW_REGEN_GOLDEN=1`.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import pytest

pytestmark = pytest.mark.usefixtures("no_sparse_rule")   # task 717: these test the drain/build plumbing, not the rule

import drain_transaction_harness as H
from cytir.index import promote_queue as Q                       # noqa: E402  (the alias of cytir.index.promote_queue)
from cytir.index import promote_sealed_pack as PSP  # noqa: E402

GOLDEN = H.HERE / "fixtures" / "drain_transaction" / "twin_first.golden.json"


# ---- prose normalisation (task 2393) ----------------------------------------------------------
# The documents the drain writes carry free-text explanations beside the contract fields. A reworded
# comment or label is not a change of what the drain does, yet it broke this golden twice (#1682,
# #1714). Each field below is a sentence written for a human reader; no consumer parses it. Everything
# else (ids, urls, counts, ratios, flags, status, stages, collection names) stays byte-exact.
# A path is the tail of the key chain; lists are transparent. Table with writers:
# the task 2393 doc.
PROSE_PATHS = {
    ("identity_check", "reason"),                  # cytir/index/promote_sealed_pack.py (identity_check block)
    ("identity_check", "note"),                    # cytir/index/promote_sealed_pack.py:1592
    ("identity_reason",),                          # manifest field, copied into the receipt
    ("vector_key", "why"),                         # cytir/index/promote_sealed_pack.py:1606
    ("layout_note",),                              # cytir/index/promote_sealed_pack.py:1617
    ("payload_carry", "fields", "counts"),         # cytir/index/payload_carry.py:251 (POLICY_CONDITIONED label)
    ("proof", "scope"),                            # cytir/index/promote_queue.py ledger proof (:5087)
    ("rollback",),                                 # cytir/index/twin_first.py:182,237 ledger line
}
PROSE_PARENTS = {"checks_composed_from"}          # cytir/gates/mechanical_content_check.py:509 -- every value is a label
PROSE_LIST_AFTER_COLON = {"skipped_checks"}       # cytir/gates/mechanical_content_check.py:506 -- "<check>: <why>"
PROSE_MARK = "<PROSE>"


def _prose(doc, seen: set, path: tuple = ()):
    if isinstance(doc, list):
        return [_prose(v, seen, path) for v in doc]
    if not isinstance(doc, dict):
        return doc
    out = {}
    for key, val in doc.items():
        here = path + (key,)
        if isinstance(val, str) and (key in PROSE_PARENTS or path[-1:] and path[-1] in PROSE_PARENTS
                                      or any(here[-len(t):] == t for t in PROSE_PATHS)):
            assert val.strip(), f"prose field {'/'.join(here)} is empty"
            seen.add(here[-2:] if len(here) > 1 else here)
            out[key] = PROSE_MARK
        elif key in PROSE_LIST_AFTER_COLON and isinstance(val, list):
            assert val and all(isinstance(v, str) and ": " in v for v in val), f"{key}: {val!r}"
            seen.add((key,))
            out[key] = [v.split(": ", 1)[0] + ": " + PROSE_MARK for v in val]
        else:
            out[key] = _prose(val, seen, here)
    return out


def _prose_text(name: str, text: str, seen: set) -> str:
    """A queue/artifact file is stored as text: parse JSON / JSON-lines, normalise, leave the rest."""
    try:
        if name.endswith(".jsonl"):
            return "\n".join(json.dumps(_prose(json.loads(l), seen), sort_keys=True)
                             for l in text.splitlines())
        if name.endswith(".json"):
            return json.dumps(_prose(json.loads(text), seen), sort_keys=True)
    except ValueError:
        pass
    return text


def normalise_prose(doc: dict, seen: set | None = None) -> dict:
    """`doc` = the run's (or golden's) dict; returns a copy with every known prose field masked."""
    seen = set() if seen is None else seen
    out = dict(doc)
    out["results"] = _prose(doc["results"], seen)
    for section in ("queue", "artifacts"):
        out[section] = {n: _prose_text(n, t, seen) for n, t in doc[section].items()}
    return out


@pytest.fixture
def url():
    u = H.qdrant_url()
    yield u
    # Teardown (task 1478): hand the throwaway back as a stranger's -- no `aw_prod` alias, none of the
    # collections this test creates -- so a second local run is proven afresh instead of refused.
    # Never masks the test result: a failed clean-up is reported on stderr, not raised.
    try:
        H.clear_live(u)
    except Exception as exc:  # noqa: BLE001 -- teardown must not turn a pass into an error
        print(f"teardown: could not clear {u}: {exc}", file=sys.stderr)


def run_twin_first(tmp: pathlib.Path, url: str, monkeypatch, **extra) -> dict:
    for name in (H.TWIN, H.PROD):
        H.reset_collection(url, name)
    H.set_alias(url, H.ALIAS, H.PROD)
    queue = H.sealed_pack(tmp)
    calls = H.install_fakes(monkeypatch, Q, PSP)
    artifacts = tmp / "artifacts"
    results = Q.drain(queue, twin_url=url, prod_url=url, task="696-char", apply=True, batch=3,
                      pause_s=0, artifact_root=artifacts, refresh_index=False,
                      prod_alias=H.ALIAS, **extra)
    return {
        "results": json.loads(H.normalise(json.dumps(results, sort_keys=True), tmp)),
        "queue": H.snapshot_tree(queue, tmp),
        "artifacts": H.snapshot_tree(artifacts, tmp),
        "lock_protocol": calls,
        "twin_points": H.dump_points(url, H.TWIN),
        "prod_points": H.dump_points(url, H.PROD),
    }


def test_twin_first_is_byte_identical_to_the_recorded_golden(tmp_path, url, monkeypatch):
    got = run_twin_first(tmp_path, url, monkeypatch)
    assert got["results"][0]["status"] == "LANDED", got["results"]
    assert len(got["twin_points"]) == len(got["prod_points"]) == 4
    if os.environ.get("AW_REGEN_GOLDEN") == "1":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(got, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
        pytest.skip("golden regenerated")
    want = json.loads(GOLDEN.read_text())
    got = json.loads(json.dumps(got, sort_keys=True))      # same JSON round trip as the golden
    got, want = normalise_prose(got), normalise_prose(want)   # prose masked on BOTH sides (task 2393)
    assert sorted(got["queue"]) == sorted(want["queue"])
    assert sorted(got["artifacts"]) == sorted(want["artifacts"])
    for section in ("queue", "artifacts"):
        for name in want[section]:
            assert got[section][name] == want[section][name], f"{section}/{name} differs"
    assert got["lock_protocol"] == want["lock_protocol"]
    assert got["results"] == want["results"]
    assert got["twin_points"] == want["twin_points"]
    assert got["prod_points"] == want["prod_points"]


def test_twin_first_is_deterministic_run_to_run(tmp_path, url, monkeypatch):
    """The comparison above is only a proof if two runs of the SAME code agree."""
    a = run_twin_first(tmp_path / "a", url, monkeypatch)
    b = run_twin_first(tmp_path / "b", url, monkeypatch)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_change_log_segment_lists_the_ids_the_drain_landed(tmp_path, url, monkeypatch):
    """Task 2314: with `change_log_dir` set the drain writes one segment beside (not inside) its files.

    The segment's ids are the ids the twin and prod hold, the receipt agrees (`verify`), and every
    byte the golden records is unchanged: the new file is the only difference."""
    from cytir.core import paths
    from cytir.index import change_log as CL
    log_dir = tmp_path / "changelog"
    monkeypatch.setenv("AW_CHANGE_LOG_DIR", str(log_dir))
    paths.reset_settings_cache()
    got = run_twin_first(tmp_path / "run", url, monkeypatch)
    segments = sorted(log_dir.glob("*.jsonl"))
    assert len(segments) == 1, segments
    header, body = CL.read_segment(segments[0])
    twin_ids = sorted(p["id"] for p in got["twin_points"])
    assert sorted(x["id"] for x in body) == twin_ids == sorted(p["id"] for p in got["prod_points"])
    assert {x["op"] for x in body} == {"add"} and header["n"] == len(twin_ids) == 4
    assert len(header["pack_sha"]) == 64
    assert CL.verify(segments[0], got["results"][0])["add"] == 4
    want = json.loads(GOLDEN.read_text())
    got = json.loads(json.dumps(got, sort_keys=True))
    for section in ("queue", "artifacts"):
        assert got[section] == want[section], f"{section} differs with the change log on"
    assert got["lock_protocol"] == want["lock_protocol"] and got["results"] == want["results"]


OLD_ROW = "00000000-0000-5000-8000-0000000000e1"


def test_change_log_tombstones_equal_the_ids_prod_deleted_for_a_url_group_supersede(
        tmp_path, url, monkeypatch):
    """Task 2337: land a sealed `supersede: url-group` pack onto a real Qdrant holding one old row at
    a url the pack writes. The segment's `tombstone` lines are exactly the ids prod no longer holds,
    its `add` lines the ids prod now holds from the pack, and `verify` passes against the receipt."""
    import hashlib
    from cytir.core import paths
    from cytir.index import change_log as CL
    from cytir.index import url_group_supersede as UGS
    for name in (H.TWIN, H.PROD):
        H.reset_collection(url, name)
    H.set_alias(url, H.ALIAS, H.PROD)
    queue = H.sealed_pack(tmp_path)
    pack_dir = next(p for p in queue.iterdir() if p.is_dir())
    mp = pack_dir / "manifest.json"
    manifest = json.loads(mp.read_text())
    H.install_fakes(monkeypatch, Q, PSP)
    real_plan = Q.dry_plan
    prov = "ce_juri"      # the fixture family serves two provenances; the allowlist names one

    def one_provenance(*a, **k):
        r = real_plan(*a, **k)
        for v in r["_joined"].values():
            v[2]["provenance"] = prov
        return r
    monkeypatch.setattr(Q, "dry_plan", one_provenance)
    joined = Q.dry_plan(manifest, url)["_joined"]
    pack_payload = next(iter(joined.values()))[2]
    # the stage's own guard block, bound to the pack's chunks (as a guarded family's stage writes it)
    stage = pathlib.Path(manifest["pack_chunks"]).parent
    pj = stage / "PACK.json"
    meta = json.loads(pj.read_text())
    meta["pack_staged_text_guard"] = {
        "docs_checked": len({v[2]["url"] for v in joined.values()}), "docs_unchecked": 0,
        "docs_shortfall": 0, "threshold_declared": True, "min_coverage": 1.0, "lowest_coverage": 1.0}
    pj.write_text(json.dumps(meta))
    manifest.update({"supersede": "url-group", "allow_supersede": "task 2337 test"})
    body = json.dumps(manifest, indent=1, sort_keys=True).encode()
    mp.write_bytes(body)
    (pack_dir / "READY").write_text(hashlib.sha256(body).hexdigest() + "\n")
    monkeypatch.setitem(UGS.URL_GROUP_FAMILIES, manifest["family"], prov)
    # one OLD row of that family at a url the pack writes, on the twin and on prod
    old = {"id": OLD_ROW, "payload": {"url": pack_payload["url"], "provenance": prov,
                                      "source_type": pack_payload.get("source_type", "docs"),
                                      "snippet": "old"},
           "vector": {"dense": [0.1] * H.MRL, "bm25": {"indices": [1], "values": [1.0]}}}
    for coll in (H.TWIN, H.PROD):
        H.api(url, f"/collections/{coll}/points?wait=true", {"points": [old]})
    export = tmp_path / "exports"
    export.mkdir()
    log_dir = tmp_path / "changelog"
    monkeypatch.setenv("AW_REMOVED_ROWS_EXPORT_DIR", str(export))
    monkeypatch.setenv("AW_CHANGE_LOG_DIR", str(log_dir))
    monkeypatch.setattr(UGS, "volatile_reason", lambda d: None)
    monkeypatch.setattr(UGS, "PROD_READ_SLEEP_S", 0)
    paths.reset_settings_cache()
    results = Q.drain(queue, twin_url=url, prod_url=url, task="2337-test", apply=True, batch=3,
                      pause_s=0, artifact_root=tmp_path / "artifacts", refresh_index=False,
                      prod_alias=H.ALIAS)
    rec = results[0]
    assert rec["status"] == "LANDED", rec
    assert rec["prod"]["superseded"] == 1
    prod_ids = {str(p["id"]) for p in H.dump_points(url, H.PROD)}
    assert OLD_ROW not in prod_ids, "prod still holds the superseded row"
    segments = sorted(log_dir.glob("*.jsonl"))
    assert len(segments) == 1, segments
    _header, lines = CL.read_segment(segments[0])
    assert sorted(x["id"] for x in lines if x["op"] == "tombstone") == [OLD_ROW]
    assert sorted(x["id"] for x in lines if x["op"] == "add") == sorted(prod_ids)
    assert all(x["reason"] == "superseded" for x in lines if x["op"] == "tombstone")
    assert CL.verify(segments[0], rec)["tombstone"] == 1


def _recorded() -> dict:
    return json.loads(GOLDEN.read_text())


def test_prose_normaliser_ignores_rewording_but_not_a_changed_count():
    """Task 2393: the golden compare tolerates a reworded sentence, never a changed contract value.
    Pure: runs on the recorded golden itself, no Qdrant."""
    seen: set = set()
    base = normalise_prose(_recorded(), seen)
    assert len(seen) >= 8, sorted(seen)          # every kind of prose field was met, not vacuous

    reworded = _recorded()
    receipt = json.loads(reworded["queue"]["dila-juri-v1-shard0/receipt.json"])
    receipt["twin"]["layout_note"] = "Reworded: a different sentence entirely."
    receipt["twin"]["identity_check"]["reason"] = "another reason"
    receipt["mechanical_content"]["skipped_checks"] = ["coverage: reworded why"]
    reworded["queue"]["dila-juri-v1-shard0/receipt.json"] = json.dumps(receipt)
    reworded["results"][0]["twin"]["vector_key"]["why"] = "reworded"
    assert normalise_prose(reworded) == base

    emptied = _recorded()
    emptied["results"][0]["twin"]["layout_note"] = ""
    with pytest.raises(AssertionError):
        normalise_prose(emptied)

    recount = _recorded()
    receipt = json.loads(recount["queue"]["dila-juri-v1-shard0/receipt.json"])
    receipt["twin"]["payload_carry"]["rows_checked"] += 1
    recount["queue"]["dila-juri-v1-shard0/receipt.json"] = json.dumps(receipt)
    assert normalise_prose(recount) != base

    status = _recorded()
    status["results"][0]["status"] = "REFUSED"
    assert normalise_prose(status) != base


def test_2938_a_row_whose_page_the_twin_serves_under_another_id_is_dropped_from_both_legs(tmp_path, url, monkeypatch):
    """S14: the twin already serves the PAGE of one pack row under a different point id (below the refusal share).
    The row is dropped from this apply, on BOTH legs, and the receipt says so (`already_served_skipped`,
    `mechanical_content_dropped`); the other rows land on twin and prod; the foreign row is untouched."""
    first = run_twin_first(tmp_path / "learn", url, monkeypatch)
    rows = first["twin_points"]
    assert len(rows) == 4
    victim = sorted(rows, key=lambda p: str(p["id"]))[0]
    foreign = {"id": "00000000-0000-5000-8000-0000000000f1",
               "payload": {"url": victim["payload"]["url"], "source_type": "official", "snippet": "served elsewhere"},
               "vector": {"dense": [0.1] * H.MRL, "bm25": {"indices": [1], "values": [1.0]}}}
    for name in (H.TWIN, H.PROD):
        H.reset_collection(url, name)
    H.set_alias(url, H.ALIAS, H.PROD)
    H.api(url, f"/collections/{H.TWIN}/points?wait=true", {"points": [foreign]})
    monkeypatch.setattr(PSP, "ALREADY_SERVED_FAIL_SHARE_MAX", 0.5)       # one page of four is below the refusal share
    queue = H.sealed_pack(tmp_path / "second")
    H.install_fakes(monkeypatch, Q, PSP)
    res = Q.drain(queue, twin_url=url, prod_url=url, task="2938-pin", apply=True, batch=3, pause_s=0,
                  artifact_root=tmp_path / "second" / "artifacts", refresh_index=False, prod_alias=H.ALIAS)
    assert res[0]["status"] == "LANDED", res
    assert res[0]["already_served_skipped"] == 1 and res[0]["mechanical_content_dropped"] == 1
    twin_ids = {str(p["id"]) for p in H.dump_points(url, H.TWIN)}
    prod_ids = {str(p["id"]) for p in H.dump_points(url, H.PROD)}
    want = {str(p["id"]) for p in rows} - {str(victim["id"])}
    assert twin_ids == want | {foreign["id"]} and prod_ids == want

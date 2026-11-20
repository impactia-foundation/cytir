"""The `Single` transaction of the drain (slice 8 step 2, specs/aw-index-serve-gate.md §2).

Same sealed real pack, same code, one target collection and no twin leg. The point of the step is
the first test: the collection a `Single` drain builds holds, point for point, what the twin-first
drain puts on PROD. Needs a throwaway Qdrant (`AW_TEST_QDRANT_URL`); skips without one.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil

import pytest

pytestmark = pytest.mark.usefixtures("no_sparse_rule")   # task 717: these test the drain/build plumbing, not the rule

import drain_transaction_harness as H
from cytir.index import promote_queue as Q                       # noqa: E402  (the alias of cytir.index.promote_queue)
from cytir.index import promote_sealed_pack as PSP  # noqa: E402
from test_drain_transaction import run_twin_first  # noqa: E402

TARGET = "aw_single_target"          # NOT a declared collection: it stores aw_e9's payload form
PACK = "dila-juri-v1-shard0"


@pytest.fixture
def url():
    return H.qdrant_url()


def state(tmp, pack=PACK):
    """The Single transaction's own pack dir: its receipts and DONE live here, never in the queue."""
    return tmp / "artifacts" / "single-state" / TARGET / pack


def single(url, profile_of=H.PROD):
    return Q.Single(url=url, collection=TARGET, payload_profile_of=profile_of)


def drain_single(queue, tmp, url, **kw):
    return Q.drain(queue, task="696-single", apply=True, batch=3, pause_s=0,
                   artifact_root=tmp / "artifacts", refresh_index=False,
                   transaction=kw.pop("transaction", None) or single(url), **kw)


def run_single(tmp: pathlib.Path, url: str, monkeypatch, *, profile_of: str | None = H.PROD):
    H.clear_live(url)                     # a build target's endpoint holds no live name
    H.reset_collection(url, TARGET)
    queue = H.sealed_pack(tmp)
    calls = H.install_fakes(monkeypatch, Q, PSP)
    # the persisted-input check must NOT run in a Single transaction that has no index
    monkeypatch.delenv("AW_PERSISTED_INDEX_SQLITE_DIR", raising=False)
    monkeypatch.setattr(PSP.PI, "assert_pages_persisted", lambda *a, **k: pytest.fail(
        "the persisted-input check ran in a Single transaction with no index"))
    results = drain_single(queue, tmp, url, transaction=single(url, profile_of))
    return queue, results, calls


def test_single_lands_the_same_points_twin_first_puts_on_prod(tmp_path, url, monkeypatch):
    """The equivalence the whole refactor rests on: the Single transaction's collection holds, point
    for point, what the twin-first transaction put on PROD (ids, dense vectors, sparse vectors,
    payload), because it is the same compose -> plan -> project -> upsert code."""
    tf = run_twin_first(tmp_path / "tf", url, monkeypatch)
    queue, results, _calls = run_single(tmp_path / "single", url, monkeypatch)
    assert results[0]["status"] == "LANDED", results
    assert (state(tmp_path / "single") / "DONE").exists() and not (queue / PACK / "DONE").exists()
    got = H.dump_points(url, TARGET)
    assert len(got) == 4
    assert got == tf["prod_points"]
    assert json.dumps(got, sort_keys=True) == json.dumps(tf["prod_points"], sort_keys=True)


def test_single_has_no_twin_leg_no_locks_and_no_ledger(tmp_path, url, monkeypatch):
    queue, results, calls = run_single(tmp_path, url, monkeypatch)
    r = results[0]
    # no TWIN/PROD lock, no assert_locks_free: ONE lock of its own, taken and released per pack
    lock = f"SINGLE-WRITE-LOCK-{TARGET}"
    assert [c[:2] for c in calls] == [("acquire", lock), ("release", lock)], calls
    assert "twin" not in r and "prod" not in r
    assert r["transaction"] == {"kind": "single", "url": url, "collection": TARGET,
                                "payload_profile_of": H.PROD, "persisted_inputs": "skipped"}
    assert r["target"]["persisted_inputs"]["verdict"] == "SKIPPED"       # recorded, never "clean"
    assert r["target"]["written"] == 4 and set(r["payload_profile"]) == {"target"}
    assert r["check5"]["status"] == "PASS" and r["check5"]["manifest_rows"] == 4
    assert sorted(p.name for p in (tmp_path / "artifacts").iterdir()) == [
        f"{PACK}-single-written-ids.json", "single-state"]                # no ledger.jsonl


def test_single_rerun_after_a_kill_before_done_marks_done_without_a_write(tmp_path, url, monkeypatch):
    queue, _results, _ = run_single(tmp_path, url, monkeypatch)
    (state(tmp_path) / "DONE").unlink()
    writes = []
    monkeypatch.setattr(PSP, "upsert", lambda *a, **k: writes.append(a) or 0)
    again = drain_single(queue, tmp_path, url)
    assert again[0]["status"] == "LANDED" and "already held every id" in again[0]["note"]
    assert writes == [] and (state(tmp_path) / "DONE").exists()
    assert len(H.dump_points(url, TARGET)) == 4


def test_single_refuses_a_partly_written_target(tmp_path, url, monkeypatch):
    queue, _results, _ = run_single(tmp_path, url, monkeypatch)
    (state(tmp_path) / "DONE").unlink()
    victim = H.dump_points(url, TARGET)[0]["id"]
    H.api(url, f"/collections/{TARGET}/points/delete?wait=true", {"points": [victim]}, method="POST")
    again = drain_single(queue, tmp_path, url)
    assert again[0]["status"] == "REFUSED" and "partly written" in again[0]["reason"]
    assert not (state(tmp_path) / "DONE").exists()


def test_single_needs_a_declared_payload_form(tmp_path, url, monkeypatch):
    """No `payload_profile_of` and an undeclared target name: the writer refuses, before any write."""
    _queue, results, _ = run_single(tmp_path, url, monkeypatch, profile_of=None)
    assert results[0]["status"] == "REFUSED" and "no declared payload profile" in results[0]["reason"]
    assert H.dump_points(url, TARGET) == []


def test_single_check5_aborts_a_write_that_does_not_match_the_target(tmp_path, url, monkeypatch):
    H.clear_live(url)
    H.reset_collection(url, TARGET)
    queue = H.sealed_pack(tmp_path)
    H.install_fakes(monkeypatch, Q, PSP)
    real = PSP.upsert
    monkeypatch.setattr(PSP, "upsert", lambda u, c, pts, b, **k: real(u, c, pts, b, **k) + 1)
    with pytest.raises(Q.QueueError, match="points_count moved by 4 while this run wrote 5"):
        drain_single(queue, tmp_path, url)
    receipt = json.loads((state(tmp_path) / "receipt.json").read_text())
    assert receipt["status"] == "ABORTED"          # a failure after a write started stops the drain
    assert not (state(tmp_path) / "DONE").exists()


def test_single_refuses_a_content_pack_and_does_not_guess(tmp_path, url, monkeypatch):
    queue, _results, _ = run_single(tmp_path, url, monkeypatch)
    (state(tmp_path) / "DONE").unlink()
    cp = queue / "a-content-pack"
    cp.mkdir()
    (cp / "manifest.json").write_text(json.dumps({
        "schema": "aw.promotion-queue-pack.v1", "pack_id": "a-content-pack", "kind": "content_pack",
        "family": "news", "pack_points": str(cp / "points.jsonl"), "rows": 1, "collection": TARGET}))
    (cp / "points.jsonl").write_text(json.dumps({"id": "00000000-0000-5000-8000-000000000001",
                                                 "payload": {"url": "https://x/"}, "vector": {}}) + "\n")
    (cp / "READY").write_text(hashlib.sha256((cp / "manifest.json").read_bytes()).hexdigest() + "\n")
    res = drain_single(queue, tmp_path, url, only=["a-content-pack"])
    assert res[0]["status"] == "REFUSED" and res[0]["stage"] == "single-transaction"


def test_drain_arguments_cannot_mix_the_two_transactions():
    nowhere = pathlib.Path("/nonexistent")
    common = dict(task="t", apply=False, batch=1, pause_s=0, artifact_root=nowhere)
    with pytest.raises(Q.QueueError, match="no twin or prod url"):
        Q.drain(nowhere, twin_url="http://x", transaction=Q.Single(url="http://x", collection="c"),
                **common)
    with pytest.raises(Q.QueueError, match="disagree"):
        Q.drain(nowhere, twin_url="http://a", prod_url="http://b",
                transaction=Q.TwinFirst("http://c", "http://d"), **common)
    with pytest.raises(Q.QueueError, match="required unless"):
        Q.drain(nowhere, **common)


def test_the_default_is_the_explicit_twin_first_transaction(tmp_path, url, monkeypatch):
    """`transaction=None` and `transaction=TwinFirst(...)` are the same run, byte for byte."""
    a = run_twin_first(tmp_path / "a", url, monkeypatch)
    b = run_twin_first(tmp_path / "b", url, monkeypatch, transaction=Q.TwinFirst(url, url, H.ALIAS))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_the_cli_door_applies_a_single_target(tmp_path, url, monkeypatch, capsys):
    """`promote_queue.main` is the door `cyt index apply` names: `--target-url/--target-collection`
    select the Single transaction; without them the argv a deploy unit passes is unchanged."""
    H.clear_live(url)
    H.reset_collection(url, TARGET)
    queue = H.sealed_pack(tmp_path)
    H.install_fakes(monkeypatch, Q, PSP)
    monkeypatch.delenv("AW_PERSISTED_INDEX_SQLITE_DIR", raising=False)
    rc = Q.main(["--queue-dir", str(queue), "--target-url", url, "--target-collection", TARGET,
                 "--target-payload-profile-of", H.PROD, "--apply", "--task", "696-cli",
                 "--batch", "3", "--artifact-root", str(tmp_path / "artifacts")])
    assert rc == 0 and (state(tmp_path) / "DONE").exists() and not (queue / PACK / "DONE").exists()
    assert len(H.dump_points(url, TARGET)) == 4
    assert "1 landed" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="go together"):
        Q.main(["--queue-dir", str(queue), "--target-url", url])


# ============================ rounds 2 and 3 (reviews of #553: F1, F2, R1, R2, R3; rebuilt on main)
import socket
import threading
import urllib.parse

from cytir.core import paths as PATHS            # noqa: E402
from cytir.index import live_target as LT        # noqa: E402


def drain_twin_first(queue, tmp, url, **kw):
    return Q.drain(queue, twin_url=url, prod_url=url, task="696-char", apply=kw.pop("apply", True),
                   batch=3, pause_s=0, artifact_root=tmp / "artifacts", refresh_index=False,
                   prod_alias=H.ALIAS, **kw)


def fresh_twin_first_world(url):
    for name in (H.TWIN, H.PROD):
        H.reset_collection(url, name)
    H.set_alias(url, H.ALIAS, H.PROD)


@pytest.fixture
def live_queue(tmp_path, monkeypatch):
    """A box that HAS a promotion queue: the `queue_dir` setting names a directory."""
    live = tmp_path / "the-live-queue"
    live.mkdir()
    monkeypatch.setenv("AW_QUEUE_DIR", str(live))
    PATHS.reset_settings_cache()
    yield live
    PATHS.reset_settings_cache()


@pytest.fixture
def no_queue_setting(monkeypatch):
    """A stranger's box: no work root, no queue setting, no overlay. `queue_dir` REFUSES."""
    # also the overlay the private suite opts into (the private build tree's conftest points it at the repo's own
    # impactia.json, which carries `work_root`, hence a derived `queue_dir`; #589)
    for name in ("AW_WORK_ROOT", "AW_QUEUE_DIR", "AW_PROMOTION_QUEUE_DIR", PATHS.OVERLAY_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(PATHS.TEST_LOOKUP_ENV, "")
    PATHS.reset_settings_cache()
    yield
    PATHS.reset_settings_cache()


def assert_twin_first_untouched_by(tmp, url, monkeypatch, before_twin_first):
    """Run `before_twin_first(queue)` (a Single in some form) on a queue, then the twin-first drain
    over the SAME queue dir: it must plan and land the pack exactly as on a queue no Single ever saw
    -- same results, same queue tree (receipts, DONE, READY), same artifacts, same points."""
    clean = run_twin_first(tmp / "clean", url, monkeypatch)
    H.clear_live(url)                          # the Single's endpoint holds no live name ...
    queue = H.sealed_pack(tmp / "dirty")
    calls = H.install_fakes(monkeypatch, Q, PSP)
    before_twin_first(queue)
    fresh_twin_first_world(url)                # ... and the twin-first world is built after it
    del calls[:]
    results = drain_twin_first(queue, tmp / "dirty", url)
    assert results[0]["status"] == "LANDED", results
    got = {
        "results": json.loads(H.normalise(json.dumps(results, sort_keys=True), tmp / "dirty")),
        "queue": H.snapshot_tree(queue, tmp / "dirty"),
        "artifacts": H.snapshot_tree(tmp / "dirty" / "artifacts", tmp / "dirty"),
        "lock_protocol": calls,
        "twin_points": H.dump_points(url, H.TWIN), "prod_points": H.dump_points(url, H.PROD)}
    got = json.loads(json.dumps(got, sort_keys=True))
    want = json.loads(json.dumps(clean, sort_keys=True))
    assert got == want


def test_a_single_refusal_does_not_poison_the_twin_first_drain(tmp_path, url, monkeypatch):
    """The review's F1 demo: a Single run against a wrong-width scratch collection cached a
    REFUSED(plan) that made the normal drain SKIP the pack. It writes nowhere the twin-first drain
    reads (here with --apply, which is the run that writes receipts)."""
    def single_refusal(queue):
        H.reset_collection(url, TARGET, dim=64)          # the wrong width: the plan refuses
        res = Q.drain(queue, task="696-single", apply=True, batch=3, pause_s=0,
                      artifact_root=tmp_path / "single-artifacts", refresh_index=False,
                      transaction=single(url))
        assert res[0]["status"] == "REFUSED" and res[0]["stage"] == "plan", res
        assert sorted(p.name for p in (queue / PACK).iterdir()) == ["READY", "manifest.json"]
    assert_twin_first_untouched_by(tmp_path, url, monkeypatch, single_refusal)


def test_a_single_apply_does_not_hide_the_pack_from_the_twin_first_drain(tmp_path, url, monkeypatch):
    """... and `single --apply` used to write DONE into the queue, so prod never got the pack."""
    def single_apply(queue):
        H.reset_collection(url, TARGET)
        res = Q.drain(queue, task="696-single", apply=True, batch=3, pause_s=0,
                      artifact_root=tmp_path / "single-artifacts", refresh_index=False,
                      transaction=single(url))
        assert res[0]["status"] == "LANDED" and len(H.dump_points(url, TARGET)) == 4
        assert sorted(p.name for p in (queue / PACK).iterdir()) == ["READY", "manifest.json"]
    assert_twin_first_untouched_by(tmp_path, url, monkeypatch, single_apply)


def test_the_refusal_cache_key_carries_the_transaction_and_the_target(tmp_path, url, monkeypatch):
    """The refusal cache cannot cross-poison: a Single's refusal is cached per (transaction,
    target), so another target on the SAME state dir is not skipped, and the twin-first drain
    never reads it. (Round 4: the state dir is NOT the queue; a state dir in the queue is refused,
    see `test_f2_*`.) Receipts are written by `--apply` runs only: a dry run writes none, #588."""
    fresh_twin_first_world(url)
    queue = H.sealed_pack(tmp_path)
    H.install_fakes(monkeypatch, Q, PSP)
    shared_state = tmp_path / "shared-state"       # ONE state dir for both targets, outside the queue
    shared = Q.Single(url=url, collection=TARGET, payload_profile_of=H.PROD, state_dir=shared_state)
    kw = dict(task="696", batch=3, pause_s=0, artifact_root=tmp_path / "artifacts",
              refresh_index=False)
    receipt = queue / PACK / "receipt.json"
    single_receipt = shared_state / PACK / "receipt.json"

    def twin_first():
        return Q.drain(queue, twin_url=url, prod_url=url, apply=True, prod_alias=H.ALIAS, **kw)

    def run_single(tx=shared):
        return Q.drain(queue, apply=True, transaction=tx, **kw)

    # 1. a Single REFUSED(plan) (its target has the wrong width) is cached in ITS state dir, keyed
    #    by transaction and target; the queue's pack dir gets no receipt at all
    H.clear_live(url)
    H.reset_collection(url, TARGET, dim=64)
    res = run_single()
    assert res[0]["status"] == "REFUSED" and res[0]["stage"] == "plan", res
    assert json.loads(single_receipt.read_text())["transaction_key"] == f"single|{url}|{TARGET}"
    assert not receipt.exists()
    # 2. the SAME transaction and target, unchanged manifest: skipped (the cache still works)
    assert run_single()[0]["status"] == "SKIPPED-REFUSED-UNCHANGED"
    # 3. ... and the Single's refusal is not a refusal of the twin-first drain: it plans for itself
    fresh_twin_first_world(url)
    H.reset_collection(url, H.TWIN, dim=64)
    res = twin_first()
    assert res[0]["status"] == "REFUSED" and res[0]["stage"] == "plan", res    # NOT skipped
    assert "transaction_key" not in json.loads(receipt.read_text())
    assert json.loads(single_receipt.read_text())["transaction_key"] == f"single|{url}|{TARGET}"
    # 4. nor of a Single aimed at another target on the SAME state dir: the key carries the target
    H.clear_live(url)
    H.reset_collection(url, "aw_single_other")
    other = Q.Single(url=url, collection="aw_single_other", payload_profile_of=H.PROD,
                     state_dir=shared_state)
    assert run_single(other)[0]["status"] == "LANDED"


# ------------------------------- round 4 F2: the queue that is READ, whatever the settings know
@pytest.fixture
def unknown_live_queue(tmp_path, monkeypatch):
    """The reviewer's half-configured box: `AW_WORK_ROOT` is exported (scratch mode), so the
    settings do NOT know the queue the drain reads: `live_queue_dir()` points elsewhere."""
    monkeypatch.setenv("AW_WORK_ROOT", str(tmp_path / "scratch-root"))
    monkeypatch.delenv("AW_QUEUE_DIR", raising=False)
    monkeypatch.delenv("AW_PROMOTION_QUEUE_DIR", raising=False)
    monkeypatch.setenv(PATHS.TEST_LOOKUP_ENV, "")
    PATHS.reset_settings_cache()
    assert Q.live_queue_dir() != (tmp_path / "queue").resolve()
    yield
    PATHS.reset_settings_cache()


def _tree(root: pathlib.Path) -> dict:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("where", ["queue", "queue-parent", "pack-dir", "inside-pack-dir"])
@pytest.mark.parametrize("apply", [False, True])
def test_f2_a_state_dir_in_the_queue_that_is_read_is_refused_whatever_the_settings(
        tmp_path, unknown_live_queue, monkeypatch, where, apply):
    """The reviewer's exact trigger: `--target-state-dir` = the queue it reads wrote DONE into the
    live pack dir (and replaced an ABORTED receipt) when the settings did not know the queue. Now
    refused from the path alone, with nothing written; the queue tree is byte-identical."""
    queue = H.sealed_pack(tmp_path)
    (queue / PACK / "receipt.json").write_text('{"status": "ABORTED", "note": "twin-first evidence"}')
    before = _tree(queue)
    state = {"queue": queue, "queue-parent": queue.parent, "pack-dir": queue / PACK,
             "inside-pack-dir": queue / PACK / "x"}[where]
    tx = Q.Single(url=NOT_LIVE, collection="aw_build", payload_profile_of=H.PROD, state_dir=state)
    with pytest.raises(Q.QueueError, match="pack dir of the queue it reads"):
        Q.drain(queue, task="t", apply=apply, batch=1, pause_s=0,
                artifact_root=tmp_path / "art", transaction=tx, refresh_index=False)
    assert _tree(queue) == before and not (tmp_path / "art").exists()


@pytest.mark.parametrize("root", ["queue", "queue-parent", "pack-dir"])
def test_f2_an_artifact_root_in_the_queue_that_is_read_is_refused_whatever_the_settings(
        tmp_path, unknown_live_queue, root):
    queue = H.sealed_pack(tmp_path)
    before = _tree(queue)
    art = {"queue": queue, "queue-parent": queue.parent, "pack-dir": queue / PACK}[root]
    tx = Q.Single(url=NOT_LIVE, collection="aw_build", payload_profile_of=H.PROD,
                  state_dir=tmp_path / "state")
    with pytest.raises(Q.QueueError, match="pack dir of the queue it reads"):
        Q.drain(queue, task="t", apply=True, batch=1, pause_s=0, artifact_root=art,
                transaction=tx, refresh_index=False)
    assert _tree(queue) == before and not (tmp_path / "state").exists()


def test_f2_the_default_state_dir_under_the_queues_underscore_artifacts_is_still_allowed(
        tmp_path, unknown_live_queue, monkeypatch):
    """`<queue>/_artifacts` is not a pack dir (the drain skips `_` children), so the CLI's default
    artifact root, and the default state dir under it, stay legal. (The endpoint guard, a different
    check, is stubbed: only the PATH rules are under test here.)"""
    queue = H.sealed_pack(tmp_path)
    monkeypatch.setattr(LT, "refuse_live_target", lambda *a, **k: None)
    tx = Q.Single(url=NOT_LIVE, collection="aw_build", payload_profile_of=H.PROD)
    root = Q.refuse_live_single(tx, queue, queue / "_artifacts")
    assert root == (queue / "_artifacts" / "single-state" / "aw_build").resolve()


def test_f2_a_state_entry_that_resolves_into_the_queue_read_is_refused_whatever_the_settings(
        tmp_path, unknown_live_queue, monkeypatch):
    queue = H.sealed_pack(tmp_path)
    monkeypatch.setattr(LT, "refuse_live_target", lambda *a, **k: None)
    state = tmp_path / "state"
    state.mkdir()
    (state / PACK).symlink_to(queue / PACK, target_is_directory=True)
    before = _tree(queue)
    tx = Q.Single(url=NOT_LIVE, collection="aw_build", payload_profile_of=H.PROD, state_dir=state)
    for apply in (False, True):
        with pytest.raises(Q.QueueError, match="symlink or leads into the live queue or the queue"):
            Q.drain(queue, task="t", apply=apply, batch=1, pause_s=0,
                    artifact_root=tmp_path / "art", transaction=tx, refresh_index=False)
    assert _tree(queue) == before


@pytest.mark.parametrize("held", ["manifest.json", "READY", "DONE", "receipt.json"])
@pytest.mark.parametrize("apply", [False, True])
def test_a_state_entry_that_holds_a_pack_or_a_receipt_it_did_not_write_is_refused(
        tmp_path, unknown_live_queue, monkeypatch, held, apply):
    """Follow-up A of #553's review (the reviewer's X4): `--target-state-dir` = a DIFFERENT queue that
    holds the same pack id and that the settings do not know. The `single` wrote DONE into that
    pack dir, and its receipt went from ABORTED to LANDED. Now the entry (an existing directory with
    the pack's id, no `SINGLE-STATE` marker) is refused whatever file shows it is not ours; the other
    queue is byte-identical afterwards."""
    queue = H.sealed_pack(tmp_path)
    monkeypatch.setattr(LT, "refuse_live_target", lambda *a, **k: None)
    other = tmp_path / "real-live-queue"
    shutil.copytree(queue / PACK, other / PACK)
    for name in ("manifest.json", "READY", "DONE", "receipt.json"):       # leave exactly ONE of them
        if name != held:
            (other / PACK / name).unlink(missing_ok=True)
    (other / PACK / held).write_text('{"status": "ABORTED", "note": "twin-first evidence"}')
    before, read_before = _tree(other), _tree(queue)
    tx = Q.Single(url=NOT_LIVE, collection="aw_build", payload_profile_of=H.PROD, state_dir=other)
    with pytest.raises(Q.QueueError, match="no SINGLE-STATE marker"):
        Q.drain(queue, task="t", apply=apply, batch=1, pause_s=0,
                artifact_root=tmp_path / "art", transaction=tx, refresh_index=False)
    assert _tree(other) == before and _tree(queue) == read_before


def test_a_state_entry_the_single_marked_itself_is_reused_and_an_empty_one_is_claimed(
        tmp_path, url, monkeypatch):
    """The marker is written on first use; a rerun (and `--resume`) reuses its own entry, a dry run
    writes none, and an empty directory with the pack's id is claimed, not refused."""
    H.clear_live(url)
    H.reset_collection(url, TARGET)
    queue = H.sealed_pack(tmp_path)
    H.install_fakes(monkeypatch, Q, PSP)
    monkeypatch.delenv("AW_PERSISTED_INDEX_SQLITE_DIR", raising=False)
    Q.drain(queue, task="t", apply=False, batch=3, pause_s=0, artifact_root=tmp_path / "artifacts",
            refresh_index=False, transaction=single(url))
    assert not (tmp_path / "artifacts").exists()                       # dry: nothing written
    entry = state(tmp_path)
    entry.mkdir(parents=True)                                          # empty entry: claimed
    assert drain_single(queue, tmp_path, url)[0]["status"] == "LANDED"
    assert (entry / Q.SINGLE_STATE_MARKER).is_file() and (entry / "DONE").is_file()
    drain_single(queue, tmp_path, url)                                 # rerun: its own entry, no refusal
    # without the marker the same entry (now holding DONE and a receipt) is somebody else's
    (entry / Q.SINGLE_STATE_MARKER).unlink()
    with pytest.raises(Q.QueueError, match="no SINGLE-STATE marker"):
        drain_single(queue, tmp_path, url)


def test_f2_the_cli_refuses_a_target_state_dir_in_the_queue_it_reads(tmp_path, unknown_live_queue):
    queue = H.sealed_pack(tmp_path)
    before = _tree(queue)
    with pytest.raises(Q.QueueError, match="pack dir of the queue it reads"):
        Q.main(["--queue-dir", str(queue), "--target-url", NOT_LIVE, "--target-collection", "aw_build",
                "--target-payload-profile-of", H.PROD, "--target-state-dir", str(queue), "--apply",
                "--task", "t"])
    assert _tree(queue) == before


def test_f3_an_env_override_of_the_cards_collection_adds_to_the_live_names(monkeypatch):
    """`AW_CARDS_COLLECTION=cards_other` used to REPLACE `source_cards` in the live names."""
    monkeypatch.setenv(LT.CARDS_COLLECTION_ENV, "cards_other")
    names = LT.live_collection_names()
    assert {"source_cards", "cards_other"} <= names
    listing(monkeypatch, collections=["source_cards"])
    with pytest.raises(Q.QueueError, match="LIVE endpoint"):
        Q.assert_single_target_not_live(Q.Single(url=NOT_LIVE, collection="aw_build"))
    monkeypatch.delenv(LT.CARDS_COLLECTION_ENV)
    assert "cards_other" not in LT.live_collection_names()


# ----------------------------------------------------------- the live queue (the settings API)
def test_a_single_refuses_the_live_queue_dir_before_reading_it(tmp_path, live_queue):
    """Equal, inside and above the configured queue; a state dir or artifact root in it."""
    common = dict(task="t", apply=False, batch=1, pause_s=0, artifact_root=tmp_path / "art")
    tx = Q.Single(url=NOT_LIVE, collection="aw_build")
    for qd in (live_queue, live_queue / "some-pack", live_queue.parent):
        with pytest.raises(Q.QueueError, match="live promotion queue"):
            Q.drain(qd, transaction=tx, **common)
    bq = tmp_path / "build-queue"
    bq.mkdir()
    with pytest.raises(Q.QueueError, match="state dir may not sit in the live queue"):
        Q.drain(bq, transaction=Q.Single(url=NOT_LIVE, collection="aw_build",
                                               state_dir=live_queue / "_single"), **common)
    assert list(live_queue.iterdir()) == [] and not (tmp_path / "art").exists()


def test_r2_the_artifact_root_gets_the_same_live_queue_check(tmp_path, live_queue):
    """R2 (review of #553 round 2): `--artifact-root` = the queue's `_artifacts` wrote
    `<pack>-single-written-ids.json` into the live queue tree."""
    tx = Q.Single(url=NOT_LIVE, collection="aw_build", state_dir=tmp_path / "state")
    bq = tmp_path / "build-queue"
    bq.mkdir()
    for root in (live_queue / "_artifacts", live_queue, live_queue / "x" / "y"):
        with pytest.raises(Q.QueueError, match="artifact root may not sit in the live queue"):
            Q.drain(bq, task="t", apply=True, batch=1, pause_s=0, artifact_root=root,
                    transaction=tx)
    assert list(live_queue.iterdir()) == []


def test_r1_a_state_entry_that_is_a_symlink_or_leads_into_the_live_queue(tmp_path, live_queue,
                                                                        monkeypatch):
    """R1: a state dir whose `<pack_id>` entry is a symlink into the live queue received a
    `receipt.json` (and, with --apply, a DONE) in the LIVE pack dir."""
    live_pack = live_queue / PACK
    live_pack.mkdir()
    queue = H.sealed_pack(tmp_path)
    monkeypatch.setattr(LT, "refuse_live_target", lambda *a, **k: None)   # the endpoint is not under test
    state_root = tmp_path / "state"
    state_root.mkdir()
    (state_root / PACK).symlink_to(live_pack, target_is_directory=True)
    tx = Q.Single(url=NOT_LIVE, collection="aw_build", state_dir=state_root)
    for apply in (False, True):
        with pytest.raises(Q.QueueError, match="symlink or leads into the live queue"):
            Q.drain(queue, task="t", apply=apply, batch=1, pause_s=0,
                    artifact_root=tmp_path / "art", transaction=tx, refresh_index=False)
    assert list(live_pack.iterdir()) == [], "a receipt or a DONE reached the live pack dir"
    # a symlink to somewhere harmless is refused too: a state entry is a real directory of its own
    (state_root / PACK).unlink()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (state_root / PACK).symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(Q.QueueError, match="symlink or leads into the live queue"):
        Q.drain(queue, task="t", apply=False, batch=1, pause_s=0, artifact_root=tmp_path / "art",
                transaction=tx, refresh_index=False)


def test_a_stranger_with_no_live_queue_proceeds_with_its_explicit_queue_dir(
        tmp_path, url, monkeypatch, no_queue_setting):
    """The `queue_dir` setting REFUSES on a box that has none: nothing to protect, so the Single
    runs against its explicit --queue-dir; but it never DEFAULTS to a queue."""
    with pytest.raises(PATHS.SettingRefused):
        PATHS.operator_settings().queue_dir
    assert Q.live_queue_dir() is None
    H.clear_live(url)
    H.reset_collection(url, TARGET)
    queue = H.sealed_pack(tmp_path)
    H.install_fakes(monkeypatch, Q, PSP)
    monkeypatch.delenv("AW_PERSISTED_INDEX_SQLITE_DIR", raising=False)
    rc = Q.main(["--queue-dir", str(queue), "--target-url", url, "--target-collection", TARGET,
                 "--target-payload-profile-of", H.PROD, "--apply", "--task", "696-cli",
                 "--batch", "3", "--artifact-root", str(tmp_path / "artifacts")])
    assert rc == 0 and len(H.dump_points(url, TARGET)) == 4
    with pytest.raises(SystemExit, match="needs an explicit --queue-dir"):
        Q.main(["--target-url", url, "--target-collection", TARGET])


def test_the_cli_refuses_a_target_on_the_live_queue_dir(live_queue):
    with pytest.raises(SystemExit, match="needs an explicit --queue-dir"):
        Q.main(["--target-url", NOT_LIVE, "--target-collection", "aw_build"])
    with pytest.raises(SystemExit, match="live promotion queue"):
        Q.main(["--queue-dir", str(live_queue) + "/", "--target-url", NOT_LIVE,
                "--target-collection", "aw_build"])


# ------------------------------------- R3: a live endpoint is what it HOLDS, not how it is spelled
NOT_LIVE = "http://127.0.0.1:9"


def listing(monkeypatch, *, aliases=None, collections=()):
    """Stub the endpoint listing (`GET /aliases`, `GET /collections`) the guard reads."""
    def api(url, path, *a, **k):
        if path == "/aliases":
            return {"aliases": [{"alias_name": a_, "collection_name": c}
                                for a_, c in (aliases or {}).items()]}
        if path == "/collections":
            return {"collections": [{"name": c} for c in collections]}
        raise AssertionError(f"the guard made a call it should not: {path}")
    monkeypatch.setattr(PSP, "_api", api)


def refused(tx):
    with pytest.raises(Q.QueueError):
        Q.assert_single_target_not_live(tx)


@pytest.mark.parametrize("held", [
    dict(collections=[H.TWIN]), dict(collections=[H.PROD]),
    dict(collections=["source_cards"]), dict(collections=["aw_dnews256_swissinfo"]),
    dict(aliases={"aw_prod": "aw_e10_future"}, collections=["aw_e10_future"]),
    dict(aliases={"aw_prod": "x"}, collections=["x"])],
    ids=["twin", "prod", "cards", "news-staging", "post-flip-alias", "alias-only"])
def test_r3_an_endpoint_holding_a_live_name_is_refused_for_every_collection(monkeypatch, held):
    listing(monkeypatch, **held)
    for spelling in ("http://some-spelling-nobody-listed:1234", "http://0.0.0.0:6334",
                     "http://[::ffff:127.0.0.1]:6334", "http://10.1.2.3:6333", "http://[::1]:6457/"):
        refused(Q.Single(url=spelling, collection="aw_innocent_name"))


def test_r3_a_configured_prod_alias_other_than_the_default_is_honoured(monkeypatch):
    listing(monkeypatch, aliases={"serving": "aw_e11"}, collections=["aw_e11"])
    Q.assert_single_target_not_live(Q.Single(url=NOT_LIVE, collection="aw_build"))    # not named
    refused(Q.Single(url=NOT_LIVE, collection="aw_build", prod_alias="serving"))


def test_r3_a_clean_endpoint_passes_and_an_alias_name_never_does(monkeypatch):
    listing(monkeypatch, aliases={"my_alias": "aw_other"}, collections=["aw_other", "scratch"])
    Q.assert_single_target_not_live(Q.Single(url=NOT_LIVE, collection="aw_build"))
    refused(Q.Single(url=NOT_LIVE, collection="my_alias"))


@pytest.mark.parametrize("how", ["raises", "none", "no-result", "wrong-shape"])
def test_r3_it_fails_closed_when_the_listing_cannot_be_read(monkeypatch, how):
    def api(url, path, *a, **k):
        if how == "raises":
            raise OSError("connection refused")
        return {"none": None, "no-result": {}, "wrong-shape": {"aliases": [{"x": 1}]}}[how]
    monkeypatch.setattr(PSP, "_api", api)
    with pytest.raises(Q.QueueError, match="cannot read what"):
        Q.assert_single_target_not_live(Q.Single(url=NOT_LIVE, collection="aw_build"))


class Forwarder:
    """A TCP forwarder on a free port: a tunnel, the case a list of URL spellings can never name."""

    def __init__(self, target_port: int):
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(16)
        self.port = self.srv.getsockname()[1]
        self.target = target_port
        self.stop = False
        threading.Thread(target=self._serve, daemon=True).start()

    def _pipe(self, a, b):
        try:
            while data := a.recv(65536):
                b.sendall(data)
        except OSError:
            pass
        finally:
            for s in (a, b):
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def _serve(self):
        while not self.stop:
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            u = socket.create_connection(("127.0.0.1", self.target))
            threading.Thread(target=self._pipe, args=(c, u), daemon=True).start()
            threading.Thread(target=self._pipe, args=(u, c), daemon=True).start()

    def close(self):
        self.stop = True
        self.srv.close()


def spellings(url: str) -> list[str]:
    port = urllib.parse.urlparse(url).port
    fwd = Forwarder(port)
    # (`[::ffff:127.0.0.1]` is a spelling the suite's offline socket guard does not know as local; it
    # is covered with a stubbed listing above, which is spelling-blind by construction.)
    return [f"http://0.0.0.0:{port}", f"http://localhost:{port}",
            f"http://127.0.0.1:{fwd.port}", url + "/"], fwd


def test_r3_the_review_triggers_on_a_real_endpoint_under_every_spelling(tmp_path, url, monkeypatch):
    """The review's exact triggers, on a real Qdrant: `0.0.0.0:<port>` and `[::ffff:127.0.0.1]`
    reached the live endpoint and `source_cards` took 12,571 points. The throwaway plays the live
    box (it holds the twin, prod, the alias, `source_cards`); a tunnel on another port too. Every
    spelling is refused, for the declared name AND an innocent one, and nothing is written."""
    fresh_twin_first_world(url)
    H.reset_collection(url, "source_cards")
    names, fwd = spellings(url)
    try:
        for spelling in names:
            for collection in ("source_cards", "aw_innocent_name", H.PROD):
                refused(Q.Single(url=spelling, collection=collection))
        queue = H.sealed_pack(tmp_path)
        H.install_fakes(monkeypatch, Q, PSP)
        before = H.dump_points(url, "source_cards")
        for spelling in names:
            with pytest.raises(Q.QueueError, match="LIVE endpoint"):
                Q.drain(queue, task="t", apply=True, batch=3, pause_s=0,
                        artifact_root=tmp_path / "art", refresh_index=False,
                        transaction=Q.Single(url=spelling, collection="source_cards",
                                             payload_profile_of=H.PROD))
        assert H.dump_points(url, "source_cards") == before == []
        assert not (tmp_path / "art").exists()
        # the same endpoint, stripped of every live name, is a stranger's: accepted
        H.clear_live(url)
        for spelling in names:
            Q.assert_single_target_not_live(Q.Single(url=spelling, collection="aw_innocent_name"))
    finally:
        fwd.close()


def test_r3_the_live_names_are_derived_from_the_declarations():
    names = LT.live_collection_names()
    assert {"aw_e8", "aw_e9", "source_cards"} <= names
    # task 847: a `staged` declaration (aw_e10 before its flip) is the build's target, not live
    declared = LT.PP.load_profiles()
    # task 2254: a non-serving `twin-fixture` (the CI job's throwaway collection) is not live either
    fixtures = {n for n, e in declared.items() if str(e.get("role", "")).startswith("twin-fixture")
                and e.get("serving") is not True}
    assert set(declared) - LT.PP.staged_collections(declared) - fixtures <= names
    assert not (fixtures & names)
    assert not (LT.PP.staged_collections(declared) & names)
    assert {p.name[:-len(".config.v1.json")]
            for p in LT.SERVING_PROFILE_FIXTURES.glob("*.config.v1.json")} <= names


def test_r3_the_cards_default_is_the_servers_default():
    import re
    server = LT.ROOT / "cytir" / "serve" / "settings.py"    # S10 step 2: the typed settings
    if not server.is_file():
        pytest.skip("the server package is not in this tree")
    found = re.search(r'AW_CARDS_COLLECTION",\s*STR,\s*"([^"]+)"', server.read_text())
    assert found and found.group(1) == LT.CARDS_COLLECTION_DEFAULT


def test_a_single_refusal_at_plan_time_writes_nothing(tmp_path, url, monkeypatch):
    """The prod alias names the target: the endpoint holds the alias, refused at drain start."""
    H.clear_live(url)
    H.reset_collection(url, TARGET)
    H.set_alias(url, H.ALIAS, TARGET)
    queue = H.sealed_pack(tmp_path)
    H.install_fakes(monkeypatch, Q, PSP)
    try:
        with pytest.raises(Q.QueueError, match="LIVE endpoint"):
            drain_single(queue, tmp_path, url)
        assert H.dump_points(url, TARGET) == []
        assert not (tmp_path / "artifacts").exists()
    finally:
        H.drop_alias(url, H.ALIAS)


def test_a_single_rechecks_the_endpoint_per_pack_under_its_lock(tmp_path, url, monkeypatch):
    """The prod alias lands on the endpoint AFTER plan time, while the lock is being taken: the pack
    is refused before the write, the lock is released, nothing is written."""
    H.clear_live(url)
    H.reset_collection(url, TARGET)
    queue = H.sealed_pack(tmp_path)
    calls = H.install_fakes(monkeypatch, Q, PSP)
    real_load = Q._load_run_batch()

    class MovingRB(real_load):
        @staticmethod
        def acquire(name, task, purpose, *, pid=None):
            real_load.acquire(name, task, purpose, pid=pid)
            H.set_alias(url, H.ALIAS, TARGET)           # the flip lands between plan and write
    monkeypatch.setattr(Q, "_load_run_batch", lambda: MovingRB)
    try:
        res = drain_single(queue, tmp_path, url)
        assert res[0]["status"] == "REFUSED" and res[0]["stage"] == "live-apply-preflight", res
        assert "LIVE endpoint" in res[0]["reason"]
        assert H.dump_points(url, TARGET) == []
        assert [c[0] for c in calls] == ["acquire", "release"]
        assert not (state(tmp_path) / "DONE").exists()
    finally:
        H.drop_alias(url, H.ALIAS)


def test_a_held_lock_refuses_before_any_write(tmp_path, url, monkeypatch):
    H.clear_live(url)
    H.reset_collection(url, TARGET)
    queue = H.sealed_pack(tmp_path)
    calls = H.install_fakes(monkeypatch, Q, PSP)
    real_load = Q._load_run_batch()

    class HeldRB(real_load):
        @staticmethod
        def acquire(name, task, purpose, *, pid=None):
            calls.append(("acquire-refused", name))
            raise RuntimeError(f"could not acquire {name}: held by another build")
    monkeypatch.setattr(Q, "_load_run_batch", lambda: HeldRB)
    res = drain_single(queue, tmp_path, url)
    assert res[0]["status"] == "REFUSED" and "held by another build" in res[0]["reason"]
    assert calls == [("acquire-refused", f"SINGLE-WRITE-LOCK-{TARGET}")]
    assert H.dump_points(url, TARGET) == []


def test_2938_a_superseded_pack_is_not_a_state_entry_the_single_checks(tmp_path, unknown_live_queue, monkeypatch):
    """Q4: a SUPERSEDED pack is left out of the Single's pack list BEFORE the state-entry refusal runs (it will not
    be built, so its state entry is not touched). A foreign, non-empty entry under that pack's id therefore does
    not refuse the drain: the run returns nothing and the entry is byte-identical afterwards. The same entry under
    a pack that is NOT superseded refuses."""
    queue = H.sealed_pack(tmp_path)
    monkeypatch.setattr(LT, "refuse_live_target", lambda *a, **k: None)
    other = tmp_path / "foreign-state"
    (other / PACK).mkdir(parents=True)
    (other / PACK / "receipt.json").write_text('{"status": "ABORTED", "note": "another run"}')
    before = _tree(other)
    tx = Q.Single(url=NOT_LIVE, collection="aw_build", payload_profile_of=H.PROD, state_dir=other)

    def run():
        return Q.drain(queue, task="t", apply=True, batch=1, pause_s=0, artifact_root=tmp_path / "art",
                       transaction=tx, refresh_index=False)
    with pytest.raises(Q.QueueError, match="no SINGLE-STATE marker"):
        run()
    (queue / PACK / "SUPERSEDED").write_text("proven wrong\n")
    assert run() == [] and _tree(other) == before
    assert not (queue / PACK / "receipt.json").exists() and not (queue / PACK / "DONE").exists()

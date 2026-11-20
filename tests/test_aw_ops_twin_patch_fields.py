"""`cyt ops twin-patch-fields` (task 845, ruling 29 T17): a closed-key payload patch on TWIN rows by id.

THE ROWS are the six real rows of `fixtures/mirror_guards/rows.v1.json` (task 833, read by id from the
twin `aw_e8` and its 09-22 copy). The three `prod_only_old` rows are L1 re-fetch rows rewritten in
place: on the twin they lost `license` (2 of 3), `stamps` (3 of 3) and moved `rule_epochs.chunk` 5 -> 1
with the chunk text unchanged (the digest is equal on both sides). The copy row holds the old values.
The patch maps are built from these two sides exactly as the T17 maps are: `set` = the copy value,
`expect` = the twin's value now + the digest. The twin is a fake store that keeps payloads as a
Qdrant `set_payload` / `delete_payload` would.

What each group pins:
  plan        dry by default: classify only, no write, a receipt; terminals patch/already/absent/
              precondition_failed; exit 0 / 3 / 4
  apply       writes only the patched keys; the export is the prior state and is itself a valid input
              (rollback round-trips byte for byte); a second run writes nothing; readback refuses a row
              not at its target; the written ids are a receipt `rebuild_e10.load_receipts` (deleted, task 1312) accounts
  contract    closed key list; prod target refused before any call; an export is never overwritten;
              a content pack waiting for its prod leg (or CARRIED-BY-FLIP) refuses before any lock
"""
from __future__ import annotations

import contextlib
import copy
import datetime as dt
import json
import pathlib
import types

import pytest

from cytir import cli
from cytir.index import prod_guard as PG
from cytir.ops import twin_patch_fields as TP

HERE = pathlib.Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "mirror_guards" / "rows.v1.json"
TWIN = "http://twin.invalid:16457"
COLL = "aw_e8"
NOW = dt.datetime(2026, 10, 10, 6, 0, tzinfo=dt.timezone.utc)


@pytest.fixture(scope="module")
def real():
    return json.loads(FIXTURE.read_text())


class Store:
    """A twin that answers `fetch_fn` and applies `write_fn` like Qdrant's payload endpoints."""

    def __init__(self, payloads: dict[str, dict]):
        self.rows = copy.deepcopy(payloads)
        self.writes: list[tuple] = []
        self.reads = 0

    def fetch(self, url, coll, ids, keys, batch):
        assert url == TWIN and coll == COLL and batch <= TP.BATCH
        self.reads += 1
        return {i: {k: self.rows[i][k] for k in keys if k in self.rows[i]} for i in ids if i in self.rows}

    def write(self, url, coll, op, what, ids):
        assert len(ids) <= TP.BATCH
        self.writes.append((op, copy.deepcopy(what), list(ids)))
        for i in ids:
            if op == "set":
                self.rows[i].update(copy.deepcopy(what))
            else:
                for k in what:
                    self.rows[i].pop(k, None)


class Locks:
    def __init__(self):
        self.taken = []

    @contextlib.contextmanager
    def __call__(self, task):
        self.taken.append(task)
        yield f"/locks/TWIN-WRITE-LOCK task={task}"


def twin_of(real):
    return copy.deepcopy({e["id"]: e["twin"]["payload"] for e in real["prod_only_old"] + real["equal"]})


def licence_rows(real):
    """The Q1 map rule: the copy holds a licence, the twin row lost it."""
    out = []
    for e in real["prod_only_old"]:
        tw, cp = e["twin"]["payload"], e["copy"]["payload"]
        if cp.get("license") and not tw.get("license"):
            out.append({"id": e["id"], "set": {"license": cp["license"]},
                        "expect": {"content_digest": tw["content_digest"], "license": tw.get("license"),
                                   "provenance": tw["provenance"]}})
    return out


def stamp_rows(real):
    """The stamp map rule: same digest on both sides, the copy's chunk epoch differs."""
    out = []
    for e in real["prod_only_old"]:
        tw, cp = e["twin"]["payload"], e["copy"]["payload"]
        assert tw["content_digest"] == cp["content_digest"]
        out.append({"id": e["id"], "set": {"rule_epochs": cp["rule_epochs"], "stamps": cp["stamps"]},
                    "expect": {"content_digest": tw["content_digest"], "rule_epochs": tw["rule_epochs"],
                               "stamps": tw.get("stamps")}})
    return out


def write_rows(path: pathlib.Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


@pytest.fixture
def env(tmp_path):
    queue = tmp_path / "queue"
    (queue / "_artifacts").mkdir(parents=True)
    settings = types.SimpleNamespace(work_root=str(tmp_path / "work"), queue_dir=str(queue))
    return types.SimpleNamespace(tmp=tmp_path, queue=queue, settings=settings)


def run(env, store, argv, locks=None, out=None):
    lines = [] if out is None else out
    code = TP.run(["--twin-url", TWIN, "--twin-collection", COLL, *argv], settings=env.settings,
                  fetch_fn=store.fetch, write_fn=store.write, locks=locks or Locks(),
                  now=lambda: NOW, out=lines.append)
    return code, lines


def receipt(env, mode):
    files = sorted((pathlib.Path(env.settings.work_root) / "twin-patch-fields").glob(f"*-{mode}.json"))
    return json.loads(files[-1].read_text())


# ------------------------------------------------------------------------------------ the plan
def test_dry_run_classifies_writes_nothing_and_keeps_a_receipt(real, env):
    rows = licence_rows(real)
    assert len(rows) == 2, "two of the three real rows lost a licence the copy holds"
    store, locks = Store(twin_of(real)), Locks()
    code, lines = run(env, store, ["--from", str(write_rows(env.tmp / "q1.jsonl", rows)),
                                   "--fields", "license"], locks=locks)
    assert code == TP.EXIT_OK
    assert store.writes == [] and locks.taken == []
    r = receipt(env, "dry")
    assert r["counts"] == {"patch": 2, "already": 0, "absent": 0, "precondition_failed": 0}
    assert r["fields"] == ["license"] and r["rows"] == 2 and len(r["input_sha256"]) == 64
    assert "dry" in lines[0] and "patch=2" in lines[0]


def test_absent_rows_are_counted_and_do_not_block(real, env):
    rows = licence_rows(real) + [{"id": "00000000-0000-5000-8000-000000000000", "set": {"license": "x"},
                                  "expect": {"content_digest": "ab" * 32}}]
    store = Store(twin_of(real))
    code, _ = run(env, store, ["--from", str(write_rows(env.tmp / "q.jsonl", rows)), "--fields", "license",
                               "--apply", "--export", str(env.tmp / "export.jsonl")])
    assert code == TP.EXIT_OK
    assert receipt(env, "apply")["counts"]["absent"] == 1
    assert sorted(i for _, _, ids in store.writes for i in ids) == sorted(r["id"] for r in rows[:2])


def test_a_moved_row_is_a_precondition_failure_and_nothing_is_written(real, env):
    """A row whose text changed since the map was built (a later pack, the Q2 re-run) is never patched."""
    rows = licence_rows(real)
    twin = twin_of(real)
    twin[rows[0]["id"]]["content_digest"] = "ff" * 32
    store, locks = Store(twin), Locks()
    code, _ = run(env, store, ["--from", str(write_rows(env.tmp / "q1.jsonl", rows)), "--fields", "license",
                               "--apply", "--export", str(env.tmp / "export.jsonl")], locks=locks)
    assert code == TP.EXIT_PRECONDITION
    assert store.writes == [] and not (env.tmp / "export.jsonl").exists()
    r = receipt(env, "apply")
    assert r["counts"]["precondition_failed"] == 1 and r["moved_keys"] == {"content_digest": 1}
    assert r["precondition_examples"][0]["found"] == {"content_digest": "ff" * 32}


# ----------------------------------------------------------------------------------- the apply
def test_q1_apply_carries_the_copy_licence_and_only_that_key(real, env):
    rows = licence_rows(real)
    store, locks = Store(twin_of(real)), Locks()
    before = copy.deepcopy(store.rows)
    code, _ = run(env, store, ["--from", str(write_rows(env.tmp / "q1.jsonl", rows)), "--fields", "license",
                               "--apply", "--export", str(env.tmp / "export.jsonl")], locks=locks)
    assert code == TP.EXIT_OK and locks.taken == ["845"]
    by_id = {e["id"]: e for e in real["prod_only_old"]}
    for r in rows:
        after, was = store.rows[r["id"]], before[r["id"]]
        assert after["license"] == by_id[r["id"]]["copy"]["payload"]["license"]
        assert {k: v for k, v in after.items() if k != "license"} == {k: v for k, v in was.items() if k != "license"}
    assert all(op == "set" and set(what) == {"license"} for op, what, _ in store.writes)
    r = receipt(env, "apply")
    assert r["verified"] == {"read_back": 2, "not_at_target": 0, "examples": []}
    assert r["export"]["rows"] == 2


def test_the_export_rolls_back_exactly_and_a_second_run_writes_nothing(real, env):
    rows = licence_rows(real)
    store = Store(twin_of(real))
    original = copy.deepcopy(store.rows)
    q1 = write_rows(env.tmp / "q1.jsonl", rows)
    assert run(env, store, ["--from", str(q1), "--fields", "license", "--apply",
                            "--export", str(env.tmp / "e1.jsonl")])[0] == TP.EXIT_OK
    patched = copy.deepcopy(store.rows)
    n = len(store.writes)
    assert run(env, store, ["--from", str(q1), "--fields", "license", "--apply",
                            "--export", str(env.tmp / "e2.jsonl")])[0] == TP.EXIT_OK
    assert len(store.writes) == n, "already at target: idempotent"
    assert receipt(env, "apply")["counts"]["already"] == 2
    export = [json.loads(line) for line in (env.tmp / "e1.jsonl").read_text().splitlines()]
    assert all(e["unset"] == ["license"] and e["set"] == {} for e in export)
    assert run(env, store, ["--from", str(env.tmp / "e1.jsonl"), "--fields", "license", "--apply",
                            "--export", str(env.tmp / "e3.jsonl")])[0] == TP.EXIT_OK
    assert store.rows == original
    assert patched != original


def test_stamp_patch_restores_epochs_and_stamps_on_byte_identical_rows(real, env):
    rows = stamp_rows(real)
    store = Store(twin_of(real))
    code, _ = run(env, store, ["--from", str(write_rows(env.tmp / "q3.jsonl", rows)),
                               "--fields", "rule_epochs,stamps", "--apply", "--export", str(env.tmp / "e.jsonl")])
    assert code == TP.EXIT_OK
    for e in real["prod_only_old"]:
        got, cp = store.rows[e["id"]], e["copy"]["payload"]
        assert got["rule_epochs"] == cp["rule_epochs"] and got["rule_epochs"]["chunk"] == 5
        assert got["stamps"] == cp["stamps"]
        assert got["content_digest"] == cp["content_digest"]


def test_a_row_not_at_its_target_after_the_write_refuses(real, env):
    rows = licence_rows(real)
    store = Store(twin_of(real))
    store.write = lambda *a, **k: None          # a write that did not land
    code, lines = run(env, store, ["--from", str(write_rows(env.tmp / "q1.jsonl", rows)), "--fields", "license",
                                   "--apply", "--export", str(env.tmp / "e.jsonl")])
    assert code == TP.EXIT_REFUSED
    r = receipt(env, "apply")
    assert r["verified"]["not_at_target"] == 2 and "roll back" in r["refused"]




# -------------------------------------------------------------------------------- the contract
@pytest.mark.parametrize("fields", ["url", "license,title", "search_ok"])
def test_only_the_closed_key_list_is_patchable(real, env, fields):
    store = Store(twin_of(real))
    code, _ = run(env, store, ["--from", str(write_rows(env.tmp / "q1.jsonl", licence_rows(real))),
                               "--fields", fields])
    assert code == TP.EXIT_REFUSED and store.reads == 0
    assert "not patchable" in receipt(env, "dry")["refused"]


def test_a_row_touching_a_key_outside_fields_refuses_the_file(real, env):
    rows = stamp_rows(real)
    store = Store(twin_of(real))
    code, _ = run(env, store, ["--from", str(write_rows(env.tmp / "q3.jsonl", rows)), "--fields", "rule_epochs"])
    assert code == TP.EXIT_REFUSED and store.reads == 0
    assert "outside --fields" in receipt(env, "dry")["refused"]


def test_a_row_without_its_digest_precondition_refuses_the_file(real, env):
    rows = licence_rows(real)
    del rows[0]["expect"]["content_digest"]
    store = Store(twin_of(real))
    code, _ = run(env, store, ["--from", str(write_rows(env.tmp / "q1.jsonl", rows)), "--fields", "license"])
    assert code == TP.EXIT_REFUSED and "content_digest" in receipt(env, "dry")["refused"]


def test_a_prod_target_is_refused_before_any_call(real, env):
    store = Store(twin_of(real))
    code = TP.run(["--twin-url", "http://127.0.0.1:%d" % PG.PROD_PORT, "--twin-collection", COLL,
                   "--from", str(write_rows(env.tmp / "q1.jsonl", licence_rows(real))), "--fields", "license",
                   "--apply", "--export", str(env.tmp / "e.jsonl")],
                  settings=env.settings, fetch_fn=store.fetch, write_fn=store.write, locks=Locks(),
                  now=lambda: NOW, out=lambda *_: None)
    assert code == TP.EXIT_REFUSED and store.reads == 0 and store.writes == []
    assert "prod target" in receipt(env, "apply")["refused"]


def test_apply_needs_a_fresh_export(real, env):
    store = Store(twin_of(real))
    q1 = write_rows(env.tmp / "q1.jsonl", licence_rows(real))
    assert run(env, store, ["--from", str(q1), "--fields", "license", "--apply"])[0] == TP.EXIT_REFUSED
    (env.tmp / "old.jsonl").write_text("keep me\n")
    assert run(env, store, ["--from", str(q1), "--fields", "license", "--apply",
                            "--export", str(env.tmp / "old.jsonl")])[0] == TP.EXIT_REFUSED
    assert (env.tmp / "old.jsonl").read_text() == "keep me\n" and store.writes == []


@pytest.mark.parametrize("state", ["twin-write-complete", "carried-by-flip"])
def test_a_pack_waiting_for_its_prod_leg_refuses_before_any_lock(real, env, state):
    """Review 836 S1: a twin-only write beside a pending prod leg stops the drain."""
    pack = env.queue / "news-daily-20261009"
    pack.mkdir()
    if state == "carried-by-flip":
        (pack / "CARRIED-BY-FLIP").write_text("2026-10-09T00:00:00Z\n")
    else:
        (env.queue / "_artifacts" / f"{pack.name}-content-pack-twin-state.json").write_text(
            json.dumps({"stage": state, "pack_id": pack.name}))
    store, locks = Store(twin_of(real)), Locks()
    code, _ = run(env, store, ["--from", str(write_rows(env.tmp / "q1.jsonl", licence_rows(real))),
                               "--fields", "license", "--apply", "--export", str(env.tmp / "e.jsonl")], locks=locks)
    assert code == TP.EXIT_REFUSED and locks.taken == [] and store.writes == [] and store.reads == 0
    assert "prod leg" in receipt(env, "apply")["refused"]
    (pack / "DONE").write_text("ok\n")
    if state == "carried-by-flip":
        (pack / "CARRIED-BY-FLIP").unlink()
    assert run(env, store, ["--from", str(write_rows(env.tmp / "q1.jsonl", licence_rows(real))),
                            "--fields", "license", "--apply", "--export", str(env.tmp / "e.jsonl")],
               locks=locks)[0] == TP.EXIT_OK


def test_the_verb_is_reachable_from_aw_ops(monkeypatch):
    seen = {}
    monkeypatch.setattr(TP, "main", lambda argv, prog: seen.setdefault("call", (argv, prog)) and 0)
    assert cli._ops_main(["twin-patch-fields", "--from", "x", "--fields", "license"]) == 0
    assert seen["call"] == (["--from", "x", "--fields", "license"], "cyt ops twin-patch-fields")


# ---------------------------------------------------------------------------- the real locks
@pytest.fixture
def unit_lock(tmp_path, monkeypatch):
    from cytir.core import paths as PATHS
    lock = tmp_path / "aw-promotion-nightly.lock"
    monkeypatch.setenv("AW_PROMOTION_LOCK", str(lock))
    monkeypatch.setenv("AW_WORK_ROOT", str(tmp_path / "work"))
    PATHS.reset_settings_cache()
    yield lock
    PATHS.reset_settings_cache()


def test_apply_holds_the_drains_flock_and_the_twin_write_lock(unit_lock):
    import subprocess
    from cytir.ops import write_lock as WL
    with TP.apply_locks("845") as held:
        assert held.endswith("TWIN-WRITE-LOCK")
        assert "task=845" in WL.read_lock("TWIN-WRITE-LOCK")
        assert subprocess.run(["flock", "-n", str(unit_lock), "true"]).returncode == 1, \
            "the hourly drain's `flock -n` must fail while the patch runs"
    assert WL.read_lock("TWIN-WRITE-LOCK") is None
    assert subprocess.run(["flock", "-n", str(unit_lock), "true"]).returncode == 0


def test_no_promotion_lock_setting_is_a_refusal_not_a_free_pass(real, env, monkeypatch):
    """`hold_unit_lock` is inert on a box with no `promotion_lock` setting (the build box's own
    file does not set it yet): for a twin write that would mean no exclusion from the drain."""
    from cytir.core import paths as PATHS
    from cytir.ops import write_lock as WL
    monkeypatch.delenv("AW_PROMOTION_LOCK", raising=False)
    PATHS.reset_settings_cache()
    try:
        store = Store(twin_of(real))
        code, _ = run(env, store, ["--from", str(write_rows(env.tmp / "q1.jsonl", licence_rows(real))),
                                   "--fields", "license", "--apply", "--export", str(env.tmp / "e.jsonl")],
                      locks=TP.apply_locks)
        assert code == TP.EXIT_REFUSED and store.writes == [] and store.reads == 0
        assert "CYTIR_PROMOTION_LOCK" in receipt(env, "apply")["refused"]
        assert WL.read_lock("TWIN-WRITE-LOCK") is None
    finally:
        PATHS.reset_settings_cache()


def test_a_twin_write_lock_held_by_another_task_refuses_and_writes_nothing(real, env, unit_lock):
    from cytir.ops import write_lock as WL
    WL.acquire("TWIN-WRITE-LOCK", "999", "another lane")
    try:
        store = Store(twin_of(real))
        code, _ = run(env, store, ["--from", str(write_rows(env.tmp / "q1.jsonl", licence_rows(real))),
                                   "--fields", "license", "--apply", "--export", str(env.tmp / "e.jsonl")],
                      locks=TP.apply_locks)
        assert code == TP.EXIT_REFUSED and store.writes == [] and store.reads == 0
        assert "task=999" in receipt(env, "apply")["refused"]
        assert "task=999" in WL.read_lock("TWIN-WRITE-LOCK"), "never clears a lock it did not take"
    finally:
        WL.release("TWIN-WRITE-LOCK", "999")

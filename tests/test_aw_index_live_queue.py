"""`cyt index promote` and the Single transaction refuse the LIVE queue whatever the environment holds.

Round 6 of #559 (reviewer's B4 trigger): with `AW_WORK_ROOT=<scratch>` or `AW_QUEUE_DIR` exported, the
`queue_dir` setting moved away from the live queue, so the live queue given explicitly was not
refused (rc 0). "Live" is now the union of the setting, the BOX file's own `queue_dir` (read with no
override) and a `LIVE-QUEUE` marker in the queue directory. Also here: a hand run takes the unit's
lock when the box has one, and a state entry is claimed only when it is absent or empty.
No Qdrant: every drain here is refused before it reads or writes anything.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

from cytir import cli
from cytir.core import paths as PATHS
from cytir.index import promote_queue as Q

NOT_LIVE = "http://127.0.0.1:9"      # a Single built on it is refused before it is ever asked


def _box_file(tmp_path: pathlib.Path, name="box-operator.json", **settings) -> pathlib.Path:
    f = tmp_path / name
    f.write_text(json.dumps({"schema": PATHS.OVERLAY_SCHEMA, "settings": settings}))
    return f


IMPACTIA_BOX_FILE = pathlib.Path(__file__).resolve().parents[1] / "deploy" / "operator" / "impactia.json"


def _impactia_shaped_box_file(tmp_path: pathlib.Path, work_root: pathlib.Path) -> pathlib.Path:
    """The real `deploy/operator/impactia.json`, loaded, with every path under its `work_root`
    (and the other absolute paths) rewritten into `tmp_path`. It names `work_root` and NO
    `queue_dir`: the queue is derived."""
    if IMPACTIA_BOX_FILE.is_file():
        doc = json.loads(IMPACTIA_BOX_FILE.read_text())
    else:   # the public cut does not carry the operator file: the same shape, inline
        doc = {"schema": PATHS.OVERLAY_SCHEMA,
               "settings": {"work_root": "/srv/aw/work", "hdd_root": "/srv/aw/hdd",
                            "releases_root": "/srv/aw/releases"}}
    assert "queue_dir" not in doc["settings"] and "work_root" in doc["settings"]
    old = doc["settings"]["work_root"]
    for k, v in doc["settings"].items():
        if isinstance(v, str) and v.startswith(old):
            doc["settings"][k] = str(work_root) + v[len(old):]
        elif isinstance(v, str) and v.startswith("/") and k not in ("box_rsh",):
            doc["settings"][k] = str(tmp_path / "other" / k)
    f = tmp_path / "box-operator.json"
    f.write_text(json.dumps(doc))
    return f


@pytest.fixture
def box(tmp_path, monkeypatch):
    """A box with a live queue, shaped like impactIA's: the BOX file (the test lookup stands in for
    /etc/aw/operator.json) is `deploy/operator/impactia.json` with its paths moved into tmp. It names
    `work_root` only; the live queue is `<work_root>/promotion-queue`."""
    work = tmp_path / "box-work"
    live = work / "promotion-queue"
    (live / "pack-a").mkdir(parents=True)
    monkeypatch.setenv(PATHS.TEST_LOOKUP_ENV, str(_impactia_shaped_box_file(tmp_path, work)))
    for name in ("AW_WORK_ROOT", "AW_QUEUE_DIR", "AW_PROMOTION_QUEUE_DIR", PATHS.OVERLAY_ENV,
                 "AW_PROMOTION_LOCK"):
        monkeypatch.delenv(name, raising=False)
    PATHS.reset_settings_cache()
    yield live
    PATHS.reset_settings_cache()


@pytest.fixture
def drain_never_starts(monkeypatch):
    monkeypatch.setattr(Q, "main", lambda argv: pytest.fail("the drain was started"))


def _promote(*args) -> int:
    return cli.main(["index", "promote", *args])


def _scratch(monkeypatch, tmp_path, how):
    """The environments that moved `queue_dir` away from the live queue."""
    if how == "work-root":
        monkeypatch.setenv("AW_WORK_ROOT", str(tmp_path / "scratch-root"))
    elif how == "queue-dir":
        monkeypatch.setenv("AW_QUEUE_DIR", str(tmp_path / "scratch-queue"))
    elif how == "other-overlay":
        monkeypatch.setenv(PATHS.OVERLAY_ENV, str(_box_file(
            tmp_path, "other-overlay.json", queue_dir=str(tmp_path / "overlay-queue"))))
    PATHS.reset_settings_cache()


# ------------------------------------------------ the reviewer's trigger and its variants
@pytest.mark.parametrize("how", ["none", "work-root", "queue-dir", "other-overlay"])
@pytest.mark.parametrize("apply", [False, True])
def test_the_live_queue_is_refused_whatever_the_environment_holds(
        tmp_path, monkeypatch, box, drain_never_starts, capsys, how, apply):
    """`--queue-dir <live>` with `AW_WORK_ROOT=<scratch>` returned rc 0 and started the drain."""
    _scratch(monkeypatch, tmp_path, how)
    if how != "none":
        assert Q.live_queue_dir() != box.resolve(), "the environment did move the setting"
    for queue in (str(box), str(box) + "/", str(box / "pack-a"), str(box.parent),
                  str(box.parent / "x" / ".." / box.name)):
        argv = ["--queue-dir", queue] + (["--apply"] if apply else [])
        assert _promote(*argv) == 2, (how, argv)
        err = capsys.readouterr().err
        assert "systemctl start aw-promotion-nightly.service" in err, err
        if queue.rstrip("/") == str(box):          # the live queue itself: who said it is live
            assert ("`queue_dir` setting" if how == "none" else "box file's `queue_dir`") in err, err
    link = tmp_path / "link-to-live"
    link.symlink_to(box, target_is_directory=True)
    assert _promote("--queue-dir", str(link)) == 2


def test_an_unreadable_box_file_refuses_fail_closed(tmp_path, monkeypatch, box, drain_never_starts,
                                                     capsys):
    pathlib.Path(os.environ[PATHS.TEST_LOOKUP_ENV]).write_text("{not json")
    monkeypatch.setenv("AW_WORK_ROOT", str(tmp_path / "scratch-root"))
    PATHS.reset_settings_cache()
    scratch = tmp_path / "scratch-queue"
    scratch.mkdir()
    reason = Q.refuse_live_queue_run(["--queue-dir", str(scratch)])
    assert reason and "box file cannot be read" in reason


@pytest.mark.parametrize("apply", [False, True])
def test_a_queue_that_is_not_live_still_runs(tmp_path, monkeypatch, box, apply):
    monkeypatch.setenv("AW_WORK_ROOT", str(tmp_path / "scratch-root"))
    PATHS.reset_settings_cache()
    other = tmp_path / "scratch-queue"
    other.mkdir()
    ran = []
    monkeypatch.setattr(Q, "main", lambda argv: ran.append(argv) or 0)
    assert _promote("--queue-dir", str(other), *(["--apply"] if apply else [])) == 0 and ran
    assert _promote("--help") == 0


def test_a_stranger_with_no_box_file_and_no_marker_is_not_refused(tmp_path, monkeypatch):
    monkeypatch.setenv(PATHS.TEST_LOOKUP_ENV, "")
    for name in ("AW_WORK_ROOT", "AW_QUEUE_DIR", "AW_PROMOTION_QUEUE_DIR", PATHS.OVERLAY_ENV):
        monkeypatch.delenv(name, raising=False)
    PATHS.reset_settings_cache()
    try:
        monkeypatch.setattr(Q, "main", lambda argv: 0)
        assert _promote("--queue-dir", str(tmp_path)) == 0
    finally:
        PATHS.reset_settings_cache()


# ------------------------------------------------------------------------------ the marker
@pytest.fixture
def stranger(tmp_path, monkeypatch):
    """No box file, no settings: only the marker can say a queue is live."""
    monkeypatch.setenv(PATHS.TEST_LOOKUP_ENV, "")
    for name in ("AW_WORK_ROOT", "AW_QUEUE_DIR", "AW_PROMOTION_QUEUE_DIR", PATHS.OVERLAY_ENV):
        monkeypatch.delenv(name, raising=False)
    PATHS.reset_settings_cache()
    yield tmp_path
    PATHS.reset_settings_cache()


@pytest.mark.parametrize("where", ["queue", "slash", "pack-dir", "parent", "symlink"])
@pytest.mark.parametrize("apply", [False, True])
def test_a_queue_carrying_the_marker_is_refused_dry_and_apply(
        stranger, monkeypatch, drain_never_starts, capsys, where, apply):
    work = stranger / "work"
    queue = work / "promotion-queue"
    (queue / "pack-a").mkdir(parents=True)
    (queue / Q.LIVE_QUEUE_MARKER).write_text("")
    (stranger / "lnk").symlink_to(queue, target_is_directory=True)
    named = {"queue": queue, "slash": str(queue) + "/", "pack-dir": queue / "pack-a",
             "parent": work, "symlink": stranger / "lnk"}[where]
    assert _promote("--queue-dir", str(named), *(["--apply"] if apply else [])) == 2
    err = capsys.readouterr().err
    assert Q.LIVE_QUEUE_MARKER in err and "systemctl start aw-promotion-nightly.service" in err


def test_the_marker_decides_with_a_scratch_work_root_and_an_explicit_live_path(
        tmp_path, monkeypatch, stranger, drain_never_starts):
    queue = tmp_path / "live" / "promotion-queue"
    queue.mkdir(parents=True)
    (queue / Q.LIVE_QUEUE_MARKER).write_text("")
    monkeypatch.setenv("AW_WORK_ROOT", str(tmp_path / "scratch-root"))
    monkeypatch.setenv("AW_QUEUE_DIR", str(tmp_path / "scratch-queue"))
    PATHS.reset_settings_cache()
    assert _promote("--queue-dir", str(queue), "--apply") == 2


def test_an_unmarked_sibling_is_not_refused(stranger, monkeypatch):
    queue = stranger / "work" / "promotion-queue"
    queue.mkdir(parents=True)
    (queue / Q.LIVE_QUEUE_MARKER).write_text("")
    sibling = stranger / "work" / "build-queue"
    sibling.mkdir()
    assert Q.live_queue_refusal(sibling) is None


# ------------------------------------------------------------- the Single reads the marker too
@pytest.mark.parametrize("apply", [False, True])
@pytest.mark.parametrize("where", ["queue", "inside", "parent"])
def test_a_single_refuses_a_marked_queue_before_it_reads_it(stranger, where, apply):
    work = stranger / "work"
    queue = work / "promotion-queue"
    (queue / "pack-a").mkdir(parents=True)
    (queue / Q.LIVE_QUEUE_MARKER).write_text("")
    before = sorted(p.name for p in queue.rglob("*"))
    named = {"queue": queue, "inside": queue / "pack-a", "parent": work}[where]
    tx = Q.Single(url=NOT_LIVE, collection="aw_build")
    with pytest.raises(Q.QueueError, match=Q.LIVE_QUEUE_MARKER):
        Q.drain(named, task="t", apply=apply, batch=1, pause_s=0,
                artifact_root=stranger / "art", transaction=tx, refresh_index=False)
    assert sorted(p.name for p in queue.rglob("*")) == before and not (stranger / "art").exists()


def test_a_single_refuses_a_state_dir_or_artifact_root_inside_a_marked_queue(stranger):
    queue = stranger / "promotion-queue"
    queue.mkdir()
    (queue / Q.LIVE_QUEUE_MARKER).write_text("")
    bq = stranger / "build-queue"
    bq.mkdir()
    for kw in ({"state_dir": queue / "_single", "root": stranger / "art"},
               {"state_dir": stranger / "state", "root": queue / "_artifacts"}):
        tx = Q.Single(url=NOT_LIVE, collection="aw_build", state_dir=kw["state_dir"])
        with pytest.raises(Q.QueueError, match=Q.LIVE_QUEUE_MARKER):
            Q.drain(bq, task="t", apply=True, batch=1, pause_s=0, artifact_root=kw["root"],
                    transaction=tx, refresh_index=False)
    assert list(queue.iterdir()) == [queue / Q.LIVE_QUEUE_MARKER]


# ----------------------------------------------------- the drain ignores the marker (intake)
def test_the_drains_intake_skips_a_marker_file(tmp_path, monkeypatch, capsys):
    """The default twin-first drain lists `queue_dir.iterdir()` filtered by `is_dir()`: the marker
    never reaches its pack list. An empty queue with only the marker is "empty"; with a pack dir the
    list it orders holds that pack and nothing else."""
    queue = tmp_path / "queue"
    queue.mkdir()
    (queue / Q.LIVE_QUEUE_MARKER).write_text("")
    common = dict(twin_url=NOT_LIVE, prod_url=NOT_LIVE, task="t", apply=False, batch=1, pause_s=0,
                  artifact_root=tmp_path / "art", refresh_index=False)
    assert Q.drain(queue, **common) == []
    assert "empty" in capsys.readouterr().out
    (queue / "pack-a").mkdir()
    seen = []

    class Stop(Exception):
        pass

    def order(dirs, artifact_root):
        seen.append([p.name for p in dirs])
        raise Stop

    monkeypatch.setattr(Q, "order_pack_dirs_for_drain", order)
    with pytest.raises(Stop):
        Q.drain(queue, **common)
    assert seen == [["pack-a"]]


# ------------------------------------------------------- a hand run takes the unit's lock
HOLDER = textwrap.dedent("""
    import fcntl, sys, time
    f = open(sys.argv[1], "rb")
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print("held", flush=True)
    time.sleep(120)
""")


def test_a_hand_run_refuses_by_name_while_another_process_holds_the_unit_lock(
        tmp_path, monkeypatch, box, capsys):
    lock = tmp_path / "unit.lock"
    lock.write_text("")
    scratch = tmp_path / "scratch-queue"
    scratch.mkdir()
    monkeypatch.setenv("AW_PROMOTION_LOCK", str(lock))
    PATHS.reset_settings_cache()
    ran = []
    monkeypatch.setattr(Q, "main", lambda argv: ran.append(argv) or 0)
    holder = subprocess.Popen([sys.executable, "-c", HOLDER, str(lock)], stdout=subprocess.PIPE,
                              text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        assert _promote("--queue-dir", str(scratch), "--apply") == 2 and not ran
        err = capsys.readouterr().err
        assert str(lock) in err and "held by another process" in err
        assert _promote("--queue-dir", str(scratch)) == 2      # a dry run too
    finally:
        holder.kill()
        holder.wait()
    # released: the same call now runs, and the lock is free again afterwards
    assert _promote("--queue-dir", str(scratch), "--apply") == 0 and len(ran) == 1
    with Q.hold_unit_lock() as held:
        assert held == str(lock)


def test_a_missing_unit_lock_file_is_created_and_taken(tmp_path, monkeypatch, box):
    """/run is cleared at reboot: the file is absent until the first firing. The hand run creates it
    (as flock(1) does) and holds it, so the unit's `flock -n` fails while the hand run lives."""
    scratch = tmp_path / "scratch-queue"
    scratch.mkdir()
    lock = tmp_path / "no-such.lock"
    monkeypatch.setenv("AW_PROMOTION_LOCK", str(lock))
    PATHS.reset_settings_cache()
    with Q.hold_unit_lock() as held:
        assert held == str(lock) and lock.is_file()
        probe = subprocess.run(["flock", "-n", str(lock), "true"])
        assert probe.returncode == 1
    assert subprocess.run(["flock", "-n", str(lock), "true"]).returncode == 0
    monkeypatch.setattr(Q, "main", lambda argv: 0)
    assert _promote("--queue-dir", str(scratch)) == 0


def test_a_lock_whose_parent_is_missing_or_is_a_directory_refuses(tmp_path, monkeypatch, box, capsys):
    scratch = tmp_path / "scratch-queue"
    scratch.mkdir()
    monkeypatch.setattr(Q, "main", lambda argv: pytest.fail("the drain was started"))
    for bad in (tmp_path / "no-dir" / "unit.lock", tmp_path):
        monkeypatch.setenv("AW_PROMOTION_LOCK", str(bad))
        PATHS.reset_settings_cache()
        assert _promote("--queue-dir", str(scratch)) == 2
        assert "cannot be used" in capsys.readouterr().err


def test_no_promotion_lock_setting_is_inert(tmp_path, monkeypatch, box):
    scratch = tmp_path / "scratch-queue"
    scratch.mkdir()
    monkeypatch.delenv("AW_PROMOTION_LOCK", raising=False)
    PATHS.reset_settings_cache()
    with Q.hold_unit_lock() as held:
        assert held is None
    monkeypatch.setattr(Q, "main", lambda argv: 0)
    assert _promote("--queue-dir", str(scratch)) == 0
    assert list(tmp_path.glob("*.lock")) == []


def test_the_drains_own_error_is_not_reported_as_a_lock_refusal(tmp_path, monkeypatch, box):
    scratch = tmp_path / "scratch-queue"
    scratch.mkdir()

    def boom(argv):
        raise Q.QueueError("a drain error")

    monkeypatch.setattr(Q, "main", boom)
    with pytest.raises(Q.QueueError, match="a drain error"):
        _promote("--queue-dir", str(scratch))


# --------------------------------------------- a state entry is claimed only when absent or empty
@pytest.mark.parametrize("held", [
    ["vectors", "passages.jsonl"],                   # a pack still being sealed: no manifest yet
    ["manifest.json"], ["READY"], ["DONE"], ["receipt.json"], ["scratch.txt"]])
def test_a_state_entry_that_is_not_empty_and_not_ours_is_refused(tmp_path, held):
    entry = tmp_path / "state" / "pack-a"
    entry.mkdir(parents=True)
    for name in held:
        if name == "vectors":
            (entry / name).mkdir()
        else:
            (entry / name).write_text("x")
    before = sorted(p.name for p in entry.iterdir())
    with pytest.raises(Q.QueueError, match="no SINGLE-STATE marker"):
        Q.claim_state_entry(entry)
    with pytest.raises(Q.QueueError, match="no SINGLE-STATE marker"):
        Q.refuse_state_entries(tmp_path / "state", ["pack-a"])
    assert sorted(p.name for p in entry.iterdir()) == before          # nothing was marked


def test_a_state_entry_that_is_absent_empty_or_ours_is_claimed(tmp_path):
    root = tmp_path / "state"
    root.mkdir()
    Q.refuse_state_entries(root, ["absent"])
    Q.claim_state_entry(root / "absent")
    assert (root / "absent" / Q.SINGLE_STATE_MARKER).is_file()
    (root / "empty").mkdir()
    Q.refuse_state_entries(root, ["empty"])
    Q.claim_state_entry(root / "empty")
    assert [p.name for p in (root / "empty").iterdir()] == [Q.SINGLE_STATE_MARKER]
    (root / "empty" / "receipt.json").write_text("{}")      # ours now: it may hold what it wrote
    Q.claim_state_entry(root / "empty")
    Q.refuse_state_entries(root, ["empty"])


def test_a_file_where_a_state_entry_belongs_is_refused(tmp_path):
    (tmp_path / "pack-a").write_text("x")
    with pytest.raises(Q.QueueError, match="no SINGLE-STATE marker"):
        Q.claim_state_entry(tmp_path / "pack-a")


def test_a_crashed_claim_leaves_a_tmp_file_and_the_entry_is_still_claimed(tmp_path):
    """SIGKILL between `atomic_write_bytes`'s tmp write and its rename leaves `.SINGLE-STATE.tmp<pid>`."""
    root = tmp_path / "state"
    entry = root / "pack-a"
    entry.mkdir(parents=True)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    stale = entry / f".{Q.SINGLE_STATE_MARKER}.tmp{dead.pid}"
    stale.write_text("half")
    Q.refuse_state_entries(root, ["pack-a"])
    Q.claim_state_entry(entry)
    assert (entry / Q.SINGLE_STATE_MARKER).is_file() and not stale.exists()
    # a tmp file of a LIVE pid is a claim in progress: still refused; real content is still refused
    other = root / "pack-b"
    other.mkdir()
    (other / f".{Q.SINGLE_STATE_MARKER}.tmp{os.getpid()}").write_text("x")
    with pytest.raises(Q.QueueError, match="no SINGLE-STATE marker"):
        Q.claim_state_entry(other)
    third = root / "pack-c"
    third.mkdir()
    (third / f".{Q.SINGLE_STATE_MARKER}.tmp{dead.pid}").write_text("half")
    (third / "manifest.json").write_text("{}")
    with pytest.raises(Q.QueueError, match="manifest.json"):
        Q.claim_state_entry(third)


# ----------------------------------- the box file names only `work_root` (impactia.json's shape)
def test_the_box_leg_resolves_the_derived_queue_from_the_box_file_alone(box, monkeypatch, tmp_path):
    monkeypatch.setenv("AW_WORK_ROOT", str(tmp_path / "scratch-root"))
    monkeypatch.setenv("AW_QUEUE_DIR", str(tmp_path / "scratch-queue"))
    PATHS.reset_settings_cache()
    assert Q.live_queue_dir() != box.resolve()
    assert Q.box_file_queue_dir() == box.resolve()


@pytest.mark.parametrize("apply", [False, True])
def test_work_root_scratch_and_explicit_live_queue_without_marker_is_refused(
        tmp_path, monkeypatch, box, drain_never_starts, capsys, apply):
    """The reviewer's trigger on the real file shape: AW_WORK_ROOT=<scratch> + --queue-dir <live>."""
    assert not (box / Q.LIVE_QUEUE_MARKER).exists()
    monkeypatch.setenv("AW_WORK_ROOT", str(tmp_path / "scratch-root"))
    PATHS.reset_settings_cache()
    argv = ["--queue-dir", str(box)] + (["--apply"] if apply else [])
    assert _promote(*argv) == 2
    assert "box file's `queue_dir`" in capsys.readouterr().err


def test_an_ancestor_and_a_descendant_of_the_live_queue_are_refused_with_the_marker(
        tmp_path, monkeypatch, box, drain_never_starts):
    """Ancestor (grandparent) and descendant, marker present, scratch environment."""
    (box / Q.LIVE_QUEUE_MARKER).write_text("")
    monkeypatch.setenv("AW_WORK_ROOT", str(tmp_path / "scratch-root"))
    PATHS.reset_settings_cache()
    for named in (box.parent.parent, box.parent, box / "pack-a"):
        assert _promote("--queue-dir", str(named)) == 2, named


def test_an_ancestor_is_refused_by_the_box_leg_even_when_the_marker_search_does_not_reach(
        tmp_path, monkeypatch, box, drain_never_starts):
    """The marker leg looks up through ancestors and one level down; the grandparent holds the
    marker two levels down. The box leg refuses it."""
    assert marked_none(box.parent.parent)
    monkeypatch.setenv("AW_WORK_ROOT", str(tmp_path / "scratch-root"))
    PATHS.reset_settings_cache()
    assert _promote("--queue-dir", str(box.parent.parent)) == 2


def marked_none(path) -> bool:
    return Q.marked_live_queue(path) is None

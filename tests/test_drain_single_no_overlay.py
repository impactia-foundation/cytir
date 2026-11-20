"""A `Single` drain on a box with NO operator overlay and NO box file (task 696 slice 4 step C,
round 6, after the drain's `transaction=` parameter, #553).

A stranger's `cyt index build` has none of our settings: no `prod_lock_host`, no `twin_ssh_host`, no
live lock location, no live queue. Step C removed the implicit `./aw-work`, so every path setting
refuses by name unless somebody configured it. The twin-first drain must refuse those at START
(round 5). A `Single` must NOT: it is asked only for what it reads (`single_startup_refusal`):
the lock registry under `--apply` (`AW_WORK_ROOT`, or `AW_LOCK_HOST` + `AW_LOCK_DIR`), and
`corpus_root` only when the persisted-input check is on (`AW_PERSISTED_INDEX_SQLITE_DIR`).
Queue, state dir and artifact root are explicit arguments. Needs a throwaway Qdrant; skips without.
"""
from __future__ import annotations

import pathlib

import pytest

import drain_transaction_harness as H
from cytir.index import promote_queue as Q                       # noqa: E402  (the alias of cytir.index.promote_queue)
from cytir.core import paths as P                  # noqa: E402

TARGET = "aw_single_no_overlay"
PACK = "dila-juri-v1-shard0"
_SETTING_ENVS = [s.env for s in P._SPECS if getattr(s, "env", None)]
_EXTRA_ENVS = [P.OVERLAY_ENV, "AW_PERSISTED_INDEX_SQLITE_DIR", "AW_APPLY_NO_TWIN", "AW_TWIN_SSH_HOST"]


def _stranger(monkeypatch):
    """A box with no overlay, no box file and no setting in the environment."""
    for name in _SETTING_ENVS + _EXTRA_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(P.TEST_LOOKUP_ENV, "")
    # the session's isolated lock registry (root conftest) pins these on the module: a stranger's
    # box has no such pin, so the settings and `AW_LOCK_*` decide
    from cytir.ops import write_lock                       # noqa: PLC0415
    for pinned in ("LOCK_HOST", "LOCK_DIR"):
        monkeypatch.delitem(write_lock.__dict__, pinned, raising=False)
    P.reset_settings_cache()


@pytest.fixture
def stranger(tmp_path, monkeypatch):
    _stranger(monkeypatch)
    yield tmp_path
    P.reset_settings_cache()


def _argv(queue, tmp, url, *extra, apply=True):
    return ["--queue-dir", str(queue), "--target-url", url, "--target-collection", TARGET,
            "--artifact-root", str(tmp / "artifacts"), "--target-state-dir", str(tmp / "state"),
            "--task", "696-no-overlay", "--batch", "3", "--pause-s", "0",
            "--no-refresh-persisted-index", *(["--apply"] if apply else []), *extra]


def test_a_single_demands_only_its_lock_location_and_names_it(stranger, capsys):
    """No lock location configured: exit 2 naming lock_host/lock_dir, and NOT `prod_lock_host`,
    `twin_ssh_host` or `corpus_root`, which a stranger has none of. The queue is never opened."""
    queue = stranger / "queue"           # does not exist: a startup refusal precedes `no such queue`
    assert Q.main(_argv(queue, stranger, "http://127.0.0.1:1")) == 2
    err = capsys.readouterr().err
    assert "lock_host/lock_dir" in err
    for operator_only in ("prod_lock_host", "twin_ssh_host", "corpus_root"):
        assert operator_only not in err, err
    assert not queue.exists() and not (stranger / "state").exists()


def test_a_single_dry_run_needs_no_setting_at_all(stranger, capsys):
    """A dry single takes no lock and reads no store: it passes the startup check with nothing set
    (it then refuses the missing queue, by its own message, not by a setting)."""
    with pytest.raises(SystemExit) as exc:
        Q.main(_argv(stranger / "queue", stranger, "http://127.0.0.1:1", apply=False))
    assert "no such queue dir" in str(exc.value)
    assert "setting" not in capsys.readouterr().err


def test_a_single_with_the_persisted_check_on_names_corpus_root(stranger, monkeypatch, capsys):
    monkeypatch.setenv("AW_PERSISTED_INDEX_SQLITE_DIR", str(stranger / "idx"))
    monkeypatch.setenv("AW_LOCK_HOST", "local")
    monkeypatch.setenv("AW_LOCK_DIR", str(stranger / "locks"))
    assert Q.main(_argv(stranger / "queue", stranger, "http://127.0.0.1:1")) == 2
    err = capsys.readouterr().err
    assert "corpus_root" in err and "prod_lock_host" not in err and "lock_host" not in err


def test_the_twin_first_drain_still_refuses_the_operator_settings_at_start(stranger, capsys):
    """The other side of the same rule: without `--target-url` the operator-only settings stay
    demanded (round 5), before the queue is opened."""
    assert Q.main(["--queue-dir", str(stranger / "queue"), "--apply"]) == 2
    err = capsys.readouterr().err
    for name in ("corpus_root", "prod_lock_host", "twin_ssh_host"):
        assert name in err, err


def test_a_single_plans_and_applies_on_a_box_with_no_overlay(tmp_path, monkeypatch):
    """The build path of a stranger, end to end through the CLI entry point: no overlay, no box
    file, explicit queue/state/artifact dirs, an explicit lock directory (`AW_LOCK_HOST=local`,
    `AW_LOCK_DIR`, the REAL write-lock registry, not a stub) and a throwaway Qdrant. It plans, takes
    its one lock, lands the pack and releases the lock. No queue setting exists, so
    `live_queue_dir()` is None: the stranger's normal state, nothing to protect."""
    url = H.qdrant_url()
    H.clear_live(url)
    H.reset_collection(url, TARGET)
    queue = H.sealed_pack(tmp_path)
    _stranger(monkeypatch)
    locks = tmp_path / "locks"
    locks.mkdir()
    monkeypatch.setenv("AW_LOCK_HOST", "local")
    monkeypatch.setenv("AW_LOCK_DIR", str(locks))
    P.reset_settings_cache()
    try:
        assert Q.live_queue_dir() is None
        from cytir.ops import write_lock                  # the real registry, in the explicit dir
        assert write_lock.effective_location() == ("local", str(locks))
        monkeypatch.setattr(Q.PSP.PI, "assert_pages_persisted", lambda *a, **k: pytest.fail(
            "the persisted-input check ran with no index"))
        assert Q.main(_argv(queue, tmp_path, url, "--target-payload-profile-of", H.PROD)) == 0
    finally:
        P.reset_settings_cache()
    assert (tmp_path / "state" / PACK / "DONE").exists()
    assert not (queue / PACK / "DONE").exists()          # the queue's pack dir is untouched
    assert len(H.dump_points(url, TARGET)) == 4
    assert list(locks.iterdir()) == []                    # the lock was released
    assert not (pathlib.Path.cwd() / "aw-work").exists()


def test_2938_a_twin_first_dry_run_needs_only_the_plan_settings_not_the_apply_ones(stranger, monkeypatch, capsys):
    """Startup refusal, dry side: with only `corpus_root` configured (what every drain reads for its plan) a dry
    twin-first run is NOT refused for the writer-lock, prod-lock-host or twin-ssh settings; the same box with
    `--apply` exits 2 naming them. The drain itself is stubbed: the exit code is the subject."""
    (stranger / "queue").mkdir()
    monkeypatch.setenv("AW_CORPUS_ROOT", str(stranger / "corpus"))
    P.reset_settings_cache()
    seen = []
    monkeypatch.setattr(Q, "drain", lambda *a, **k: seen.append(k) or [])
    assert Q.main(["--queue-dir", str(stranger / "queue"), "--no-refresh-persisted-index"]) == 0 and len(seen) == 1
    assert Q.main(["--queue-dir", str(stranger / "queue"), "--no-refresh-persisted-index", "--apply"]) == 2
    err = capsys.readouterr().err
    assert "prod_lock_host" in err and "twin_ssh_host" in err and "corpus_root" not in err and len(seen) == 1

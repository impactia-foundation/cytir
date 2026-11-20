"""Round 4 of #559, the refusals that need no Qdrant: the build's RUN_DIR scan and endpoint
normalisation, the target lock, and `cyt index promote`'s honest door (B1-B4; see
`test_drain_build_guards.py` for the Qdrant-backed half).
"""
from __future__ import annotations

import pathlib

import pytest

from cytir.index import promote_queue as Q                       # noqa: E402  (the alias of cytir.index.promote_queue)
from cytir.core import paths as PATHS  # noqa: E402
from cytir.index import build as B  # noqa: E402
from cytir.index import live_target as LT  # noqa: E402


def test_b1_the_lock_is_taken_before_the_check_and_released_after_a_refusal(monkeypatch):
    events = []

    class RB:
        @staticmethod
        def acquire(name, task, purpose, *, pid=None):
            events.append(("acquire", name))

        @staticmethod
        def release(name, task):
            events.append(("release", name))

    def refuse(single):
        events.append(("check",))
        raise Q.QueueError("refused for the test")

    monkeypatch.setattr(Q, "_load_run_batch", lambda: RB)
    monkeypatch.setattr(Q, "assert_single_target_not_live", refuse)
    single = Q.Single(url="http://127.0.0.1:9", collection="aw_build_target")
    with pytest.raises(Q.QueueError):
        with Q.single_target_lock(single, "t", "p"):
            Q.assert_single_target_not_live(single)
    lock = "SINGLE-WRITE-LOCK-aw_build_target"
    assert events == [("acquire", lock), ("check",), ("release", lock)]


@pytest.fixture
def unknown_live_queue(tmp_path, monkeypatch):
    """The half-configured box: `AW_WORK_ROOT` exported, so the settings do NOT know the live queue.
    The queue holds a READY pack, a DONE pack and an ABORTED receipt."""
    monkeypatch.setenv("AW_WORK_ROOT", str(tmp_path / "scratch-root"))
    monkeypatch.delenv("AW_QUEUE_DIR", raising=False)
    monkeypatch.delenv("AW_PROMOTION_QUEUE_DIR", raising=False)
    monkeypatch.delenv(PATHS.OVERLAY_ENV, raising=False)
    monkeypatch.setenv(PATHS.TEST_LOOKUP_ENV, "")
    PATHS.reset_settings_cache()
    live = tmp_path / "work" / "promotion-queue"
    (live / "pack-a").mkdir(parents=True)
    (live / "pack-a" / "READY").write_text("sha\n")
    (live / "pack-a" / "manifest.json").write_text("{}")
    (live / "pack-b").mkdir()
    (live / "pack-b" / "DONE").write_text("done\n")
    (live / "pack-b" / "receipt.json").write_text('{"status": "ABORTED"}')
    assert Q.live_queue_dir() != live.resolve()          # the settings do not know it
    yield live
    PATHS.reset_settings_cache()


def _tree(root: pathlib.Path) -> dict:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_b2_the_scan_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "_SCAN_BUDGET", 3)
    run = tmp_path / "big-run"
    for i in range(5):
        (run / f"d{i}").mkdir(parents=True)
    with pytest.raises(B.BuildRefusal, match="too many directories"):
        B.check_run_dir(run)


def test_b3_normalize_endpoint():
    n = LT.normalize_endpoint
    assert n("http://127.0.0.1") == "http://127.0.0.1:6333"
    assert n("HTTP://LocalHost/") == "http://localhost:6333"
    assert n("https://q.example.org") == "https://q.example.org:6333"      # the client's default, not 443
    assert n("http://[::1]") == "http://[::1]:6333"
    assert n("http://127.0.0.1:6334/") == "http://127.0.0.1:6334"
    for bad in ("127.0.0.1:6333", "ftp://h:1", "http://", "http://u:p@h:1", "http://h:1/prefix",
                "http://h:1?x=1", "http://h:notaport"):
        with pytest.raises(LT.LiveTargetRefused):
            n(bad)


@pytest.mark.parametrize("persisted_index", [None, "/nonexistent", "set-to-a-real-dir"])
def test_b4_aw_index_promote_refuses_the_live_queue_whatever_the_environment_holds(
        tmp_path, monkeypatch, unknown_live_queue, capsys, persisted_index):
    """Round 5: on the configured LIVE queue the verb refuses outright, dry or `--apply`, and names the
    unit. Round 4 let any value of `AW_PERSISTED_INDEX_SQLITE_DIR` through (`/nonexistent` too)."""
    from cytir import cli
    monkeypatch.setenv("AW_QUEUE_DIR", str(unknown_live_queue))      # now the settings DO know it
    PATHS.reset_settings_cache()
    if persisted_index is None:
        monkeypatch.delenv("AW_PERSISTED_INDEX_SQLITE_DIR", raising=False)
    else:
        real = tmp_path / "idx"
        real.mkdir()
        monkeypatch.setenv("AW_PERSISTED_INDEX_SQLITE_DIR",
                           str(real) if persisted_index == "set-to-a-real-dir" else persisted_index)
    monkeypatch.setattr(Q, "main", lambda argv: pytest.fail("the drain was started"))
    for args in ([], ["--queue-dir", str(unknown_live_queue)],
                 ["--queue-dir", str(unknown_live_queue.parent)],
                 ["--queue-dir=" + str(unknown_live_queue / "pack-a")], ["--apply"],
                 ["--apply", "--queue-dir", str(unknown_live_queue)]):
        assert cli.main(["index", "promote", *args]) == 2, args
        err = capsys.readouterr().err
        assert "systemctl start aw-promotion-nightly.service" in err, err


def test_b4_aw_index_promote_runs_on_a_queue_that_is_not_live(tmp_path, monkeypatch, unknown_live_queue):
    from cytir import cli
    monkeypatch.setenv("AW_QUEUE_DIR", str(unknown_live_queue))
    PATHS.reset_settings_cache()
    monkeypatch.setattr(Q, "main", lambda argv: 0)
    other = tmp_path / "other-queue"
    other.mkdir()
    assert cli.main(["index", "promote", "--queue-dir", str(other)]) == 0


def test_b4_a_box_with_no_queue_setting_is_not_refused(tmp_path, monkeypatch):
    from cytir import cli
    for name in ("AW_WORK_ROOT", "AW_QUEUE_DIR", "AW_PROMOTION_QUEUE_DIR", PATHS.OVERLAY_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(PATHS.TEST_LOOKUP_ENV, "")
    monkeypatch.delenv("AW_PERSISTED_INDEX_SQLITE_DIR", raising=False)
    PATHS.reset_settings_cache()
    try:
        monkeypatch.setattr(Q, "main", lambda argv: 0)
        assert cli.main(["index", "promote", "--queue-dir", str(tmp_path)]) == 0
    finally:
        PATHS.reset_settings_cache()


def test_b4_the_help_does_not_call_promote_the_units_door(capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "10000")
    from cytir import cli
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "own door" not in out and "NOT its unit" in out and "flock" in out
    assert "REFUSES" in out and "systemctl start aw-promotion-nightly.service" in out

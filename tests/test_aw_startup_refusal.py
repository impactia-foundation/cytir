"""An entry point with no operator settings fails at START (task 696 slice 4, round 3; review of #567).

Round 2 ran the live drain on a copy with no overlay and an explicit `--queue-dir` (every live unit
passes one): it printed `write-lock = local:REFUSED`, opened the queue, marked EACH pack REFUSED at
`live-apply-preflight`, wrote a receipt into every pack directory and exited 0. A green unit that
landed nothing, with the refusals cached. These tests run the real entry points in a fresh
interpreter with NO configuration and pin: exit code not 0, the setting's name on stderr, and the
queue tree byte-identical before and after (no receipt, no marker, nothing created).
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import site
import subprocess
import sys

import pytest

APP = pathlib.Path(__file__).resolve().parents[1]


def _bare_env(home: pathlib.Path) -> dict:
    # No setting variable, no overlay, and the box-file lookup replaced by "none".
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home),
            "PYTHONPATH": str(APP), "PYTHONUSERBASE": site.getuserbase(), "PYTHONDONTWRITEBYTECODE": "1", "AW_TEST_OVERLAY_LOOKUP": "",
            "AW_OPERATOR_URL": "https://example.invalid/test",
            "AW_OPERATOR_CONTACT": "nobody@example.invalid"}


def _tree(root: pathlib.Path) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        rel = str(p.relative_to(root))
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "<dir>"
    return out


def _queue(tmp_path) -> pathlib.Path:
    q = tmp_path / "queue"
    for name in ("pack-a", "pack-b"):
        (q / name).mkdir(parents=True)
        (q / name / "manifest.json").write_text('{"schema": "aw.promotion-queue-pack.v1"}')
        (q / name / "READY").write_text("00\n")
    return q


def _run(argv: list[str], tmp_path) -> subprocess.CompletedProcess:
    for dep in ("qdrant_client", "httpx", "yaml", "lxml"):
        pytest.importorskip(dep)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    cwd = tmp_path / "cwd"
    cwd.mkdir(exist_ok=True)
    return subprocess.run([sys.executable, *argv], cwd=str(cwd), env=_bare_env(home),
                          capture_output=True, text=True, timeout=300)


def test_the_drain_apply_without_settings_exits_nonzero_names_them_and_touches_no_pack(tmp_path):
    q = _queue(tmp_path)
    before = _tree(q)
    proc = _run(["-m", "cytir.index.promote_queue", "--apply", "--queue-dir", str(q)], tmp_path)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "prod_lock_host" in proc.stderr and "lock_host" in proc.stderr, proc.stderr
    assert "twin_ssh_host" in proc.stderr, proc.stderr
    assert _tree(q) == before, "the drain wrote into the queue before it refused (a receipt?)"
    assert not list(q.rglob("receipt*")) and not list(q.rglob("REFUSED*"))
    assert "live-apply-preflight" not in proc.stdout        # it never got as far as a pack


def test_a_dry_drain_without_settings_refuses_by_name_before_it_reads_the_queue(tmp_path):
    """A dry drain reads the queue, the persisted store and both endpoints for its plan. With no
    operator settings it must not plan against defaults: it exits non-zero naming `corpus_root`
    (and its chain), with the queue tree untouched. The apply-only settings are NOT demanded."""
    q = _queue(tmp_path)
    before = _tree(q)
    proc = _run(["-m", "cytir.index.promote_queue", "--queue-dir", str(q),
                 "--twin-url", "http://127.0.0.1:1", "--prod-url", "http://127.0.0.1:1"], tmp_path)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "corpus_root" in proc.stderr, proc.stderr
    assert "prod_lock_host" not in proc.stderr and "twin_ssh_host" not in proc.stderr, proc.stderr
    assert "[queue]" not in proc.stdout and "[drain]" not in proc.stdout     # never reached the queue
    assert _tree(q) == before


def test_help_needs_no_settings(tmp_path):
    q = _queue(tmp_path)
    proc = _run(["-m", "cytir.index.promote_queue", "--queue-dir", str(q), "--help"], tmp_path)
    assert proc.returncode == 0 and "--apply" in proc.stdout


def test_the_apply_check_resolves_what_the_lock_calls_use(tmp_path, monkeypatch):
    """In-process: a value pinned on `write_lock` (a test's isolated registry) counts as the
    location; with it pinned and the three settings in the environment, the check passes."""
    from cytir.core import paths as P
    from cytir.index import promote_queue as PQ
    from cytir.ops import write_lock as WL
    for var in ("AW_PROD_LOCK_HOST", "AW_TWIN_SSH_HOST", "AW_APPLY_NO_TWIN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(P.TEST_LOOKUP_ENV, "")
    P.reset_settings_cache()
    try:
        monkeypatch.setitem(WL.__dict__, "LOCK_HOST", "local")
        monkeypatch.setitem(WL.__dict__, "LOCK_DIR", str(tmp_path))
        assert PQ.apply_startup_refusal() == 2                       # prod host + twin host missing
        monkeypatch.setenv("AW_PROD_LOCK_HOST", "somewhere")
        monkeypatch.setenv("AW_TWIN_SSH_HOST", "elsewhere")
        P.reset_settings_cache()
        assert PQ.apply_startup_refusal() is None
        monkeypatch.delenv("AW_TWIN_SSH_HOST")
        monkeypatch.setenv("AW_APPLY_NO_TWIN", "1")                  # the deliberate emergency hatch
        P.reset_settings_cache()
        assert PQ.apply_startup_refusal() is None
    finally:
        P.reset_settings_cache()


def test_growth_pending_compose_without_settings_fails_before_the_first_job(tmp_path):
    if not (APP / "cytir" / "ops" / "growth_pending_compose.py").is_file():
        pytest.skip("this tree ships no cytir/ ops module (the public cut)")
    q = _queue(tmp_path)
    before = _tree(q)
    proc = _run(["-m", "cytir.ops.growth_pending_compose", "--queue-dir", str(q)], tmp_path)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "gpu_lock_dir" in proc.stderr, proc.stderr
    assert _tree(q) == before
    assert "nothing pending" not in proc.stdout and "pending" not in proc.stdout.replace(
        "growth-pending-compose", "")                                # no discovery ran


def test_news_aged_out_without_settings_fails_before_the_twin_scroll(tmp_path):
    proc = _run(["-m", "cytir.news.aged_out"], tmp_path)
    assert proc.returncode != 0
    assert "queue_dir" in proc.stderr and "work_root" in proc.stderr, proc.stderr
    assert "aged_out=" not in proc.stderr                            # the scroll never started


def test_news_daily_without_settings_fails_before_any_day_runs(tmp_path):
    proc = _run(["-m", "cytir.news.daily", "--day", "2026-01-01"], tmp_path)
    assert proc.returncode != 0
    assert "[REFUSED] news daily" in proc.stderr and "setting `" in proc.stderr, proc.stderr
    assert "=== day=" not in proc.stderr

"""Task 769: derive mode cuts the release for a moved HEAD; `prune` bounds the releases root.

Real throwaway git repo, no network, no box."""
from __future__ import annotations

import json
import os
import subprocess
import time

import pytest

from cytir.news import release as rel
from cytir.news import release_preflight as rp

GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x",
           "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True, env=GIT_ENV).stdout.strip()


def _commit(repo, note):
    app = repo / rel.APP_SUBDIR
    for path in sorted(rp.REQUIRED_RELEASE_FILES):
        (app / path).parent.mkdir(parents=True, exist_ok=True)
        (app / path).write_text(f"# stub {note}\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", note)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture()
def box(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    return repo, tmp_path / "releases", tmp_path / "work"


def test_head_without_release_is_cut_then_resolves(box, capsys):
    repo, roots, work = box
    sha = _commit(repo, "one")
    logs = []
    pin = rel.ensure_pin(repo / rel.APP_SUBDIR, roots, work, log=logs.append)  # the APP subdir is fine
    assert logs == [f"news release: cut {sha[:8]} for HEAD (auto)"]
    assert pin.sha == sha and pin.release_dir == roots / f"news-{sha[:8]}"
    assert len((pin.release_dir / "GIT_SHA").read_text().strip()) == 40


def test_resolve_block_after_autocut(box, capsys):
    repo, roots, work = box
    sha = _commit(repo, "one")
    rel.ensure_pin(repo, roots, work, log=lambda m: None)
    capsys.readouterr()
    assert rel.main(["resolve", "--checkout", str(repo), "--releases-root", str(roots),
                     "--work", str(work)]) == 0
    block = json.loads(capsys.readouterr().out)
    assert len(block) == 5 and block["AW_NEWS_RELEASE_SHA"] == sha


def test_existing_release_is_reused_not_recut(box):
    repo, roots, work = box
    _commit(repo, "one")
    first = rel.ensure_pin(repo, roots, work, log=lambda m: None)
    marker = first.release_dir / "GIT_SHA"
    before = marker.stat().st_mtime_ns
    logs = []
    again = rel.ensure_pin(repo, roots, work, log=logs.append)
    assert logs == [] and again == first and marker.stat().st_mtime_ns == before


def test_moved_head_gets_its_own_release(box):
    repo, roots, work = box
    old = rel.ensure_pin(repo, roots, work, log=lambda m: None) if _commit(repo, "one") else None
    new_sha = _commit(repo, "two")
    new = rel.ensure_pin(repo, roots, work, log=lambda m: None)
    assert new.sha == new_sha != old.sha and old.release_dir.is_dir() and new.release_dir.is_dir()


def test_cut_failure_still_refuses_rc2(box, capsys, monkeypatch):
    repo, roots, work = box
    # the cut fails: this commit has no app subtree to archive
    (repo / "README").write_text("x\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "no app")
    with pytest.raises(Exception) as err:
        rel.ensure_pin(repo, roots, work, log=lambda m: None)
    assert isinstance(err.value, (rel.ReleaseError, rp.PreflightError))
    assert not any(roots.glob("news-*")) or not any(p for p in roots.glob("news-*") if p.suffix != ".partial")
    rc = rel.main(["ensure", "--checkout", str(repo), "--releases-root", str(roots), "--work", str(work)])
    err = capsys.readouterr().err
    assert rc == 2 and err.count("news release ensure: FAILED: ") == 1


def test_pinned_mode_is_unchanged_resolve_still_refuses_without_release(box, capsys):
    repo, roots, work = box
    _commit(repo, "one")
    assert rel.main(["resolve", "--checkout", str(repo), "--releases-root", str(roots),
                     "--work", str(work)]) == 2
    assert "no news release has been cut for HEAD" in capsys.readouterr().err
    assert not roots.exists() or not list(roots.glob("news-*"))   # resolve never cuts


def test_daily_module_pinned_mode_does_not_reference_ensure(monkeypatch):
    src = open(os.path.join(os.path.dirname(rel.__file__), "daily.py")).read()
    # the only call to ensure_pin sits inside the FROM_CHECKOUT == "1" branch
    head, tail = src.split('if os.environ.get("AW_NEWS_RELEASE_FROM_CHECKOUT") == "1":', 1)
    assert "ensure_pin" not in head and "derive_pin" not in head and "ensure_pin" in tail.split("\n\n")[0]


def _fake_release(roots, work, sha8, age_days, now):
    d = roots / f"news-{sha8}"
    d.mkdir(parents=True)
    g = d / "GIT_SHA"
    g.write_text(sha8 * 5 + "\n")
    t = now - age_days * 86400
    os.utime(g, (t, t))
    work.mkdir(parents=True, exist_ok=True)
    (work / f"release-manifest-{sha8}.json").write_text("{}")
    (work / f"release-manifest-{sha8}.sha256").write_text("0" * 64)


def test_prune_keeps_current_and_n_newest_and_nothing_young(tmp_path):
    roots, work, now = tmp_path / "r", tmp_path / "w", time.time()
    ages = {"aaaaaaa1": 30, "aaaaaaa2": 25, "aaaaaaa3": 20, "aaaaaaa4": 15, "aaaaaaa5": 10,
            "aaaaaaa6": 9, "aaaaaaa7": 3, "aaaaaaa8": 1}
    for k, v in ages.items():
        _fake_release(roots, work, k, v, now)
    (roots / "task625-news-catchup-r22").mkdir()        # not a tool-cut dir: untouched
    gone = rel.prune(roots, work, keep=4, protect=("aaaaaaa1",), now=now)
    assert sorted(gone) == ["news-aaaaaaa2", "news-aaaaaaa3", "news-aaaaaaa4"]
    left = sorted(p.name for p in roots.iterdir())
    assert left == ["news-aaaaaaa1", "news-aaaaaaa5", "news-aaaaaaa6",
                    "news-aaaaaaa7", "news-aaaaaaa8", "task625-news-catchup-r22"]
    assert not (work / "release-manifest-aaaaaaa2.sha256").exists()
    assert (work / "release-manifest-aaaaaaa1.sha256").exists()
    # young releases survive even with keep=0
    assert sorted(rel.prune(roots, work, keep=0, protect=("aaaaaaa1",), now=now)) == \
        ["news-aaaaaaa5", "news-aaaaaaa6"]
    assert sorted(p.name for p in roots.glob("news-*")) == ["news-aaaaaaa1", "news-aaaaaaa7", "news-aaaaaaa8"]


def test_autocut_prunes_and_never_prunes_head(box):
    repo, roots, work = box
    now = time.time()
    for i in range(6):
        _fake_release(roots, work, f"bbbbbbb{i}", 30 + i, now)
    sha = _commit(repo, "one")
    logs = []
    rel.ensure_pin(repo, roots, work, keep=2, log=logs.append)
    assert (roots / f"news-{sha[:8]}").is_dir()
    assert len(list(roots.glob("news-*"))) == 2          # head + 1 newest old (keep=2 counts head)
    assert any(l.startswith("news release: pruned") for l in logs)


def test_refusal_and_reuse_paths_open_no_lock_file(box):
    repo, roots, work = box
    _commit(repo, "one")
    rel.ensure_pin(repo, roots, work, log=lambda m: None)
    (roots / ".news-release-cut.lock").unlink()
    rel.ensure_pin(repo, roots, work, log=lambda m: None)           # reuse: lock-free
    assert not (roots / ".news-release-cut.lock").exists()
    (work / next(work.glob("release-manifest-*.sha256")).name).write_text("junk\n")
    with pytest.raises(rel.ReleaseError):                            # broken release: refusal
        rel.ensure_pin(repo, roots, work, log=lambda m: None)
    assert not (roots / ".news-release-cut.lock").exists()
    assert not (work / ".release-cut.lock").exists()


def test_unwritable_release_root_refuses_cleanly_rc2(box, tmp_path, capsys):
    repo, _, work = box
    _commit(repo, "one")
    blocker = tmp_path / "file"
    blocker.write_text("x")
    roots = blocker / "releases"                                     # mkdir under a file: OSError
    with pytest.raises(rel.ReleaseError) as err:
        rel.ensure_pin(repo, roots, work, log=lambda m: None)
    assert "no news release has been cut for HEAD" in str(err.value) and "auto-cut failed" in str(err.value)
    assert rel.main(["ensure", "--checkout", str(repo), "--releases-root", str(roots),
                     "--work", str(work)]) == 2
    assert "news release ensure: FAILED: " in capsys.readouterr().err


def _ensure_cli(repo, roots, work, capsys):
    rc = rel.main(["ensure", "--checkout", str(repo), "--releases-root", str(roots), "--work", str(work)])
    cap = capsys.readouterr()
    return rc, cap.out, [l for l in cap.err.splitlines() if l.startswith("news release ensure:")]


def test_ensure_cli_one_outcome_line_on_stderr_created_then_reused(box, capsys):
    repo, roots, work = box
    sha = _commit(repo, "one")
    rc, out, lines = _ensure_cli(repo, roots, work, capsys)
    assert rc == 0 and lines == [f"news release ensure: created news-{sha[:8]}"]
    assert json.loads(out)                        # the env block stays on stdout
    rc, out, lines = _ensure_cli(repo, roots, work, capsys)
    assert rc == 0 and lines == [f"news release ensure: reused news-{sha[:8]}"]


def test_ensure_cli_unwritable_root_prints_failed_rc2(box, tmp_path, capsys):
    repo, _, work = box
    _commit(repo, "one")
    blocker = tmp_path / "file"
    blocker.write_text("x")
    rc, out, lines = _ensure_cli(repo, blocker / "r", work, capsys)
    assert rc == 2 and len(lines) == 1 and lines[0].startswith("news release ensure: FAILED: ")

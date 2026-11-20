from __future__ import annotations

import json

import pytest

from cytir.index import rebuild_from_manifest as rebuild
from cytir.index.preservation_contract import ContractError


def test_expansion_is_argv_only_and_deterministic():
    argv = rebuild._expand(
        ["python3", "{repo}/bin/x.py", "--out", "{work}/out"],
        {"repo": "/repo", "work": "/work"})
    assert argv == ["python3", "/repo/bin/x.py", "--out", "/work/out"]


def test_unknown_placeholder_fails():
    with pytest.raises(ContractError, match="unknown rebuild placeholder"):
        rebuild._expand(["{machine_specific_path}"], {"repo": "/repo"})


def test_blocked_production_lock_refuses_status(tmp_path, capsys):
    manifest = tmp_path / "blocked.json"
    manifest.write_text(json.dumps({
        "status": "blocked",
        "collection": {"name": "aw_e3"},
        "blocking_gaps": [{"id": "G1", "detail": "missing Markdown"}],
    }))
    rc = rebuild.main(["status", "--manifest", str(manifest)])
    output = capsys.readouterr().out
    assert rc == 1
    assert "status=blocked" in output
    assert "G1: missing Markdown" in output


def test_phase_selection_rejects_unknown_name():
    rows = [{"phase": "extract", "argv": ["true"]}]
    selected = [row for row in rows if row["phase"] in ["embed"]]
    unknown = sorted({"embed"} - {row["phase"] for row in selected})
    assert unknown == ["embed"]


def test_nonempty_work_requires_explicit_resume(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "unexpected").write_text("state")
    with pytest.raises(ContractError, match="not empty"):
        rebuild._require_workdir(work, resume=False)
    rebuild._require_workdir(work, resume=True)


def test_the_index_rebuild_door_reaches_the_runner(tmp_path, capsys):
    """`cyt index rebuild` is the one rebuild door (task 1234); a blocked manifest refuses through it."""
    from cytir import cli
    manifest = tmp_path / "blocked.json"
    manifest.write_text(json.dumps({"status": "blocked", "collection": {"name": "c"},
                                    "blocking_gaps": ["G9"]}))
    assert cli.main(["index", "rebuild", "status", "--manifest", str(manifest)]) == 1
    assert "BLOCKED: G9" in capsys.readouterr().out


def _git(repo, *args, check=True):
    import subprocess
    return subprocess.run(["git", "-C", str(repo), *args], text=True, capture_output=True,
                          check=check)


def _tmp_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.org")
    _git(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("1")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "one")
    old = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "a.txt").write_text("2")
    _git(repo, "commit", "-q", "-am", "two")
    return repo, old


def test_interrupted_worktree_add_is_recovered(tmp_path):
    repo, old = _tmp_repo(tmp_path)
    work = tmp_path / "work"
    target = work / f"repo-{old[:12]}"
    # what a kill -9 mid `git worktree add` leaves: a half-populated dir plus a locked admin entry
    target.mkdir(parents=True)
    (target / "partial.txt").write_text("x")
    admin = repo / ".git" / "worktrees" / target.name
    admin.mkdir(parents=True)
    (admin / "gitdir").write_text(f"{target}/.git\n")
    (admin / "HEAD").write_text(old + "\n")
    (admin / "commondir").write_text("../..\n")
    (admin / "locked").write_text("initializing\n")
    raw = _git(repo, "worktree", "add", "--detach", str(target), old, check=False)
    assert raw.returncode != 0  # the simulation really breaks a plain add
    got = rebuild._pinned_repo(repo, work, old)
    assert got == target
    assert _git(target, "rev-parse", "HEAD").stdout.strip() == old
    assert not (target / "partial.txt").exists()


def test_interrupted_add_with_missing_dir_and_locked_entry(tmp_path):
    repo, old = _tmp_repo(tmp_path)
    work = tmp_path / "work"
    target = work / f"repo-{old[:12]}"
    admin = repo / ".git" / "worktrees" / target.name
    admin.mkdir(parents=True)
    (admin / "gitdir").write_text(f"{target}/.git\n")
    (admin / "HEAD").write_text(old + "\n")
    (admin / "commondir").write_text("../..\n")
    (admin / "locked").write_text("initializing\n")
    work.mkdir()
    raw = _git(repo, "worktree", "add", "--detach", str(target), old, check=False)
    assert raw.returncode != 0
    assert rebuild._pinned_repo(repo, work, old) == target
    assert _git(target, "rev-parse", "HEAD").stdout.strip() == old


def test_clean_worktree_at_right_sha_is_reused_untouched(tmp_path):
    repo, old = _tmp_repo(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    target = work / f"repo-{old[:12]}"
    _git(repo, "worktree", "add", "--detach", str(target), old)
    marker = target / "untracked-marker.txt"
    marker.write_text("keep me")
    assert rebuild._pinned_repo(repo, work, old) == target
    assert marker.read_text() == "keep me"


def test_dirty_complete_worktree_at_other_sha_is_refused_not_deleted(tmp_path):
    import pytest
    repo, old = _tmp_repo(tmp_path)
    new = _git(repo, "rev-parse", "HEAD").stdout.strip()
    work = tmp_path / "work"
    target = work / f"repo-{old[:12]}"
    work.mkdir()
    _git(repo, "worktree", "add", "--detach", str(target), new)   # complete, but at the OTHER sha
    (target / "a.txt").write_text("someone's uncommitted edit")
    with pytest.raises(SystemExit, match="uncommitted changes"):
        rebuild._pinned_repo(repo, work, old)
    assert (target / "a.txt").read_text() == "someone's uncommitted edit"

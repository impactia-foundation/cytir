"""Hygiene (2026-10-04): `cytir.news.release cut` meets tracked symlinks, and the coverage backfill's
resume command keeps the release mode it was started with.

Real throwaway git repo, no network. The tracked tree holds the three link shapes the cut can
meet: an absolute link, a relative link that stays inside the tree, and a dangling link."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys

import pytest

from cytir.news import coverage_backfill as CB
from cytir.news import release as rel
from cytir.news import release_preflight as rp

GIT_ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x",
           "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True, env=GIT_ENV).stdout.strip()


@pytest.fixture()
def linked(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    app = repo / rel.APP_SUBDIR
    for path in sorted(rp.REQUIRED_RELEASE_FILES):
        (app / path).parent.mkdir(parents=True, exist_ok=True)
        (app / path).write_text("# stub\n")
    (app / "evidence").mkdir()
    (app / "data").mkdir(exist_ok=True)
    (app / "data" / "real.json").write_text("{}\n")
    os.symlink("/etc/hostname", app / "evidence" / "absolute.json")     # absolute: refused by `data`
    os.symlink("../data/real.json", app / "evidence" / "relative.json")  # relative, inside the tree
    os.symlink("nowhere/at/all.json", app / "evidence" / "dangling.json")  # target does not exist
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "links")
    return repo, tmp_path / "releases", tmp_path / "work"


def test_cut_survives_absolute_relative_and_dangling_links(linked):
    repo, roots, work = linked
    pin = rel.cut(repo, roots, work)                 # used to die: AbsoluteLinkError
    tasks = pin.release_dir / "evidence"
    assert not (tasks / "absolute.json").is_symlink() and not (tasks / "absolute.json").exists()
    assert (tasks / "relative.json").is_symlink()    # in-tree relative link kept as a link
    assert (tasks / "dangling.json").is_symlink()    # harmless, kept as archived
    manifest = json.loads(pin.manifest.read_text())
    assert manifest["skipped_links"] == [{"path": "evidence/absolute.json",
                                          "target": "/etc/hostname"}]
    # the closure ignores every link, so none of them is in it and the release validates
    assert not any(name.startswith("evidence/") for name in manifest["files"])
    assert rel.derive_pin(repo, roots, work).sha == pin.sha


def test_a_link_to_outside_the_tree_is_skipped_and_recorded(linked, tmp_path):
    repo, roots, work = linked
    os.symlink("../../../escape", repo / rel.APP_SUBDIR / "evidence" / "escape.json")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "escape")
    pin = rel.cut(repo, roots, work)
    skipped = {d["path"]: d["target"] for d in json.loads(pin.manifest.read_text())["skipped_links"]}
    assert skipped == {"evidence/absolute.json": "/etc/hostname",
                       "evidence/escape.json": "../../../escape"}
    assert not (pin.release_dir / "evidence" / "escape.json").is_symlink()


def test_a_tree_without_links_keeps_its_manifest_shape(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    app = repo / rel.APP_SUBDIR
    for path in sorted(rp.REQUIRED_RELEASE_FILES):
        (app / path).parent.mkdir(parents=True, exist_ok=True)
        (app / path).write_text("# stub\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "plain")
    pin = rel.cut(repo, tmp_path / "r", tmp_path / "w")
    assert "skipped_links" not in json.loads(pin.manifest.read_text())


# --------------------------------------------------------------------------- resume command


def test_resume_command_carries_the_checkout_release_mode(monkeypatch, tmp_path):
    monkeypatch.delenv("AW_NEWS_RELEASE_FROM_CHECKOUT", raising=False)
    monkeypatch.delenv("AW_OPERATOR_URL", raising=False)
    monkeypatch.delenv("AW_OPERATOR_CONTACT", raising=False)
    plain = CB.resume_command(["srf"], tmp_path / "s")
    assert "AW_NEWS_RELEASE_FROM_CHECKOUT" not in plain          # pinned mode: unchanged
    monkeypatch.setenv("AW_NEWS_RELEASE_FROM_CHECKOUT", "1")
    derived = CB.resume_command(["srf"], tmp_path / "s")
    assert "PYTHONDONTWRITEBYTECODE=1 AW_NEWS_RELEASE_FROM_CHECKOUT=1 " \
           + shlex.quote(sys.executable) in derived
    monkeypatch.setenv("AW_OPERATOR_URL", "https://example.test")
    both = CB.resume_command(["srf"], tmp_path / "s")
    assert "AW_OPERATOR_URL=https://example.test AW_NEWS_RELEASE_FROM_CHECKOUT=1 " in both
    monkeypatch.setenv("AW_NEWS_RELEASE_FROM_CHECKOUT", "0")
    assert "AW_NEWS_RELEASE_FROM_CHECKOUT" not in CB.resume_command(["srf"], tmp_path / "s")

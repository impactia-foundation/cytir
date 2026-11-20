"""task 1505 (S7): the github_repo_docs refresh adapter calls the acquirer in-process."""
from __future__ import annotations

import pathlib

import pytest

from cytir.core import paths
from cytir.sources.github_repo_docs import acquire
from cytir.sources.refresh_direct import github_repo_docs as G


def test_adapter_runs_the_acquirer_in_process_with_the_former_subprocess_arguments(monkeypatch, tmp_path):
    seen = []

    def fake_run(args):
        seen.append(args)
        return 0

    monkeypatch.setattr(acquire, "run", fake_run)
    q, root = tmp_path / "queue-0.jsonl", tmp_path / "run"
    G.run_acquire(q, root, 4, "docs", 3)
    (args,) = seen
    # the exact argv the subprocess call passed after the script name
    assert G.acquire_argv(q, root, 4, "docs", 3) == [
        "run", "--queue", str(q), "--root", str(root), "--max-files", "4", "--tree", "docs", "--workers", "3"]
    assert (args.queue, args.root, args.max_files, args.tree, args.workers) == (q, root, 4, "docs", 3)
    G.run_acquire(q, root, None)
    assert seen[-1].max_files is None and seen[-1].tree == "capped" and seen[-1].workers == 1


def test_a_nonzero_acquirer_result_and_an_acquisition_error_become_runtime_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(acquire, "run", lambda a: 1)
    with pytest.raises(RuntimeError, match="exit 1"):
        G.run_acquire(tmp_path / "q", tmp_path, 1)

    def boom(a):
        raise acquire.AcquisitionError("no token")

    monkeypatch.setattr(acquire, "run", boom)
    with pytest.raises(RuntimeError, match="no token"):
        G.run_acquire(tmp_path / "q", tmp_path, 1)


def test_the_preserve_root_defaults_under_the_work_root_and_the_env_overrides_it(tmp_path):
    s = paths.load_settings(env={"AW_WORK_ROOT": str(tmp_path)}, overlay=None)
    assert s.refresh_github_docs_root == str(tmp_path / "refresh_lane_direct" / "github-repo-docs")
    s = paths.load_settings(env={"AW_REFRESH_GH_DOCS_ROOT": "/elsewhere"}, overlay=None)
    assert (s.refresh_github_docs_root, s.source("refresh_github_docs_root")) == (
        "/elsewhere", "env AW_REFRESH_GH_DOCS_ROOT")


def test_the_package_modules_name_no_box_path_or_subprocess():
    for mod in (acquire, G):
        text = pathlib.Path(mod.__file__).read_text("utf-8")
        assert "/media" not in text and "subprocess" not in text and "sys.path" not in text

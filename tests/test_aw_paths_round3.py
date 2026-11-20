"""Round 3 of #567 (task 696 slice 4): the low items and the box-file drift warning.

* `lock_host` has no silent default: `local` only beside a configured work root.
* A `LazyPath` nobody configured RAISES on comparison; it never answers `False`.
* `python -m cytir.core.paths` prints a WARNING when the box file and the checkout's
  `deploy/operator/impactia.json` both exist and differ (the box file wins until re-installed).
"""
from __future__ import annotations

import json

import pytest

from cytir.core import paths as P
from tests.helpers.repo_files import repo_file_or_skip


@pytest.fixture(autouse=True)
def clean_settings_env(monkeypatch):
    for spec in P._SPECS:
        for name in (spec.env, *spec.aliases):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(P.OVERLAY_ENV, raising=False)
    monkeypatch.setenv(P.TEST_LOOKUP_ENV, "")
    P.reset_settings_cache()
    yield
    P.reset_settings_cache()


def _overlay(path, settings):
    path.write_text(json.dumps({"schema": P.OVERLAY_SCHEMA, "settings": settings}))
    return path


def test_the_lock_host_refuses_without_a_work_root_and_names_it():
    s = P.load_settings()
    with pytest.raises(P.SettingRefused, match=r"`lock_host` derives from `work_root`"):
        s.lock_host
    assert P.load_settings(env={"AW_WORK_ROOT": "/w"}, overlay=None).lock_host == "local"


def test_a_lazy_path_that_nobody_configured_raises_on_comparison_never_answers_false():
    lp = P.lazy_setting_path("news_release_dir")
    with pytest.raises(P.SettingRefused, match="news_release_dir"):
        lp == "/x"  # noqa: B015
    with pytest.raises(P.SettingRefused):
        lp != "/x"  # noqa: B015
    assert (lp == 3) is False                                  # a type mismatch is not a setting


def test_a_configured_lazy_path_compares_by_value(monkeypatch):
    monkeypatch.setenv("AW_NEWS_RELEASE_DIR", "/rel")
    P.reset_settings_cache()
    lp = P.lazy_setting_path("news_release_dir")
    assert lp == "/rel" and lp != "/other"


def test_the_drift_warning_names_the_differing_settings_only(tmp_path):
    repo = _overlay(tmp_path / "impactia.json", {"work_root": "/w", "qpy": "/a"})
    box = _overlay(tmp_path / "operator.json", {"work_root": "/w", "qpy": "/b"})
    warn = P.box_overlay_drift(box=box, repo=repo)
    assert warn and warn.startswith("WARNING:") and "qpy" in warn and "work_root" not in warn
    box.write_text(repo.read_text())
    assert P.box_overlay_drift(box=box, repo=repo) is None      # equal: silent
    assert P.box_overlay_drift(box=tmp_path / "absent.json", repo=repo) is None


def test_python_m_aw_core_paths_prints_the_drift_warning(tmp_path, monkeypatch, capsys):
    box = _overlay(tmp_path / "operator.json", {"work_root": "/not-the-repo-copy"})
    monkeypatch.setenv(P.TEST_LOOKUP_ENV, str(box))
    P.reset_settings_cache()
    repo_file_or_skip("deploy/operator/impactia.json")
    assert P.main([]) == 0
    out = capsys.readouterr().out
    assert "WARNING:" in out and "work_root" in out


def test_python_m_aw_core_paths_is_silent_when_there_is_no_box_file(capsys):
    assert P.main([]) == 0
    assert "WARNING" not in capsys.readouterr().out

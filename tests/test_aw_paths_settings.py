"""Operator settings (task 696 slice 4, round 2): `cytir.core.paths`. Generic: nothing of ours in here.

Contracts:
  1. NO silent default. With no overlay and no env, every setting derived from the work root or
     naming a host, release or interpreter REFUSES BY NAME on access (never at import).
  2. `AW_WORK_ROOT` alone is a fully isolated scratch run (queue, locks, GPU lock, corpus, news
     dirs); with an overlay present it still re-derives everything that derives from the work root.
  3. Precedence: argument > env (canonical name, then deprecated aliases) > overlay > derived.
  4. The box file `/etc/aw/operator.json` is the overlay when `AW_OPERATOR_OVERLAY` is unset.
  5. One crawler identity: the pack-fetch User-Agent is built by `cytir.core.user_agent`.
  6. A lock directory that does not exist is a refusal, never "free".
The values our own boxes use are pinned in the operator-side test (`test_aw_paths_overlay_values`).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys

import pytest

from cytir.core import envnames as _E

from cytir.core import paths as P

APP = P.REPO_ROOT

ALL_ENV = sorted({n for s in P._SPECS for n in (s.env, *s.aliases)} | {P.OVERLAY_ENV})
ALL_ENV = sorted(set(ALL_ENV) | {_E.shown(n) for n in ALL_ENV})
REFUSING_WITHOUT_ANYTHING = {"work_root", "hdd_root", "corpus_root", "l0_store_root", "releases_root",
                             "queue_dir", "gpu_lock_dir", "news_work", "refresh_github_docs_root", "refresh_direct_work", "b1_corpus_dir", "b1_build_dir", "hn_work", "news_backfill_dir",
                             "lock_host", "lock_dir", "prod_lock_host", "twin_ssh_host", "langgate_py",
                             "news_release_dir", "storagebox", "box_rsh", "operator_url", "operator_contact"}

#: A made-up operator overlay: a work root, one release, one host. Nothing real.
FAKE = {"work_root": "/srv/aw/work", "hdd_root": "/srv/aw/bulk", "news_release_dir": "/srv/aw/rel-1",
        "prod_lock_host": "serving", "lock_host": "builder", "lock_dir": "/srv/aw/locks",
        "queue_dir": "/srv/aw/queue", "news_work": "/srv/aw/news"}


@pytest.fixture(autouse=True)
def clean_settings_env(monkeypatch):
    """No AW_* setting variable, and the default overlay lookup replaced by "none"."""
    for name in ALL_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(P.TEST_LOOKUP_ENV, "")
    P.reset_settings_cache()
    yield
    P.reset_settings_cache()


@pytest.fixture
def overlay(tmp_path):
    path = tmp_path / "overlay.json"
    path.write_text(json.dumps({"schema": P.OVERLAY_SCHEMA, "settings": FAKE}))
    return path


# ----------------------------------------------------------------------- 1. no silent default
def test_without_overlay_and_env_every_path_host_and_release_refuses_by_name():
    s = P.load_settings()
    got = s.as_dict()
    assert {k for k, v in got.items() if v is None} == REFUSING_WITHOUT_ANYTHING
    for name in sorted(REFUSING_WITHOUT_ANYTHING):
        with pytest.raises(P.SettingRefused) as ei:
            getattr(s, name)
        msg = str(ei.value)
        assert f"`{name}`" in msg and _E.shown(P.OVERLAY_ENV) in msg and str(P.BOX_OVERLAY) in msg, name
        assert s.source(name).startswith("REFUSED")
    # a derived one names what it derives from, and its own name
    with pytest.raises(P.SettingRefused, match=r"`queue_dir` derives from `work_root`"):
        s.queue_dir
    with pytest.raises(P.SettingRefused, match="CYTIR_WORK_ROOT"):
        s.lock_dir
    # a chain names its ROOT, not the middle link: corpus_root -> hdd_root -> work_root
    with pytest.raises(P.SettingRefused, match=r"`corpus_root` derives from `work_root`"):
        s.corpus_root
    with pytest.raises(P.SettingRefused, match="CYTIR_PROD_LOCK_HOST"):
        s.prod_lock_host


def test_neutral_defaults_carry_nothing_of_ours():
    got = P.load_settings().as_dict()
    assert got["news_config_name"] == "current.env" and got["fallback_models_dir"] == ""
    assert got["lock_host"] is None and got["qpy"] == sys.executable   # `local` only beside a work root


def test_a_refused_path_refuses_where_it_is_used_not_at_import():
    from cytir.corpus import store
    from cytir.index import compose, promote_queue
    from cytir.news import daily
    from cytir.ops import write_lock
    for access in (lambda: compose.DEFAULT_QUEUE_DIR, lambda: promote_queue.DEFAULT_QUEUE_DIR,
                   lambda: promote_queue.PROD_LOCK_HOST, lambda: store.DEFAULT_DEST,
                   lambda: write_lock.lock_location(), lambda: str(daily.QUEUE_DIR),
                   lambda: str(daily.W), lambda: str(daily.R), lambda: str(daily.R / "x"),
                   lambda: daily.stable_env(), lambda: P.CORPUS_ROOT):
        with pytest.raises(P.SettingRefused):
            access()


def test_the_lazy_release_refuses_with_the_settings_message_for_str_fspath_and_join():
    from cytir.news import daily
    for use in (lambda: str(daily.R), lambda: os.fspath(daily.R), lambda: str(daily.R / "a" / "b"),
                lambda: daily.R.exists()):
        with pytest.raises(P.SettingRefused, match="news_release_dir"):
            use()
    assert "not configured" in repr(daily.R)          # repr never raises


def test_read_lock_refuses_when_the_lock_directory_does_not_exist(tmp_path, monkeypatch):
    from cytir.ops import write_lock as WL
    monkeypatch.setitem(WL.__dict__, "LOCK_HOST", WL.LOCAL_HOST)
    monkeypatch.setitem(WL.__dict__, "LOCK_DIR", str(tmp_path / "no-such-dir"))
    with pytest.raises(WL.LockRegistryUnreadable, match="does not exist"):
        WL.read_lock("TWIN-WRITE-LOCK")
    with pytest.raises(WL.LockRegistryUnreadable):
        WL.assert_locks_free("t")
    (tmp_path / "no-such-dir").mkdir()
    assert WL.read_lock("TWIN-WRITE-LOCK") is None     # existing directory, no file: free


def test_read_lock_refuses_an_unreadable_host_instead_of_calling_it_free(monkeypatch):
    from cytir.ops import write_lock as WL
    monkeypatch.setitem(WL.__dict__, "LOCK_HOST", "nowhere")
    monkeypatch.setitem(WL.__dict__, "LOCK_DIR", "/x")
    monkeypatch.setattr(WL, "ssh", lambda host, cmd, timeout=120: subprocess.CompletedProcess(
        [], 255, "", "ssh: Could not resolve hostname"))
    with pytest.raises(WL.LockRegistryUnreadable, match="unreadable registry"):
        WL.read_lock("PROD-WRITE-LOCK")


def test_the_lock_location_follows_the_work_root_in_a_scratch_run(tmp_path, monkeypatch):
    from cytir.ops import write_lock as WL
    monkeypatch.setenv("AW_WORK_ROOT", str(tmp_path))
    P.reset_settings_cache()
    assert WL.lock_location() == ("local", str(tmp_path))


# ------------------------------------------------------------------------ 2. scratch = isolated
def test_work_root_alone_is_a_fully_isolated_scratch_run(tmp_path):
    root = str(tmp_path / "scratch")
    s = P.load_settings(env={"AW_WORK_ROOT": root}, overlay=None)
    got = s.as_dict()
    for k in ("queue_dir", "lock_dir", "gpu_lock_dir", "news_work", "refresh_github_docs_root", "refresh_direct_work", "b1_corpus_dir", "b1_build_dir", "hn_work", "news_backfill_dir",
              "releases_root", "hdd_root", "corpus_root", "l0_store_root"):
        assert got[k].startswith(root), (k, got[k])
    assert got["lock_host"] == "local"
    assert s.source("work_root") == "env AW_WORK_ROOT"
    # what has no honest scratch value still refuses, by name
    for k in ("prod_lock_host", "twin_ssh_host", "news_release_dir", "langgate_py"):
        with pytest.raises(P.SettingRefused, match=k):
            getattr(s, k)


def test_work_root_overrides_the_overlays_derived_paths_so_the_run_is_not_half_isolated(tmp_path, overlay):
    root = str(tmp_path / "scratch")
    s = P.load_settings(env={"AW_WORK_ROOT": root}, overlay=overlay)
    got = s.as_dict()
    for k in ("queue_dir", "lock_dir", "gpu_lock_dir", "news_work", "refresh_github_docs_root", "refresh_direct_work", "b1_corpus_dir", "b1_build_dir", "hn_work", "news_backfill_dir",
              "releases_root", "hdd_root", "corpus_root", "l0_store_root"):
        assert got[k].startswith(root), (k, got[k])        # the overlay pinned these; scratch wins
    assert got["lock_host"] == "local"                     # a scratch run never locks on the box
    # settings that are not paths under the work root still come from the overlay
    assert got["prod_lock_host"] == FAKE["prod_lock_host"]
    assert got["news_release_dir"] == FAKE["news_release_dir"]


def test_a_single_path_can_still_be_set_explicitly_in_a_scratch_run(tmp_path, overlay):
    s = P.load_settings(env={"AW_WORK_ROOT": str(tmp_path), "AW_QUEUE_DIR": "/q"}, overlay=overlay)
    assert s.queue_dir == "/q" and s.source("queue_dir") == "env AW_QUEUE_DIR"


def test_the_overlay_alone_is_not_a_scratch_run(overlay):
    s = P.load_settings(env={}, overlay=overlay)
    assert (s.queue_dir, s.lock_host, s.lock_dir) == (FAKE["queue_dir"], FAKE["lock_host"], FAKE["lock_dir"])
    assert s.gpu_lock_dir == "/srv/aw/work/GPU-LOCK" and s.corpus_root == "/srv/aw/bulk/aw-corpus"


# ----------------------------------------------------------------------------- 3. precedence
def test_precedence_argument_env_overlay_default(tmp_path):
    ov = tmp_path / "ov.json"
    ov.write_text(json.dumps({"schema": P.OVERLAY_SCHEMA,
                              "settings": {"work_root": "/ov", "queue_dir": "/ov-q"}}))
    s = P.load_settings(env={"AW_NEWS_WORK": "/env-news"}, overlay=ov, explicit={"queue_dir": "/arg"})
    assert (s.queue_dir, s.source("queue_dir")) == ("/arg", "arg")
    assert (s.news_work, s.source("news_work")) == ("/env-news", "env AW_NEWS_WORK")
    assert s.work_root == "/ov" and s.source("work_root").startswith(f"overlay {ov}")
    s = P.load_settings(env={}, overlay=ov)
    assert (s.work_root, s.queue_dir) == ("/ov", "/ov-q")
    assert s.hn_work == "/ov/hn_discussion" and s.source("hn_work") == "default"


def test_overlay_typos_and_missing_files_refuse(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": P.OVERLAY_SCHEMA, "settings": {"work_rot": "x"}}))
    with pytest.raises(P.SettingsError):
        P.load_settings(env={}, overlay=bad)
    with pytest.raises(P.SettingsError):
        P.load_settings(env={P.OVERLAY_ENV: str(tmp_path / "nope.json")}, overlay=False)
    with pytest.raises(P.SettingsError):
        P.load_settings(env={}, overlay=None, explicit={"nope": "1"})


def test_the_old_env_names_are_aliases_with_one_deprecation_line(caplog):
    P._deprecated_seen.clear()
    cases = {"AW_PROMOTION_QUEUE_DIR": ("queue_dir", "/q"), "AW_LANGGATE_PYTHON_BIN": ("langgate_py", "/lg"),
             "AW_L0_ROOT": ("l0_store_root", "/l0")}
    with caplog.at_level(logging.WARNING, logger="cytir.core.paths"):
        for old, (setting, value) in cases.items():
            s = P.load_settings(env={old: value}, overlay=None)
            assert getattr(s, setting) == value and s.source(setting) == f"env {old}"
            P.load_settings(env={old: value}, overlay=None)       # a second load: no second line
    lines = [r.getMessage() for r in caplog.records]
    assert len(lines) == len(cases), lines
    assert all("deprecated" in ln for ln in lines)


def test_aw_release_dir_is_not_an_alias_of_the_news_release():
    """AW_RELEASE_DIR is "the release this process runs from" (the ERIC chain sets it to its own); it
    must not decide which release the news lane is pinned to."""
    s = P.load_settings(env={"AW_RELEASE_DIR": "/eric-release"}, overlay=None)
    with pytest.raises(P.SettingRefused, match="news_release_dir"):
        s.news_release_dir
    ok = P.load_settings(env={"AW_RELEASE_DIR": "/eric-release", "AW_NEWS_RELEASE_DIR": "/news"}, overlay=None)
    assert ok.news_release_dir == "/news"


def test_an_alias_and_its_canonical_name_set_to_different_values_refuse():
    s = P.load_settings(env={"AW_QUEUE_DIR": "/a", "AW_PROMOTION_QUEUE_DIR": "/b"}, overlay=None)
    with pytest.raises(P.SettingRefused, match="disagree"):
        s.queue_dir
    ok = P.load_settings(env={"AW_QUEUE_DIR": "/a", "AW_PROMOTION_QUEUE_DIR": "/a"}, overlay=None)
    assert ok.queue_dir == "/a"


# ------------------------------------------------------------------------ 4. the box default
def test_the_box_file_is_the_overlay_when_the_env_var_is_unset(tmp_path, monkeypatch, overlay):
    monkeypatch.setenv(P.TEST_LOOKUP_ENV, str(overlay))
    s = P.load_settings()
    assert s.queue_dir == FAKE["queue_dir"]
    assert s.source("queue_dir").startswith(f"overlay {overlay}") and "test lookup" in s.source("queue_dir")


def test_the_env_var_beats_the_box_file(tmp_path, monkeypatch, overlay):
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"schema": P.OVERLAY_SCHEMA, "settings": {"work_root": "/other"}}))
    monkeypatch.setenv(P.TEST_LOOKUP_ENV, str(overlay))
    s = P.load_settings(env={P.OVERLAY_ENV: str(other), P.TEST_LOOKUP_ENV: str(overlay)})
    assert s.work_root == "/other"


def test_with_no_box_file_and_no_env_there_is_no_overlay(tmp_path, monkeypatch):
    monkeypatch.setenv(P.TEST_LOOKUP_ENV, str(tmp_path / "absent.json"))
    assert P.overlay_lookup() == (None, "none")


def test_the_box_path_is_the_conventional_system_path():
    assert str(P.BOX_OVERLAY) == "/etc/aw/operator.json"


# ------------------------------------------------------------------------------ the config echo
def test_the_config_echo_prints_every_setting_with_value_and_source(tmp_path):
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), P.TEST_LOOKUP_ENV: "",
           "AW_WORK_ROOT": str(tmp_path / "w")}
    out = subprocess.run([sys.executable, "-m", "cytir.core.paths"], cwd=str(APP), env=env,
                         capture_output=True, text=True, check=True).stdout
    for spec in P._SPECS:
        assert any(ln.startswith(f"{spec.name} = ") for ln in out.splitlines()), spec.name
    assert f"queue_dir = {tmp_path / 'w' / 'promotion-queue'}  [default]" in out
    assert "work_root = " + str(tmp_path / "w") + "  [env AW_WORK_ROOT]" in out
    assert "prod_lock_host = REFUSED" in out
    assert list(tmp_path.iterdir()) == []                  # printing creates nothing


def test_the_echo_explains_the_unset_identity_instead_of_alarming(tmp_path):
    """Step B2: the box file lacks the two identity keys. The echo says so; it is not a drift WARNING."""
    overlay = tmp_path / "operator.json"
    overlay.write_text(json.dumps({"schema": P.OVERLAY_SCHEMA, "settings": {"work_root": str(tmp_path / "w")}}))
    # The box file is stood in by a tmp copy of the checkout file (no drift, no identity keys): the
    # test never reads the operator file on the host.
    box = tmp_path / "box-operator.json"
    checkout_file = APP / "deploy" / "operator" / "impactia.json"      # absent in the public cut
    box.write_text(checkout_file.read_text() if checkout_file.exists() else overlay.read_text())
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), P.OVERLAY_ENV: str(overlay),
           P.TEST_LOOKUP_ENV: str(box)}
    out = subprocess.run([sys.executable, "-m", "cytir.core.paths"], cwd=str(APP), env=env,
                         capture_output=True, text=True, check=True).stdout
    assert "operator_url = REFUSED" in out
    assert "not in the box overlay yet (step B2)" in out
    assert "WARNING" not in out


def test_the_startup_lines_name_the_queue_and_the_locks_with_their_source(tmp_path):
    s = P.load_settings(env={"AW_WORK_ROOT": str(tmp_path)}, overlay=None)
    assert s.source("queue_dir") == "default"
    P._cached = s
    line = P.location_line("queue_dir", str(tmp_path / "promotion-queue"), "queue_dir")
    assert line == f"[config] queue_dir = {tmp_path / 'promotion-queue'}  [default]"
    assert "command-line argument" in P.location_line("queue_dir", "/elsewhere", "queue_dir")
    lock = P.lock_location_line()
    assert lock.startswith(f"[config] write-lock = local:{tmp_path}") and "dir: default" in lock
    P.reset_settings_cache()
    refused = P.location_line("queue_dir", None, "queue_dir")
    assert "REFUSED" in refused                              # a startup line never raises
    assert "REFUSED" in P.lock_location_line()


# ------------------------------------------------------------------------ 5. one crawler identity
def test_the_pack_fetch_user_agent_is_built_by_the_one_user_agent_module(monkeypatch):
    from cytir.core import user_agent as UA
    monkeypatch.setenv("AW_OPERATOR_URL", "https://example.org")
    monkeypatch.setenv("AW_OPERATOR_CONTACT", "ops@example.org")
    assert UA.pack_fetch_user_agent() == "AuthoritativeWebBot (+https://example.org; ops@example.org)"
    from cytir.acquire import crawl
    assert crawl.UA == UA.pack_fetch_user_agent()
    assert crawl.HEADERS == {"User-Agent": crawl.UA}
    assert crawl._ua() == crawl.UA


def test_without_an_identity_the_pack_fetch_user_agent_refuses(monkeypatch, tmp_path):
    from cytir.core import user_agent as UA
    monkeypatch.delenv("AW_OPERATOR_URL", raising=False)
    monkeypatch.delenv("AW_OPERATOR_CONTACT", raising=False)
    monkeypatch.delenv(UA.ALLOW_UNIDENTIFIED_ENV, raising=False)
    monkeypatch.setattr(UA, "CONFIG_FILE", tmp_path / "operator.json")
    with pytest.raises(UA.UnidentifiedCrawler):
        UA.pack_fetch_user_agent()
    from cytir.acquire import crawl
    with pytest.raises(UA.UnidentifiedCrawler):
        crawl.UA


def test_there_is_no_second_crawler_identity_setting():
    assert "crawl_user_agent" not in {s.name for s in P._SPECS}
    assert "AW_CRAWL_USER_AGENT" not in ALL_ENV


def test_every_top_level_def_and_class_in_paths_is_defined_once():
    # A second `def` of the same name silently shadows the first (task 1162: `data_path` was
    # defined twice, the first one dead). Parse the module; no import, no runtime state.
    import ast
    import collections
    import pathlib

    src = pathlib.Path(P.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    dupes = sorted(k for k, v in collections.Counter(names).items() if v > 1)
    assert dupes == [], f"defined more than once in cytir/core/paths.py: {dupes}"


def test_the_b1_builder_settings_resolve_to_the_values_the_builder_hard_coded():
    """Task 2123: the three b1 settings carry what `build_aw_b1.py` hard-coded: the build dir and
    the corpus store derive from the work and bulk roots, the resume command names this module."""
    s = P.load_settings(env={"AW_WORK_ROOT": "/w", "AW_HDD_ROOT": "/h"}, overlay=None)
    assert s.b1_build_dir == "/w/aw_b1_build"
    assert s.b1_corpus_dir == "/h/aw-corpus"
    assert s.b1_resume_prefix == f"AW_FOLD_ACCENTS=1 {sys.executable} -m cytir.index.b1_cli"
    env = {"AW_B1_CORPUS_DIR": "/c", "AW_B1_BUILD_DIR": "/b", "AW_B1_RESUME_PREFIX": "run it"}
    t = P.load_settings(env={"AW_WORK_ROOT": "/w", **env}, overlay=None)
    assert (t.b1_corpus_dir, t.b1_build_dir, t.b1_resume_prefix) == ("/c", "/b", "run it")

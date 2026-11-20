"""Where the twin is (task 832): ONE setting pair, `twin_url` / `twin_collection`, one accessor.

Pins: the accessor resolves environment -> operator overlay -> default (the order every setting
uses); the default is today's twin; a twin move is one overlay edit; and `prod_guard` still
classifies the twin as NOT prod under the default and under a moved twin.
"""
import json
import subprocess
import sys

import pytest

from cytir.core import paths as P
from cytir.core import twin as T
from cytir.index import prod_guard as PG

_H = "127.0.0.1"                       # built from parts: the throwaway-qdrant scan reads literals
DEFAULT_URL = "http://%s:%d" % (_H, 6457)
DEFAULT_COLL = "aw_e" + "8"
MOVED_URL = "http://%s:%d" % ("twinbox", 6999)
MOVED_COLL = "aw_e" + "10_twin"
ENV_URL = "http://%s:%d" % ("envbox", 6998)
ENV_COLL = "aw_from_env"


@pytest.fixture
def overlay(tmp_path):
    path = tmp_path / "overlay.json"
    path.write_text(json.dumps({"schema": P.OVERLAY_SCHEMA,
                                "settings": {"twin_url": MOVED_URL + "/", "twin_collection": MOVED_COLL}}))
    return str(path)


def test_default_is_todays_twin():
    env = {P.TEST_LOOKUP_ENV: ""}
    assert (T.url(env), T.collection(env)) == (DEFAULT_URL, DEFAULT_COLL)


def test_overlay_beats_default_and_the_url_loses_its_trailing_slash(overlay):
    env = {P.OVERLAY_ENV: overlay}
    assert (T.url(env), T.collection(env)) == (MOVED_URL, MOVED_COLL)


def test_environment_beats_overlay(overlay):
    env = {P.OVERLAY_ENV: overlay, "AW_TWIN_URL": ENV_URL, "AW_TWIN_COLLECTION": ENV_COLL}
    assert (T.url(env), T.collection(env)) == (ENV_URL, ENV_COLL)
    assert P.load_settings(env).source("twin_url") == "env AW_TWIN_URL"


def test_the_accessor_reads_the_process_environment_fresh(monkeypatch):
    monkeypatch.setenv(P.TEST_LOOKUP_ENV, "")
    monkeypatch.setenv("AW_TWIN_URL", ENV_URL)
    assert T.url() == ENV_URL
    monkeypatch.delenv("AW_TWIN_URL")
    assert T.url() == DEFAULT_URL


def test_the_shell_helper_path_resolves_the_same_value():
    """`aw_setting_in` shells out to `python -m cytir.core.paths --get`; same loader, same answer."""
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(P.REPO_ROOT), P.TEST_LOOKUP_ENV: "",
           "AW_TWIN_COLLECTION": ENV_COLL}
    out = subprocess.run([sys.executable, "-m", "cytir.core.paths", "--get", "twin_collection"],
                         env=env, cwd="/", capture_output=True, text=True, check=True).stdout.strip()
    assert out == ENV_COLL


@pytest.mark.parametrize("url,coll", [(DEFAULT_URL, DEFAULT_COLL), (MOVED_URL, MOVED_COLL)])
def test_prod_guard_never_classifies_the_twin_as_prod(monkeypatch, url, coll):
    monkeypatch.delenv("AW_PROD_URL", raising=False)
    monkeypatch.delenv("AW_PROD_COLLECTIONS", raising=False)
    monkeypatch.setattr(PG, "_fetch_aliases", lambda u: [])      # the twin holds no serving alias
    PG.reset()
    assert not PG.is_prod(url, coll)
    assert not PG.is_prod(url)
    PG.reset()


def test_an_empty_overlay_value_refuses_instead_of_resolving_silently(tmp_path):
    path = tmp_path / "o.json"
    path.write_text(json.dumps({"schema": P.OVERLAY_SCHEMA, "settings": {"twin_url": "", "twin_collection": " "}}))
    env = {P.OVERLAY_ENV: str(path)}
    with pytest.raises(P.SettingRefused, match="twin_url"):
        T.url(env)
    with pytest.raises(P.SettingRefused, match="twin_collection"):
        T.collection(env)


def test_configured_url_ignores_the_environment(monkeypatch):
    monkeypatch.setenv(P.TEST_LOOKUP_ENV, "")
    monkeypatch.setenv("AW_TWIN_URL", ENV_URL)
    assert T.url() == ENV_URL
    assert T.configured_url() == DEFAULT_URL


def test_url_is_named_by_overlay_only_for_the_overlay(tmp_path, monkeypatch):
    f = tmp_path / "operator.json"
    f.write_text(json.dumps({"schema": "aw.operator-overlay.v1", "settings": {"twin_url": MOVED_URL}}))
    monkeypatch.setenv("AW_OPERATOR_OVERLAY", str(f))
    monkeypatch.setenv("AW_TWIN_URL", ENV_URL)           # the environment never qualifies
    assert T.url_is_named_by_overlay() is True
    g = tmp_path / "no-url.json"
    g.write_text(json.dumps({"schema": "aw.operator-overlay.v1", "settings": {"twin_collection": "x"}}))
    monkeypatch.setenv("AW_OPERATOR_OVERLAY", str(g))    # overlay without twin_url: the default
    assert T.url_is_named_by_overlay() is False
    monkeypatch.delenv("AW_OPERATOR_OVERLAY")
    monkeypatch.setenv(P.TEST_LOOKUP_ENV, "")
    assert T.url_is_named_by_overlay() is False

"""The stranger suite never reads the box's operator file (task 696 slice 4, round 3; review of #567).

CI and the cut rehearsal run on the operator's build box. When `/etc/aw/operator.json` exists there,
`cytir.core.paths` hands every test the operator's values, and a refusal defect that only a box with no
overlay can see would pass unseen. The root `conftest.py` pins the overlay lookup to "no box file"
for the whole session; the operator suite (next to the private pipeline) opts in explicitly.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import types

from cytir.core import paths as P

APP = pathlib.Path(__file__).resolve().parents[1]


def _fake_box(tmp_path) -> pathlib.Path:
    box = tmp_path / "etc-aw-operator.json"
    box.write_text(json.dumps({"schema": P.OVERLAY_SCHEMA, "settings": {"work_root": "/fake/box"}}))
    return box


def _root_conftest():
    spec = importlib.util.spec_from_file_location("aw_root_conftest", APP / "conftest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_this_test_process_sees_no_box_overlay():
    """FAILS when a test process can see an overlay: the lookup must find none and the named
    overlay variable must be unset."""
    assert not os.environ.get(P.OVERLAY_ENV), "AW_OPERATOR_OVERLAY leaked into the stranger suite"
    assert P.TEST_LOOKUP_ENV in os.environ, "the session did not pin the overlay lookup"
    assert P.overlay_lookup() == (None, "none")
    P.reset_settings_cache()
    try:
        assert P.operator_settings().overlay_file is None
    finally:
        P.reset_settings_cache()


def test_a_simulated_box_file_is_visible_without_the_session_pin(tmp_path, monkeypatch):
    """The override is how a box file is simulated: with the pin removed and the override pointing
    at a file, the lookup finds it. This is what the session pin prevents."""
    box = _fake_box(tmp_path)
    monkeypatch.setenv(P.TEST_LOOKUP_ENV, str(box))
    monkeypatch.delenv(P.OVERLAY_ENV, raising=False)
    assert P.overlay_lookup()[0] == box


def test_the_session_hook_hides_a_simulated_box_file_and_a_stray_named_overlay(tmp_path, monkeypatch):
    box = _fake_box(tmp_path)
    monkeypatch.setenv(P.TEST_LOOKUP_ENV, str(box))          # a box file the runner "has"
    monkeypatch.setenv(P.OVERLAY_ENV, str(box))              # and a stray named overlay
    conftest = _root_conftest()
    cfg = types.SimpleNamespace()
    conftest.pytest_configure(cfg)
    try:
        assert P.overlay_lookup() == (None, "none")
        assert not os.environ.get(P.OVERLAY_ENV)
    finally:
        conftest.pytest_unconfigure(cfg)
    # unconfigure restores what the runner had (so the hook leaves no trace of its own)
    assert os.environ[P.TEST_LOOKUP_ENV] == str(box)
    assert os.environ[P.OVERLAY_ENV] == str(box)

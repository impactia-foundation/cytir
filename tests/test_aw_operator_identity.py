"""The crawler identity can come from the operator overlay (task 696 slice 4, round 3; review of #567).

Order: the environment (`AW_OPERATOR_URL` / `AW_OPERATOR_CONTACT`), then the checkout's own
`operator.json`, then the overlay's `operator_url` / `operator_contact`. Every entry point on a box
then has ONE identity without a per-unit environment line. A stranger has no overlay and still fails
closed: there is no default, and `user_agent()` refuses.
"""
from __future__ import annotations

import json

import pytest

from cytir.core import paths as P
from cytir.core import user_agent as UA


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    for name in ("AW_OPERATOR_URL", "AW_OPERATOR_CONTACT", UA.ALLOW_UNIDENTIFIED_ENV, P.OVERLAY_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(P.TEST_LOOKUP_ENV, "")
    monkeypatch.setattr(UA, "CONFIG_FILE", tmp_path / "operator.json")
    monkeypatch.setattr(UA, "_warned", False)
    P.reset_settings_cache()
    yield
    P.reset_settings_cache()


def _overlay(tmp_path, **settings):
    path = tmp_path / "overlay.json"
    path.write_text(json.dumps({"schema": P.OVERLAY_SCHEMA, "settings": settings}))
    return path


ID = {"operator_url": "https://op.example", "operator_contact": "ops@op.example"}


def test_a_stranger_with_no_overlay_fails_closed():
    with pytest.raises(UA.UnidentifiedCrawler):
        UA.user_agent()
    assert UA.is_configured() is False


def test_the_overlay_supplies_the_identity_when_nothing_else_does(monkeypatch, tmp_path):
    monkeypatch.setenv(P.OVERLAY_ENV, str(_overlay(tmp_path, **ID)))
    assert UA.user_agent() == "authoritative-web/0.1 (+https://op.example; ops@op.example)"
    assert UA.pack_fetch_user_agent() == "AuthoritativeWebBot (+https://op.example; ops@op.example)"
    assert UA.is_configured()


def test_the_box_file_supplies_it_too(monkeypatch, tmp_path):
    monkeypatch.setenv(P.TEST_LOOKUP_ENV, str(_overlay(tmp_path, **ID)))      # the box file, emulated
    assert UA.pack_fetch_user_agent() == "AuthoritativeWebBot (+https://op.example; ops@op.example)"


def test_the_environment_wins_value_by_value(monkeypatch, tmp_path):
    monkeypatch.setenv(P.OVERLAY_ENV, str(_overlay(tmp_path, **ID)))
    monkeypatch.setenv("AW_OPERATOR_URL", "https://env.example")
    assert UA.pack_fetch_user_agent() == "AuthoritativeWebBot (+https://env.example; ops@op.example)"
    monkeypatch.setenv("AW_OPERATOR_CONTACT", "env@env.example")
    assert UA.pack_fetch_user_agent() == "AuthoritativeWebBot (+https://env.example; env@env.example)"


def test_the_checkout_operator_json_beats_the_overlay(monkeypatch, tmp_path):
    (tmp_path / "operator.json").write_text(json.dumps({"url": "https://file.example", "contact": "f@file.example"}))
    monkeypatch.setenv(P.OVERLAY_ENV, str(_overlay(tmp_path, **ID)))
    assert UA.pack_fetch_user_agent() == "AuthoritativeWebBot (+https://file.example; f@file.example)"


def test_half_an_overlay_identity_is_not_an_identity(monkeypatch, tmp_path):
    monkeypatch.setenv(P.OVERLAY_ENV, str(_overlay(tmp_path, operator_url="https://op.example")))
    assert UA.is_configured() is False
    with pytest.raises(UA.UnidentifiedCrawler):
        UA.user_agent()


def test_a_malformed_overlay_never_falls_back_to_a_placeholder(monkeypatch, tmp_path):
    bad = tmp_path / "overlay.json"
    bad.write_text("{not json")
    monkeypatch.setenv(P.OVERLAY_ENV, str(bad))
    with pytest.raises(P.SettingsError):
        UA.user_agent()


def test_the_identity_setting_refuses_by_name_when_unconfigured():
    with pytest.raises(P.SettingRefused, match="operator_url"):
        P.operator_settings().operator_url

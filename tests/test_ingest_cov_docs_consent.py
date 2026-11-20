#!/usr/bin/env python3
"""Consent-boundary tests for the shared Task-351 fetch engine."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cytir.sources.cov import engine as COV  # noqa: E402


class Rules:
    def __init__(self, allowed, delay=0.0):
        self._allowed = allowed
        self.delay = delay

    def allowed(self, path):
        return self._allowed(path)


def test_fetch_origin_hosts_includes_secondary_origins_once():
    cfg = {"host": "www.suva.ch", "ok_hosts": [
        "www.suva.ch", "suva-p-001.sitecorecontenthub.cloud"]}
    assert COV.fetch_origin_hosts(cfg) == [
        "www.suva.ch", "suva-p-001.sitecorecontenthub.cloud"]


def test_fetch_origin_hosts_rejects_urls_disguised_as_hosts():
    with pytest.raises(ValueError, match="invalid fetch origin"):
        COV.fetch_origin_hosts(
            {"host": "x.test", "ok_hosts": ["https://other.test/path"]})


def test_exact_url_uses_secondary_origins_own_policy():
    robots = {
        "www.suva.ch": Rules(lambda _path: True),
        "suva-p-001.sitecorecontenthub.cloud": Rules(
            lambda path: path.startswith("/api/public/")),
    }
    assert COV.robots_allow_exact_url(
        robots,
        "https://suva-p-001.sitecorecontenthub.cloud/api/public/content/abc?v=1")
    assert not COV.robots_allow_exact_url(
        robots,
        "https://suva-p-001.sitecorecontenthub.cloud/api/private/content/abc")


def test_exact_url_fails_closed_for_unresolved_or_cross_host_origin():
    robots = {"www.suva.ch": Rules(lambda _path: True)}
    assert not COV.robots_allow_exact_url(
        robots,
        "https://suva-p-001.sitecorecontenthub.cloud/api/public/content/abc")
    assert not COV.robots_allow_exact_url(robots, "not-a-url")


def test_route_preflight_resolves_primary_and_secondary_before_activation(
        monkeypatch):
    calls = []
    rules = {
        "www.suva.ch": Rules(lambda _path: True, delay=1.0),
        "suva-p-001.sitecorecontenthub.cloud": Rules(
            lambda path: path.startswith("/api/public/"), delay=3.0),
    }
    monkeypatch.setattr(COV, "fetch_robots",
                        lambda _client, host: calls.append(host) or rules[host])
    monkeypatch.setattr(COV.time, "sleep", lambda _seconds: None)
    resolved = {}
    unavailable, delay = COV.resolve_fetch_origin_robots(
        object(),
        {"host": "www.suva.ch", "delay": 2.0,
         "ok_hosts": ["www.suva.ch",
                      "suva-p-001.sitecorecontenthub.cloud"]},
        resolved, lambda _message: None)
    assert calls == ["www.suva.ch", "suva-p-001.sitecorecontenthub.cloud"]
    assert unavailable == []
    assert delay == 3.0
    assert set(resolved) == set(rules)


def test_route_preflight_holds_when_secondary_policy_is_unresolved(monkeypatch):
    monkeypatch.setattr(
        COV, "fetch_robots",
        lambda _client, host: Rules(lambda _path: True)
        if host == "primary.test" else None)
    monkeypatch.setattr(COV.time, "sleep", lambda _seconds: None)
    unavailable, _delay = COV.resolve_fetch_origin_robots(
        object(), {"host": "primary.test", "delay": 2.0,
                   "ok_hosts": ["secondary.test"]}, {}, lambda _message: None)
    assert unavailable == ["secondary.test"]

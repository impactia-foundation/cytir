"""The refresh-lane summary-mail recipient has no code default (task 2988): AW_MAILTO, else the
operator contact, else none."""
from __future__ import annotations

import re
from pathlib import Path

from cytir.core import user_agent
from cytir.sources.refresh_direct import lane_run


def test_env_wins_over_operator(monkeypatch):
    monkeypatch.setattr(user_agent, "operator", lambda: ("u", "op@example.test", True))
    assert lane_run._alert_recipient({"AW_MAILTO": " me@example.test "}, print) == "me@example.test"


def test_operator_contact_used_when_env_absent(monkeypatch):
    monkeypatch.setattr(user_agent, "operator", lambda: ("u", "op@example.test", True))
    assert lane_run._alert_recipient({}, print) == "op@example.test"
    assert lane_run._alert_recipient({"AW_MAILTO": ""}, print) == "op@example.test"


def test_unconfigured_operator_means_no_recipient(monkeypatch):
    monkeypatch.setattr(user_agent, "operator", lambda: ("u", "placeholder", False))
    assert lane_run._alert_recipient({}, print) == ""


def test_module_carries_no_address_literal():
    src = Path(lane_run.__file__).read_text(encoding="utf-8")
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", src)


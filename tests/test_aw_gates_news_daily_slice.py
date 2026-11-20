"""`cyt gate news-daily-slice`: the ADR-043 daily-gate slice builder (task 2056).

The slice file is sealed by the daily-gate contract's sha256, so the builder's `--check` must hold
against it byte for byte in everything but the `generated_by` label.
"""
from __future__ import annotations
from cytir import cli
from cytir.gates import news_daily_slice as builder


def test_the_cli_routes_the_verb(monkeypatch):
    seen = {}
    monkeypatch.setattr(builder, "main", lambda argv, prog=None: seen.update(argv=argv, prog=prog) or 0)
    assert cli._gate_main(["news-daily-slice", "--check"]) == 0
    assert seen == {"argv": ["--check"], "prog": "cyt gate news-daily-slice"}

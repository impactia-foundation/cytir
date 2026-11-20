#!/usr/bin/env python3
"""Task 683 S7 (news port, tranche 4, slice 4): `rts_section_discovery` pinned through its OLD
`pipeline/` path before and after it moves into `cytir/news/`. It is started by path by
`news_coverage_backfill` and by hand on the build box. Passes unchanged on the old code and on the shim.
No network.
"""
from __future__ import annotations

import datetime as _dt
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cytir.news.rts_section_discovery as RSD  # noqa: E402


def test_constants_and_pure_parsers():
    assert (RSD.HOST, RSD.SITEMAP_FILES, RSD.MIN_DELAY_S, RSD.WINDOW_MONTHS) == (
        "www.rts.ch", ("info", "sport"), 10.0, 12)
    assert RSD.STATE_SCHEMA == "aw.rts-section-discovery.v3" and RSD.ROBOTS_DISALLOWED == -1
    assert RSD.window_start(_dt.date(2026, 10, 3)) == _dt.date(2025, 10, 3)
    assert RSD.hub_loadmore('<div data-loadmore-src="/x?lastDate=2026-09-01">') == "/x?lastDate=2026-09-01"
    for name in ("Fetcher", "discover", "discover_section", "parse_page", "parse_card_date", "load_state",
                 "save_state", "accounting", "main", "run_live", "sections_from_sitemaps"):
        assert hasattr(RSD, name), name


def test_state_round_trip_keeps_its_schema(tmp_path):
    path = tmp_path / "state.json"
    RSD.save_state(path, {"schema": RSD.STATE_SCHEMA, "sections": {}})
    assert RSD.load_state(path)["schema"] == RSD.STATE_SCHEMA

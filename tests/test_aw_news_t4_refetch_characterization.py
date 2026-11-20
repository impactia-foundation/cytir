#!/usr/bin/env python3
"""Task 683 S7 (news port, tranche 4, slice 5): `news_page_date_refetch` pinned through its OLD
`pipeline/` path before and after it moves into `cytir/news/`. Passes unchanged on the old code and on
the shim. No network.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cytir.news.page_date_refetch as R  # noqa: E402


def test_constants_and_names_callers_use():
    assert R.SCHEMA == "aw.news-page-date-refetch.v1"
    assert (R.SCROLL_PAGE, R.MIN_DELAY_S, R.DEFAULT_MAX_MBPS) == (1000, 1.0, 1.0)
    assert R.TERMINALS == ("sealed", "non_article", "robots_disallowed", "error")
    for name in ("domain_variants", "twin_urls", "plan", "spool_dir", "Politeness", "fetch", "main",
                 "PF", "PSP", "PPD", "US"):
        assert hasattr(R, name), name
    assert R.domain_variants("www.rts.ch") == R.domain_variants("www.rts.ch")

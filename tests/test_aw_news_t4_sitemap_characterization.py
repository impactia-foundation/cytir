#!/usr/bin/env python3
"""Task 683 S7 (news port, tranche 4, slice 3): `sitemap_ingest` pinned through its OLD `pipeline/` path
before and after it moves into `cytir/acquire/`, plus the clamp check `news_coverage_backfill` runs against
a pinned release. Passes unchanged on the old code and on the shim. No network.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import os
import pathlib
import re
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cytir.acquire import sitemap_ingest as SI  # noqa: E402
from cytir.news import coverage_backfill as NCB  # noqa: E402


def test_module_level_names_and_pure_functions():
    assert (SI.SAVE_EVERY, SI.MIN_FREE_BYTES) == (25, 700 * 1024 * 1024)
    assert SI.ROBOT_AGENT == "authoritative-web" and SI.UA == SI.di.UA
    assert SI.NEWS_DOCUMENT_SCHEMA == "aw.news-lane-document.v1"
    url = "https://www.rts.ch/info/x.html"
    assert SI.news_document_id(url) == hashlib.sha256(f"news-document\0{url}".encode()).hexdigest()
    assert SI.child_sitemap_end("https://h/sitemap-2026.xml?mm=09&dd=30") == _dt.date(2026, 9, 30)
    assert SI.child_sitemap_end("https://h/2026_12.xml") == _dt.date(2026, 12, 31)
    assert SI.child_sitemap_end("https://h/other.xml") is None
    assert SI.window_cutoff_date(_dt.date(2026, 3, 31), 1) == _dt.date(2026, 2, 28)
    today = _dt.date(2026, 10, 3)
    assert SI.page_date_clamp_day(_dt.date(2026, 9, 1), today=today) == today
    assert SI.page_date_clamp_day(_dt.date(2026, 10, 9), today=today) == _dt.date(2026, 10, 9)
    for name in ("process_host", "cmd_fetch", "main", "select_recent", "fetch_page", "load_robots",
                 "discover_sitemap_urls", "TokenBucket", "HostResult", "di", "cr", "ccx", "l0_store"):
        assert hasattr(SI, name), name


_PROG_SITEMAP = re.compile(r"usage: (\S+ -m cytir\.acquire\.sitemap_ingest|sitemap_ingest\.py) ")


def test_prog_regex_accepts_both_argparse_spellings():
    assert _PROG_SITEMAP.match("usage: python -m cytir.acquire.sitemap_ingest [-h] ...")
    assert _PROG_SITEMAP.match("usage: sitemap_ingest.py [-h] ...")
    assert not _PROG_SITEMAP.match("usage: pipeline/sitemap_ingest.py [-h] ...")


def test_script_start_from_another_directory_with_no_pythonpath_keeps_its_prog_name(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(HERE.parent)        # the package root (the module runs by -m, no shim to put it on the path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    done = subprocess.run([sys.executable, "-m", "cytir.acquire.sitemap_ingest", "--help"], cwd=tmp_path,
                          env=env, capture_output=True, text=True, timeout=180)
    assert done.returncode == 0, done.stderr[-500:]
    # task 1797: Python 3.14 argparse prints `python -m cytir.acquire.sitemap_ingest`; CI Python prints the basename.
    assert _PROG_SITEMAP.match(done.stdout[:120]), done.stdout[:120]


def _release(root: pathlib.Path, rel: str, text: str) -> pathlib.Path:
    f = root / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)
    return root


MARKED = "def page_date_clamp_day(now_ref, today=None):\n    pass\nx = f(now_ref=page_date_clamp_day(a))\n"


def test_clamp_check_reads_only_the_package_layout(tmp_path):
    """Only `cytir/acquire/sitemap_ingest.py` counts; a release holding just the pre-move file is refused."""
    new = _release(tmp_path / "new", "cytir/acquire/sitemap_ingest.py", MARKED)
    assert NCB.release_supports_page_date_clamp(new) is True
    NCB.require_page_date_clamp_release(new)
    old = _release(tmp_path / "old", "pipeline/sitemap_ingest.py", MARKED)
    assert NCB.release_supports_page_date_clamp(old) is False
    with pytest.raises(SystemExit):
        NCB.require_page_date_clamp_release(old)


def test_clamp_check_refuses_a_release_without_the_clamp(tmp_path):
    bare = _release(tmp_path / "bare", "cytir/acquire/sitemap_ingest.py", "x = 1\n")
    assert NCB.release_supports_page_date_clamp(bare) is False
    with pytest.raises(SystemExit):
        NCB.require_page_date_clamp_release(bare)
    assert NCB.release_supports_page_date_clamp(tmp_path / "empty") is False


def test_this_checkouts_own_clamp_check_passes():
    """A release cut from this tree must pass (the trap: a shim holds no marks)."""
    assert NCB.release_supports_page_date_clamp(HERE.parent) is True

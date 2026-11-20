#!/usr/bin/env python3
"""Task 683 S7 (news port, tranche 4, slice 2): `discovery_ingest` (73 callers) pinned through its OLD
`pipeline/` path before and after it moves into `cytir/acquire/`. Passes unchanged on the old code and on
the shim. The 60-test `test_discovery_ingest.py` already pins the fetch path; this pins the surface the
callers and the by-path start rely on. No network.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cytir.acquire import discovery_ingest as di  # noqa: E402


def _run(args, tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(HERE.parent)        # the package root (the module runs by -m, no shim to put it on the path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, "-m", "cytir.acquire.discovery_ingest", *args], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=180)


def test_module_level_names_callers_import():
    """The names sitemap_ingest, cc_ingest and the 73 callers take from the module."""
    assert di.CLASS_TRUST == {"gov": 0.85, "health": 0.82, "edu": 0.8, "ref": 0.78,
                              "tech": 0.75, "news": 0.6, "other": 0.55, "commercial": 0.5}
    assert di.CLASS_SOURCE["health"] == "reference" and di.CLASS_SOURCE["gov"] == "official"
    assert di.DEFAULT_CRAWL == "CC-MAIN-2026-25" and di._HTML_CAP == 1_500_000
    assert (di.SELF_SIGNAL, di.ENDORSED) == ("self-signal", "endorsed")
    for name in ("ccx", "fp", "process_html", "reg_domain", "outlinks_column_enabled",
                 "resolve_trust", "CdxError", "WarcRangeFetchError", "SmokeCheckFailed", "smoke_check",
                 "process_domain", "fetch_warc", "fetch_warc_response_body", "_extract_in_pool",
                 "reset_extraction_pool", "cdx_query", "dedup_rank", "main", "cmd_fetch", "cmd_embed"):
        assert hasattr(di, name), name


def test_class_trust_for_prices_earned_and_inherited():
    assert di.class_trust_for("gov", di.SELF_SIGNAL, 1) == 0.85
    assert di.class_trust_for("gov", di.ENDORSED, 1) == 0.6
    assert di.class_trust_for("edu", di.ENDORSED, 8) == 0.7
    assert di.class_trust_for("news", di.ENDORSED, 8) == 0.6      # never raised above base
    assert di.class_trust_for("nope", None, 1) == 0.55


def test_monkeypatching_the_old_name_patches_the_running_code(monkeypatch):
    monkeypatch.setattr(di, "_RATE", 0)
    di._throttle()                                    # returns at once when the rate is 0
    assert di._RATE == 0


_PROG_DISCOVERY = re.compile(r"usage: (\S+ -m cytir\.acquire\.discovery_ingest|discovery_ingest\.py) ")


def test_prog_regex_accepts_both_argparse_spellings():
    assert _PROG_DISCOVERY.match("usage: python -m cytir.acquire.discovery_ingest [-h] {fetch,embed} ...")
    assert _PROG_DISCOVERY.match("usage: discovery_ingest.py [-h] {fetch,embed} ...")
    assert not _PROG_DISCOVERY.match("usage: pipeline/discovery_ingest.py [-h] {fetch,embed} ...")


def test_script_start_prog_name_and_subcommands(tmp_path):
    done = _run(["--help"], tmp_path)
    assert done.returncode == 0, done.stderr[-500:]
    # task 1797: Python 3.14 argparse prints `python -m cytir.acquire.discovery_ingest`; CI Python prints the basename.
    assert _PROG_DISCOVERY.match(done.stdout[:120]), done.stdout[:120]
    assert "fetch" in done.stdout and "embed" in done.stdout


def test_script_start_fetch_with_an_empty_queue_says_nothing_to_do(tmp_path):
    q = tmp_path / "q.json"
    q.write_text("[]")
    done = _run(["fetch", "--queue", str(q), "--out", str(tmp_path / "o.jsonl"), "--skip-smoke-check"], tmp_path)
    assert done.returncode == 0, done.stderr[-500:]
    assert "nothing to do." in done.stdout


def test_script_start_embed_refuses_ungated_input_before_any_subprocess(tmp_path):
    inp = tmp_path / "p.jsonl"
    inp.write_text(json.dumps({"url": "https://x.example/", "text": "t"}) + "\n")
    done = _run(["embed", "--in", str(inp), "--intent", "measurement-only", "--collection", "c",
                 "--qdrant-url", "http://127.0.0.1:1"], tmp_path)
    assert done.returncode != 0
    assert "REFUSING to embed" in (done.stderr + done.stdout)

"""Per-host fetch depth of `cytir.acquire.discovery_ingest` (task 714, ruling 27 T2).

The page depth used to be one literal (`--max-per-domain 60`) for a whole tranche; a queue row could
only lower it with `cap`. A tranche roster now carries `max_pages` on the rows that should go deeper
(tranche 13: 120 for endorsed edu/gov/ref hosts). Pinned here:

* a row without `max_pages` behaves exactly as before (default, `cap` lowers it);
* `max_pages` replaces the default up or down, and `cap` still lowers the result;
* a malformed value refuses the whole fetch before it starts;
* `process_domain` fetches the row's depth, the columnar resolver is asked for 3x the deepest row of
  the chunk (and 3x the default when no row is deeper, as before).
"""
from __future__ import annotations

import argparse
import io
import json
import os
from contextlib import redirect_stdout
from unittest import mock

import pytest

os.environ.setdefault("AW_LANGGATE_OPTIONAL", "1")

from cytir.acquire import discovery_ingest as di  # noqa: E402


def _records(n):
    return [{"url": f"https://deep.example/p{i}", "offset": 0, "length": 10,
             "warc_filename": "f.warc.gz", "digest": None} for i in range(n)]


def _pages_fetched(item, n_records, default=60):
    with mock.patch.object(di, "fetch_warc", return_value="<html>x</html>"), \
            mock.patch.object(di, "_extract_in_pool", return_value=([], set(), {}, False)), \
            redirect_stdout(io.StringIO()):
        return di.process_domain(None, item, di.DEFAULT_CRAWL, default,
                                 preresolved=_records(n_records))[3]


def test_row_without_max_pages_uses_the_default_and_cap_lowers_it():
    assert di.host_page_limit({"domain": "a.example"}, 60) == 60
    assert di.host_page_limit({"domain": "a.example", "cap": 25}, 60) == 25
    assert di.host_page_limit({"domain": "a.example", "cap": 500}, 60) == 60   # cap never raises
    assert _pages_fetched({"domain": "deep.example"}, 300) == 60


def test_max_pages_replaces_the_default_up_and_down_and_cap_still_lowers():
    assert di.host_page_limit({"domain": "a.example", "max_pages": 120}, 60) == 120
    assert di.host_page_limit({"domain": "a.example", "max_pages": 30}, 60) == 30
    assert di.host_page_limit({"domain": "a.example", "max_pages": 120, "cap": 90}, 60) == 90
    assert _pages_fetched({"domain": "deep.example", "max_pages": 120}, 300) == 120
    assert _pages_fetched({"domain": "deep.example", "max_pages": 120}, 100) == 100  # fewer captures than depth


@pytest.mark.parametrize("bad", [0, -5, 1001, 12.5, "120", True])
def test_a_malformed_max_pages_is_refused(bad):
    with pytest.raises(ValueError, match="max_pages"):
        di.host_page_limit({"domain": "a.example", "max_pages": bad}, 60)


def _fetch_args(tmp_path, queue):
    q = tmp_path / "queue.json"
    q.write_text(json.dumps(queue))
    return argparse.Namespace(
        queue=str(q), out=str(tmp_path / "passages.jsonl"), links_out=str(tmp_path / "outlinks.jsonl"),
        state=str(tmp_path / "state.json"), only="", limit_domains=0, rate=4.0, workers=1,
        max_per_domain=60, resolver="columnar", columnar_threads=1, columnar_chunk=2,
        crawl=di.DEFAULT_CRAWL, resume=True, min_free_gb=0.0, heartbeat_pages=0,
        skip_smoke_check=True)


def _run_fetch(args):
    asked = []

    def fake_resolve(crawl, chunk, con=None, per_domain=None, **_):
        asked.append((list(chunk), per_domain))
        return {d: [] for d in chunk}

    con = mock.Mock()
    with mock.patch.object(di.ccx, "resolve_domains", fake_resolve), \
            mock.patch.object(di.ccx, "_con", return_value=con), \
            mock.patch.object(di.httpx, "Client"), redirect_stdout(io.StringIO()) as buf:
        di.cmd_fetch(args)
    return asked, buf.getvalue()


def test_resolver_depth_follows_the_deepest_row_of_each_chunk(tmp_path):
    queue = [{"domain": "a.example", "class": "edu", "max_pages": 120},
             {"domain": "b.example", "class": "other"},
             {"domain": "c.example", "class": "other"},
             {"domain": "d.example", "class": "other", "max_pages": 30}]
    asked, out = _run_fetch(_fetch_args(tmp_path, queue))
    # chunk 1 = a (120) + b (default): 3 x 120; chunk 2 = c + d (30 < default): 3 x 60, as before
    assert asked == [(["a.example", "b.example"], 360), (["c.example", "d.example"], 180)]
    assert "depth: 1 host(s) above the default, max 120" in out


def test_a_queue_without_max_pages_resolves_at_three_times_the_default(tmp_path):
    queue = [{"domain": "a.example", "class": "edu"}, {"domain": "b.example", "class": "gov"}]
    asked, out = _run_fetch(_fetch_args(tmp_path, queue))
    assert asked == [(["a.example", "b.example"], 180)]
    assert "depth:" not in out


def test_a_malformed_row_refuses_the_whole_fetch_before_any_resolve(tmp_path):
    queue = [{"domain": "a.example", "class": "edu"}, {"domain": "b.example", "max_pages": 5000}]
    with pytest.raises(ValueError, match="b.example"):
        _run_fetch(_fetch_args(tmp_path, queue))

"""Task 2041 -- the probe may read the public prod host (allow-list, https only) and the Slice B draw rotates by UTC day.

No network: the urllib opener is monkeypatched. The internal guard is asserted unchanged.
"""
import datetime
import io
import json
import pathlib
import shutil
import sys
import urllib.request

import pytest

import cytir.probe.run as run
import cytir.probe.sample_sliceb as sb

REAL_POOL = sb.POOL
from cytir.core import internal_http as IH


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_public_flag_reaches_allow_listed_host_only(monkeypatch):
    seen = []

    def fake(req, timeout=None):
        seen.append(req.full_url)
        return _Resp(json.dumps({"results": []}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    d, err = run.fetch_http("https://websear.ch", "q=x", retries=1, public=True)
    assert err is None and d == {"results": []}
    assert seen == ["https://websear.ch/search?q=x"]


@pytest.mark.parametrize("api", ["https://example.com", "http://websear.ch", "https://websear.ch.example.com"])
def test_public_flag_refuses_other_hosts_and_http(monkeypatch, api):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("network opened"))
    monkeypatch.setattr(run.time, "sleep", lambda s: None)
    d, err = run.fetch_http(api, "q=x", retries=1, public=True)
    assert d is None and "--public allows only" in err


def test_without_public_the_internal_guard_still_refuses(monkeypatch):
    from cytir.sources.http import InternalHostRefused
    monkeypatch.setattr(run.time, "sleep", lambda s: None)
    with pytest.raises(InternalHostRefused):
        IH.urlopen("https://websear.ch/search?q=x")
    d, err = run.fetch_http("https://websear.ch", "q=x", retries=1)
    assert d is None and "not loopback" in err


def _draw(tmp_path, monkeypatch, argv):
    if not pathlib.Path(REAL_POOL).is_file():
        pytest.skip("slice_b_pool.jsonl is withheld from the public cut")
    pool = tmp_path / "probes"
    pool.mkdir(exist_ok=True)
    shutil.copy(REAL_POOL, pool / "slice_b_pool.jsonl")
    monkeypatch.setattr(sb, "PROBES", str(pool))
    monkeypatch.setattr(sb, "POOL", str(pool / "slice_b_pool.jsonl"))
    monkeypatch.setattr(sys, "argv", ["sample_sliceb"] + argv)
    sb.main()
    date = argv[argv.index("--date") + 1]
    rows = [json.loads(ln) for ln in (pool / "runs" / date / "sample_sliceb.jsonl").read_text().splitlines()]
    return [r["qid"] for r in rows]


def test_draw_differs_by_date_and_repeats_for_same_date(tmp_path, monkeypatch):
    a = _draw(tmp_path, monkeypatch, ["--date", "2026-10-31"])
    a2 = _draw(tmp_path, monkeypatch, ["--date", "2026-10-31"])
    b = _draw(tmp_path, monkeypatch, ["--date", "2026-11-01"])
    assert a == a2
    assert a != b


def test_seed_reproduces_a_past_date(tmp_path, monkeypatch):
    off = datetime.date(2026, 10, 31).toordinal()
    by_date = _draw(tmp_path, monkeypatch, ["--date", "2026-10-31"])
    by_seed = _draw(tmp_path, monkeypatch, ["--date", "2026-11-05", "--seed", str(off)])
    assert by_date == by_seed

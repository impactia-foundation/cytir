import gzip
import json
import sqlite3

import pytest

from cytir.corpus import l0_census as lc
from cytir.corpus.persisted import Hit, page_keys


class StubIndex:
    def __init__(self, hits):
        self.hits = hits  # {(layer, url): True}

    def lookup(self, url, sha=None, order=("l1", "l0", "l2")):
        for layer in order:
            if (layer, url) in self.hits:
                return Hit(url=url, layer=layer, store_id="s", store_path="p", sha_matched=None)
        return None


def test_terminal_precedence():
    idx = StubIndex({("l0", "https://a.org/1"): 1, ("l1", "https://a.org/1"): 1,
                     ("l1", "https://a.org/2"): 1, ("l2", "https://a.org/3"): 1,
                     ("l1", "https://a.org/4"): 1})
    warc = {k for k in page_keys("https://a.org/4")}
    assert lc.classify("https://a.org/1", index=idx) == "l0"
    assert lc.classify("https://a.org/2", index=idx) == "l1"
    assert lc.classify("https://a.org/3", index=idx) == "l2"
    assert lc.classify("https://a.org/9", index=idx) == "none"
    # a WARC locator outranks L1 (CC is the durable L0); a local L0 outranks a locator
    assert lc.classify("https://a.org/4", index=idx, warc_keys=warc) == "warc"
    assert lc.classify("https://a.org/1", index=idx, warc_keys=warc) == "l0"
    assert lc.classify("https://a.org/9", index=idx, dump_l0=lambda u: True) == "l0"
    extra = set(page_keys("https://a.org/9"))
    assert lc.classify("https://a.org/9", index=idx, extra_l0=extra) == "l0"


def test_wilson_bounds():
    assert lc.wilson(0, 0) == (0.0, 0.0)
    lo, hi = lc.wilson(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert lc.wilson(0, 2000)[0] == 0.0 and lc.wilson(2000, 2000)[1] == 1.0


def test_census_rows_and_merge():
    a, b = lc.Census(), lc.Census()
    for t in ("none", "none", "l1", "l0"):
        a.add("fam", t)
    b.add("fam", "none")
    a.merge(b)
    row = a.rows(served={"fam": 1000})[0]
    assert row["sample"] == 5 and row["none"] == 3 and row["none_pct"] == 60.0
    assert row["none_rows_est"] == 600
    with pytest.raises(ValueError):
        a.add("fam", "bogus")


def test_sample_urls_jsonl_gz_and_payload(tmp_path):
    p = tmp_path / "x.jsonl.gz"
    with gzip.open(p, "wt") as fh:
        for i in range(10):
            fh.write(json.dumps({"url": f"https://a.org/{i}"}) + "\n")
        fh.write(json.dumps({"payload": {"url": "https://b.org/p"}}) + "\n")
    assert len(list(lc.sample_urls(str(p), 4))) == 4
    got = list(lc.sample_urls(str(tmp_path), 100))
    assert "https://b.org/p" in got and len(got) == 11


def test_roster_from_index():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE scanned_files (layer TEXT, path TEXT, size INTEGER, "
              "mtime_ns INTEGER, rows INTEGER, err TEXT)")
    c.executemany("INSERT INTO scanned_files VALUES (?,?,?,?,?,?)",
                  [("l0", "a", 10, 0, 5, None), ("l0", "b", 20, 0, 0, "bad"),
                   ("l1", "c", 7, 0, 3, None)])
    r = {x["layer"]: x for x in lc.roster_from_index(c)}
    assert r["l0"] == {"layer": "l0", "files": 2, "bytes": 30, "rows": 5, "unreadable_files": 1}
    assert r["l1"]["rows"] == 3

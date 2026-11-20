"""`cytir.ops.spam_tombstones` (task 701): a pure-tombstone pack for spam rows that already serve."""
from __future__ import annotations

import gzip
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cytir.index import point_delete as PD  # noqa: E402
from cytir.ops import spam_tombstones as T  # noqa: E402

LISTS = HERE.parent / "data" / "spam-host-lists.v1.json"


def _exclude(tmp_path):
    rows = [{"point_id": f"p{i}", "reason": "host_hijacked" if i % 2 else "R1",
             "url": f"https://giffconstable.com/x{i}", "domain": "giffconstable.com"} for i in range(6)]
    f = tmp_path / "exclude.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return f


@pytest.fixture
def twin(monkeypatch):
    served = {"p0", "p1", "p3", "p5"}  # p2, p4 never landed

    def fetch(url, collection, ids, fields, **kw):
        return {i: {"url": f"https://giffconstable.com/stored-{i}", "domain": "giffconstable.com"}
                for i in ids if i in served}
    monkeypatch.setattr(PD, "fetch_payload_fields", fetch)
    return served


def test_plan_counts_only_the_rows_that_serve(tmp_path, twin):
    res = T.plan("http://twin", "c", _exclude(tmp_path))
    assert res["excluded_ids"] == 6 and res["serving"] == 4
    assert res["by_detail"] == {"R1": 1, "host_hijacked": 3}


def test_compose_writes_a_pure_tombstone_pack_the_drain_shape(tmp_path, twin):
    live = T.serving("http://twin", "c", T.load_exclude(_exclude(tmp_path)))
    m = T.write_pack(str(tmp_path / "q"), "cc-x-spam", live, family="cc_authority_frontier_w7",
                     collection="aw_e8", note="n")
    d = tmp_path / "q" / "cc-x-spam"
    assert m["kind"] == "content_pack" and m["rows"] == 0 and m["tombstones_count"] == 4
    lines = [json.loads(x) for x in (d / "tombstones.jsonl").read_text().splitlines()]
    assert {x["reason"] for x in lines} == {"takedown"} and all(x["url"].endswith(x["id"]) for x in lines)
    assert gzip.open(m["pack_points"]).read() == b""
    import hashlib
    assert (d / "READY").read_text().strip() == hashlib.sha256((d / "manifest.json").read_bytes()).hexdigest()
    from cytir.index import promote_queue as PQ
    assert PQ.TOMBSTONE_REASONS.count("takedown") == 1  # the reason the drain accepts


def test_nothing_serving_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(PD, "fetch_payload_fields", lambda *a, **k: {})
    with pytest.raises(SystemExit, match="0 serving"):
        T.write_pack(str(tmp_path / "q"), "p", T.serving("u", "c", T.load_exclude(_exclude(tmp_path))),
                     family="f", collection="c", note="n")


def test_domain_counts_ask_the_indexed_domain_field_for_both_spellings(monkeypatch):
    asked = []

    def api(url, path, body=None, method=None):
        asked.append((path, body["filter"]["must"][0]["match"]["value"], body["exact"]))
        return {"count": 3 if body["filter"]["must"][0]["match"]["value"].startswith("www.") else 2}
    monkeypatch.setattr(PD, "_api", api)
    assert T.domain_counts("u", "c", ["x.example"]) == {"x.example": 5}
    assert [a[1] for a in asked] == ["x.example", "www.x.example"] and all(a[2] for a in asked)

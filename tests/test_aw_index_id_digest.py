"""`cytir.index.id_digest`: the order-independent, payload-blind MEMBERSHIP digest over point ids.

The flip check must survive a scroll reorder and ignore payload changes; `--compare` and `--diff`
are the two offline verbs of its command line (the live scroll is `digest_collection`, exercised
here against a stubbed transport).
"""
from __future__ import annotations

import inspect
import json
import uuid

from cytir.index import id_digest as cid

IDS = [str(uuid.UUID(int=i * 7919 + 1)) for i in range(50)] + [17, 18, 19]


def _acc(seq):
    a = [0] * cid.MODULI
    for pid in seq:
        for i, t in enumerate(cid.id_terms(pid)):
            a[i] = (a[i] + t) & cid.MASK
    return tuple(a)


def test_id_digest_is_order_independent_and_payload_blind():
    assert _acc(IDS) == _acc(list(reversed(IDS))), "a reorder must not change it"
    assert _acc(IDS) != _acc(IDS[:-1]), "a missing id must change it"
    assert _acc(IDS) != _acc(IDS + IDS[:1]), "a duplicate id must change it"
    src = inspect.getsource(cid.id_terms)
    assert "payload" not in src and "vector" not in src


def _pages(ids, size):
    pages = [ids[i:i + size] for i in range(0, len(ids), size)]

    def req(url, body=None, method=None, timeout=300):
        if url.endswith("/points/scroll"):
            k = body.get("offset") or 0
            nxt = k + 1 if k + 1 < len(pages) else None
            return {"result": {"points": [{"id": i} for i in pages[k]], "next_page_offset": nxt}}
        return {"result": {"points_count": len(ids)}}
    return req


def test_digest_collection_equals_digest_of_and_dumps_every_id(monkeypatch, tmp_path):
    monkeypatch.setattr(cid, "_req", _pages(IDS, 7))
    dump = tmp_path / "ids.txt"
    rep = cid.digest_collection("http://q.invalid", "c", batch=7, dump_ids=str(dump))
    ref = cid.digest_of(IDS)
    assert rep["digest"] == ref["digest"] and rep["accumulators"] == ref["accumulators"]
    assert rep["ids_scrolled"] == len(IDS) and rep["schema"] == "aw.collection-id-digest.v1"
    assert dump.read_text().split() == [str(i) for i in IDS]


def test_compare_and_diff_name_the_difference(tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    for p, ids in ((a, IDS), (b, IDS[:-1])):
        d = cid.digest_of(ids)
        p.write_text(json.dumps({"collection": p.stem, "ids_scrolled": d["ids"], "digest": d["digest"],
                                 "measured_at": "t"}))
    assert cid.main(["--compare", str(a), str(a)]) == 0
    assert cid.main(["--compare", str(a), str(b)]) == 2
    capsys.readouterr()
    ia, ib, out = tmp_path / "ia", tmp_path / "ib", tmp_path / "only_a"
    ia.write_text("".join(f"{i}\n" for i in IDS))
    ib.write_text("".join(f"{i}\n" for i in IDS[:-1]))
    assert cid.main(["--diff", str(ia), str(ib), "--diff-out", str(out)]) == 2
    assert out.read_text().split() == [str(IDS[-1])]

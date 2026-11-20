"""cytir.index.loda_resurface (task 886): the E6 loda dense store the aw_e10 build reads.

Real records: the twin payloads are the real loda/other rows of `fixtures/row_decode/twin_rows.v1.json`
(two tier-A loda, two tier-B loda, sixteen others); the staged texts are those rows' real
title + snippet. The embedder is a tiny deterministic fake.
"""
from __future__ import annotations

import gzip
import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from cytir.index import loda_resurface as LR  # noqa: E402
from cytir.index import row_decode as RD  # noqa: E402

ROWS = json.loads((HERE / "fixtures" / "row_decode" / "twin_rows.v1.json").read_text())["rows"]
TIER_A = sorted(r["id"] for r in ROWS if LR.is_tier_a_loda(RD.decode_row(r["payload"], profile=None)))
WIDTH = 8


class FakeClient:
    """Scroll over the real rows; ignores the server filter on purpose (the module re-checks)."""
    def scroll(self, collection, scroll_filter=None, limit=10, offset=None, **kw):
        return [SimpleNamespace(id=r["id"], payload=r["payload"]) for r in ROWS], None


def write_texts(d, rows=ROWS):
    d.mkdir(exist_ok=True)
    with gzip.open(d / "shard-00000.texts.jsonl.gz", "wt", encoding="utf-8") as fh:
        for r in rows:
            p = r["payload"]
            fh.write(json.dumps({"point_id": r["id"], "title": p.get("title"),
                                 "text": f"{p.get('title')}\n\n{p.get('snippet')}"}) + "\n")


def fake_embed(chunks, vecs):
    """Row k -> a deterministic 1024-d fp16 vector keyed by the id; writes the shard contract."""
    import glob, os
    os.makedirs(vecs, exist_ok=True)
    for tp in sorted(glob.glob(chunks + "/*.texts.jsonl.gz")):
        tag = os.path.basename(tp)[: -len(".texts.jsonl.gz")]
        ids = [json.loads(x)["point_id"] for x in gzip.open(tp, "rt")]
        m = np.stack([np.random.default_rng(int(i[:8], 16)).standard_normal(1024) for i in ids])
        m = (m / np.linalg.norm(m, axis=1, keepdims=True)).astype("<f2")
        m.tofile(f"{vecs}/{tag}.vectors.f16")
        open(f"{vecs}/{tag}.row_ids.jsonl", "w").write(
            "".join(json.dumps({"i": k, "point_id": i}) + "\n" for k, i in enumerate(ids)))
        json.dump({"n": len(ids), "dim": 1024}, open(f"{vecs}/{tag}.vectors.manifest.json", "w"))


def go(tmp_path, ids=None, embed=fake_embed, width=WIDTH, texts_dirs=None):
    texts = tmp_path / "texts"
    if texts_dirs is None:
        if not texts.exists():
            write_texts(texts)
        texts_dirs = [str(texts)]
    return LR.run(ids=ids if ids is not None else TIER_A, texts_dirs=[str(d) for d in texts_dirs],
                  out_dir=str(tmp_path / "out"), width=width, twin_url="http://twin",
                  twin_collection="aw_t", points_count=len(ROWS), embed=embed, log=lambda *_: None)


def test_ids_are_tier_a_loda_only():
    assert len(TIER_A) == 2
    assert LR.twin_loda_ids(FakeClient(), "c") == TIER_A




def test_restage_ids_equal_store_ids_and_manifest_verifies(tmp_path):
    man = go(tmp_path)
    out = tmp_path / "out"
    assert {x.strip() for x in open(out / LR.IDS_NAME)} == set(TIER_A)
    assert LR.verify(str(out))["ids_digest"] == man["ids_digest"] == LR.ids_digest(TIER_A)
    assert man["twin"]["collection"] == "aw_t" and man["model"]["revision"]
    (out / LR.IDS_NAME).write_text("tampered\n")
    with pytest.raises(LR.ResurfaceRefused):
        LR.verify(str(out))




def test_resume_skips_stored_ids(tmp_path):
    go(tmp_path)
    def boom(*a):
        raise AssertionError("embed called on a complete store")
    man = go(tmp_path, embed=boom)
    assert man["embedded_this_run"] == 0 and man["skipped_resumed"] == 2


def test_resume_embeds_only_the_rest(tmp_path):
    seen = []
    def spy(chunks, vecs):
        seen.append(sum(1 for _ in gzip.open(f"{chunks}/shard-00000.texts.jsonl.gz", "rt")))
        fake_embed(chunks, vecs)
    go(tmp_path, ids=TIER_A[:1], embed=spy)
    man = go(tmp_path, embed=spy)
    assert seen == [1, 1] and man["rows"] == 2


def test_missing_text_and_foreign_store_ids_refuse(tmp_path):
    write_texts(tmp_path / "texts", [r for r in ROWS if r["id"] != TIER_A[0]])
    with pytest.raises(LR.ResurfaceRefused, match="no text"):
        go(tmp_path)
    db = LR.open_store(str(tmp_path / "out2.sqlite")); db.close()
    t2 = tmp_path / "b"; t2.mkdir(); write_texts(t2 / "texts")
    go(t2)
    with pytest.raises(LR.ResurfaceRefused, match="not twin tier-A"):
        go(t2, ids=TIER_A[:1])


def test_filter_matches_the_coded_spelling_of_loda():
    assert LR.loda_provenance_values() == ["loda", 27]
    coded = {"provenance": 27, "tier": "A"}
    assert LR.is_tier_a_loda(RD.decode_row(coded, profile=None))


class UpperClient:
    """The twin returns ids in another spelling (upper case, no hyphens): the join must still hold."""
    def scroll(self, collection, **kw):
        return [SimpleNamespace(id=r["id"].upper().replace("-", ""), payload=r["payload"])
                for r in ROWS if r["id"] in TIER_A], None


def tier_a_row(i):
    return next(r for r in ROWS if r["id"] == TIER_A[i])


def test_two_texts_dirs_are_joined(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_texts(a, [tier_a_row(0)])
    write_texts(b, [r for r in ROWS if r["id"] != TIER_A[0]])
    man = go(tmp_path, texts_dirs=[a, b])
    assert man["rows"] == 2 and man["embedded_this_run"] == 2


def test_duplicates_across_dirs_are_counted_and_conflicts_refused(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_texts(a)
    write_texts(b)                                    # the same rows again: identical text
    stats = LR.write_chunks([str(a), str(b)], set(TIER_A), str(tmp_path / "chunks"), log=lambda *_: None)
    assert stats == {"shards": 1, "rows": 2, "duplicates": 2}
    c = tmp_path / "c"
    c.mkdir()
    with gzip.open(c / "x.texts.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"point_id": TIER_A[0], "text": "other", "title": "t"}) + "\n")
    with pytest.raises(LR.ResurfaceRefused, match="two different texts"):
        LR.write_chunks([str(a), str(c)], set(TIER_A), str(tmp_path / "chunks2"))


def test_embedded_text_is_the_stage_text_bytes(tmp_path):
    row = tier_a_row(0)["payload"]
    text = f"{row['title']}\n\n{row['snippet']}"
    assert LR.dense_surface({"text": text, "title": row["title"]}) == text
    d = tmp_path / "texts"
    write_texts(d)
    LR.write_chunks([str(d)], set(TIER_A), str(tmp_path / "chunks"))
    got = {json.loads(x)["point_id"]: json.loads(x)["text"]
           for x in gzip.open(tmp_path / "chunks" / "shard-00000.texts.jsonl.gz", "rt", encoding="utf-8")}
    assert got[TIER_A[0]] == text


def test_loda_row_without_tier_is_tier_a():
    payload = {k: v for k, v in tier_a_row(0)["payload"].items() if k != "tier"}
    assert "tier" not in payload
    assert LR.is_tier_a_loda(RD.decode_row(payload, profile=None))


def test_id_spelling_joins_on_both_sides(tmp_path):
    assert LR.twin_loda_ids(UpperClient(), "c") == TIER_A
    d = tmp_path / "texts"
    d.mkdir()
    odd = [" " + TIER_A[0].upper().replace("-", "") + " ", TIER_A[1].upper()]
    with gzip.open(d / "s.texts.jsonl.gz", "wt", encoding="utf-8") as fh:
        for pid in odd:
            fh.write(json.dumps({"point_id": pid, "text": "x", "title": "t"}) + "\n")
    stats = LR.write_chunks([str(d)], set(TIER_A), str(tmp_path / "chunks"))
    assert stats["rows"] == 2
    ids = [json.loads(x)["point_id"]
           for x in gzip.open(tmp_path / "chunks" / "shard-00000.texts.jsonl.gz", "rt", encoding="utf-8")]
    assert sorted(ids) == TIER_A


def test_cli_refuses_prod_endpoint_before_any_client_call(tmp_path, monkeypatch):
    import cytir.cli as CLI
    import cytir.core.qdrant as Q

    def no_client(*a, **k):
        raise AssertionError("a client was built against the prod endpoint")
    monkeypatch.setattr(Q, "qdrant_client", no_client)
    d = tmp_path / "texts"
    write_texts(d)
    prod_host = "prod-box"  # a prod-looking endpoint: the verb must refuse it before any client call
    rc = CLI._index_main(["loda-resurface", "--twin", f"http://{prod_host}:6334", "--collection", "aw_prod",
                          "--texts", str(d), "--out", str(tmp_path / "out")])
    assert rc != 0


def test_resume_refuses_width_or_model_pin_mismatch(tmp_path):
    def boom(*a):
        raise AssertionError("embed called on a refused resume")
    go(tmp_path)
    with pytest.raises(LR.ResurfaceRefused, match="width"):
        go(tmp_path, width=16, embed=boom)
    man_p = tmp_path / "out" / LR.MANIFEST_NAME
    man = json.loads(man_p.read_text())
    man["model"]["revision"] = "other-revision"
    man_p.write_text(json.dumps(man))
    with pytest.raises(LR.ResurfaceRefused, match="model"):
        go(tmp_path, embed=boom)
    man_p.unlink()                                     # no manifest: the stored blobs still refuse
    with pytest.raises(LR.ResurfaceRefused, match="dims"):
        go(tmp_path, width=16, embed=boom)

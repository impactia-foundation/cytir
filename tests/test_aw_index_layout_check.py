"""Task 2405 (S6): `cyt index layout-check` against a fake collection config. No Qdrant: the one GET is faked."""
import copy

import pytest

from cytir import cli
from cytir.core import serving_profile as SP
from cytir.index import layout_check as LC

URL, COLL = "http://qdrant.invalid:6333", "fixture"


def _result(**over):
    """The `result` of `GET /collections/<c>` for a collection created from PROD_E9."""
    p = SP.PROD_E9
    r = {"config": {
        "params": {
            "vectors": {"dense": {"size": p.dense_dim, "distance": "Cosine", "on_disk": True,
                                  "datatype": p.dense_datatype}},
            "sparse_vectors": {"bm25": {"modifier": "idf", "index": {"on_disk": True, "datatype": "uint8"}}},
            "shard_number": 1, "replication_factor": 1, "write_consistency_factor": 1,
            "on_disk_payload": True},
        "hnsw_config": {"m": 8, "ef_construct": 100, "full_scan_threshold": 10000,
                        "max_indexing_threads": 8, "on_disk": False},
        "optimizer_config": dict(p.optimizers),
        "quantization_config": {"product": {"compression": "x8", "always_ram": False}}},
        "payload_schema": {}}
    for path, value in over.items():
        cur = r
        *head, last = path.split("/")
        for k in head:
            cur = cur[k]
        if value is None:
            cur.pop(last, None)
        else:
            cur[last] = value
    return r


def _run(monkeypatch, capsys, result):
    monkeypatch.setattr(LC.IH, "fetch_json", lambda url, **kw: {"result": copy.deepcopy(result)})
    rc = LC.main(["--url", URL, "--collection", COLL])
    return rc, capsys.readouterr()


def test_the_declared_served_layout_is_256_float16_pq_x8():
    assert (SP.PROD_E9.dense_dim, SP.PROD_E9.dense_datatype, SP.PROD_E9.quantizer) == (256, "float16", "pq_x8")


def test_match(monkeypatch, capsys):
    rc, out = _run(monkeypatch, capsys, _result())
    assert rc == 0 and out.out.startswith("layout: MATCH ") and "256-d float16" in out.out
    assert "product/x8" in out.out and len(out.out.strip().splitlines()) == 1


def test_dim_mismatch(monkeypatch, capsys):
    rc, out = _run(monkeypatch, capsys, _result(**{"config/params/vectors/dense/size": 128}))
    assert rc == 1 and out.out.startswith("layout: MISMATCH ")
    assert "vectors.dense.size: read 128, profile 256" in out.out


def test_datatype_mismatch(monkeypatch, capsys):
    rc, out = _run(monkeypatch, capsys, _result(**{"config/params/vectors/dense/datatype": None}))
    assert rc == 1 and "datatype: read 'float32', profile 'float16'" in out.out


@pytest.mark.parametrize("quant", [
    {"scalar": {"type": "int8", "always_ram": False}},
    {"product": {"compression": "x16", "always_ram": False}},
    None,
])
def test_quantization_mismatch(monkeypatch, capsys, quant):
    rc, out = _run(monkeypatch, capsys, _result(**{"config/quantization_config": quant}))
    assert rc == 1 and out.out.startswith("layout: MISMATCH ") and "quantization" in out.out


def test_unreadable_collection_is_not_a_verdict(monkeypatch, capsys):
    def boom(url, **kw):
        raise OSError("refused")
    monkeypatch.setattr(LC.IH, "fetch_json", boom)
    assert LC.main(["--url", URL, "--collection", COLL]) == 2
    assert "UNREADABLE" in capsys.readouterr().err


def test_the_verb_is_registered_and_reaches_the_module(monkeypatch):
    assert cli._INDEX_TOOL_VERBS["layout-check"] == "cytir.index.layout_check"
    monkeypatch.setattr(LC, "main", lambda argv, prog: 7)
    assert cli.main(["index", "layout-check", "--url", URL, "--collection", COLL]) == 7

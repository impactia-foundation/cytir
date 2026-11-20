"""The producers of committed artifacts (task 1074): each regenerates its file from sealed mini inputs."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cytir import cli
from cytir.ops import _provenance, bind_assembly_vector_rollups as bind, build_authority_table as bat, build_gazetteer

APP = Path(__file__).resolve().parents[1]

# a sealed mini-graph: three seeds, each of a class, endorsing a few hosts
GRAPH = {"edges": {
    "seed1.example.edu": {"class": "universities-research", "targets": {"roche.ch": 9, "uni-a.example.edu": 9, "other.example.gov": 5}},
    "seed2.example.gov": {"class": "disc-gov", "targets": {"roche.ch": 9, "other.example.gov": 9, "uni-a.example.edu": 4}},
    "seed3.example.org": {"class": "universities-research", "targets": {"roche.ch": 9, "other.example.gov": 9}},
}}


def _hosts(override):
    old = bat.HOST_CLASS_OVERRIDE
    bat.HOST_CLASS_OVERRIDE = override
    try:
        return bat.build(copy.deepcopy(GRAPH), min_score=0.0)[0]
    finally:
        bat.HOST_CLASS_OVERRIDE = old


def test_authority_override_changes_only_the_class_of_the_named_host():
    base, over = _hosts({}), _hosts({"roche.ch": "corporate"})
    assert "roche.ch" in base and base["roche.ch"]["class"] != "corporate"
    assert over["roche.ch"]["class"] == "corporate"
    assert {k: v for k, v in over["roche.ch"].items() if k != "class"} == \
        {k: v for k, v in base["roche.ch"].items() if k != "class"}
    assert {h: v for h, v in over.items() if h != "roche.ch"} == {h: v for h, v in base.items() if h != "roche.ch"}
    assert list(over) == list(base)


def test_authority_declared_table_names_roche_ch():
    assert bat.HOST_CLASS_OVERRIDE == {"roche.ch": "corporate"}


def test_authority_cli_writes_the_table_and_a_provenance_sidecar(tmp_path):
    src, out = tmp_path / "endorse.json", tmp_path / "authority_hosts.json"
    src.write_text(json.dumps(GRAPH))
    assert bat.main(["--endorse", str(src), "--out", str(out), "--min-score", "0"]) == 0
    doc = json.loads(out.read_text())
    assert doc["hosts"]["roche.ch"]["class"] == "corporate"
    side = json.loads((tmp_path / "authority_hosts.json.provenance.json").read_text())
    assert side["producer"] == "aw ops build-authority-table"
    assert side["inputs"]["endorse"]["sha256"] == _provenance.sha256_file(src)
    assert side["output"]["sha256"] == _provenance.sha256_file(out)


def test_gazetteer_regenerates_the_committed_file_byte_for_byte(tmp_path):
    out = tmp_path / "geo_gazetteer.json"
    build_gazetteer.main(["--out", str(out)])
    assert out.read_bytes() == (APP / "cytir/serve/geo_gazetteer.json").read_bytes()


def test_bind_vector_rollups_binds_rows_from_the_witness_receipt(tmp_path):
    root = tmp_path / "corpus"
    (root / "w").mkdir(parents=True)
    receipt = root / "w" / "receipt.json"
    receipt.write_text(json.dumps({"rows": 7, "dim": 1024, "dtype": "fp16", "status": "sealed"}))
    run = root / "w" / "RUN.json"
    run.write_text(json.dumps({"inputs": {"portable_vector_receipt": {
        "uri": "artifact://w/receipt.json", "sha256": _provenance.sha256_file(receipt)}}}))
    census = tmp_path / "census.json"
    census.write_text(json.dumps({"inputs": {"replacement_point_witnesses": {
        "setA": {"run": {"uri": "artifact://w/RUN.json"}}}}}))
    manifests = tmp_path / "in.json"
    manifests.write_text(json.dumps({"sets": [
        {"set_id": "setA", "contributes_vectors": True, "class": "atomic_epoch1", "claims": {"count": 7},
         "adjudication": {"points_lost_to_task427": 2}},
        {"set_id": "setB", "contributes_vectors": True, "class": "exact", "claims": {"count": 1}},
        {"set_id": "setC", "contributes_vectors": False}]}))
    out = tmp_path / "out.json"
    assert bind.main(["--corpus-root", str(root), "--census-run", str(census), "--in", str(manifests),
                      "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    a, b = doc["sets"][0], doc["sets"][1]
    assert a["vector"]["declared"] and a["vector"]["rows"] == 7 and a["vector"]["receipt_sha256"] == _provenance.sha256_file(receipt)
    assert a["adjudication"]["rows_suppressed"] == 2 and a["adjudication"]["rows_loaded"] == 5
    assert b["vector"]["declared"] is False
    assert doc["vector_binding"] == {"producer": "aw ops bind-assembly-vector-rollups", "bound": 1,
                                     "unbound": 1, "unbound_detail": ["setB: no witness RUN to follow"]}
    assert (tmp_path / "out.json.provenance.json").exists()


def test_bind_refuses_an_atomic_family_whose_rows_differ_from_its_claims():
    with pytest.raises(SystemExit):
        bind._derive_row_split({"set_id": "s", "class": "atomic_epoch1", "vector": {"rows": 5},
                                "claims": {"count": 4}})


def test_arena_labels_from_a_sealed_mini_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    from cytir.ops import extract_arena_labels as ex

    def meta(*urls):
        return {"web_search_trace": [[["[1]", u] for u in urls]]}
    rows = {
        "messages_a": [[{"role": "user", "content": "what does article 336 say about notice"}]] * 2,
        "languages": [["English"], ["English"]],
        "winner": ["model_a", "tie"],
        "turn": [1, 1],
        "system_a_metadata": [meta("https://a.example/x", "https://c.example/z"), meta("https://a.example/x")],
        "system_b_metadata": [meta("https://b.example/y", "https://c.example/z"), meta("https://b.example/y")],
        "model_a": ["ma", "ma"], "model_b": ["mb", "mb"]}
    pq.write_table(pa.table(rows), tmp_path / "s.parquet")
    out = tmp_path / "labels.jsonl"
    ex.main(["--parquet", str(tmp_path / "s.parquet"), "--out", str(out)])
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(lines) == 1 and lines[0]["preferred_urls"] == ["https://a.example/x"]
    assert lines[0]["rejected_urls"] == ["https://b.example/y"] and lines[0]["shared_urls"] == 1
    assert (tmp_path / "labels.jsonl.provenance.json").exists()


def test_assembly_winner_rule_and_cli_doors():
    from cytir.ops import produce_assembly_artifacts as pa
    with pytest.raises(SystemExit):
        pa.winner_of("no_such_class|x")
    for verb, mod in cli._PRODUCER_VERBS.items():
        import importlib.util
        if importlib.util.find_spec(mod) is None:   # a producer the public cut does not ship (task 1088)
            continue
        assert callable(importlib.import_module(mod).main), verb
    with pytest.raises(SystemExit) as e:
        cli._ops_main(["scan-membership-assembly-facts", "--help"])
    assert e.value.code == 0


def test_graduate_qhead_country_raises_only_the_named_knob_and_writes_a_sidecar(tmp_path):
    from cytir.ops import graduate_qhead_country as g
    src, out = tmp_path / "qhead.json", tmp_path / "out.json"
    src.write_text(json.dumps({"knobs": {"off_country": {"driver": "head", "alpha": 0.0, "b": -2.0},
                                         "other": {"driver": "head", "alpha": 0.0},
                                         "static": {"driver": "constant"}}}))
    assert g.main(["--in", str(src), "--out", str(out), "--alpha", "1.0"]) == 0
    art = json.loads(out.read_text())
    assert art["knobs"]["off_country"]["alpha"] == 1.0 and art["knobs"]["other"]["alpha"] == 0.0
    assert art["knobs"]["off_country"]["graduated"]["prev_alpha"] == 0.0 and art["phase"] == "2-country"
    side = json.loads((tmp_path / "out.json.provenance.json").read_text())
    assert side["producer"] == "aw ops graduate-qhead-country"
    with pytest.raises(SystemExit):
        g.main(["--in", str(src), "--out", str(out), "--knob", "static"])

"""`cyt ops source-cards`: the build/embed/search commands ported into `cytir.corpus.source_cards`
(task 1235). The collection request is compared with the request the former script sent."""
from __future__ import annotations

import json

from cytir import cli
from cytir.corpus import source_cards as SC


class _Recorder:
    def __init__(self, exists=False):
        self.exists = exists
        self.calls = []

    def collection_exists(self, name):
        return self.exists

    def delete_collection(self, name):
        self.calls.append(("delete_collection", name))

    def create_collection(self, **kw):
        self.calls.append(("create_collection", kw))

    def create_payload_index(self, name, field_name, field_schema):
        self.calls.append(("create_payload_index", name, field_name, field_schema))


def _request(client):
    return next(c[1] for c in client.calls if c[0] == "create_collection")


def test_the_collection_request_is_the_one_the_script_sent():
    from qdrant_client import models as qm
    client = _Recorder()
    SC.ensure_source_cards_collection(client, qm, "source_cards", 128, recreate=False)
    kw = _request(client)
    # no field beyond the ones the former script sent (an added knob is a drift too)
    assert set(kw) == {"collection_name", "vectors_config", "sparse_vectors_config",
                       "quantization_config", "hnsw_config", "on_disk_payload"}
    assert kw["collection_name"] == "source_cards"
    assert kw["vectors_config"] == {"dense": qm.VectorParams(
        size=128, distance=qm.Distance.COSINE, on_disk=True)}
    assert kw["sparse_vectors_config"] == {"bm25": qm.SparseVectorParams(modifier=qm.Modifier.IDF)}
    assert kw["quantization_config"] == qm.ScalarQuantization(scalar=qm.ScalarQuantizationConfig(
        type=qm.ScalarType.INT8, always_ram=False))
    # the old script also sent ef_construct=100, which is the server default
    assert kw["hnsw_config"] == qm.HnswConfigDiff(m=16, on_disk=True)
    assert kw["on_disk_payload"] is True
    idx = [(c[2], c[3]) for c in client.calls if c[0] == "create_payload_index"]
    assert idx == [("disc_class", qm.PayloadSchemaType.KEYWORD),
                   ("jurisdiction", qm.PayloadSchemaType.KEYWORD),
                   ("disposition", qm.PayloadSchemaType.KEYWORD),
                   ("domain", qm.PayloadSchemaType.KEYWORD),
                   ("trust_tier", qm.PayloadSchemaType.FLOAT)]


def test_an_existing_collection_is_kept_unless_recreate_is_asked():
    from qdrant_client import models as qm
    kept = _Recorder(exists=True)
    SC.ensure_source_cards_collection(kept, qm, "source_cards", 128, recreate=False)
    assert kept.calls == []
    redone = _Recorder(exists=True)
    SC.ensure_source_cards_collection(redone, qm, "source_cards", 128, recreate=True)
    assert redone.calls[0] == ("delete_collection", "source_cards")
    assert _request(redone)["collection_name"] == "source_cards"


def test_build_writes_cards_offline(tmp_path, capsys):
    disc = tmp_path / "disc.json"
    disc.write_text(json.dumps([]))
    out = tmp_path / "cards.jsonl"
    try:
        SC.main(["build", "--discovery", str(disc), "--out", str(out)])
    except SystemExit:
        raise
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows and all("domain" in r and "disposition" in r for r in rows)
    assert "WROTE" in capsys.readouterr().out


def test_the_verb_is_registered_on_the_ops_door():
    assert cli._PRODUCER_VERBS["source-cards"] == "cytir.ops.source_cards"

"""Builder corpus KV (moved to cytir/index/b1_corpus_kv.py, task 2106): the folded
text and the join keys are unchanged, and `lmdb` stays optional."""
from __future__ import annotations

import gzip
import json
import subprocess
import sys

import pytest

from cytir.index import b1_corpus_kv as KV


class Txn:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        self.data[key] = value


def test_importing_the_module_does_not_import_lmdb():
    code = ("import sys; import cytir.index.b1_corpus_kv; "
            "sys.exit(1 if 'lmdb' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_canon_key_and_kv_key_are_the_frozen_join():
    key = KV._canon_key("https://www.fedlex.admin.ch/eli/cc/27/317_321_377/de",
                        "art_336")
    assert key.endswith("#art_336")
    assert KV._canon_key("https://example.org/a", None).endswith("#")
    k = KV._kv_key(key)
    assert len(k) == 16 and k == KV._kv_key(key)
    assert KV._kv_channel_key(k) == b"C" + k


def test_multi_text_value_keeps_every_distinct_chunk_text():
    txn, k = Txn(), b"y" * 16
    assert KV._kv_put_multi(txn, k, "Art. 336 erster Satz")
    assert KV._kv_put_multi(txn, k, "Art. 336 zweiter Satz")
    assert not KV._kv_put_multi(txn, k, "Art. 336 erster Satz")
    assert KV._kv_get_texts(txn, k) == ["Art. 336 erster Satz",
                                        "Art. 336 zweiter Satz"]
    assert txn.data[k] == "Art. 336 erster Satz".encode() + b"\x00" + \
        "Art. 336 zweiter Satz".encode()
    assert KV._kv_get_texts(txn, b"z" * 16) is None


def test_fold_jsonl_puts_text_and_channels(tmp_path):
    pytest.importorskip("lmdb")
    path = tmp_path / "law.jsonl.gz"
    rows = [
        {"url": "https://example.org/law", "anchor": "a1", "text": "Text eins",
         "dense_text": "dense eins", "sparse_text": "titel. Text eins",
         "bundle_digest": "d" * 64},
        {"url": "https://example.org/law", "anchor": "a1", "text": "Text eins"},
        {"url": "https://example.org/law", "anchor": "a2", "text": ""},
    ]
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    env = KV.kv_open(str(tmp_path / "kv"), readonly=False)
    done, st = set(), {}
    put = KV._fold_jsonl_into_kv(env, str(path), "__fedlex__", done, st,
                                 lambda *_: None)
    assert put == 1 and st["kv_done_files"] == ["__fedlex__"]
    key = KV._kv_key(KV._canon_key("https://example.org/law", "a1"))
    with env.begin() as txn:
        assert KV._kv_get_texts(txn, key) == ["Text eins"]
        assert KV._kv_get_channels(txn, key)["dense_text"] == "dense eins"
    env.close()


def test_published_and_conf_maps(tmp_path):
    pub = tmp_path / "pub.jsonl"
    pub.write_text(json.dumps({"url": "https://www.swissinfo.ch/eng/x/1",
                               "published": "2026-08-01"}) + "\n\n")
    m = KV.load_published_map([str(pub), str(tmp_path / "missing")],
                              log=lambda *_: None)
    assert list(m.values()) == ["2026-08-01"]
    conf = tmp_path / "conf.jsonl"
    conf.write_text('{"domain": ".Example.CH", "conf": 0.75}\nnot json\n')
    assert KV.load_resolve_conf_map(str(conf), log=lambda *_: None) == {
        "example.ch": 0.75}


def test_corpus_text_index_collects_all_texts_per_key(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    wave = tmp_path / "wave=w1"
    wave.mkdir()
    tbl = pa.table({"url": ["https://example.org/p"] * 3,
                    "anchor": [None, None, None],
                    "text": ["eins", "zwei", "eins"]})
    pq.write_table(tbl, wave / "f.parquet")
    need = {KV._canon_key("https://example.org/p", None)}
    out = KV.build_corpus_text_index(need, log=lambda *_: None,
                                     corpus_dir=str(tmp_path))
    assert out[next(iter(need))]["texts"] == ["eins", "zwei"]

"""Characterization of the BM25 IDF table tool (ADR-099 Ruling 2), written BEFORE it moved from
the old measurer script to `cytir/index/idf_table.py` (slice 8 step 2) and kept green after it.

The tool reads a collection through a tiny, stubbed Qdrant (two routes) and writes the table
`cytir/serve/bm25_idf.json` is made of. The test pins every field except the wall-clock
`measured_at`. It does NOT regenerate the shipped table: that is a ranking change and needs the
eval battery (spec §6 Q4).
"""
from __future__ import annotations

import json
import math
import pathlib
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cytir.index import idf_table as M  # noqa: E402

POINTS = 1234
#: the idf the stub reports per token hash (weight 2.0 stored, so score = 2 * idf)
IDF_OF = {}


class _Qdrant(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.server.seen.append(("GET", self.path))
        self._send({"result": {"points_count": POINTS}})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        self.server.seen.append(("POST", self.path))
        out = []
        for s in body["searches"]:
            h = s["query"]["nearest"]["indices"][0]
            idf = IDF_OF.get(h)
            if idf is None:
                out.append({"points": []})              # df == 0: absent from the index
                continue
            out.append({"points": [{"score": 2.0 * idf,
                                    "vector": {"bm25": {"indices": [h], "values": [2.0]}}}]})
        self._send({"result": out})


@pytest.fixture
def qdrant():
    srv = HTTPServer(("127.0.0.1", 0), _Qdrant)
    srv.seen = []
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def corpus(tmp_path):
    f = tmp_path / "q.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in [
        {"query": "the law of contracts"}, {"question": "Art. 336 CO Kuendigung"},
        {"q": "what is the law"}, {"other": "ignored"}]) + "\nnot json\n")
    y = tmp_path / "q.yaml"
    y.write_text("eval:\n  - {query: 'the contract'}\n  - {task: 'fedlex Obligationenrecht'}\n")
    return [str(f), str(y)]


def _vocab(corpus):
    vocab, n_texts, n_files = M.collect_vocabulary(corpus)
    return vocab, n_texts, n_files


def test_vocabulary_is_the_servers_tokenizer_over_every_query_key(corpus):
    vocab, n_texts, n_files = _vocab(corpus)
    assert (n_texts, n_files) == (5, 2)          # 3 from jsonl (the `other` key is ignored); 2 from yaml
    assert vocab and all(isinstance(h, int) for h in vocab)
    assert "law" in set(vocab.values()) and "the" in set(vocab.values())


def test_implied_df_inverts_qdrants_formula():
    n, df = 10_000, 250
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    assert M.implied_df(idf, n) == pytest.approx(df)
    assert M.implied_df(-1.0, n) == float(n)


def test_measure_is_score_over_stored_weight_and_skips_absent_tokens(qdrant):
    IDF_OF.clear()
    IDF_OF.update({7: 0.4442, 9: 5.5})
    got = M.measure(f"http://127.0.0.1:{qdrant.server_port}", "c", [7, 8, 9], 2, 0)
    assert got == {7: pytest.approx(0.4442), 9: pytest.approx(5.5)}      # 8 absent: df == 0
    assert [m for m, _ in qdrant.seen] == ["POST", "POST"]             # batch 2 -> two requests
    assert all(p == "/collections/c/points/query/batch" for _, p in qdrant.seen)


def test_main_writes_the_table_exactly(qdrant, corpus, tmp_path, monkeypatch):
    vocab, _, _ = _vocab(corpus)
    hashes = sorted(vocab)
    IDF_OF.clear()
    IDF_OF.update({h: 0.5 + i * 0.25 for i, h in enumerate(hashes)})    # some above the 6.0 cap
    IDF_OF[hashes[0]] = 7.0                                              # above capture_max_idf
    out = tmp_path / "idf.json"
    monkeypatch.delenv("AW_BM25_QSTOP", raising=False)
    monkeypatch.setattr(sys, "argv", ["measure_bm25_idf.py", "--qdrant",
                                      f"http://127.0.0.1:{qdrant.server_port}", "--collection", "cx",
                                      "--out", str(out), "--corpus", *corpus,
                                      "--batch", "5", "--sleep-ms", "0"])
    M.main()
    got = json.loads(out.read_text())
    assert got.pop("measured_at").endswith("Z")
    kept = {h: v for h, v in IDF_OF.items() if v < 6.0}
    order = sorted(kept, key=lambda h: kept[h])
    assert got == {
        "_comment": "ADR-099 Ruling 2 — measured live BM25 IDF per query token. An absent token is "
                    "treated as high-IDF (kept). Regenerate with `python3 -m cytir.index.idf_table`.",
        "collection": "cx", "points_count": POINTS,
        "fold_accents": got["fold_accents"],
        "method": "idf = single_token_score / stored_weight (qdrant modifier=idf)",
        "capture_max_idf": 6.0, "n_vocabulary": len(vocab), "n_measured": len(IDF_OF),
        "idf": {str(h): round(kept[h], 4) for h in order},
        "tokens": {str(h): vocab[h] for h in order},
    }
    assert isinstance(got["fold_accents"], bool)
    # key ORDER is part of the file's bytes
    assert list(json.loads(out.read_text()).keys()) == [
        "_comment", "collection", "points_count", "measured_at", "fold_accents", "method",
        "capture_max_idf", "n_vocabulary", "n_measured", "idf", "tokens"]


def test_main_refuses_while_qstop_is_on(monkeypatch, tmp_path):
    monkeypatch.setenv("AW_BM25_QSTOP", "1")
    monkeypatch.setattr(sys, "argv", ["measure_bm25_idf.py", "--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit) as e:
        M.main()
    assert "refusing to measure with AW_BM25_QSTOP=1" in str(e.value)


def test_the_script_path_still_runs_and_names_its_flags():
    """`cytir.index.idf_table seal` runs this module (`-m`) as a subprocess; it must keep working."""
    r = subprocess.run([sys.executable, "-m", "cytir.index.idf_table", "--help"], cwd=str(HERE.parent),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "--capture-max-idf" in r.stdout and "--collection" in r.stdout

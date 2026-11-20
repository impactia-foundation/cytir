"""A STAND-IN embedder for the index-serve gate's CI job: NOT the pinned model.

The `index-serve-gate` job builds a 70-row index and queries it. Embedding with the real model needs
torch, the 2.3 GB `Snowflake/snowflake-arctic-embed-l-v2.0` weights and minutes of CPU, none of which
a small CI runner holds. So this module makes the document vectors AND the query
vectors with one deterministic function: a signed hashed bag of folded word unigrams and bigrams,
1024-wide, L2-normalised. It has the property the gate needs (texts that share rare words are close),
and none of the model's: no synonyms, no cross-language meaning, no semantics at all.

WHAT A GREEN RUN OF THE JOB THEREFORE PROVES: the plumbing. The packs build, the collection has the
declared shape, `cyt serve` starts on it and answers over HTTP, the gold documents are in the index,
the lexical leg (real BM25 weights, real IDF) and the fusion rank them, and `cyt gate` scores them
the same way twice. It proves NOTHING about the real embedder's ranking: the baseline verdict
records that (`vectors.kind == "stand-in"`), and a gate run on real vectors needs its own baseline.

Everything this module writes is marked `stand_in_for_tests` in the vector manifest, next to the
pinned embedder's identity (the build's pin check reads the latter).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import re
import unicodedata

import numpy as np

KIND = "hashed-bow-v1"
DIM = 1024
_WORD = re.compile(r"[a-z0-9]{2,}")


def _fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn")


def _bucket(token: str) -> tuple[int, float]:
    h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "little")
    return h % DIM, 1.0 if (h >> 40) & 1 else -1.0


def embed(text: str) -> np.ndarray:
    """One 1024-d unit vector (float32) for `text`. A pure function of the text."""
    words = _WORD.findall(_fold(text))
    v = np.zeros(DIM, dtype=np.float64)
    for tok in words + [f"{a}_{b}" for a, b in zip(words, words[1:])]:
        i, s = _bucket(tok)
        v[i] += s
    n = float(np.linalg.norm(v))
    return (v / n if n else v).astype(np.float32)


def doc_text(row: dict) -> str:
    return f"{row.get('title') or ''}\n{row.get('text') or ''}"


def write_vectors(stage: pathlib.Path) -> int:
    """`<stage>/vectors/shard-0000.*` for the rows of `<stage>/chunks/shard-0000.texts.jsonl.gz`,
    in the sealed shard format `cyt embed shards` writes. Returns the row count."""
    from cytir.core.model_pin import EMBED_MODEL_ID, embed_model_revision
    with gzip.open(stage / "chunks" / "shard-0000.texts.jsonl.gz", "rt", encoding="utf-8") as fh:
        rows = [json.loads(x) for x in fh]
    out = stage / "vectors"
    out.mkdir(parents=True, exist_ok=True)
    np.stack([embed(doc_text(r)) for r in rows]).astype("<f2").tofile(out / "shard-0000.vectors.f16")
    with open(out / "shard-0000.row_ids.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({"point_id": r["point_id"]}) + "\n")
    (out / "shard-0000.vectors.manifest.json").write_text(json.dumps({
        "schema": "aw.413a.vectors.v1", "shard": "shard-0000", "n": len(rows), "dim": DIM,
        "dtype": "float16",
        "embedder": {"model": EMBED_MODEL_ID, "revision_pinned": embed_model_revision()},
        "stand_in_for_tests": True, "stand_in_kind": KIND}), encoding="utf-8")
    return len(rows)


# ------------------------------------------------------------------ a TEI-shaped query server
def tei_handler():
    """A request handler class that answers `/info` and `/embed` like a text-embeddings-inference
    server of the pinned model, with `embed()` of each input (a stand-in; see the module doc)."""
    import http.server

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, doc):
            body = json.dumps(doc).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            from cytir.core.model_pin import EMBED_MODEL_ID, embed_model_revision
            # the stand-in announces itself: `cyt serve` records this in its receipt and `cyt gate`
            # reads it to say its verdict proves plumbing only
            self._send({"model_id": EMBED_MODEL_ID, "model_sha": embed_model_revision(),
                        "stand_in": KIND})

        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length", 0))
            inputs = json.loads(self.rfile.read(n))["inputs"]
            if isinstance(inputs, str):
                inputs = [inputs]
            self._send([embed(t).tolist() for t in inputs])

        def log_message(self, *a):
            pass

    return Handler


def main(argv=None) -> int:
    """The two things a walkthrough needs outside pytest:

        python tests/standin_embedder.py vectors STAGE_DIR [STAGE_DIR ...]   # vectors/ beside chunks/
        python tests/standin_embedder.py tei PORT                            # a TEI-shaped query server
    """
    import http.server
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) >= 2 and args[0] == "vectors":
        for d in args[1:]:
            print(f"{d}: {write_vectors(pathlib.Path(d))} stand-in vectors ({KIND}, NOT the pinned model)")
        return 0
    if len(args) == 2 and args[0] == "tei":
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", int(args[1])), tei_handler())
        print(f"stand-in query embedder on http://127.0.0.1:{args[1]} ({KIND}, NOT the pinned model)",
              flush=True)
        srv.serve_forever()
        return 0
    print(main.__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

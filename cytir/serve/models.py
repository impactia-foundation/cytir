"""Lazy-loaded query embedder + reranker (CPU on the serving box — the GPU-light
serve path; spec §5.2). Set AW_RERANK=0 to disable reranking.

The reranker uses bge-reranker-v2-m3 directly via transformers
AutoModelForSequenceClassification (the official model-card usage) rather than
FlagEmbedding's FlagReranker, which calls `tokenizer.prepare_for_model` and
breaks on current transformers (XLMRobertaTokenizer has no such attribute).
"""
from __future__ import annotations

from cytir.core import internal_http as IH  # the one door to the operator's own services (consent step 5, task 1673)
import logging
import threading
from functools import lru_cache

from .settings import SETTINGS as _S

log = logging.getLogger("aw_server.models")

# Full-precision (1024-dim) query vector side-channel for the ranking head
# (task 205). The dense leg needs only MRL-128, but the learned rank head
# conditions on the FULL arctic vector. TEI already returns the full 1024
# (normalized) BEFORE we slice to 128 — capturing it here is free (no extra
# embed). Thread-local because FastAPI runs sync routes on a threadpool and
# concurrent queries must not clobber each other's vector. Read via
# last_full_query_vec() immediately after embed_query() in the same thread.
_tls = threading.local()


def last_full_query_vec():
    """The full 1024-dim normalized query vector from the most recent
    embed_query() call ON THIS THREAD, or None (BGE path / not captured)."""
    return getattr(_tls, "full_qvec", None)

RERANK_ENABLED = _S.runtime.rerank
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_MAX_LEN = 512


# Scale stack (ADR-018; bake-off + 1.13M scale-validation): arctic-embed-l-v2
# truncated to Matryoshka-256, dense-only (no learned-sparse; BM25 is the lexical
# leg). AW_EMBEDDER=bge-m3 falls back to the P0 incumbent.
# Task 432 — the embedder identity is a MIRROR CONTRACT with the build side:
# vectors in the index and vectors computed for a query must come from the same
# model at the same revision, or the geometry silently stops matching. Both
# sides read `cytir.core.model_pin` -- the aw-api image installs the `aw`
# package (task 646 S2; see server/Dockerfile), so there is no more hand-
# maintained copy to keep byte-identical (was `cytir/serve/model_pin.py`
# until this move).
from cytir.core.model_pin import EMBED_MODEL_ID, embed_model_revision, pin_note

EMBED_MODEL = EMBED_MODEL_ID
EMBED_REVISION = embed_model_revision()
# The MRL truncation width. This is a MIRROR CONTRACT with three other places
# and it is the one number that must move ATOMICALLY with a collection flip:
#
#   * the collection's `VectorParams(size=...)`
#   * the build side's `mrl_trunc(vec, dim)`
#   * this slice, which produces the query vector
#
# A mismatch is not a quality bug, it is an outage: Qdrant rejects a query whose
# vector width differs from the collection's. Until task 478 this was a literal,
# which meant a 256-dim collection could not be QUERIED AT ALL without editing
# and redeploying the server — so the serving-config bake could not be run, and
# the flip would have had to couple a code edit to an alias swap with no way to
# stage or roll back the two independently.
#
# Env-driven with the shipped default, so: the default deploy is byte-identical
# in behaviour to before, the bake can point one process at a 256-dim collection,
# and the flip sets AW_MRL_DIM alongside AW_COLLECTION in the same compose change.
# `cytir/core/serving_config.py` owns the registry of legal widths and carries the
# evidence for each; the serving-config test asserts this env name and
# default still agree with it.
MRL_DIM = _S.runtime.mrl_dim
_USE_BGE = _S.runtime.embedder == "bge-m3"

# TEI sidecar (task 071): when AW_TEI_URL is set (e.g. http://tei:80), query
# embedding goes over HTTP to text-embeddings-inference instead of loading the
# 2.6GB torch model in-process. Parity validated at cosine ~ 1.0 (task 070):
# prompt_name="query" + slice[:MRL_DIM] + L2-renorm reproduces the ST path.
# On ANY TEI failure we fall back to the in-process embedder (logged).
TEI_URL = _S.runtime.tei_url.rstrip("/")
TEI_TIMEOUT = _S.runtime.tei_timeout
_tei_announced = False

# Task 432: the read side of the mirror contract. TEI is the embedder that
# actually answers production queries, and it was launched with no --revision:
# `GET /info` returned `"model_sha": null`, i.e. serving could not report which
# revision it was embedding queries with. `deploy/docker-compose.yaml` now
# passes --revision; this check proves it rather than assuming it.
#
# It LOGS rather than refuses on purpose. Refusing would turn a pin drift into a
# search outage, and the fallback (in-process, pinned) is not obviously safer
# than the sidecar. `tei_revision_status()` surfaces the finding so /health and
# an operator can see it. Deployment doctrine: make the drift visible, do not
# make it fatal in the request path.
_tei_rev = {"checked": False, "model_sha": None, "ok": None, "detail": None}


def tei_revision_status() -> dict:
    """{'ok': True|False|None, 'model_sha': ..., 'detail': ...}. ok=None means
    not checked yet (or TEI not configured)."""
    return dict(_tei_rev)


def _check_tei_revision() -> None:
    """Once per process: compare TEI's served model + revision to the pin."""
    if _tei_rev["checked"] or not TEI_URL:
        return
    _tei_rev["checked"] = True
    try:
        info = IH.fetch_json(f"{TEI_URL}/info", timeout=TEI_TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        _tei_rev.update(ok=None, detail=f"/info unreachable: {exc!r}")
        return
    served_id = info.get("model_id")
    sha = info.get("model_sha")
    _tei_rev["model_sha"] = sha
    want = embed_model_revision()
    if served_id != EMBED_MODEL_ID:
        _tei_rev.update(ok=False, detail=f"TEI serves {served_id!r}, pin is {EMBED_MODEL_ID!r}")
        log.error("EMBEDDER PIN: TEI model id mismatch — %s", _tei_rev["detail"])
    elif sha is None:
        _tei_rev.update(ok=False, detail="TEI reports model_sha=null — launched without "
                                         "--revision, so the served revision is unknown")
        log.error("EMBEDDER PIN: TEI cannot report its model revision (model_sha=null). "
                  "Add `--revision %s` to the tei command in deploy/docker-compose.yaml "
                  "and recreate the container.", want)
    elif sha != want:
        _tei_rev.update(ok=False, detail=f"TEI serves revision {sha}, pin is {want}")
        log.error("EMBEDDER PIN MISMATCH: query vectors come from %s but the index was "
                  "built at %s. Retrieval quality is silently degraded.", sha, want)
    else:
        _tei_rev.update(ok=True, detail=f"TEI serves the pinned revision {sha}")
        log.info("EMBEDDER PIN: %s", _tei_rev["detail"])


@lru_cache(maxsize=1)
def embedder():
    if _USE_BGE:
        from FlagEmbedding import BGEM3FlagModel
        return BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    from sentence_transformers import SentenceTransformer
    # Pinned: the in-process fallback must not answer queries from a different
    # snapshot than the one the index was built with (task 429 blocker E).
    log.info("embed_query in-process: %s", pin_note())
    m = SentenceTransformer(EMBED_MODEL, revision=EMBED_REVISION,
                            trust_remote_code=True)
    m.max_seq_length = 512
    return m


@lru_cache(maxsize=1)
def _reranker():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(RERANKER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
    model.eval()
    torch.set_grad_enabled(False)
    return tok, model


def _embed_query_tei(q: str):
    """Embed via the TEI sidecar. Raises on any failure (caller falls back).

    TEI applies the model's "query: " prefix via prompt_name and normalizes the
    full 1024-dim vector; we slice to MRL_DIM and renormalize client-side —
    identical post-processing to the in-process path.
    """
    global _tei_announced
    import numpy as np
    _check_tei_revision()
    vecs = IH.fetch_json(
        f"{TEI_URL}/embed",
        body={"inputs": [q], "prompt_name": "query",
              "normalize": True, "truncate": True},
        timeout=TEI_TIMEOUT,
    )
    full = np.asarray(vecs[0], dtype="float32")  # 1024-dim, TEI-normalized
    _tls.full_qvec = full                             # head side-channel (task 205)
    v = full[:MRL_DIM]
    n = float(np.linalg.norm(v)) or 1.0
    if not _tei_announced:
        log.info("embed_query: using TEI sidecar at %s", TEI_URL)
        _tei_announced = True
    return (v / n).tolist(), None


# Task 342: one-entry, per-thread memo of the last embedded query string.
# `/search?authorities=1` embeds the SAME query TWICE — once in
# retrieval.search() for the passage legs, then again in
# authority.find_authorities() for the source-card sidecar — a second TEI round
# trip (measured 0.09-0.12 s on prod, ~5 % of a novel-query response) for a
# vector that is deterministic in the query string. On a hit the memo returns
# the IDENTICAL objects and restores the full-1024 head side-channel, so the
# ranking inputs cannot differ; it is not a cache with a staleness question,
# because the embedder cannot change without restarting the process.
# Default OFF => the second embed still happens; rollback = AW_EMBED_MEMO=0.
_EMBED_MEMO = _S.runtime.embed_memo


def embed_query(q: str):
    """Return (dense_list, lexical_or_None). arctic → MRL dense, no lexical."""
    if not _EMBED_MEMO:
        return _embed_query_uncached(q)
    memo = getattr(_tls, "embed_memo", None)
    if memo is not None and memo[0] == q:
        _tls.full_qvec = memo[2]      # head side-channel (task 205)
        return memo[1]
    out = _embed_query_uncached(q)
    _tls.embed_memo = (q, out, getattr(_tls, "full_qvec", None))
    return out


def _embed_query_uncached(q: str):
    if not _USE_BGE and TEI_URL:
        try:
            return _embed_query_tei(q)
        except Exception as e:
            log.warning("TEI embed failed (%s: %s) — falling back to "
                        "in-process embedder", type(e).__name__, e)
    m = embedder()
    if _USE_BGE:
        out = m.encode([q], return_dense=True, return_sparse=True,
                       return_colbert_vecs=False)
        dense = out["dense_vecs"][0]
        return (dense.tolist() if hasattr(dense, "tolist") else list(dense)), \
            out["lexical_weights"][0]
    import numpy as np
    avail = set(getattr(m, "prompts", {}) or {})
    kw = {"normalize_embeddings": True, "convert_to_numpy": True}
    if "query" in avail:
        kw["prompt_name"] = "query"
    full = np.asarray(m.encode([q], **kw)[0], dtype="float32")  # 1024, normalized
    _tls.full_qvec = full                                       # head side-channel
    v = full[:MRL_DIM]
    n = float(np.linalg.norm(v)) or 1.0
    return (v / n).tolist(), None


def rerank(q: str, docs: list[str]) -> list[float]:
    """Cross-encoder relevance scores in [0,1] for (q, doc) pairs."""
    if not docs:
        return []
    import torch
    tok, model = _reranker()
    pairs = [[q, d] for d in docs]
    scores: list[float] = []
    # modest batches to keep CPU memory + latency bounded
    for i in range(0, len(pairs), 16):
        batch = pairs[i:i + 16]
        inputs = tok(batch, padding=True, truncation=True,
                     max_length=RERANK_MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits.view(-1).float()
            scores.extend(torch.sigmoid(logits).tolist())
    return scores

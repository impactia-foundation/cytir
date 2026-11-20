"""The embedder registry and loader (task 682, migration S6; moved verbatim from
the old `index_bakeoff` script, which re-exports both names).

`EMBEDDERS` names the models and `load_embedder` builds `(encode_docs,
encode_queries, spec)` for one of them. The embed stage
(`cytir.embed.collection_stage`) calls it, so the package no longer reaches into
`pipeline/` for its encoder (task 643 section 6.1: `embed/model.py`).
Only arctic-l-v2 is revision-pinned (`cytir.core.model_pin`); the other three are
bake-off history (ADR-018).
"""
from __future__ import annotations

from cytir.core.model_pin import EMBED_MODEL_ID, embed_model_revision, pin_note

# --- embedder registry ------------------------------------------------------
# dim   : dense vector size
# kind  : "flag" (FlagEmbedding BGE-M3) | "st" (sentence-transformers)
# model : HF id
# has_lexical : exposes a learned-sparse leg (only bge-m3)
# query_prompt_name : sentence-transformers prompt to use for QUERIES (None=raw)
# doc_prompt_name   : sentence-transformers prompt to use for DOCUMENTS (None=raw)
EMBEDDERS = {
    "bge-m3": dict(dim=1024, kind="flag", model="BAAI/bge-m3",
                   has_lexical=True, query_prompt_name=None, doc_prompt_name=None),
    "qwen3-0.6b": dict(dim=1024, kind="st", model="Qwen/Qwen3-Embedding-0.6B",
                       has_lexical=False, query_prompt_name="query", doc_prompt_name=None),
    "qwen3-4b": dict(dim=2560, kind="st", model="Qwen/Qwen3-Embedding-4B",
                     has_lexical=False, query_prompt_name="query", doc_prompt_name=None),
    # revision: the model_pin.py commit sha. Task 429 blocker E — load_embedder
    # took no revision, so document vectors came from whatever HuggingFace
    # served that day and "the same index" was not reproducible across months.
    # Only arctic is pinned: the other three are bake-off history, never used to
    # build a shipped index (ADR-018), and pinning them would assert a
    # reproducibility we never established for them.
    "arctic-l-v2": dict(dim=1024, kind="st", model=EMBED_MODEL_ID,
                        has_lexical=False, query_prompt_name="query", doc_prompt_name=None,
                        revision=None),   # filled from model_pin at load time
}


def load_embedder(name: str, use_fp16: bool, max_seq: int = 512,
                  model_path: str | None = None, revision: str | None = None):
    """Return (encode_docs, encode_queries, spec). encode_* return
    (dense_list, lexical_list_or_None). max_seq truncates inputs uniformly —
    Qwen3 defaults to 32k tokens which OOMs on long passages and makes the
    embedder comparison unfair; 512 is right for passage retrieval.

    `revision` pins the HuggingFace commit. For arctic it defaults to the
    committed pin in `model_pin.py` — the same sha the production embed driver
    passes on the command line — so callers get a reproducible embedder without
    having to know the sha. Pass it explicitly only to re-embed against a
    different snapshot on purpose; `spec["revision"]` always records what was
    actually used, for receipts.
    """
    spec = dict(EMBEDDERS[name])
    if spec.get("revision", "absent") is None and revision is None:
        # Registry entry opts in to the pin; resolve it now (honours
        # AW_MODEL_REVISION and validates the sha shape).
        revision = embed_model_revision()
    if revision is not None:
        spec["revision"] = revision
    if model_path:
        # A caller that has already verified an immutable model snapshot may
        # name it explicitly.  This is useful on sealed/offline GPU workers;
        # callers still record the model id/revision in their own manifest.
        # A local directory has no HuggingFace revision to resolve, so the pin
        # is recorded but not passed to the loader.
        spec["model"] = model_path
        spec["model_path_override"] = True
    if spec["kind"] == "flag":
        # BGE-M3 ships only .bin (pickle) weights; with torch<2.6 newer
        # transformers blocks torch.load via check_torch_load_is_safe()
        # (CVE-2025-32434). The model is trusted (BAAI), so neutralize the check.
        _noop = lambda *a, **k: None
        try:
            import transformers.utils.import_utils as _iu
            _iu.check_torch_load_is_safe = _noop
            import transformers.modeling_utils as _mu
            if hasattr(_mu, "check_torch_load_is_safe"):
                _mu.check_torch_load_is_safe = _noop
        except Exception:  # noqa: BLE001
            pass
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel(spec["model"], use_fp16=use_fp16)

        def encode(texts, is_query):
            out = model.encode(texts, return_dense=True,
                               return_sparse=spec["has_lexical"],
                               return_colbert_vecs=False, max_length=max_seq)
            dense = [v.tolist() for v in out["dense_vecs"]]
            lex = out["lexical_weights"] if spec["has_lexical"] else None
            return dense, lex

        return (lambda t: encode(t, False), lambda t: encode(t, True), spec)

    # sentence-transformers
    import torch
    from sentence_transformers import SentenceTransformer
    dtype = torch.float16 if use_fp16 else torch.float32
    st_kwargs = dict(device="cuda" if use_fp16 else "cpu",
                     model_kwargs={"torch_dtype": dtype},
                     trust_remote_code=True)
    if spec.get("revision") and not spec.get("model_path_override"):
        st_kwargs["revision"] = spec["revision"]
        print(f"[embedder] {pin_note()}")
    model = SentenceTransformer(spec["model"], **st_kwargs)
    model.max_seq_length = max_seq   # cap Qwen3's 32k default → no OOM, fair compare
    avail = set(getattr(model, "prompts", {}) or {})

    def _st_encode(texts, prompt_name, batch):
        kw = dict(batch_size=batch, normalize_embeddings=True,
                  convert_to_numpy=True, show_progress_bar=False)
        if prompt_name and prompt_name in avail:
            kw["prompt_name"] = prompt_name
        vecs = model.encode(list(texts), **kw)
        return [v.tolist() for v in vecs], None

    def encode_docs(texts, batch=64):
        return _st_encode(texts, spec["doc_prompt_name"], batch)

    def encode_queries(texts, batch=64):
        return _st_encode(texts, spec["query_prompt_name"], batch)

    spec = dict(spec, _st_prompts=sorted(avail))
    return encode_docs, encode_queries, spec

"""BM25 tokenizer + accent-fold epoch gate (moved out of `cytir.core.common`, task 2333).

One tokenization rule for the index side; `cytir.serve.schema` is the query-side mirror.
`_FOLD_ACCENTS` lives HERE: set `cytir.core.bm25_tokenizer._FOLD_ACCENTS`, never `cytir.core.common._FOLD_ACCENTS`.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import unicodedata


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


# apostrophe-joined names (O'Hear, O'Brien, d'Angelo) — index the joined form
# additively so the distinctive surname survives. Must match server/schema.py.
_APOS_RE = re.compile(r"\w+(?:['’]\w+)+", re.UNICODE)


# German compound splitter. For the BM25 leg we additively index the component words
# of German compounds so a query for "Schutzdauer" also matches "Schutz"/"Dauer"
# (ADR-004). Index- and query-side MUST use the same tokenization: the rule lives in
# ONE place, `cytir.core.decompound` (CharSplit, MIT; ADR-121), and the server's
# schema.py calls the same function. If the pinned splitter or its table is missing it
# RAISES: no silent "no splitting".
from cytir.core.decompound import decompound_parts as _decompound_parts  # noqa: E402


def _decompound(token: str) -> list[str]:
    return list(_decompound_parts(token))


def _hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


# --- AW_FOLD_ACCENTS (task 290, validated prototype in task 290 fold_index.py) --
# Accent-fold the BM25 leg on BOTH the write (this file) and query
# (cytir/serve/schema.py) paths so accented query terms hit the index
# ("Genève"→"geneve", "psychédélique"→"psychedelique"). NFKD-normalize + strip
# combining marks, applied to each token/decompound-part/apostrophe-joined form
# *before* `_hash()`. Env-gated, default OFF: when OFF the tokenizer is
# byte-identical to the pre-fold behaviour so ASCII exact-lookups (art/336/co/
# fedlex/UIDs/signatures) cannot regress. This is a tokenizer change, so it can
# only land via a full rebuild with the serving flag flipped in lockstep
# (folding a query token against an un-folded index is identity). The write- and
# query-side `_fold`/`bm25_tokens` MUST stay byte-identical (mirror contract);
# tests/test_bm25_fold.py asserts both paths produce the same tokens.
def _env_truthy(v: "str | None") -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


# Module-level so the hot path (bulk build ~50M chunks) pays no per-call env
# cost; tests toggle by setting `aw_common._FOLD_ACCENTS` directly.
_FOLD_ACCENTS = _env_truthy(os.environ.get("AW_FOLD_ACCENTS"))


#: The tokenizer epoch every collection in service was built at:
#: `tok=2` == accent-folded (`aw_b1_identity` §"epochs.md §2, flips at aw_b1"),
#: which is what `deploy/docker-compose.yaml` sets (`AW_FOLD_ACCENTS: "1"`).
TOK_EPOCH_FOLD = {1: False, 2: True}


def require_index_fold(expected_epoch: int | None = None, who: str = "") -> bool:
    """Refuse to tokenise under a fold setting the index was not built at.

    WHY THIS EXISTS. `_FOLD_ACCENTS` defaults OFF on BOTH sides, deliberately,
    so a pre-fold index cannot regress -- and the value is read at import time.
    A script that imports either tokenizer without the env therefore hashes
    `chômage` differently from the tok@2 collection it is querying, silently,
    and every accented term reads as ABSENT.

    MEASURED (task 564): `task 562 bisect_562.py` has no runner that
    sets it, while its sibling `run_probe_562.sh` runs under `prod_env_run.py`
    and does. 4 of the 27 golds task 562 attributed to family composition are
    that artefact -- their "lost" words are in the candidate's own posting --
    and three more lose fewer words than reported
    (`task 564 refold-words.json`). At the time of the pin (task 564)
    several pipeline modules imported the query-side tokenizer with no pin at
    all; see the mechanism-callers data for the current list.

    Fails CLOSED: a caller that cannot state the epoch it is measuring against
    should not be measuring. `AW_TOK_EPOCH` overrides the default for a
    deliberate pre-fold run, and the value used travels in the message.
    """
    if expected_epoch is None:
        expected_epoch = int(os.environ.get("AW_TOK_EPOCH", "2"))
    if expected_epoch not in TOK_EPOCH_FOLD:
        raise RuntimeError(f"unknown tokenizer epoch {expected_epoch!r}")
    want = TOK_EPOCH_FOLD[expected_epoch]
    # task 2333: the flag moved here from `cytir.core.common`. A write to the OLD module is a
    # silent no-op on the tokenizer, so fail closed on it instead of hashing at the wrong epoch.
    legacy = sys.modules.get("cytir.core.common")
    if legacy is not None and "_FOLD_ACCENTS" in vars(legacy):
        raise RuntimeError(
            "cytir.core.common._FOLD_ACCENTS was assigned, but the flag now lives in "
            "cytir.core.bm25_tokenizer._FOLD_ACCENTS (task 2333); the write has no effect")
    sides = {"aw_common": _FOLD_ACCENTS}
    schema = sys.modules.get("cytir.serve.schema")
    if schema is not None:
        sides["cytir.serve.schema"] = bool(schema._FOLD_ACCENTS)
    wrong = {k: v for k, v in sides.items() if v is not want}
    if wrong:
        raise RuntimeError(
            f"{who or 'this process'} tokenises at fold={sides} but the index "
            f"is tok@{expected_epoch} (fold={want}); token hashes would not "
            f"match. Set AW_FOLD_ACCENTS=1 (or import aw_b1_projection, which "
            f"pins it), or state AW_TOK_EPOCH for a deliberate pre-fold run.")
    return want


def _fold(s: str) -> str:
    """NFKD-normalize and strip combining marks (accent-fold). Must be
    byte-identical to cytir/serve/schema.py::_fold."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def bm25_tokens(text: str, lang: str | None = None, *,
                decompound: bool = True) -> dict[int, float]:
    """{term-hash: term-frequency} sparse vector for the BM25 leg (Qdrant
    applies IDF at query time). For German (lang 'de' or unknown), additively
    index decompounded parts at half weight. Must match query-side tokenization.

    `decompound=False` is NOT a second tokenizer: it is the posting a writer
    WITHOUT the splitter produced (task 710: three writers ran so). The rebuild
    (the deleted `aw.index.rebuild_e10`, E2, used it only to PROVE that it composed the same
    surface as the stored no-parts row before it replaces that row's posting.
    Every writer and the query side keep the default.
    """
    tf: dict[int, float] = {}
    de = decompound and lang in (None, "de")
    fold = _FOLD_ACCENTS
    for tok in _TOKEN_RE.findall(text.lower()):
        if len(tok) < 2:
            continue
        key = _fold(tok) if fold else tok
        tf[_hash(key)] = tf.get(_hash(key), 0.0) + 1.0
        if de:
            # Decompound the raw (un-folded) token so char_split still sees the
            # umlauts it splits on, then fold each part before hashing.
            for part in _decompound(tok):
                pk = _fold(part) if fold else part
                tf[_hash(pk)] = tf.get(_hash(pk), 0.0) + 0.5
    for m in _APOS_RE.findall(text.lower()):
        j = m.replace("'", "").replace("’", "")
        if len(j) >= 2:
            jk = _fold(j) if fold else j
            tf[_hash(jk)] = tf.get(_hash(jk), 0.0) + 1.0
    return tf

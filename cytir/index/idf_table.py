"""Measure the LIVE per-token BM25 IDF of a query vocabulary (ADR-099 Ruling 2).

Why this exists
---------------
ADR-099 Ruling 2 suppresses query terms whose IDF falls below a threshold. The
serving path cannot measure IDF at query time: Qdrant owns the `modifier=idf`
statistic and exposes it only through scores, and the single-token df-window
probe the server already runs (`AW_BM25_PER_TOKEN_LIMIT`) proves `df < limit`
only for RARE terms — it says nothing about how common a common term is, which
is exactly the regime the flood lives in. So the IDF numbers are MEASURED
offline, from the live index, and shipped as a committed table
(`cytir/serve/bm25_idf.json`) that the tokenizer reads.

Method (exact, not an estimate)
-------------------------------
Qdrant scores a sparse query with `modifier=idf` as

    score(d) = sum_t  idf(t) * q_value(t) * d_value(t)

For a SINGLE-token query with `q_value = 1.0` this collapses to
`score(d) = idf(t) * d_value(t)`. Ask for the single best-scoring document with
`with_vector=true`, read that document's stored weight `d_value(t)` out of the
returned sparse vector, and

    idf(t) = score / d_value(t)

is exact arithmetic on the live collection's own statistics — no sampling, no
scroll, no full-scan (see notes/prod-serving-box-fragile-scrolls). Two cheap
read-only requests' worth of work per token, batched. The measured
`idf('the') = 0.4442` reproduces task 446's independently-derived number.

The implied document frequency follows from Qdrant's IDF formula
`idf = ln(1 + (N - n + 0.5) / (n + 0.5))`, inverted here for reporting only.

Vocabulary
----------
The tokens measured are the ones AW's own query traffic actually produces: the
eval suites plus every committed probe run. That keeps the table corpus-derived
and language-uniform (EN/DE/FR/IT fall out of the corpus, ADR-099's explicit
requirement) instead of curated per-language stopword lists.

Tokens are keyed by the tokenizer's HASH, so the table is only valid for the
fold setting it was measured under; `AW_FOLD_ACCENTS` is recorded in the header
and checked by the loader.

NOTE on German de-compounding: the splitter (`cytir.core.decompound`, CharSplit; ADR-121) needs
its fetched n-gram table (`python3 -m cytir.embed.fetch_models`). It raises without it, so a
measurement on a box without the table fails loudly instead of leaving compound HALVES out
of the vocabulary (the pre-ADR-121 behaviour). Measure where the table is fetched.

Usage
-----
    python3 -m cytir.index.idf_table \
        --qdrant http://127.0.0.1:16333 --collection aw_prod \
        --out cytir/serve/bm25_idf.json

Read-only: it issues `points/query/batch` searches and nothing else.

`python -m cytir.index.idf_table seal ...` seals a GATE table on top of that measurement (the second
half of this module; moved from the pipeline tree in task 1234, behaviour unchanged except that
the receipt's `sealed_by` names this door).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os
import pathlib
import subprocess
import sys
import time
from cytir.core import internal_http as IH

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# The tokenizer under measurement must be the SERVING one, not a copy.
from cytir.core.serve_layout import SERVER_REL  # noqa: E402
from cytir.serve import schema as SC  # noqa: E402
from cytir.serve import settings as ST  # noqa: E402
from cytir.core import twin as TWIN  # noqa: E402
from cytir.index import layout as LAYOUT  # noqa: E402

DEFAULT_CORPUS = [
    "data/eval_tasks.yaml", "data/eval_names.yaml",
    "data/eval_legal_authority.yaml", "data/eval_edgecases.yaml",
    "data/eval_authoritative.yaml", "data/eval_agentic.yaml",
    "probes/slice_b_pool.jsonl",
    "probes/runs/*/sample.jsonl", "probes/runs/*/sample_sliceb.jsonl",
]
QUERY_KEYS = ("query", "question", "q", "task")


def _texts_from(path: str) -> list[str]:
    out: list[str] = []
    if path.endswith((".yaml", ".yml")):
        import yaml
        doc = yaml.safe_load(open(path))
        items = doc.get("eval", doc) if isinstance(doc, dict) else doc
        for it in items if isinstance(items, list) else []:
            if isinstance(it, dict):
                out.extend(str(it[k]) for k in QUERY_KEYS if it.get(k))
    else:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                it = json.loads(line)
            except ValueError:
                continue
            if isinstance(it, dict):
                out.extend(str(it[k]) for k in QUERY_KEYS if it.get(k))
    return out


def collect_vocabulary(patterns: list[str]) -> tuple[dict[int, str], int, int]:
    """{token hash -> a surface word that emits it}, n_texts, n_files."""
    vocab: dict[int, str] = {}
    n_texts = n_files = 0
    for pat in patterns:
        for path in sorted(glob.glob(str(REPO / pat))):
            n_files += 1
            for text in _texts_from(path):
                n_texts += 1
                for word, _primary, hashes in SC.bm25_token_groups(text, None):
                    for h in hashes:
                        vocab.setdefault(h, word)
    return vocab, n_texts, n_files


def implied_df(idf: float, n_points: int) -> float:
    """Invert Qdrant's `idf = ln(1 + (N - n + 0.5)/(n + 0.5))` for reporting."""
    r = math.exp(idf) - 1.0
    if r <= 0:
        return float(n_points)
    return (n_points + 0.5 - 0.5 * r) / (1.0 + r)


def measure(qdrant: str, collection: str, hashes: list[int],
            batch: int, sleep_ms: int, log=print) -> dict[int, float]:
    """{hash -> idf} for every token present in the collection (df > 0)."""
    url = f"{qdrant.rstrip('/')}/collections/{collection}/points/query/batch"
    idf: dict[int, float] = {}
    for i in range(0, len(hashes), batch):
        chunk = hashes[i:i + batch]
        body = {"searches": [{
            "query": {"nearest": {"indices": [int(h)], "values": [1.0]}},
            "using": "bm25", "limit": 1,
            "with_payload": False, "with_vector": ["bm25"],
        } for h in chunk]}
        res = IH.fetch_json(url, body=body, timeout=120)["result"]
        for h, r_ in zip(chunk, res):
            pts = r_["points"] if isinstance(r_, dict) else r_
            if not pts:
                continue                      # df == 0: absent from the index
            p = pts[0]
            vec = (p.get("vector") or {}).get("bm25") or {}
            w = dict(zip(vec.get("indices", []), vec.get("values", []))).get(h)
            if not w:
                continue                      # cannot divide: skip, never guess
            idf[h] = float(p["score"]) / float(w)
        log(f"[idf] {min(i + batch, len(hashes)):,}/{len(hashes):,} tokens  "
            f"measured={len(idf):,}")
        if sleep_ms:
            time.sleep(sleep_ms / 1000.0)
    return idf


#: Terms whose snapshot IDF is below `capture_max_idf` + this margin are measured through Qdrant:
#: a term above the bound cannot be a suppression candidate, and the margin absorbs a measured
#: value that sits a little below the integer-count formula (stale postings, rounding).
CANDIDATE_MARGIN = 0.5


def measure_built(qdrant: str, collection: str, df, n_points: int, *, as_of: str | None = None,
                  capture_max_idf: float = 6.0, batch: int = 64, log=print) -> dict:
    """The qstop IDF table of a BUILT collection (design question Q4: a stranger measures the
    table per build; prod's profile pins its own and changes it only through the battery).

    The vocabulary is the build's own: every term hash of the sparse rows (`df`, the `{hash: df}`
    of `cytir.distribute.idf`'s snapshot over the built rows), with `n_points` the row count. The
    values are MEASURED the way prod's table is (`measure`: single-token score / stored weight,
    through Qdrant's own statistics), not computed from `df`: `df` only picks which terms can be
    below the capture bound. The document has the shape the server's loader reads
    (`cytir/serve/schema.py::_load_idf_table`) plus `collection` and `points_count`, which `cyt serve
    --idf-table` checks against its target. `fold_accents` is True: the projection that wrote the
    rows forces the fold, so the hashes in the index are folded ones."""
    from cytir.distribute import idf as SNAP
    candidates = sorted(h for h, d in df.items()
                        if SNAP.idf(n_points, d) < capture_max_idf + CANDIDATE_MARGIN)
    measured = measure(qdrant, collection, candidates, batch, 0, log=lambda _m: None)
    log(f"[idf] measured {len(measured):,} of {len(candidates):,} candidate terms on {collection}")
    kept = {h: v for h, v in measured.items() if v < capture_max_idf}
    order = sorted(kept, key=lambda h: (kept[h], h))
    return {
        "_comment": "BM25 IDF measured on the collection this build wrote (specs/aw-index-serve-gate.md "
                    "section 6, Q4). Regenerated by every `aw index build`; an absent token is "  # rename-codemod: keep (persisted: the _comment text is written into bm25_idf.json, whose sha256 serve_receipt.idf_table pins)
                    "treated as high-IDF (kept).",
        "collection": collection,
        "points_count": n_points,
        "measured_at": as_of or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fold_accents": True,
        "method": "idf = single_token_score / stored_weight (qdrant modifier=idf)",
        "capture_max_idf": capture_max_idf,
        "n_vocabulary": len(df),
        "n_measured": len(measured),
        "idf": {str(h): round(kept[h], 4) for h in order},
    }


def main(argv=None) -> None:
    """Measure (the default) or, with a leading `seal`, seal a GATE table (`seal_main`)."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["seal"]:
        raise SystemExit(seal_main(argv[1:], prog="python -m cytir.index.idf_table seal"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdrant", default=os.environ.get("AW_QDRANT_URL",
                                                       "http://127.0.0.1:16333"))
    ap.add_argument("--collection", default=os.environ.get("AW_COLLECTION", "aw_prod"))
    ap.add_argument("--out", default=str(REPO / SERVER_REL / "bm25_idf.json"))
    ap.add_argument("--corpus", nargs="*", default=DEFAULT_CORPUS)
    ap.add_argument("--capture-max-idf", type=float, default=6.0,
                    help="tokens at or above this IDF are NOT stored (the "
                         "loader treats an absent token as high-IDF, so the "
                         "table only needs the suppression candidates)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--sleep-ms", type=int, default=100)
    ap.add_argument("--report", default="", help="optional full-distribution dump")
    a = ap.parse_args(argv)

    if ST.current("AW_BM25_QSTOP"):
        sys.exit("refusing to measure with AW_BM25_QSTOP=1: the tokenizer would "
                 "suppress the very tokens this table has to measure")

    n_points = int(IH.fetch_json(
        f"{a.qdrant.rstrip('/')}/collections/{a.collection}", timeout=30)["result"]["points_count"])

    vocab, n_texts, n_files = collect_vocabulary(a.corpus)
    print(f"[vocab] {len(vocab):,} distinct tokens from {n_texts:,} texts "
          f"in {n_files} files", flush=True)

    idf = measure(a.qdrant, a.collection, sorted(vocab), a.batch, a.sleep_ms)
    print(f"[idf] measured {len(idf):,} of {len(vocab):,} tokens "
          f"({len(vocab) - len(idf):,} absent from the index)", flush=True)

    kept = {h: v for h, v in idf.items() if v < a.capture_max_idf}
    order = sorted(kept, key=lambda h: kept[h])
    out = {
        "_comment": "ADR-099 Ruling 2 — measured live BM25 IDF per query token. "
                    "An absent token is treated as high-IDF (kept). Regenerate "
                    "with `python3 -m cytir.index.idf_table`.",
        "collection": a.collection,
        "points_count": n_points,
        "measured_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fold_accents": bool(SC._FOLD_ACCENTS),
        "method": "idf = single_token_score / stored_weight (qdrant modifier=idf)",
        "capture_max_idf": a.capture_max_idf,
        "n_vocabulary": len(vocab),
        "n_measured": len(idf),
        "idf": {str(h): round(kept[h], 4) for h in order},
        "tokens": {str(h): vocab[h] for h in order},
    }
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False,
                                              sort_keys=False) + "\n")
    print(f"[out] {a.out}: {len(kept):,} tokens below idf {a.capture_max_idf}")

    if a.report:
        with open(a.report, "w") as f:
            for h, v in sorted(idf.items(), key=lambda kv: kv[1]):
                f.write(f"{v:8.4f}  df~{implied_df(v, n_points):12,.0f}  "
                        f"{vocab[h]}\n")
        print(f"[out] {a.report}: full distribution, {len(idf):,} tokens")



# --- seal: a GATE table with proof of the state it was measured on (task 629 P4 / ADR-109 R2) --- #

_SEAL_DOC = """Seal a BM25 IDF table for the GATE, with proof of the state it was measured on.

Task 629 P4 / ADR-109 R2. This module's `measure` path already measures IDF
exactly (`idf = single_token_score / stored_weight`) and already takes
`--capture-max-idf`, so this module adds no measurement. It adds the two things a
GATE table needs and the serving table never did:

1. **Proof of settledness.** Task 627 §6-R3 cost (3): the method reads whatever `n`
   and `df` the collection had at measurement time, so a table sealed from a
   drifted layout bakes that drift in permanently *and ships it to every
   self-hoster* — converting a self-correcting error into a fixed one. So this
   asserts ADR-043 A8's predicate (the optimizer log quiet **and**
   `n == points_count`) BEFORE measuring, records the layout fingerprint, re-checks
   it AFTER, and refuses if the layout moved underneath — because a table measured
   across a merge is measured against two different collections.
2. **A `settled: true` stamp.** `gate_leg_arms.load_idf_table` requires it
   positively rather than accepting its absence, so a producer cannot bypass the
   refusal by not mentioning the question. This is the only thing that writes it,
   and it writes it only when the assertion actually passed.

WHAT IT DOES NOT TOUCH
----------------------
`cytir/serve/bm25_idf.json`, the SERVING table. Replacing that is a ranking
change under ADR-109 decision (2) — the ADR-043 ranking bar plus a fusion re-tune
(ADR-029) — and is a deliberate, ruled event. A gate artefact and a serving
artefact that share a path are one artefact with two meanings, which is how a gate
change becomes a silent serving change.
"""

SCHEMA = "aw.gate-idf-table.v1"

#: Lifted well above the observed range. The shipped serving table's stored maximum
#: is 5.9967 against a 6.0 ceiling — i.e. it is truncated exactly where the
#: distribution still has mass, which is why it covers only 3,117 of 6,894 tokens
#: as a WEIGHT (task 627 §6-R3 cost 1). A gate table must weigh the whole
#: vocabulary, so nothing is dropped.
GATE_CAPTURE_MAX_IDF = 99.0


class SealError(RuntimeError):
    """A refusal. A table is not sealed on a state that cannot be proven."""


#: The largest stale-postings residual a GATE table may be sealed over, when the
#: caller names one. Default is 0.0 -- strict -- and strict is MANDATORY for any
#: table destined for the serving path (ADR-109 decision 1/R3), where a baked-in
#: drift ships to every self-hoster.
#:
#: Why a gate table may tolerate a residual at all, with the arithmetic rather
#: than an assurance (ruled by the orchestrator 2026-09-22, ADR-103-amendment
#: pending):
#:
#:  * A settled twin un-settles the moment the news lane writes: an
#:    overwrite-heavy pack retains its predecessors' postings, and the 2026-09-17
#:    pack was 13,026 overwrites behind a net +974. So demanding stale == 0 means
#:    re-running a 52-minute settle before every seal, in a window the daily lane
#:    closes.
#:  * The size of what that residual does to an IDF: Qdrant computes
#:    `idf = ln((n+1)/(df+0.5))`, so 13k stale on 33.7M points is a 0.04% change
#:    in `n` and moves any token's IDF by about `ln(1.0004) ~= 0.0004` -- against
#:    the **max |Δidf| 0.0159** task 627 measured across a single optimizer merge.
#:    Two orders of magnitude below the noise that motivated the whole exercise.
#:  * And the deeper reason: a sealed table's value to a gate arm is its
#:    FIXEDNESS, not its currency. Task 627 §3.5's result -- 30/30 golds' leg order
#:    collapsing onto each other under one shared IDF vector -- holds for ANY
#:    shared vector. Settledness is what stops a SERVING table baking in drift; for
#:    a gate arm using the same vector on both sides it is a cleanliness property,
#:    not the mechanism.
#:
#: The measured residual is always stamped, whether it was zero or tolerated, so a
#: verdict taken over a table with drift can be placed on that drift.
GATE_MAX_STALE_RATIO = 0.001            # 0.1%, the ruled ceiling for a gate table

#: A segment below the indexing threshold is NEVER tolerated, at any residual: a
#: plain segment scores a filtered sparse query differently (task 606) and
#: `status: green` does not report it. That is a different defect from staleness
#: and no stale-ratio allowance covers it.
PLAIN_SEGMENT_FLOOR = 10_000


def assert_settled(url: str, collection: str, *,
                   max_stale_ratio: float = 0.0) -> dict:
    """ADR-043 A8's predicate, as a precondition rather than a note.

    Judged on the fingerprint's own NUMBERS, not on `settled()`'s prose -- string
    matching a reason to decide whether it is tolerable would break the first time
    someone improved the wording. `settled()`'s reasons are still recorded, for the
    operator who reads the receipt.
    """
    if max_stale_ratio > GATE_MAX_STALE_RATIO:
        raise SealError(
            f"--max-stale-ratio {max_stale_ratio} exceeds the ruled gate ceiling "
            f"{GATE_MAX_STALE_RATIO} (0.1%). Raising it is a ruling, not a flag.")
    fingerprint = LAYOUT.fingerprint(url, collection)
    reasons = LAYOUT.settled(fingerprint)

    points = fingerprint.get("points_count")
    n = fingerprint.get("n_per_vector")
    if n is None or not points:
        raise SealError(
            f"{collection} at {url}: `n` is unreadable, so the IDF divisor this "
            "table would inherit cannot be checked at all. Refusing rather than "
            "sealing against an unknown state.")
    ratio = abs(float(fingerprint.get("stale_ratio") or 0.0))

    blocking = []
    if round(n) != points and ratio > max_stale_ratio:
        blocking.append(
            f"stale postings {fingerprint['stale_postings']:+,.0f} "
            f"({ratio:.4%}) exceed the allowed residual {max_stale_ratio:.4%}")
    for size in fingerprint.get("segment_sizes") or ():
        if 0 < size < PLAIN_SEGMENT_FLOOR:
            blocking.append(
                f"a {size:,}-point segment is below the indexing threshold; a "
                "plain segment scores a filtered sparse query differently "
                "(task 606) and no stale-ratio allowance covers it")
            break
    if blocking:
        raise SealError(
            f"{collection} at {url} may not be sealed from:\n  - "
            + "\n  - ".join(blocking)
            + "\nForce a vacuum (optimizers_config.deleted_threshold, bounded "
              "concurrency) and wait for the optimizer log to go quiet with "
              "n == points_count.")
    fingerprint = dict(fingerprint)
    fingerprint["settled_reasons_at_seal"] = reasons
    fingerprint["stale_ratio_tolerated"] = max_stale_ratio
    fingerprint["stale_ratio_measured"] = ratio
    return fingerprint


def measure_table(*, qdrant: str, collection: str, out: pathlib.Path,
            capture_max_idf: float, extra: list[str] | None = None) -> dict:
    """Run the existing measurer. One home for the arithmetic, not a second."""
    command = [sys.executable, "-m", "cytir.index.idf_table", "--qdrant", qdrant,
               "--collection", collection, "--out", str(out),
               "--capture-max-idf", str(capture_max_idf), *(extra or [])]
    run = subprocess.run(command, capture_output=True, text=True, cwd=str(REPO))
    if run.returncode:
        raise SealError(f"the IDF measurer failed ({run.returncode}):\n"
                        f"{run.stdout[-2000:]}\n{run.stderr[-2000:]}")
    return {"command": command, "stdout_tail": run.stdout[-4000:]}


def seal(*, qdrant: str, collection: str, out: pathlib.Path,
         capture_max_idf: float = GATE_CAPTURE_MAX_IDF,
         extra: list[str] | None = None,
         max_stale_ratio: float = 0.0) -> dict:
    """Assert settled -> measure -> re-assert -> stamp. Refuses on any movement."""
    before = assert_settled(qdrant, collection, max_stale_ratio=max_stale_ratio)
    started = dt.datetime.now(dt.timezone.utc)
    run = measure_table(qdrant=qdrant, collection=collection, out=out,
                  capture_max_idf=capture_max_idf, extra=extra)
    after = LAYOUT.fingerprint(qdrant, collection)

    moved = LAYOUT.optimizer_ran_between(before, after)
    quiet, why = LAYOUT.quiescent(before, after)
    if moved or not quiet:
        raise SealError(
            "the layout MOVED while the table was being measured "
            f"({why}; {len(moved)} optimizer pass(es) in between), so the table "
            "was measured against two different collections. Re-seal on a quiet "
            "one; do not keep this file.")

    doc = json.loads(out.read_text())
    doc["schema"] = SCHEMA
    doc["settled"] = True
    doc["settled_predicate"] = (
        "ADR-043 A8: optimizer log quiet AND n == points_count, asserted before "
        "the measurement and re-asserted after it"
        + ("" if not max_stale_ratio else
           f"; a stale residual up to {max_stale_ratio:.4%} was ALLOWED for this "
           "gate table (ruled 2026-09-22), and the measured residual is recorded "
           "in layout_fingerprint.stale_ratio_measured"))
    doc["stale_ratio_measured"] = before.get("stale_ratio_measured")
    doc["stale_ratio_tolerated"] = max_stale_ratio
    doc["layout_fingerprint"] = after
    doc["layout_fingerprint_before"] = before
    doc["sealed_at"] = started.strftime("%Y-%m-%dT%H:%M:%SZ")
    doc["sealed_by"] = "cytir.index.idf_table seal"
    doc["purpose"] = (
        "GATE arm only (ADR-043 A8 rule 4 / ADR-109 R2). NOT the serving table: "
        "replacing cytir/serve/bm25_idf.json is a ranking change under "
        "ADR-109 decision (2) and takes the ADR-043 ranking bar plus a fusion "
        "re-tune.")
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return {"out": str(out), "tokens": len(doc.get("idf") or {}),
            "n_vocabulary": doc.get("n_vocabulary"),
            "n_measured": doc.get("n_measured"),
            "points_count": doc.get("points_count"),
            "capture_max_idf": capture_max_idf,
            "fingerprint": after, "run": run}


def seal_main(argv=None, prog=None) -> int:
    """`python -m cytir.index.idf_table seal ...`: seal a GATE table (see `seal`)."""
    ap = argparse.ArgumentParser(prog=prog, description=_SEAL_DOC.split("\n")[0])
    ap.add_argument("--qdrant", required=True)
    ap.add_argument("--collection", default=TWIN.collection())
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--capture-max-idf", type=float,
                    default=GATE_CAPTURE_MAX_IDF)
    ap.add_argument("--corpus", nargs="*", default=None,
                    help="passed through to the measurer's --corpus")
    ap.add_argument("--max-stale-ratio", type=float, default=0.0,
                    help="GATE tables only: the largest stale-postings residual "
                         "to seal over, up to %(const)s. Default 0.0 (strict), "
                         "which is MANDATORY for any serving-path table."
                         % {"const": GATE_MAX_STALE_RATIO})
    args = ap.parse_args(argv)
    extra = ["--corpus", *args.corpus] if args.corpus else None
    result = seal(qdrant=args.qdrant, collection=args.collection, out=args.out,
                  capture_max_idf=args.capture_max_idf, extra=extra,
                  max_stale_ratio=args.max_stale_ratio)
    print(f"[sealed] {result['out']}: {result['tokens']:,} tokens "
          f"({result['n_measured']} measured of {result['n_vocabulary']} "
          f"vocabulary) at points_count={result['points_count']:,}")
    print(LAYOUT.render(result["fingerprint"], "sealed against"))
    return 0


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""`aw.chunk-provenance.v1` — the one home for per-row chunking/vector lineage.

WHY (Tim, 2026-08-13): "we [must] have full traceability of the exact chunking
methodology (including differences between dense and sparse) used for each row".

Before this module, per-row lineage was UNEVEN — which is the real defect, not
a uniform absence. Measured 2026-08-13:

  * chunk@3 structured rows carry `rule_epochs.chunk` AND a legend set
    (`chunk_rule`, `chunk_unit`, `chunk_n_words`, `chunk_header_path`,
    `chunk_ordinal`, `extraction_rule`) via
    `aw_b1_projection::CHUNK2_PASSTHROUGH`;
  * legacy epoch-1 rows — including every new news-lane row — carry
    `rule_epochs.chunk` and nothing else; `chunk_rule` is null;
  * source-native, document-native and card rows carry no chunk lineage at all.

So a legend exists on structured rows and nowhere else, and even there it is an
assertion rather than a proof: a legend cannot detect a policy artifact edited
in place under an unchanged name, only its sha can. Seven chunk-policy
artifacts live in `data/`; **five of them declare epoch 3**, so five
byte-different rule sets all answer to the same epoch int. Tim's 2026-08-13
ruling makes that concrete rather than hypothetical: the go-forward policy is
`struct-md-768w-o0-adjacent-sections-tail-1152-tok2560` (sha `7dcb54c4…`,
epoch 4), whose ONLY substantive difference from the shipped epoch-3 policy
(sha `cf0f5244…`) is `token_hard_max.exact: 2000 -> 2560`. Every other boundary
rule is byte-identical, so epoch-3 and epoch-4 rows are indistinguishable in
content *except* for the units that hit the 2,000-token backstop. Two epochs
that differ in one field and mostly produce identical output is exactly the
case where "which policy made this row" becomes unanswerable after the fact —
which is why the **policy sha**, not the epoch int, is the load-bearing stamp.

Design
------
The row carries ONE short interned string, ``prov_id`` — 16 hex chars of
sha256 over the canonical JSON of the full provenance record. The full record
lives once in a legend, not once per row (at 54M rows the difference is ~1.3 GB
vs ~19 GB). A changed rule mints a new ``prov_id`` automatically, because the
id is a hash of the rule set: there is no way to change a rule and keep an id.

ADR-039 compact/master split
----------------------------
The ``producer`` block (absolute paths, host layout, input digests, git
revisions) is **master-only** — same class as `bundle_digest` and `_master`.
Everything else is **compact**: `Passage.stamps`' own docstring already states
the requirement — "a redistributor must be able to tell which chunking rule a
record carries" — and a redistributor cannot do that without the channel
surfaces and the policy hash.

Honesty rule (verification-doctrine)
------------------------------------
``confidence`` is one of ``stamped`` (written by the producer at write time),
``recovered`` (reconstructed from artifacts, with ``evidence`` naming them), or
``unknown``. A field that cannot be recovered is **absent** and named in
``unknown_fields`` with a ``unknown_reason``. A plausible value is never
invented to fill a hole: an unrecoverable row must read unrecoverable.
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterable, Optional

from cytir.core.digest import canonical_json_utf8

SCHEMA = "aw.chunk-provenance.v1"
WAVE_SCHEMA = "aw.wave-provenance.v1"
PROV_ID_CHARS = 16

CONFIDENCE = ("stamped", "recovered", "unknown")

# Chunker families. `chunk` is ONE epoch family in specs/epochs.md, but at
# least four independently-evolvable chunkers hide inside it (ADR-079 scoped
# chunk@2 to the generic path precisely because they are not the same rule).
# The family is what an auditor needs before the legend even resolves.
FAMILIES = (
    "generic-structured",     # cytir/chunk/chunker.py under a pinned policy artifact
    "generic-legacy-window",  # fetch_process.chunk paragraph windows
    "law-article-117",        # per-article splitter + task-117 -cK windowing
    # The retired fetch_corpus.py's fedlex render path (deleted, task 1901): a
    # source article anchor, then fetch_process.chunk at target_words=400. A
    # FOURTH generic windower variant, never named anywhere until task 407
    # measured a partial anchor fraction in the P0-era waves and went looking
    # for what produced it. Kept as a legend for rows already in the index.
    "law-article-fedlex-render",
    "source-native",          # source's own record boundary, no AW windowing
    "document-native",        # one row per source document; never windowed
    "card-recipe",            # composed entity/site card (ADR-060 two-channel)
    "unknown",                # not recoverable — see unknown_reason
)

# Dense/sparse surface vocabulary. These are the ACTUAL divergences in AW, and
# naming them per row is the point of Tim's ask: the two channels are not the
# same text on any structured, fedlex, or card row.
SURFACES = (
    "body",                    # the chunk body alone
    "hierarchy-plus-body",     # full breadcrumb / AKN heading path + body
    "leaf-heading-plus-body",  # deepest retained heading + body
    "card-dense-prose",        # ADR-060 natural-language description
    "card-sparse-hooks",       # ADR-060 name + aliases + domain + product terms
    "unknown",
)


class ProvenanceError(ValueError):
    """A malformed or internally inconsistent provenance record."""


# Byte-stable JSON. The id is a hash of these bytes, so this encoding is a mirror
# contract with one home (engineering-doctrine 6).
canonical_json = canonical_json_utf8


# --- ADR-039 compact/master split -------------------------------------------
# The producer block is traceability evidence about OUR machines, not a fact
# about the record's content. It never ships in the redistributable dump.
MASTER_ONLY_BLOCKS = frozenset({"producer"})


def prov_id_for(record: dict) -> str:
    """Mint the interned id over the RULE, and only the rule.

    Excluded, deliberately:
      * `prov_id` itself;
      * `evidence` / `confidence` — how we learned the rule, not what it is, so
        a stamped row and a row recovered from a manifest intern identically;
      * the master-only blocks — which RUN produced a row (host paths, input
        digests, git revision) is traceability about our machines, not part of
        the rule. Including them would hand a redistributor two different ids
        whose compact content is byte-identical, which is exactly the kind of
        distinction-without-a-difference this schema exists to remove. The run
        detail is still carried, per row's wave, in the master-only block.
    """
    body = {k: v for k, v in record.items()
            if k not in ("prov_id", "evidence", "confidence")
            and k not in MASTER_ONLY_BLOCKS}
    digest = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return digest[:PROV_ID_CHARS]


def build(
    *,
    family: str,
    chunker: dict,
    channels: dict,
    embed: Optional[dict] = None,
    tok: Optional[dict] = None,
    extract: Optional[dict] = None,
    producer: Optional[dict] = None,
    confidence: str,
    evidence: Iterable[str] = (),
    unknown_fields: Iterable[str] = (),
    unknown_reason: Optional[str] = None,
) -> dict:
    """Assemble + validate one provenance record and mint its `prov_id`."""
    if family not in FAMILIES:
        raise ProvenanceError(f"unknown chunker family {family!r}")
    if confidence not in CONFIDENCE:
        raise ProvenanceError(f"confidence must be one of {CONFIDENCE}")
    for key in ("dense", "sparse"):
        if channels.get(key) not in SURFACES:
            raise ProvenanceError(
                f"channels.{key}={channels.get(key)!r} is not a named surface; "
                f"expected one of {SURFACES}")
    unknown_fields = sorted(set(unknown_fields))
    if unknown_fields and not unknown_reason:
        raise ProvenanceError(
            "unknown_fields requires an explicit unknown_reason — an "
            "unrecoverable value is named, never quietly omitted")
    if confidence == "recovered" and not list(evidence):
        raise ProvenanceError(
            "a recovered record must cite the artifacts it was recovered from")

    record: dict[str, Any] = {
        "schema": SCHEMA,
        "family": family,
        "chunker": chunker,
        "channels": channels,
    }
    for key, value in (("embed", embed), ("tok", tok), ("extract", extract),
                       ("producer", producer)):
        if value is not None:
            record[key] = value
    if unknown_fields:
        record["unknown_fields"] = unknown_fields
        record["unknown_reason"] = unknown_reason
    record["prov_id"] = prov_id_for(record)
    record["confidence"] = confidence
    if evidence:
        record["evidence"] = list(evidence)
    return record


def compact(record: dict) -> dict:
    """The redistributable projection of one legend entry (ADR-039)."""
    return {k: v for k, v in record.items() if k not in MASTER_ONLY_BLOCKS}


def compact_legend(legend: dict) -> dict:
    """Strip master-only blocks from every entry of a legend."""
    return {pid: compact(rec) for pid, rec in legend.items()}


def validate_legend(legend: dict) -> None:
    """Every entry must intern to the key it is filed under."""
    for pid, record in legend.items():
        minted = prov_id_for(record)
        if minted != pid:
            raise ProvenanceError(
                f"legend key {pid} does not match its content ({minted}); "
                "a legend entry is immutable — mint a new id instead")
        if record.get("prov_id") != pid:
            raise ProvenanceError(f"legend entry {pid} carries prov_id "
                                  f"{record.get('prov_id')!r}")

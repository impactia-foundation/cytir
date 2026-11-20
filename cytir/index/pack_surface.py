#!/usr/bin/env python3
"""Document-first retrieval surfaces for the structured law/research packs.

WHY THIS EXISTS (task 527 -> task 528b).  The task-417 pack materializers open
every document's Markdown with their own machine-identity block -- Canonical
URL, archive SHA-256, PMID, OAI page, ELI.  For a one-sentence statute article
that block is *most of the document*: task 527 measured the share of the
embedded surface's non-blank lines at legifrance 0.816, gesetze 0.815, pubmed
0.667, dila-loda 0.540, fedlex 0.439, arxiv 0.345.  It is byte-similar across
every article of every law in a source, so it is exactly the part a dense leg
cannot use to tell two articles apart -- the mechanism behind task 524's legal
(-34) and research (-13) dense-only losses.

WHAT THIS MODULE DOES.  It re-projects an already-sealed structured chunk onto
document-first retrieval surfaces WITHOUT touching the chunk's identity:

  * dense   = compose_index_text(composed title, content atoms)   [ADR-061]
  * sparse  = hierarchy breadcrumbs + content atoms               [ADR-060 shape]
  * identity= the dropped provenance atoms, returned as record METADATA

``canonical_text`` is never rewritten.  ``cytir.core.common.mint_passage_point_id``
keys a passage id on (url, anchor, canonical text), so leaving canonical alone
is what makes this an in-place same-id vector/posting modify (ADR-083) rather
than a rebuild.  That is deliberate and is the whole reason the delta conserves
``points_count`` exactly.

WHY IT DOES NOT RE-CHUNK.  Task 516 fixed fedlex the other way -- it changed
`materialize_r0_fedlex_anchored.py` so the block never enters the Markdown, and
re-materialized.  That is the right shape for a *fresh* build and it is what a
future re-materialization of these five families must do (the composer below is
idempotent: a Markdown with no provenance section makes it a no-op).  It is the
wrong shape for a delta onto a shipped collection, because a changed canonical
text changes the point id of every row.

HOW THE ATOM SPLIT IS EXACT, NOT HEURISTIC.  `cytir/chunk/chunker.py` builds a chunk
from ordered ``_SectionAtom``s joined by "\\n\\n" in both channels, where each
atom is::

    canonical = breadcrumb + "\\n\\n" + body     (breadcrumb "" for the page lead)
    dense     = local_heading + "\\n" + body     (local_heading "" for the lead)

`split_channel_atoms` recovers that split from the sealed (canonical, dense)
pair and then RECOMPOSES both strings from the recovered atoms, refusing the
row unless both come back byte-identical.  A row whose split cannot be proven
is quarantined and keeps its shipped surface (doctrine 3: fail loud, never a
silent fallback to a differently-shaped default).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, Optional

from cytir.core.common import compose_index_text
from cytir.core.paths import data_path

BREADCRUMB_SEP = " > "

CONTRACT_PATH = str(data_path("pack-surface-contract.v1.json"))


class PackSurfaceError(RuntimeError):
    """A sealed row whose channels cannot be re-projected provably."""


@dataclass(frozen=True)
class Atom:
    """One source-adjacent structured unit of a sealed chunk."""
    breadcrumb: str          # canonical prefix, "" for the page lead
    local: str               # dense prefix (breadcrumb leaf), "" for the lead
    body: str                # the atom body, identical on both channels

    @property
    def sparse_text(self) -> str:
        return f"{self.breadcrumb}\n\n{self.body}" if self.breadcrumb else self.body

    @property
    def dense(self) -> str:
        return f"{self.local}\n{self.body}" if self.local else self.body


def _leaf(breadcrumb: str) -> str:
    return breadcrumb.rsplit(BREADCRUMB_SEP, 1)[-1].strip() if breadcrumb else ""


def split_channel_atoms(canonical_text: str, dense_text: str) -> list[Atom]:
    """Recover a sealed chunk's atoms, or raise.

    Walks the two channel strings block-by-block.  An atom either starts with a
    breadcrumb (canonical carries one extra block that dense folds into the
    first body block as ``leaf + "\\n"``) or is a bare lead atom (both channels
    identical).  The result is verified by recomposition before it is returned.
    """
    cbl = canonical_text.split("\n\n")
    dbl = dense_text.split("\n\n")
    atoms: list[Atom] = []
    ci = di = 0
    while ci < len(cbl) and di < len(dbl):
        cblock, dblock = cbl[ci], dbl[di]
        if cblock == dblock:
            # A lead atom (no heading), or a continuation block of the atom we
            # are already inside.  Continuations are folded below.
            atoms.append(Atom("", "", cblock))
            ci += 1
            di += 1
            continue
        leaf = _leaf(cblock)
        prefix = leaf + "\n"
        if leaf and dblock.startswith(prefix) and ci + 1 < len(cbl) \
                and cbl[ci + 1] == dblock[len(prefix):]:
            atoms.append(Atom(cblock, leaf, cbl[ci + 1]))
            ci += 2
            di += 1
            continue
        raise PackSurfaceError(
            f"unalignable channels at canonical block {ci} / dense block {di}")
    if ci != len(cbl) or di != len(dbl):
        raise PackSurfaceError("channel block streams have different lengths")

    # Fold each headless block that follows a headed atom back into that atom:
    # a section body may legitimately hold several blocks, and only the FIRST
    # carries the prefix.  A headless block at position 0 is a genuine lead.
    folded: list[Atom] = []
    for atom in atoms:
        if atom.breadcrumb == "" and folded and folded[-1].breadcrumb != "":
            last = folded[-1]
            folded[-1] = Atom(last.breadcrumb, last.local,
                              last.body + "\n\n" + atom.body)
            continue
        if atom.breadcrumb == "" and folded and folded[-1].breadcrumb == "":
            last = folded[-1]
            folded[-1] = Atom("", "", last.body + "\n\n" + atom.body)
            continue
        folded.append(atom)

    if "\n\n".join(a.sparse_text for a in folded) != canonical_text:
        raise PackSurfaceError("canonical recomposition is not byte-identical")
    if "\n\n".join(a.dense for a in folded) != dense_text:
        raise PackSurfaceError("dense recomposition is not byte-identical")
    return folded


def load_contract(path: str = CONTRACT_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def spec_for_source(contract: dict, source: str) -> dict:
    """Return the declared spec for a pack source, or raise.

    An undeclared source is refused rather than defaulted: the whole point of
    the onboarding contract is that a new family states its surface before it
    can be staged (engineering doctrine 3).
    """
    spec = (contract.get("sources") or {}).get(source)
    if not spec:
        raise PackSurfaceError(f"pack source is not declared in the contract: {source}")
    return spec


def _is_provenance(atom: Atom, headings: Iterable[str]) -> bool:
    return atom.local in set(headings)


def compose_title(record: dict, spec: dict) -> str:
    """Compose the document-first title from the record's own sealed fields.

    ADR-061 shape is ``"<title>. <site_name>. <body>"``.  The site name is NOT
    dropped here: it is read off the record like any other field, and for these
    families production carried none.  Measured against the twin `aw_e3` on
    2026-09-06 (task 528b probe, n=40 per host):
    fedlex.admin.ch 40/40 `"Fedlex"`, and gesetze-im-internet.de,
    legifrance.gouv.fr, pubmed.ncbi.nlm.nih.gov, arxiv.org and
    laws-lois.justice.gc.ca 40/40 `None`.  So for every family this module
    governs, task 274's rule composes to ``"<title>. <body>"`` by itself -- which
    is exactly task 526's best-measured arm (629 dense-only: 145 -> 165 hit@1,
    legal 62 -> 82), reached without a site-name decision being taken here.

    The title is the materializer's own ``source_metadata.title``.  When the
    family declares `citation_from_identity` AND `title_citation` is on, the
    short citation a user actually types is added -- an abbreviation, an article
    number, an SR number (task 516's fedlex finding: production's legal titles
    carry the citation and the build's did not).  That addition is a SECOND
    variable on top of task 526's proven arm, so it has its own A/B arm rather
    than riding along unmeasured.
    """
    meta = record.get("source_metadata") or {}
    title = (meta.get("title") or record.get("title") or "").strip()
    title = re.sub(r"\s*\n\s*", " ", title)
    cites: list[str] = []
    for rule in (spec.get("citation_from_identity") or []
                 if spec.get("title_citation", True) else []):
        value = _identity_value(record, rule)
        if value and value not in title:
            cites.append(value)
    if cites:
        head, sep, tail = title.partition(" — ")
        cite = " (" + "; ".join(cites) + ")"
        title = head + cite + sep + tail if sep else title + cite
    return title


_BULLET = re.compile(r"^\s*-\s*(?P<label>[^:]{1,60}):\s*(?P<value>.+?)\s*$")


def identity_fields(atoms: Iterable[Atom], headings: Iterable[str]) -> dict:
    """Parse the dropped provenance atoms into record metadata.

    The block does not disappear -- it stops being the document's first
    sentence and becomes structured, sha-bindable metadata on the staged row
    (task 516's rule: the provenance survives in the receipt, the record, and
    the run manifest; it just leaves the text).
    """
    out: dict[str, dict[str, str]] = {}
    for atom in atoms:
        if not _is_provenance(atom, headings):
            continue
        block: dict[str, str] = {}
        for line in atom.body.split("\n"):
            match = _BULLET.match(line)
            if match:
                block[match.group("label").strip()] = match.group("value").strip("` ")
        out[atom.local] = block or {"_text": atom.body}
    return out


def _identity_value(record: dict, rule: str) -> str:
    """Resolve ``"<Section>/<Label>"`` or ``"meta/<key>"`` against a record."""
    scope, _, key = rule.partition("/")
    if scope == "meta":
        meta = record.get("source_metadata") or {}
        value = meta.get(key)
        if isinstance(value, dict):
            return ""
        return str(value or "").strip()
    block = ((record.get("source_identity") or {}).get(scope) or {})
    return str(block.get(key) or "").strip()


def compose_surfaces(record: dict, spec: dict) -> dict:
    """Return the document-first ``text`` / ``sparse_text`` / ``source_identity``.

    The returned dict is the fields to OVERWRITE on the staged row.  Every
    other field -- ``point_id``, ``canonical_text``, ``url``, ``prov_id`` --
    is left exactly as sealed, which is what keeps the delta same-id.
    """
    atoms = split_channel_atoms(record["canonical_text"], record["text"])
    provenance = spec.get("provenance_headings") or []
    sparse_only = spec.get("sparse_only_headings") or []
    dropped = [a for a in atoms if _is_provenance(a, provenance)]
    kept = [a for a in atoms if not _is_provenance(a, provenance)]
    # The hierarchy is a lexical asset and a dense distraction: BM25 wants
    # "Code de commerce > Partie legislative > LIVRE Ier", the embedder wants
    # the article.  That asymmetry is the whole reason the pack carries two
    # channels (specs/chunk-vector-provenance.md), and the binder has been
    # collapsing them into one (assembly_binder.py:1389).
    dense_atoms = [a for a in kept if not _is_provenance(a, sparse_only)]
    if not dense_atoms:
        # A chunk that is nothing but hierarchy still has to embed something;
        # falling back to the kept atoms is visible in the counters below, not
        # silent.
        dense_atoms = kept
    if not kept:
        raise PackSurfaceError("every atom is provenance; refusing an empty surface")

    identity = identity_fields(dropped, provenance)
    enriched = dict(record)
    enriched["source_identity"] = identity
    title = compose_title(enriched, spec)

    body_dense = "\n\n".join(a.dense for a in dense_atoms)
    sparse = "\n\n".join(a.sparse_text for a in kept)
    # `dense_title` exists so the A/B can vary ONE thing at a time: the
    # document-first surface is two changes at once (the identity block leaves,
    # the title arrives) and a table that cannot say which half earned the gain
    # is not a finding. "none" also gives arm A a free correctness check -- with
    # nothing declared, the re-projection must reproduce the sealed bytes.
    # ONE composition function, the one aw_b1_projection.py:164-166 applies for
    # every source production ever built (task 274 / ADR-061). Task 526 proved
    # that the pack -> 413a-stage boundary is the only place in the corpus that
    # never called it: 164 sealed stage receipts, 17,234,345 rows, composition
    # `precomposed-*`, `compose_index_text` applied zero times -- while
    # Wikipedia's 19.3M rows took the other path and did call it. That is the
    # mechanism behind task 524's reference +9 / legal -34 / research -13.
    site = spec.get("site_name") or record.get("site_name") or None
    if spec.get("dense_title", "composed") == "composed":
        dense = compose_index_text(
            {"title": title, "site_name": site, "text": body_dense})
    else:
        dense = body_dense
    if spec.get("sparse_prepends_title", True):
        sparse = compose_index_text(
            {"title": title, "site_name": site, "text": sparse})
    return {
        "text": dense,
        "sparse_text": sparse,
        "source_identity": identity,
        "dropped_atoms": len(dropped),
        "sparse_only_atoms": len(kept) - len(dense_atoms),
        "kept_atoms": len(dense_atoms),
        "composed_title": title,
        "site_name": site,
    }


def resurface_record(record: dict, spec: dict) -> tuple[dict, Optional[str]]:
    """Re-project one sealed row; return ``(row, quarantine_reason)``.

    On failure the row is returned UNCHANGED with a reason.  The caller counts
    quarantines and refuses the run above the family's declared ceiling; a
    silently defaulted row is exactly the failure mode this project keeps
    paying for.
    """
    try:
        fields = compose_surfaces(record, spec)
    except PackSurfaceError as exc:
        return record, str(exc)
    out = dict(record)
    out["text"] = fields["text"]
    out["sparse_text"] = fields["sparse_text"]
    if fields["source_identity"]:
        out["source_identity"] = fields["source_identity"]
    return out, None

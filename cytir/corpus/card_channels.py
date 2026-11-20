"""Resolve an entity card's ADR-060 dense/sparse channels from preserved rows.

ONE home for the question "given every preserved source-evidence row that
mentions this card, which text is its DENSE surface and which is its SPARSE
surface?".  Both the task-467 stage producer and the task-528a surface delta
import it, so the two can never answer it differently.

Two rulings shape it.

**Tim, 2026-09-06 (register-first):** *"we're supposed to be getting the names
of directors from the official source, not the entity's website."*  A card's
dense surface is the **commercial-register composition** —
``Name — <register purpose>. Registered signatories/officers (commercial
register): <officers>.`` — and the entity's own website contributes nothing to
identity or officers.  Task 467's tier resolution took
``dense_text or new_dense_text`` at whatever source ranked first, and
``task393_apply`` (homepage body text, ``adr060_phase: task393-website-content``)
outranked ``task363_people`` (the register composition).  The Vector8 card is
the measured instance: production embeds the 574-char register purpose plus
four named officers, the rebuild embedded 1,600 chars of marketing copy with
the officers pushed past the truncation point (``… Marc Maret (…); Ni``).
Officers must trace to the register evidence unit (ADR-094 §6a), never to
website prose.  The same ordering governs the sparse channel: the register hook
bag outranks a hook bag built from the website.

**ADR-060 §(b) (channels are not interchangeable):** *"the BM25 card channel is
a bag of discriminative hooks, never sentences; sentences are dense-channel
material"* — and ADR-060 §(a) measured the sparse hook as the only thing that
gets a card into the candidate pool at all.  So a card that has a sealed
``sparse_text`` / ``new_sparse_text`` keeps it as its own channel instead of
having the dense prose tokenised in its place.

**What this deliberately does NOT change.**  Within the non-register,
non-website candidates the ordering is byte-for-byte task 467's
``(tier, source-priority)``: a sourced description still beats a legacy
single-channel ``text``, which still beats a synthesized
``"{title} — {snippet}"``, and among equals the caller's source order decides.
Only two edges move per channel, and each is a ruling: a register-sourced
surface rises above every other class, and a website-derived surface falls
below every other class.  Nothing here does I/O, fetches, or composes prose the
preserved rows do not already carry: a card with no register evidence does not
gain one.
"""
from __future__ import annotations

from typing import Iterable

# A surface derived from the entity's OWN website.  Read from the producer's
# own declarations, never guessed from the text: task 393 stamps `adr060_phase`
# and `description_source` on every row it enriched.
WEBSITE_PHASES = frozenset({"task393-website-content"})
WEBSITE_DESCRIPTION_SOURCES = frozenset({"homepage", "homepage+register_field"})

# Dense classes, best first.  The class is recorded per row so a delta can be
# counted by reason instead of by source label.
DENSE_REGISTER = "register_composition"      # register purpose + officers (task 363)
DENSE_DESCRIPTION = "sourced_description"    # wikidata/wikipedia/category prose
DENSE_LEGACY_TEXT = "legacy_single_channel"  # epoch-1 `text`
DENSE_SYNTH = "recovered_title_snippet"      # nothing else survived
DENSE_WEBSITE = "website_content"            # refused while any other class exists
# The RULED composition (ADR-060a, task 528a-2): the register surface first,
# then a bounded website DESCRIPTION. Not a class the resolver ranks -- it is a
# transform applied to the winning non-website surface, so identity and officers
# still come from the register and the website can only ever follow them.
DENSE_REGISTER_PLUS_WEBSITE = "register_composition_plus_website"

# 1..4 reproduce task 467's tier order exactly; 1 and 5 are the two ruled edges.
DENSE_CLASS_RANK = {
    DENSE_REGISTER: 1,
    DENSE_DESCRIPTION: 2,
    DENSE_LEGACY_TEXT: 3,
    DENSE_SYNTH: 4,
    DENSE_WEBSITE: 5,
}

# Sparse classes, best first.
SPARSE_REGISTER_SEALED = "register_hooks_sealed"   # task 363 `new_sparse_text`
SPARSE_SEALED = "sealed_hooks"                     # the producer's own channel
SPARSE_COMPOSED = "register_hooks_plus_content"    # composed here, segmented
SPARSE_WEBSITE = "website_hooks"                   # refused while any other exists
SPARSE_NAME_ONLY = "name_hook_plus_content"        # last resort: the card's name
SPARSE_NONE = "none"

SPARSE_CLASS_RANK = {
    SPARSE_REGISTER_SEALED: 1,
    SPARSE_SEALED: 2,
    SPARSE_COMPOSED: 3,
    SPARSE_WEBSITE: 4,
}

# The register composition's officer clause, verbatim as task 363 shipped it.
OFFICER_CLAUSE = "Registered signatories/officers (commercial register): "


def is_website_derived(row: dict) -> bool:
    """Does this row's text come from the entity's own website?"""
    if (row.get("adr060_phase") or "") in WEBSITE_PHASES:
        return True
    return (row.get("description_source") or "") in WEBSITE_DESCRIPTION_SOURCES


def uid_token(row: dict) -> str:
    """The register UID as a single BM25 token: `CHE-341.859.062` -> `CHE341859062`.

    This is the hook a UID lookup types and a name lookup benefits from, and it
    is the term task 527 §5 measured missing from the rebuild's card postings
    entirely -- a query term with no posting cannot match at any weight.
    """
    uid = (row.get("uid") or "").strip()
    if uid:
        return uid
    reg = (row.get("register_id") or "").strip()
    if not reg or reg.lower() == "none":
        return ""
    return "".join(ch for ch in reg if ch.isalnum())


def register_composition(row: dict) -> str:
    """The sealed register-first dense surface, or "" if this row has none.

    Returned verbatim: `new_dense_text` is task 363's own composition of the
    Zefix *Zweck* plus the register's signatories, and it is the string
    production embedded.  Recomposing it here would fork it.
    """
    if is_website_derived(row):
        return ""
    return (row.get("new_dense_text") or "").strip()


def dense_candidate(row: dict) -> tuple[str, str] | None:
    """(dense_class, text) for ONE preserved row, or None if it carries no text.

    A website-derived row offers exactly ONE candidate, its website text, at
    the refused class -- its `snippet` is homepage prose, not the register
    purpose, so it may not re-enter through the synthesized branch either.
    """
    if is_website_derived(row):
        web = (row.get("dense_text") or row.get("text") or "").strip()
        return (DENSE_WEBSITE, web) if web else None
    reg = register_composition(row)
    if reg:
        return DENSE_REGISTER, reg
    dense = (row.get("dense_text") or "").strip()
    if dense:
        return DENSE_DESCRIPTION, dense
    text = (row.get("text") or "").strip()
    if text:
        return DENSE_LEGACY_TEXT, text
    title = (row.get("title") or "").strip()
    snippet = (row.get("snippet") or "").strip()
    if title and snippet:
        return DENSE_SYNTH, f"{title} — {snippet}"
    if title or snippet:
        return DENSE_SYNTH, title or snippet
    return None


def compose_register_hooks(row: dict) -> str:
    """`name + UID + people tokens` -- the ADR-060 hook bag, nothing else.

    Deliberately the shape of task 363's `new_sparse_text`
    (``"Vector8 AG CHE341859062 marc maret emanuele diquattro …"``): a short bag
    of discriminative hooks, so BM25 length normalisation still favours an exact
    name match.  Returns "" for a row with no register identity to hook on --
    a bare title is not a hook bag, and a card is not given one it has no
    evidence for.
    """
    uid = uid_token(row)
    tokens = row.get("people_tokens") or []
    if isinstance(tokens, str):
        tokens = [tokens]
    if not uid and not tokens:
        return ""
    parts: list[str] = []
    title = (row.get("title") or "").strip()
    if title:
        parts.append(title)
    if uid:
        parts.append(uid)
    seen = {p.lower() for p in parts}
    for tok in tokens:
        t = str(tok).strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            parts.append(t)
    return " ".join(parts)


def sparse_candidate(row: dict) -> tuple[str, str] | None:
    """(sparse_class, text) for ONE preserved row's SEALED channel, or None.

    `new_sparse_text` is task 363's register hook bag -- name, UID, people
    tokens, 9-46 terms -- and is the surface task 527 §5 proved production's
    card postings reconstruct from.  A `sparse_text` on a website-derived row
    is still admissible, but only after every register-sourced one.
    """
    reg = (row.get("new_sparse_text") or "").strip()
    if reg:
        return SPARSE_REGISTER_SEALED, reg
    sealed = (row.get("sparse_text") or "").strip()
    if sealed:
        return (SPARSE_WEBSITE if is_website_derived(row)
                else SPARSE_SEALED), sealed
    return None


def compose_sparse(hooks: str, dense_text: str) -> str:
    """`hooks` and the dense chunk as SEPARATE BM25 segments.

    ADR-060 §(b) wants a bag of discriminative hooks; ADR-060's own v2 recipe
    (`card_recipe_adr060.segmented_sparse_content`) wants sparse to cover the
    whole chunk so a curated subset cannot hide a dense word.  Both hold at
    once *because the surface is segmented*: `aw_b1_projection.saturated_bm25`
    saturates each segment independently and merges by max, so a name that is
    1 of 11 terms in the hook segment keeps its full length-normalised weight
    however long the description segment is.  That length normalisation --
    "1 of 11 terms in an 82-char document is 1 of 44-133 in a 574-1,600-char
    one" -- is the second of the two mechanisms task 527 §5 named behind Tim's
    Marc Maret result, and an UNsegmented surface is what reintroduced it.
    """
    import cytir.corpus.card_recipe as CR  # lazy: keeps this module import-light
    return CR.segmented_sparse_content(hooks, dense_text, [])


def website_description(row: dict) -> str:
    """The entity's own website text, or "" -- DESCRIPTION material only.

    Read only from a row the producer itself declared website-derived, so this
    can never pick up register prose by accident.
    """
    if not is_website_derived(row):
        return ""
    return (row.get("dense_text") or row.get("text") or "").strip()


def compose_dense_with_website(dense: str, website: str,
                               cap_tokens: int | None = None) -> str:
    """ADR-060a's ruled dense surface: register text FIRST, website description
    AFTER it, bounded.

    Tim, 2026-09-06: *"we're supposed to be getting the names of directors from
    the official source, not the entity's website"* — and, in the same message,
    *"at most a bounded description sentence after the register text, if the
    ADR-060 recipe allows it."*  Both halves hold here **because of the order**:
    the name, the register purpose and every officer are the head of the
    surface and come only from the register evidence unit; the website supplies
    a trailing description and can never displace, truncate or contribute a
    name.

    The website half is bounded by
    `entity_card_website.normalized_content_tokens` — the ratified ADR-060
    budget, in Arctic tokens, not a number invented here — and that function
    REFUSES to approximate the budget when the pinned tokenizer is absent
    rather than guessing at it.

    Measured (task 528a §6.8, one variable, 13 gold cards / 16 queries):
    register-only 0 h@1 / 2 h@10, this composition **6 / 7**, pre-528a
    website-only 7 / 10 — and all six wins are `intent-paraphrase` queries, not
    the one `site-verbatim` tautology.
    """
    dense = (dense or "").strip()
    if not website:
        return dense
    import cytir.corpus.entity_card_website as ECW
    tail = ECW.normalized_content_tokens(website, cap_tokens)
    if not tail:
        return dense
    if not dense:
        return tail
    sep = "" if dense.endswith((".", "!", "?", "…")) else "."
    return f"{dense}{sep} {tail}"


def resolve_channels(rows: Iterable[tuple[str, dict]],
                     website_tail: bool = False,
                     website_cap_tokens: int | None = None) -> dict | None:
    """Resolve one card's channels from every preserved row that mentions it.

    `rows` is an iterable of `(source_label, row)` **in the caller's tie-break
    order** -- the caller owns which source it prefers among equals; this
    function owns which CLASS of surface wins, which is the part the rulings
    constrain.  Returns None when no row carries any text at all (a tombstone).

    The result carries both channels, the class each came from, and the register
    evidence the representation contract checks the dense surface against, so a
    staged row is auditable without re-reading the 17 evidence files.

    `website_tail` selects ADR-060a's ruled composition: the winning
    register/description surface, then a bounded website description
    (`compose_dense_with_website`).  Off by default, so a caller has to ask for
    it and every stage records which branch it ran.
    """
    best_dense = None   # (rank, order, class, text, label)
    best_sparse = None  # (rank, order, class, text, label)
    hook_row = None     # (label, hooks) -- fallback when nothing sealed a channel
    web_row = None      # (label, website description) for the ruled tail
    evidence: dict = {}
    for order, (label, row) in enumerate(rows):
        d = dense_candidate(row)
        if d is not None:
            key = (DENSE_CLASS_RANK[d[0]], order)
            if best_dense is None or key < best_dense[:2]:
                best_dense = (key[0], order, d[0], d[1], label)
        s = sparse_candidate(row)
        if s is not None:
            key = (SPARSE_CLASS_RANK[s[0]], order)
            if best_sparse is None or key < best_sparse[:2]:
                best_sparse = (key[0], order, s[0], s[1], label)
        # (hook_row is collected below and only consulted after the loop)
        if hook_row is None:
            # `compose_register_hooks` reads only title, register UID and
            # people tokens -- all register-sourced fields, even when they sit
            # on a row whose PROSE came from the website. So a website row may
            # contribute hooks; it may not contribute its sealed hook bag,
            # which carries the site body as a second segment.
            hooks = compose_register_hooks(row)
            if hooks:
                hook_row = (label, hooks)
        if web_row is None:
            desc = website_description(row)
            if desc:
                web_row = (label, desc)
        for field in ("uid", "register_id", "people_tokens", "title"):
            if not evidence.get(field) and row.get(field):
                evidence[field] = row[field]
        if not evidence.get("register_officers"):
            people = row.get("people")
            if isinstance(people, dict) and people:
                evidence["register_officers"] = sorted(people.values())
        if not evidence.get("register_purpose"):
            reg = register_composition(row)
            if reg:
                evidence["register_purpose"] = reg.split(OFFICER_CLAUSE)[0].strip()
    if best_dense is None:
        return None
    # ADR-060a's ruled tail. Applied to the surface that ALREADY WON, so the
    # register text keeps the head and the website can only follow it -- and
    # never to a surface that is itself website-derived, which would put the
    # site body first and is the refusal this whole lane exists for.
    dense_text, dense_class = best_dense[3], best_dense[2]
    website_tail_source = None
    if website_tail and web_row is not None and dense_class != DENSE_WEBSITE:
        with_tail = compose_dense_with_website(dense_text, web_row[1],
                                               website_cap_tokens)
        if with_tail != dense_text:
            dense_text = with_tail
            dense_class = DENSE_REGISTER_PLUS_WEBSITE
            website_tail_source = web_row[0]
    composed_rank = SPARSE_CLASS_RANK[SPARSE_COMPOSED]
    use_composed = hook_row is not None and (
        best_sparse is None or best_sparse[0] > composed_rank)
    if use_composed:
        # Either no producer sealed a channel, or the only sealed one is the
        # website hook bag. Compose from the register evidence and segment it
        # against the dense chunk, rather than let the binder tokenise the
        # prose alone (the collapse being fixed) or ship the site body.
        sparse_text = compose_sparse(hook_row[1], dense_text)
        sparse_class, sparse_source = SPARSE_COMPOSED, hook_row[0]
    elif best_sparse is not None:
        sparse_text, sparse_class, sparse_source = (
            best_sparse[3], best_sparse[2], best_sparse[4])
    elif evidence.get("title"):
        # No register identity at all (a YC listing, a leaderboard, a seed
        # card). The card still HAS a name, and ADR-060 §(a) measured the name
        # as a card's route into the pool -- so the name gets its own BM25
        # segment ahead of the prose instead of being one term in a
        # 574-1,600-char chunk. Measured need: 2 of 339,307 cards had a title
        # token their prose does not carry AT ALL ("LMArena — Chatbot Arena LLM
        # Leaderboard": the body says "large language models", never "LLM"),
        # and 6,585 more carried the name only at prose weight.
        sparse_text = compose_sparse(str(evidence["title"]), dense_text)
        sparse_class, sparse_source = SPARSE_NAME_ONLY, "title"
    else:
        # COUNTED, never silently refilled with the dense prose.
        sparse_text, sparse_class, sparse_source = "", SPARSE_NONE, None
    return {
        "dense_text": dense_text,
        "dense_class": dense_class,
        "dense_source": best_dense[4],
        # named so a stage row says where its tail came from, and so a delta
        # can be counted by reason rather than by source label
        "website_tail_source": website_tail_source,
        "dense_register_only": best_dense[3],
        "sparse_text": sparse_text,
        "sparse_class": sparse_class,
        "sparse_source": sparse_source,
        "register_evidence": evidence,
    }

#!/usr/bin/env python3
"""Per-family REPRESENTATION contract, pinned by `prov_id` (ADR-089 draft).

`specs/chunk-vector-provenance.md` already says a row must name which SURFACE
fed which vector -- `channels.dense` / `channels.sparse`, "a closed vocabulary,
so 'what went into dense' is answerable by lookup rather than by reading the
producer's source" (invariant I-P5). What it does not do is CHECK the answer.
This module is that check, and it exists because two representation changes
shipped into a rebuild with nothing in the pipeline able to see them:

  1. the task-417 law materializers embed a surface whose first ~70% is the
     materializer's own provenance block (Canonical URL, Source record sha256,
     Jurisdiction, Archive SHA-256, build dates) with the statute as its tail;
  2. the task-467 entity cards reach BM25 through their DENSE prose, so
     ADR-060's `name + UID + discriminative hooks` sparse channel -- the leg
     ADR-060 measured as the ONLY thing that gets a card into the pool -- is
     replaced by sentences, which its own standing rule forbids.

Two independent checks, because they fail independently:

  CONFORMANCE  the surface the lane sealed matches the template its family
               declares. Cheap and total: the staged text is on disk next to
               the vectors, so this is a string check, not an inference.
  IDENTITY     the embedded dense surface actually carries the document's
               identity -- the title for a page, name+abbreviation for a law,
               name+UID for a card. A family may legitimately declare
               `body_only`; it may not legitimately lose the thing a query is
               made of without saying so.

Neither check needs a GPU, a collection, or a search. Both run at stage time,
which is the last moment a fix is cheap (verification-doctrine: cheap checks at
the front of expensive processes).
"""
from __future__ import annotations

import json
import pathlib
import re
import unicodedata

from cytir.core.paths import data_path

REGISTRY_PATH = data_path("representation-contract.v1.json")

# ---------------------------------------------------------------------------
# surface templates -- the closed vocabulary I-P5 asks for
# ---------------------------------------------------------------------------
#   body_only        the chunk body, verbatim
#   title_body       aw_common.compose_index_text: "<title>. <site_name>. <body>"
#   title_only_body  compose_index_text with title but no site_name
#                    (task 417 stage_atomic_embeddings script)
#   card_prose       ADR-060 dense channel: natural-language prose
#   card_hooks       ADR-060 sparse channel: name + UID + discriminative terms
#   record_native    the source's own record text, no AW composition
#   hierarchy_body   the law packs' sparse channel: the document's full
#                    ancestor heading path, then the body
#                    (fedlex_akn.compose_law_sparse_surface)
SURFACES = frozenset({"body_only", "title_body", "title_only_body",
                      "card_prose", "card_hooks", "record_native",
                      "hierarchy_body"})

# ---------------------------------------------------------------------------
# MARKUP THAT MUST NEVER BE TOKENISED (task 564)
#
# A materializer's own anchor identifiers are addressing, not language. The
# epoch-4 fedlex re-materialization emitted Akoma Ntoso eIds inline --
# `Art. 341 {#art_341} Art. 341 paragraph {#art_341/para_1} 1 For the period…`
# -- and every one of the 22,927 pack rows carried them into BOTH the dense and
# the sparse surface, against 0 of 299 matched rows in production (task 562
# §4.3). `aw_common.bm25_tokens` splits on `\w+`, so `{#art_341}` becomes the
# single token `art_341`; no query can produce it ("art 341" tokenises to `art`
# + `341`), and it still counts toward the `dl` that discounts every real term
# on the row. Corpus-wide in `aw_e8`: 105 of 8,000 id-stratified rows, 400 of
# 400 fedlex rows; in the twin, 0 of 7,945.
#
# Opt-in per family (`refused_index_markup`), like every other dimension here:
# a family whose source legitimately contains brace-hash text is not failed by
# a global rule nobody measured for it.
# ---------------------------------------------------------------------------
INDEX_MARKUP = {
    "akn_anchor_id": re.compile(r"\{#[^}\s]+\}"),
    # Task 2737: the `## Record identity` provenance heading (bfs/swisstopo/pubmed materializers).
    "record_identity_header": re.compile(r"^## Record identity\b", re.M),
}

# Machine identity a materializer emits about ITSELF. Not document content, and
# byte-similar across every row of a source -- which is what makes it poison
# for a dense leg: it is the part every article of every law has in common.
PROVENANCE_BULLET = re.compile(
    r"^-\s*(Canonical URL|Official URL|Source record|Source URL|Jurisdiction|"
    r"Article anchor|Official build date|Promulgation date|Archive member|"
    r"Archive SHA-?256|Parent archive|Parent archive SHA-?256|Parent OAI page|"
    r"Source status|Effective date|Document sha-?256|Parent sha-?256|Record id|"
    r"Source profile|Retrieved|Checksum|Source version|Norm build date|"
    r"Release build date|Parent identity|ELI|Language|"
    r"Consolidated source version|XML SHA-?256|Pinned source commit|PMID|"
    r"DOI|arXiv ID|OAI identifier|OAI datestamp|Point id|Content digest)\b",
    re.I)
# YAML front-matter emitted by a materializer about its own inputs
PROVENANCE_YAML = re.compile(r"^aw_[a-z0-9_]+:", re.I)
PROVENANCE_HEADING = re.compile(
    r"^(Article identity|Version and status|Document identity|"
    r"Record identity|Provenance|Source identity|Official source identity|"
    r"Legal hierarchy)\s*$", re.I)


class ContractError(ValueError):
    """A staged row cannot be represented truthfully under its contract."""


def registry_sha256(path=None) -> str:
    """The digest of the registry a run judged itself against.

    A receipt that names the family but not the version of the rule cannot be
    re-derived later: the standard may have moved. Cheap, so every producer
    that runs a contract check records it."""
    import hashlib
    p = pathlib.Path(path or REGISTRY_PATH)
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_registry(path=None):
    p = pathlib.Path(path or REGISTRY_PATH)
    doc = json.loads(p.read_text())
    if doc.get("schema") != "aw.representation-contract.v1":
        raise ContractError(f"{p}: not aw.representation-contract.v1")
    for key, fam in doc["families"].items():
        for ch in ("dense", "sparse"):
            if fam[ch] not in SURFACES:
                raise ContractError(
                    f"{key}: {ch} surface {fam[ch]!r} is not in the closed "
                    f"vocabulary {sorted(SURFACES)}")
    return doc


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------
def provenance_share(text: str) -> tuple[float, int, int]:
    """Fraction of the surface's non-blank LINES that are machine identity.

    Line-based rather than character-based on purpose: a 64-hex sha is one
    line and one line of statute is one line, so lines weigh the two the way a
    reader would, and a long document is not let off because its boilerplate is
    short in proportion.
    """
    prov = total = 0
    inblock = False
    for raw in text.split("\n"):
        s = raw.strip()
        if not s:
            continue
        total += 1
        if PROVENANCE_HEADING.match(s):
            prov += 1
            inblock = True
        elif PROVENANCE_BULLET.match(s) or PROVENANCE_YAML.match(s):
            prov += 1
        elif inblock and s.startswith("- "):
            prov += 1
        else:
            inblock = False
    return (prov / total if total else 0.0), prov, total


def _norm(s: str) -> str:
    """Lowercase, and collapse everything that is not a word character.

    `\\w` under `re.UNICODE`, not `[0-9a-z]`: an ASCII-only class erases a
    CJK title entirely, and a check whose needle normalises to the empty
    string reports "absent" for every surface on earth. Task 528a measured
    exactly one such row -- an F2-HTML page titled
    `專為編碼代理提供的可靠最佳化模型` -- and it was the checker's defect, not
    the row's. Keeping accents is the same call in the other direction:
    `Küsnacht` stays one token instead of becoming `k snacht`, and both sides
    of every comparison are normalised the same way.
    """
    return re.sub(r"[^\w]+", " ", (s or "").lower(), flags=re.UNICODE).strip()


def carries(text: str, needle: str) -> bool:
    """Is `needle` present in `text` as a token sequence, ignoring punctuation
    and case? Punctuation-insensitive because a title is `Art. 336 CO - Fedlex`
    in one place and `Art 336 CO Fedlex` in the other, and a check that a human
    would call a match must not fail on a hyphen."""
    n = _norm(needle)
    return bool(n) and n in _norm(text)


def _strip_accents(s: str) -> str:
    """NFKD-decompose and drop combining marks: 'assemblée' -> 'assemblee'."""
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                    if not unicodedata.combining(c))


def carries_foldaccents(text: str, needle: str) -> bool:
    """Like `carries`, but accent-INSENSITIVE on both sides (task 560).

    Scoped to ONE caller: the site_name_identity check, whose `needle` is
    always `aw_common.site_entity_label`'s output -- a hostname label, which
    is definitionally plain ASCII (a DNS name cannot carry an accent) --
    compared against a real entity name the page's own title may spell WITH
    one ("assemblee-nationale.fr" -> label "Assemblee-nationale", while the
    row's title says "Assemblée nationale"; `carries("...Assemblée
    nationale...", "Assemblee-nationale")` is False under plain `_norm`,
    which is a checker false-positive, not a missing site name -- confirmed
    by hand: the row's composed surface DOES carry "Assemblée nationale"
    verbatim, `identity.site_name.dense` just could not see it).

    `carries`/`_norm` deliberately do NOT fold accents everywhere ("Küsnacht
    stays one token instead of becoming k snacht" -- `_norm`'s own
    docstring), so this is a dedicated helper, not a change to `_norm`
    itself: folding is one-directional-safe only because the needle here is
    KNOWN to have already lost any accent by construction (it came from a
    hostname), so stripping accents from the haystack too can only turn a
    false "absent" into a true "present" for this one check -- it cannot
    manufacture a new false "present" that the un-folded label wouldn't
    already have been a substring of on some other row.
    """
    n = _strip_accents(_norm(needle))
    return bool(n) and n in _strip_accents(_norm(text))


def carries_composed(text: str, needle: str) -> bool:
    """Is `needle` present, allowing a citation to have been INSERTED into it?

    A family that composes `"<law name> (<abbr>) - Art. N"` from a payload title
    of `"<law name> - Art. N"` carries every word of the identity and none of it
    contiguously. Requiring contiguity there fails 80% of a family that is in
    fact carrying its title twice (task 528b measured exactly that on gesetze
    before this function existed). Splitting on the family's own separator and
    requiring EVERY part keeps the check strict -- a dropped article number
    still fails -- without failing on an insertion the contract itself asked
    for.

    NOT wired into `check_row`'s `identity.dense` check (task 534, CI repair).
    Task 528b's original commit (952642f4) wired it there directly, but that
    call site was silently dropped by the merge that landed task528b
    alongside task528a-2's sibling work, and restoring it verbatim regresses
    `test_representation_contract_cards.py::test_the_website_blurb_surface_is_refused`:
    the entity-card `register_purpose` field ("Vector8 AG -- Entwicklung ...")
    splits on the same em-dash separator, and both halves happen to occur
    separately (not contiguously) inside an unrelated marketing blurb that the
    register-first ruling exists specifically to refuse -- so the
    citation-insertion tolerance built for gesetze silently defeats the
    register-first check it was never designed against. That is a real
    product-behaviour conflict between two branches' work, not a mechanical
    restoration, so it is left for a deliberate decision (narrow the split to
    law-shaped families, require ordered/adjacent parts, or something else)
    rather than resolved here. The function itself is restored and unit-tested
    (`InsertedCitationIsNotAMissingIdentity` in test_representation_contract.py).
    """
    if carries(text, needle):
        return True
    parts = [p for p in re.split(r"\s+[-–—]\s+", needle) if p.strip()]
    return len(parts) > 1 and all(carries(text, p) for p in parts)


def carries_tokens(text: str, needle: str) -> bool:
    """Is every content token of `needle` present as a token of `text`?

    The right test for a BAG-OF-HOOKS channel, where contiguity is meaningless
    by construction: ADR-060's sparse surface is stopword-dropped and reordered
    (`"Seminar und Bildungshaus St. Wiborada"` is sealed as
    `"Seminar Bildungshaus St Wiborada Georgen …"`), and BM25 scores per token,
    not per phrase. Applied ONLY to the sparse checks -- `identity.dense` keeps
    the contiguous test, because there a scattered title is exactly the failure
    mode (the F2-HTML body that "contains" its title only by coincidence).

    The needle passes through the channel's OWN stopword filter first
    (`cytir.extract.retailer_card_channels.drop_stopwords`, imported rather than restated --
    a second copy of that list is a forked mirror contract). Without it the
    check demands tokens the channel is defined to remove: `und`, `am`, `sa`
    and `org` are all in `_STOP`, so "Kairos Holding SA" and "Mystery.org"
    read as absent from postings that carry them entirely.
    """
    want = assertable_tokens(needle)
    if not want:
        return True
    return want <= set(_norm(text).split())


def assertable_tokens(needle: str) -> set[str]:
    """The tokens of `needle` that a stopword-dropping channel can carry.

    Empty means the value is UNASSERTABLE through this channel, not absent
    from it: "L.E.S.S. SA", "AM ET SA", "NO FUR" and "And and and" are real
    card titles made entirely of terms `_STOP` removes, and demanding them back
    measures the filter rather than the surface. `check_row` reports that case
    under its own check name so the guard's reach stays visible instead of
    being quietly widened.
    """
    from cytir.extract.retailer_card_channels import drop_stopwords
    return {t for t in _norm(drop_stopwords(needle)).split() if len(t) >= 2}


def render_surface(surface: str, rec: dict) -> str:
    """The text a family's declared template says this record must embed."""
    body = rec.get("text") or ""
    if surface == "body_only" or surface == "record_native":
        return body
    if surface == "title_body":
        import cytir.core.common as AC
        return AC.compose_index_text(rec)
    if surface == "title_only_body":
        import cytir.core.common as AC
        return AC.compose_index_text(
            {"text": body, "title": rec.get("title"),
             "source_type": rec.get("source_type")})
    if surface == "card_prose":
        return rec.get("dense_text") or body
    if surface == "card_hooks":
        return rec.get("sparse_text") or ""
    if surface == "hierarchy_body":
        # The law packs' declared sparse channel, composed by the ONE home
        # `fedlex_akn.compose_law_sparse_surface` -- imported, never restated,
        # so the checker and `fedlex_dump.parse_akn` cannot disagree.
        #
        # A record that carries no hierarchy CANNOT satisfy this surface, and
        # saying so is the whole point: the epoch-4 fedlex pack sealed a
        # `sparse_text` with the law title repeated per section and no ancestor
        # path at all, and a template that quietly fell back to the body would
        # have passed it. `check_row` turns the refusal into `conformance.sparse`.
        from cytir.sources.families import fedlex_akn as FA
        if not (rec.get("heading_ctx") or "").strip():
            raise ContractError("hierarchy_body: the record declares no "
                                "`heading_ctx`, so the family's sparse surface "
                                "cannot be rendered for it")
        return FA.compose_law_sparse_surface(
            rec["heading_ctx"], rec.get("sparse_body") or body)
    raise ContractError(f"unknown surface {surface!r}")


# ---------------------------------------------------------------------------
# the two checks
# ---------------------------------------------------------------------------
def check_row(rec: dict, fam: dict, embedded: str | None = None,
              sparse_surface_text: str | None = None) -> list[dict]:
    """Return the violations for ONE staged row. Empty list == conformant.

    `embedded` is the text the lane actually sealed for the dense channel (the
    `text` field of its `shard-NNNN.texts.jsonl.gz` line). `sparse_surface_text`
    is the surface the binder will hand to `saturated_bm25` -- for every route
    in `assembly_binder.py` today that is `rec["text"] or rec["dense_text"]`,
    which is exactly the collapse this checks for.
    """
    out = []
    dense = embedded if embedded is not None else render_surface(fam["dense"], rec)

    # 1. CONFORMANCE -- dense
    want = render_surface(fam["dense"], rec)
    if embedded is not None and embedded != want:
        out.append({"check": "conformance.dense", "surface": fam["dense"],
                    "detail": "the sealed dense text is not what the family's "
                              "declared template renders",
                    "sealed_head": embedded[:120], "template_head": want[:120]})

    # 2. CONFORMANCE -- sparse
    if sparse_surface_text is not None:
        try:
            want_sp = render_surface(fam["sparse"], rec)
        except ContractError as exc:
            out.append({"check": "conformance.sparse", "surface": fam["sparse"],
                        "detail": f"the declared sparse surface cannot be "
                                  f"rendered for this record: {exc}",
                        "tokenised_head": sparse_surface_text[:120],
                        "template_head": ""})
            want_sp = ""
        if want_sp and sparse_surface_text != want_sp:
            out.append({"check": "conformance.sparse", "surface": fam["sparse"],
                        "detail": "the surface that will be tokenised is not "
                                  "the family's declared sparse channel",
                        "tokenised_head": sparse_surface_text[:120],
                        "template_head": want_sp[:120]})

    # 3. IDENTITY -- the embedded dense surface must carry what a query is
    #    made of. Fields are named per family; a missing FIELD is not a
    #    violation (an absent title is a different defect), an unrepresented
    #    present field is.
    for field in fam.get("identity_fields", []):
        val = rec.get(field)
        if isinstance(val, (list, tuple)):
            val = " ".join(str(v) for v in val[:4])
        if not val:
            continue
        if not carries(dense, str(val)):
            out.append({"check": "identity.dense", "field": field,
                        "detail": f"{field!r} is not present in the embedded "
                                  "dense surface",
                        "value": str(val)[:120]})

    # 4. IDENTITY -- sparse hooks (the ADR-060 card rule, generalised)
    for field in fam.get("sparse_identity_fields", []):
        val = rec.get(field)
        if isinstance(val, (list, tuple)):
            val = " ".join(str(v) for v in val[:4])
        if not val or sparse_surface_text is None:
            continue
        if not assertable_tokens(str(val)):
            out.append({"check": "identity.sparse_unassertable", "field": field,
                        "detail": f"{field!r} is made entirely of terms the "
                                  "hook channel's own stopword filter removes, "
                                  "so its presence cannot be asserted here",
                        "value": str(val)[:120]})
        elif not carries_tokens(sparse_surface_text, str(val)):
            out.append({"check": "identity.sparse", "field": field,
                        "detail": f"{field!r} is not present in the surface "
                                  "that becomes the BM25 posting",
                        "value": str(val)[:120]})

    # 4b. IDENTITY, ELEMENT-WISE -- for a field whose value is a LIST of
    #     independently checkable assertions (a card's register officers), each
    #     element must survive into the embedded surface. Checking the joined
    #     list as one substring would pass a surface that carries the first
    #     officer and truncates the rest, which is precisely what shipped:
    #     the Vector8 card's 1,600-char website blurb ends "… Marc Maret (…); Ni".
    for field in fam.get("identity_fields_each", []):
        vals = rec.get(field)
        if not vals:
            continue
        if isinstance(vals, str):
            vals = [vals]
        missing = [str(v) for v in vals if not carries(dense, str(v))]
        if missing:
            out.append({"check": "identity.dense_each", "field": field,
                        "detail": f"{len(missing)} of {len(vals)} {field} "
                                  "assertions are not in the embedded dense "
                                  "surface",
                        "missing": [m[:80] for m in missing[:4]]})

    # 4d. IDENTITY -- THE SITE'S OWN NAME (task 548; ADR-061 bug #1, recurring).
    #
    #     ADR-061: "the title strips the entity/site-name suffix ... DISASTROUS
    #     for a page where the suffix IS the entity name". A row can hold a
    #     present, non-empty, correctly-composed title and still be unfindable
    #     by the one token every query for it is made of, because the token was
    #     already gone when the title was captured. `identity.dense` cannot see
    #     that: it compares the surface to the title, and a stripped title
    #     matches a stripped surface perfectly.
    #
    #     MEASURED (task 548): 13 of the 19 impactia.org rows in the epoch-4
    #     candidate carry `La Fondation`, `Conseil de Fondation`, `Women in Ai`
    #     -- and `site_name: null` on every staged row, so `compose_index_text`
    #     had nothing to fold in. `impactia.org/la-fondation/` holds no
    #     `impactia` posting at any weight while its siblings, whose titles kept
    #     the suffix, score 22-27 on the same query. The `names` gold class went
    #     5 -> 2 h@1.
    #
    #     Opt-in per family, like every other dimension here: a family that has
    #     never carried a site name is not failed over a rule nobody measured
    #     for it. When declared, the expected token comes from
    #     `aw_common.site_entity_label` -- the SAME function the composer uses,
    #     never a second reading of "what is this site called".
    sn = fam.get("site_name_identity")
    if sn:
        st = rec.get("source_type")
        want_types = sn.get("source_types")
        if (want_types is None or st in want_types) and rec.get("url"):
            from cytir.extract.html_extract import site_entity_label
            label = site_entity_label(rec["url"])
            if not label:
                out.append({"check": "identity.site_name_unassertable",
                            "url": rec["url"],
                            "detail": "the registrable label names nothing "
                                      "distinctive, so the site name's presence "
                                      "cannot be asserted for this row",
                            "value": rec["url"][:120]})
            else:
                for channel in sn.get("channels", ["dense", "sparse"]):
                    text = dense if channel == "dense" else sparse_surface_text
                    if text is None:
                        continue
                    if not carries_foldaccents(text, label):
                        out.append({"check": f"identity.site_name.{channel}",
                                    "site_name": label, "url": rec["url"],
                                    "title": str(rec.get("title") or "")[:120],
                                    "detail": "the site's own name token is "
                                              f"absent from the {channel} "
                                              "surface; the title lost its "
                                              "site-name suffix and no "
                                              "`site_name` replaced it "
                                              "(ADR-061 bug #1)"})

    # 4c. REFUSED DENSE CLASS -- Tim's register-first ruling (2026-09-06), as a
    #     stage-time refusal rather than a review habit. The producer declares
    #     which CLASS of surface it resolved (`dense_class`, from
    #     `card_channels`); a family may name classes that are
    #     never an acceptable dense surface for it. For entity cards that is
    #     `website_content`: officers and identity come from the commercial
    #     register, never from the entity's own marketing copy.
    refused = set(fam.get("refused_dense_class", []))
    if refused and rec.get("dense_class") in refused:
        out.append({"check": "dense.class_refused",
                    "dense_class": rec.get("dense_class"),
                    "refused": sorted(refused),
                    "detail": "the resolved dense surface is of a class this "
                              "family refuses"})

    # 4e. INDEX MARKUP -- a materializer's anchor identifiers must not be
    #     tokenised (task 564). Checked on BOTH channels a family names,
    #     because the epoch-4 fedlex leak reached both: the pack's `text` and
    #     its `sparse_text` each carry `{#art_N}` on 22,927 of 22,927 rows.
    #     The verdict is per channel so a dense-only or sparse-only delta can
    #     be judged on the channel it moved.
    for name in fam.get("refused_index_markup", []):
        pat = INDEX_MARKUP.get(name)
        if pat is None:
            raise ContractError(f"unknown index-markup class {name!r}")
        for channel, text in (("dense", dense), ("sparse", sparse_surface_text)):
            if not text:
                continue
            found = pat.findall(text)
            if found:
                out.append({"check": f"markup.{channel}", "markup": name,
                            "detail": f"{len(found)} {name} identifier(s) are "
                                      f"in the {channel} surface; they are "
                                      "addressing, not language, and no query "
                                      "can produce the token they become",
                            "examples": found[:4]})

    #     Task 2733: the same identifiers in the DISPLAY fields. `snippet` and `title` are not
    #     tokenised, but a reader (an agent) sees them; the 22.9k twin fedlex rows served
    #     `Art. 336 c {#art_336_c} ... paragraph {#...}` for months because this check read
    #     only the two index channels. Checked when the row carries the field.
    for name in fam.get("refused_index_markup", []):
        pat = INDEX_MARKUP.get(name)
        for fld in ("snippet", "title"):
            found = pat.findall(str(rec.get(fld) or "")) if pat else []
            if found:
                out.append({"check": f"markup.{fld}", "markup": name,
                            "detail": f"{len(found)} {name} identifier(s) are in the "
                                      f"{fld}; a reader sees them",
                            "examples": found[:4]})

    # 5. PROVENANCE CAP -- the materializer's own metadata must not be the
    #    document. Off by default (None), because a family that legitimately
    #    embeds structured records should say so rather than be caught by a
    #    global rule.
    cap = fam.get("max_provenance_share")
    if cap is not None:
        share, prov, total = provenance_share(dense)
        if share > cap:
            out.append({"check": "provenance.cap", "cap": cap,
                        "share": round(share, 4),
                        "detail": f"{prov} of {total} non-blank lines of the "
                                  "embedded surface are materializer identity, "
                                  "not document"})
    return out


def evaluate(counts: dict, rows: int, fam: dict) -> dict:
    """Turn per-check violation COUNTS into a per-check verdict.

    A family may declare a tolerance per check in `max_fail_rate_by_check`,
    and every tolerance must carry its measured reason in the registry -- a
    number with no reason next to it is a waiver, and waivers escalate
    (memory: gate-waivers-escalate-to-tim). The default is zero.
    """
    limits = fam.get("max_fail_rate_by_check") or {}
    default = float(fam.get("max_fail_rate", 0.0))
    out = {}
    for check in sorted(set(counts) | set(limits)):
        n = int(counts.get(check, 0))
        rate = (n / rows) if rows else 0.0
        cap = float(limits.get(check, default))
        out[check] = {"violations": n, "rate": round(rate, 6), "cap": cap,
                      "pass": rate <= cap}
    return {"rows": rows, "by_check": out,
            "pass": all(v["pass"] for v in out.values())}


def binder_sparse_surface(rec: dict, fam: dict) -> str:
    """The surface `assembly_binder.py` will actually hand to `saturated_bm25`.

    THIS IS A MIRROR OF THE BINDER AND IT DRIFTED ONCE. The first cut hardcoded
    the pre-528a line, `rec["text"] or rec["dense_text"]`. Task 528a then gave
    a family that DECLARES `sparse_surface_field` its own channel:

        sparse_surface = (rec.get("sparse_text") or "").strip()
        if (_sparse_channel_declared and corpus_row is not None
                and sparse_surface and sparse_surface != surface):
            corpus_row["sparse_text"] = sparse_surface

    -- so from that day the checker measured a surface the binder no longer
    used for exactly the families the contract exists to protect. Task 533 ran
    it against the sealed entity-card stage and got `conformance.sparse` on
    20,000 of 20,000 rows: a 100% FAIL on a family whose sparse channel is
    correct. A checker that cries wolf on its own subject is worse than no
    checker, because the next real failure reads as the same noise.

    The condition is copied from the binder, including `!= surface`, and the
    fallback is the binder's fallback -- not "the field if it exists".
    """
    surface = rec.get("text") or rec.get("dense_text") or ""
    if not fam.get("sparse_surface_field"):
        return surface
    own = (rec.get(fam["sparse_surface_field"]) or "").strip()
    return own if (surface and own and own != surface) else surface


def check_sample(rows, fam, get_embedded=lambda r: r.get("text"),
                 get_sparse=None, max_fail_rate=0.0):
    """Run `check_row` over a staged sample and summarise.

    `get_sparse`'s default is not a guess: it is `binder_sparse_surface`, which
    mirrors `assembly_binder.py`'s own selection, so the check measures the
    surface the binder WILL tokenise rather than the one the producer hoped it
    would.
    """
    if get_sparse is None:
        def get_sparse(r, _fam=fam):
            return binder_sparse_surface(r, _fam)
    viol, per_check = [], {}
    for r in rows:
        v = check_row(r, fam, get_embedded(r), get_sparse(r))
        for x in v:
            per_check[x["check"]] = per_check.get(x["check"], 0) + 1
        if v:
            viol.append({"point_id": r.get("point_id"), "url": r.get("url"),
                         "violations": v})
    rate = len(viol) / len(rows) if rows else 0.0
    return {"rows": len(rows), "rows_with_violations": len(viol),
            "rate": round(rate, 4), "by_check": per_check,
            "pass": rate <= max_fail_rate, "examples": viol[:5]}

"""cytir.index.compose — embed a sealed chunk stage, verify its point ids, and
enqueue a `cytir/index/promote_queue.py` promotion-queue pack (task 657).

Task 656 relaunched ERIC PDF materialization through the routed extractor
(`cytir/extract/pdf.py`) and sealed 14 tranches' worth of chunk stages, but
deliberately stopped short of embed -> compose -> enqueue: its own
`task 656 NEXT.md` names the real risk ("a wrong identity/mint
declaration here is a KNOWN, real failure mode ... task 546: 53,822/84,961
staged rows silently addressing nothing from a bad id-mint guess") and asks
the next session to close it properly rather than rush it. This module is
that close.

THE ID RISK, MEASURED (task 657)
---------------------------------
Two DIFFERENT functions mint a "point_id" in this repo, and they do not
agree:

* `cytir.core.ids.mint_point_id` -- the "b1"/future scheme `build_aw_b1.py`
  mints against for a from-scratch rebuild. Its own module docstring says
  it "deliberately does NOT touch the live `aw_common.Passage.__post_init__`
  scheme (which conflates `sha1(text)` into the id -- the F1 bug I1 exists
  to kill)". It is NOT what the current staging pipeline
  (`stage_structured_markdown_for_embedding.py`) mints, and it is not what
  any already-promoted sealed-pack family's ids reproduce under.
* `cytir.core.common.mint_passage_point_id(url, anchor, text)` -- the
  LEGACY/live scheme every sealed pack today is actually minted under
  (`stage_structured_markdown_for_embedding.py`'s own producer code).
  `promote_sealed_pack.py`'s 2026-09-23 correction (superseding a WRONG
  2026-09-22 "252 combinations, none matched" finding) re-derived this
  exact function, over `canonical_text`/`text`, with anchor
  `document:<doc_id>:chunk-<ordinal>`, for all four W1 packs:
  80,117/80,117 rows, 100%.

`verify_identity()` below re-runs that same check on ERIC's OWN real
records (not inferred from the W1 families' result) -- 111/111 rows of the
task-656 smoke pack reproduce under `aw_common.mint_passage_point_id`;
`cytir.core.ids.mint_point_id` reproduces 0/111. `compose_pack()` therefore
composes every manifest with `identity_check: "pack-declared"` (trust the
pack's own, VERIFIED-reproducible id) and REFUSES to compose at all if any
row's declared id fails to reproduce -- per the task 657 brief: "Refuse to
compose on any id mismatch."

DEPENDENCY NOTE (architecture-doctrine #4)
-------------------------------------------
Everything this module imports is a package module. `promote_queue` (the live drain)
moved into `cytir/index/` with alias shims left in `pipeline/` (task 677, S5); this module
takes one READ-ONLY helper from it (`fetch_collection_vector_config`) rather than
forking it. `test_aw_does_not_import_pipeline.py` holds the arrow.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cytir.core import common as aw_common  # noqa: E402  (mint_passage_point_id -- the one legacy home)
from cytir.core import paths as PATHS  # noqa: E402
from cytir.core.paths import operator_settings  # noqa: E402
from cytir.index import normalize_sealed_pack_shape as NSP  # noqa: E402  (merge/write_shard -- backfill_lang, task 669)
from cytir.index import promote_sealed_pack as PSP  # noqa: E402  (read_records, SEALED_DIM)
from cytir.index import promote_queue as PQ  # noqa: E402  (fetch_collection_vector_config)

from cytir.consent import stamp as CONSENT_STAMP  # noqa: E402  (task 684 / ruling 20: the ADR-115 engine, stamped at compose)
from cytir.embed.stage import embed_stage  # noqa: E402
from cytir.index import exclude_file as XF  # noqa: E402
from cytir.index import id_rule as IR  # noqa: E402  (task 2218: the drain's id rule, asked at compose)
from cytir.index import german_parts_guard as GPG  # noqa: E402  (task 710 guard 1)
from cytir.index import manifest_rows as MR  # noqa: E402  (rows_staged / rows_to_write_max: one reading rule)
from cytir.index import row_decode as RD  # noqa: E402  (task 855: canonical stored rows, any profile)
from cytir.core.search_result_url import REJECTED_REASON, rejected_reason  # noqa: E402,F401
from cytir.index import served_exclusion as SE  # noqa: E402  (ruling 23.5: already-served ids excluded at compose)
from cytir.index import line_coverage as LC  # noqa: E402  (ruling 27 T6: per-id line coverage block)
from cytir.index import url_group_supersede as UGS  # noqa: E402  (ruling 27 T6: the url-group fields)

MANIFEST_SCHEMA = "aw.promotion-queue-pack.v1"


def default_queue_dir() -> str:
    """The promotion queue: operator setting `queue_dir`, refused by name when unconfigured."""
    return operator_settings().queue_dir


def __getattr__(name):
    # `compose.DEFAULT_QUEUE_DIR` stays readable but resolves on first ACCESS, never at import.
    if name == "DEFAULT_QUEUE_DIR":
        return default_queue_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class ComposeError(RuntimeError):
    """Refused before any manifest is written or any pack is enqueued."""


def write_line_coverage_block(block: dict, pack_dir) -> dict:
    """Write `<pack_dir>/line-coverage.v1.json` and return the manifest fields that bind it
    (`line_coverage_file`, `line_coverage_sha256`, `line_coverage`)."""
    path = Path(pack_dir) / LC.FILE_NAME
    data = LC.block_bytes(block)
    path.write_bytes(data)
    return LC.manifest_fields(block, data, path)


def check_supersede_options(family: str, *, supersede: str | None = None,
                            allow_supersede: str | None = None, copy_sealed_guard: bool = False,
                            line_coverage_old_texts=(), line_coverage_wanted_ids=(),
                            served_targets=None, supersede_url_group=None,
                            exclude_ids_file=None, ruled_skip=None) -> bool:
    """Ruling 27 T6: the explicit options that make a pack a url-group supersede pack. True when
    any is given (all must then be), False when none is. Raises `ComposeError` for: a family
    outside `url_group_supersede.URL_GROUP_FAMILIES` (blind families never use it); a `supersede`
    value other than `url-group`; no `allow_supersede` reason; no `copy_sealed_guard`; line-coverage
    inputs without `supersede`; wanted ids without old-text sources; and `served_targets` (the
    already-served exclusion would drop the unchanged rows that open their URL groups). Pure:
    `compose_pack` and `embed_compose_enqueue` both call it first, so a refused pack never burns
    the GPU or writes a stamp."""
    old_texts, wanted = list(line_coverage_old_texts or ()), list(line_coverage_wanted_ids or ())
    if supersede_url_group is not None and (supersede or allow_supersede or copy_sealed_guard
                                            or old_texts or wanted):
        raise ComposeError("--supersede-url-group (task 876) and --supersede url-group "
                           "(ruling 27 T6) are two doors to one manifest field; pass one")
    if (supersede or supersede_url_group is not None) and served_targets:
        raise ComposeError("a url-group supersede pack is composed without the already-served "
                           "exclusion (--served-prod-url), through either door: an unchanged row "
                           "keeps its id and opens its URL group; excluding it would leave the old "
                           "rows of that URL in place or delete them unrewritten")
    if (supersede or supersede_url_group is not None) and exclude_ids_file:
        raise ComposeError("a url-group supersede pack is composed without the caller's "
                           "--exclude-ids-file, through either door (task 892): an excluded "
                           "unchanged row keeps no served copy to rewrite, so its served copy is "
                           "deleted and never rewritten")
    if ruled_skip and not (supersede and old_texts):
        raise ComposeError("--ruled-skip (plan.md T24) needs --supersede url-group and "
                           "--line-coverage-old-texts: it waives one id of the line coverage block")
    if not (supersede or allow_supersede or copy_sealed_guard or old_texts or wanted):
        return False
    if family not in UGS.URL_GROUP_FAMILIES:
        raise ComposeError(
            f"{family!r}: the url-group supersede options are refused for this family; only "
            + ", ".join(sorted(UGS.URL_GROUP_FAMILIES)) + " may use them (URL_GROUP_FAMILIES)")
    if supersede != UGS.URL_GROUP:
        raise ComposeError(f"--supersede is {supersede!r}; the only value is {UGS.URL_GROUP!r} "
                           "(the other url-group options need it)")
    if not (allow_supersede or "").strip():
        raise ComposeError("--supersede url-group needs --allow-supersede <reason> (the drain "
                           "refuses a pack whose pages are served already without it)")
    if not copy_sealed_guard:
        raise ComposeError("--supersede url-group needs --copy-sealed-guard: the manifest carries "
                           "the sealed pack's own stage guard block, or the drain refuses it")
    if wanted and not old_texts:
        raise ComposeError("--line-coverage-wanted-ids needs --line-coverage-old-texts")
    return True


def default_anchor(record: dict) -> str:
    """`document:{document_id}:chunk-{chunk_ordinal}` -- the anchor form
    `cytir.corpus.stage_structured_markdown` mints for every
    `chunk.generic` family (confirmed identical for `de_openlegaldata_v1`
    by the pack's own family contract, and empirically for
    `eric_official_pdf` -- task 657, 111/111 real rows of the task-656
    smoke pack).
    """
    return f"document:{record['document_id']}:chunk-{record['chunk_ordinal']}"


def raw_anchor_field(record: dict) -> str | None:
    """The anchor `cytir.core.common.Passage`/`mint_passage_point_id` used
    DIRECTLY at mint time -- a `#fragment` string, or `None` for a
    whole-page passage (`Passage.anchor` defaults to `None`; CC never sets
    `document_id`/`chunk_ordinal` at all -- identity is per-URL, not
    per-document-chunk-ordinal).

    Root cause this fixes (task 664, 2026-09-26): `cytir.ops.growth_pending_compose.
    py`'s CC discovery routed `cc_authority_frontier_w7` packs through this
    module's ONE hardcoded `default_anchor`, which unconditionally reads
    `record['document_id']` -- a `KeyError` on every CC row, since CC chunks
    carry no such field (tranche4/5's first real compose attempts, journal
    13:25 CEST). tranche2/3's ORIGINAL landing (before this unattended path
    existed) worked around the same gap by composing through a bespoke
    script with `identity_check="pack-declared"` set by hand -- see the
    already-enqueued `cc-authority-frontier-w7-tranche3` manifest's own
    `identity_reason`, which records the exact prior measurement this
    docstring continues: `default_anchor` does NOT reproduce CC's ids
    (a confirmed mismatch, 2026-09-23), because CC was never minted under
    the document/chunk-ordinal convention in the first place.

    Verified 2026-09-26 on 2 real tranche4 rows (`testdata/
    task664_cc_real_rows.jsonl`): `aw_common.mint_passage_point_id(url,
    record.get("anchor"), text)` reproduces the row's declared `point_id`
    100% using THIS anchor -- the row's own raw field, not a derived or
    invented one.
    """
    return record.get("anchor")


#: Per-family anchor scheme for `verify_identity()`'s reproduction check.
#: `default_anchor` (the ERIC/W1 document/chunk-ordinal convention) is the
#: fallback for every family not listed here -- KEEP that default; adding an
#: entry below is an explicit, evidenced statement that the family's rows
#: are minted a different way (never guessed -- see the winning function's
#: own docstring for the real-record evidence backing its entry).
FAMILY_ANCHOR_FN: dict[str, Callable[[dict], str | None]] = {
    "cc_authority_frontier_w7": raw_anchor_field,
    # task 696: `cytir.extract.dila_juri` builds a legacy `Passage` (anchor None); its ids are
    # `mint_passage_point_id(url, None, text)`. Evidence: test_aw_family_stage.py on the
    # real CASS/JADE fixtures, and the real-path smoke stage (task 696 doc).
    "dila_juri": raw_anchor_field,
    "dila_kali": raw_anchor_field,      # same legacy `Passage` mint (anchor None), task 696
    "dila_jorf": raw_anchor_field,
    # task 698: `Passage` built by `passage_from_chunk` (anchor None) -> mint_passage_point_id(url, None, text)
    "us_federal_register_ecfr": raw_anchor_field,
    "govuk_content": raw_anchor_field,     # task 699: same `passage_from_chunk` mint (anchor None)
    # task 696: `passage_from_chunk` with anchor `chunk-<n>` -> mint_passage_point_id(url, anchor, text)
    "stackexchange_nonso_qa_p0": raw_anchor_field,
    # task 696: same `passage_from_chunk` mint, anchor `chunk-<n>`
    "openalex_academic_dump": raw_anchor_field,
    # task 696: same `passage_from_chunk` mint, anchor `chunk-<n>` (the six fixture fiches reproduce 6/6)
    "service_public_fiche": raw_anchor_field,
    # task 2204: `statcan_wds` stages atomic legacy `Passage` cards (`anchor` None, no document_id /
    # chunk_ordinal); the default `default_anchor` raised `KeyError: 'document_id'` at compose.
    "statcan_wds": raw_anchor_field,
    # task 2204: `uk_legislation` mints via `passage_from_chunk` with anchor `<eId>[:chunk-<n>]`
    # (the row's own `anchor`); its rows also carry document_id, so the default would have
    # composed the WRONG anchor and been refused by the identity check, not by a KeyError.
    "uk_legislation": raw_anchor_field,
}


#: The point-id SCHEME a family's sealed pack is minted under (task 696, slice 8 step 2). Two schemes
#: are in prod; compose used to check only the first for every family. A family not listed here is
#: `ID_SCHEME_LEGACY` -- the default, so every existing manifest (and its READY sha) is unchanged.
#: Adding an entry is an explicit, evidenced statement, never a guess:
#:   * `legacy-passage`: `aw_common.mint_passage_point_id(url, anchor, text)`; `anchor_fn_for_family`
#:     names the anchor form (above).
#:   * `b1-source-coordinate`: `cytir.core.ids.mint_point_id(canonical_url(url), anchor)` -- the id is
#:     the SOURCE COORDINATE (no text in it). Evidence for `fedlex_exact`: the sealed offline golden
#:     (`fixtures/fedlex_offline/`, 63 rows) reproduces 63/63 under it and 0/63 under the
#:     legacy mint (`test_aw_index_id_scheme.py`).
#:   * `legacy-passage-titled`: the legacy mint over the MARKDOWN, where the staged row's `text` is
#:     `"<title>\n\n<markdown>"` (the markdown itself is not in the row). Evidence for `loda_exact`
#:     (`cytir.extract.legi` mints over the markdown, `cytir.sources.families.loda` stages the titled
#:     text): the sealed LODA fixture (3 rows) reproduces 3/3 under it and 0/3 under the plain
#:     legacy check (the id-scheme test). A row whose text was truncated at
#:     staging cannot reproduce and is REFUSED, never re-minted.
ID_SCHEME_LEGACY = "legacy-passage"
ID_SCHEME_B1 = "b1-source-coordinate"
ID_SCHEME_LEGACY_TITLED = "legacy-passage-titled"
#:   * `i1-tier-digest` (task 876): the four regulator PDF replay families
#:     (`cytir.sources.families.pdf_replay`, contract `identity.minted_by` = `mint_point_id`):
#:     `cytir.core.ids.mint_point_id(canonical_url(url), comp)` with `comp` the tier-resolved
#:     component of the chunk's content digest (Tier B: the digest). The staged row has no
#:     `anchor`, so the b1 check above cannot reproduce it. Mirrors `pdf_replay.mint_ids`
#:     (a test pins the two equal on the same rows). Task 1035: `fedlex_exact` too (door rows carry no
#:     `anchor`; tier B; `tests/test_aw_family_fedlex_i1.py` pins ids against the twin).
ID_SCHEME_I1_TIER = "i1-tier-digest"
ID_SCHEMES = (ID_SCHEME_LEGACY, ID_SCHEME_B1, ID_SCHEME_LEGACY_TITLED, ID_SCHEME_I1_TIER)
FAMILY_ID_SCHEME: dict[str, str] = {
    "fedlex_exact": ID_SCHEME_I1_TIER,   # task 1035: the door mints I1 tier B (was b1)
    "loda_exact": ID_SCHEME_LEGACY_TITLED,
    "scratch_weko_docs_exact": ID_SCHEME_I1_TIER,
    "scratch_bav_docs_exact": ID_SCHEME_I1_TIER,
    "scratch_bazl_docs_exact": ID_SCHEME_I1_TIER,
    "scratch_elcom_docs_exact": ID_SCHEME_I1_TIER,
}


def id_scheme_for_family(family: str) -> str:
    return FAMILY_ID_SCHEME.get(family, ID_SCHEME_LEGACY)


#: A family whose ids use a scheme the template above words for another family says so here, so
#: its receipt never cites evidence that is not its own (task 696).
FAMILY_IDENTITY_NOTE: dict[str, str] = {
    f: (f" [{f}, task 696: the rows are legacy `Passage` records (no document_id/chunk_ordinal), so "
        "the 'CC' wording above names the shared raw-anchor scheme, not this family; evidence: "
        "the family's real fixtures and its smoke stage reproduce 100%.]")
    for f in ("dila_juri", "dila_kali", "dila_jorf", "us_federal_register_ecfr", "govuk_content",
              "stackexchange_nonso_qa_p0", "openalex_academic_dump", "service_public_fiche",
              "statcan_wds", "uk_legislation")
}


def anchor_fn_for_family(family: str) -> Callable[[dict], str | None]:
    return FAMILY_ANCHOR_FN.get(family, default_anchor)


#: Manifest prose per anchor scheme, keyed by the function object actually
#: used -- so the receipt states the real derivation for THIS pack's family,
#: never a boilerplate line borrowed from a different family's evidence
#: (task 664, 2026-09-26: the prior text was hardcoded to ERIC/W1's
#: measurement regardless of which family composed).
_IDENTITY_REASON_TEMPLATES: dict[Callable, str] = {
    default_anchor: (
        "task 657: aw_common.mint_passage_point_id(url, "
        "'document:<doc_id>:chunk-<ordinal>', canonical_text|text) "
        "reproduces {matched}/{checked} real rows' declared point_id "
        "(100%) -- the same scheme and anchor form task 629's 2026-09-23 "
        "correction verified for the related W1 packs (B01/B02/B03/B05, "
        "80,117/80,117). cytir.core.ids.mint_point_id (the b1/future scheme) "
        "was tried on the same {checked} rows and reproduced 0 of them, "
        "so it is refused here, not used."
    ),
    raw_anchor_field: (
        "task 664, 2026-09-26: aw_common.mint_passage_point_id(url, "
        "record.get('anchor'), canonical_text|text) reproduces "
        "{matched}/{checked} real rows' declared point_id (100%) -- CC "
        "chunks carry no document_id/chunk_ordinal at all (identity is "
        "per-URL); the row's own raw anchor field is what "
        "pipeline.aw_common.Passage minted under in the first place, per "
        "the task 664 real-rows test fixture."
    ),
}


_DECLARED_IDENTITY_REASON = {
    ID_SCHEME_B1: (
        "task 696 (slice 8 step 2): cytir.core.ids.mint_point_id(canonical_url(url), anchor) -- the b1 "
        "source-coordinate scheme -- reproduces {matched}/{checked} real rows' declared point_id "
        "(100%). The legacy mint aw_common.mint_passage_point_id is NOT used for this family "
        "(the id-scheme test: 0 of the sealed fedlex golden's rows reproduce "
        "under it)."),
    ID_SCHEME_I1_TIER: (
        "task 876: cytir.core.ids.mint_point_id(canonical_url(url), the tier-resolved component of "
        "the chunk's content digest) -- the contract's `identity.minted_by` = mint_point_id, the "
        "rule the PDF replay door runs -- reproduces {matched}/{checked} real rows' declared "
        "point_id (100%)."),
    ID_SCHEME_LEGACY_TITLED: (
        "task 696 (slice 8 step 2): aw_common.mint_passage_point_id(url, anchor, <the text after "
        "the title line>) -- the legacy mint over the markdown, which the staged row carries as "
        "'<title>\\n\\n<markdown>' -- reproduces {matched}/{checked} real rows' declared point_id "
        "(100%); the plain legacy check reproduces 0 of them (the id-scheme test)."),
}


def _identity_reason(anchor_fn: Callable[[dict], str | None],
                     identity: dict) -> str:
    template = _IDENTITY_REASON_TEMPLATES.get(anchor_fn)
    if template is None:
        return (f"custom anchor_fn {getattr(anchor_fn, '__name__', anchor_fn)!r} "
                f"reproduces {identity['matched']}/{identity['checked']} "
                "real rows' declared point_id.")
    return template.format(**identity)


def _mint_b1(rec: dict) -> str | None:
    from cytir.core import ids as aw_ids   # local: the legacy path never needs it
    anchor = rec.get("anchor")
    return (aw_ids.mint_point_id(aw_common.canonical_url(rec["url"]), anchor) if anchor else None)


def _mint_i1_tier(rec: dict) -> str | None:
    """The replay door's `mint_point_id` rule over a staged row (`pdf_replay.mint_ids`)."""
    from cytir.core import ids as aw_ids   # local: the legacy path never needs it
    digest = aw_ids.content_digest(rec["text"])
    _tier, comp = aw_ids.classify_tier(rec.get("source_type"), rec.get("anchor"), True, digest,
                                       title=rec.get("title"))
    return aw_ids.mint_point_id(aw_common.canonical_url(rec["url"]), comp)


def _mint_legacy_titled(rec: dict) -> str | None:
    title_part, sep, markdown = (rec.get("text") or "").partition("\n\n")
    if not sep or title_part != rec.get("title"):
        return None
    return aw_common.mint_passage_point_id(rec["url"], rec.get("anchor"), markdown)


#: scheme -> (the minting function over a staged row, how a refusal names it)
_DECLARED_MINTS: dict[str, tuple[Callable[[dict], str | None], str]] = {
    ID_SCHEME_B1: (_mint_b1, "cytir.core.ids.mint_point_id(canonical_url(url), anchor) (the b1 scheme)"),
    ID_SCHEME_I1_TIER: (
        _mint_i1_tier,
        "cytir.core.ids.mint_point_id(canonical_url(url), tier-resolved digest component) (the I1 "
        "rule the replay door runs)"),
    ID_SCHEME_LEGACY_TITLED: (
        _mint_legacy_titled,
        "aw_common.mint_passage_point_id(url, anchor, <text after the title line>) (the legacy "
        "mint over the markdown)"),
}


def _verify_identity_declared(chunks_dir: str, scheme: str, *, sample: int | None = None) -> dict:
    """`verify_identity` for a family that DECLARES a non-default scheme: every row's declared
    `point_id` must equal what that scheme mints from the row. Same refusal contract (any
    mismatch, an empty stage, a row with no id)."""
    mint, how = _DECLARED_MINTS[scheme]
    checked = matched = 0
    mismatches: list[dict] = []
    for rec in PSP.read_records(chunks_dir):
        declared = rec.get("point_id")
        if not declared:
            raise ComposeError(
                f"row for {rec.get('url')!r} carries no point_id -- "
                "nothing to verify or to join a vector to")
        got = mint(rec)
        checked += 1
        if got == declared:
            matched += 1
        else:
            mismatches.append({"url": rec["url"], "anchor": rec.get("anchor"),
                               "declared": declared, "tried": [[scheme, got]]})
        if sample is not None and checked >= sample:
            break
    if checked == 0:
        raise ComposeError(f"{chunks_dir}: 0 rows -- nothing to verify")
    if mismatches:
        raise ComposeError(
            f"identity check FAILED: {len(mismatches)}/{checked} rows' declared point_id does not "
            f"reproduce under {how} this family declares -- refusing to compose "
            f"(first mismatch: {mismatches[0]!r})")
    return {"checked": checked, "matched": matched, "mismatches": []}


def verify_identity(chunks_dir: str, *,
                    anchor_fn: Callable[[dict], str] = default_anchor,
                    text_fields: Iterable[str] = ("canonical_text", "text"),
                    sample: int | None = None,
                    scheme: str = ID_SCHEME_LEGACY) -> dict:
    """Recompute every (or `sample` many) row's point_id via
    `aw_common.mint_passage_point_id(url, anchor, text)` and compare
    against the row's own declared `point_id`.

    Deliberately does NOT try `cytir.core.ids.mint_point_id` here: task 657
    already measured it directly against 111 real ERIC rows (0/111
    reproduced) and the docstring above records why it is the wrong
    scheme. Re-trying it per call would re-derive an already-falsified
    result (engineering-doctrine #8, "the prior result exists until
    proven absent").

    Returns `{"checked": n, "matched": n, "mismatches": []}` (mismatches
    always empty on return -- see below). Raises `ComposeError` on ANY
    mismatch: a partial match is not evidence the scheme is right, it is
    evidence a DIFFERENT scheme sometimes agrees by construction, and the
    task 657 brief is explicit -- "refuse to compose on any id mismatch."
    """
    if scheme not in ID_SCHEMES:
        raise ComposeError(f"unknown id scheme {scheme!r} (known: {', '.join(ID_SCHEMES)})")
    if scheme != ID_SCHEME_LEGACY:
        return _verify_identity_declared(chunks_dir, scheme, sample=sample)
    checked = 0
    matched = 0
    mismatches: list[dict] = []
    for rec in PSP.read_records(chunks_dir):
        declared = rec.get("point_id")
        if not declared:
            raise ComposeError(
                f"row for {rec.get('url')!r} carries no point_id -- "
                "nothing to verify or to join a vector to")
        url = rec["url"]
        anchor = anchor_fn(rec)
        ok = False
        tried = []
        for field in text_fields:
            text = rec.get(field)
            if text is None:
                continue
            got = aw_common.mint_passage_point_id(url, anchor, text)
            tried.append([field, got])
            if got == declared:
                ok = True
                break
        checked += 1
        if ok:
            matched += 1
        else:
            mismatches.append({"url": url, "anchor": anchor,
                               "declared": declared, "tried": tried})
        if sample is not None and checked >= sample:
            break
    if checked == 0:
        raise ComposeError(f"{chunks_dir}: 0 rows -- nothing to verify "
                           "(the real-path smoke test should have refused "
                           "before this ever ran)")
    if mismatches:
        raise ComposeError(
            f"identity check FAILED: {len(mismatches)}/{checked} rows' "
            "declared point_id does not reproduce under "
            "aw_common.mint_passage_point_id(url, "
            "'document:<doc>:chunk-<n>', canonical_text|text) -- refusing "
            f"to compose (first mismatch: {mismatches[0]!r})")
    return {"checked": checked, "matched": matched, "mismatches": []}


def read_vector_native_dim(vectors_dir: str) -> int:
    """The sealed vector shard's own `dim`, from its `*.vectors.manifest.json`
    files -- refuses if none exist (embed has not run) or if shards
    disagree (a partial/mismatched embed run)."""
    manifests = sorted(Path(vectors_dir).glob("*.vectors.manifest.json"))
    if not manifests:
        raise ComposeError(
            f"no *.vectors.manifest.json under {vectors_dir} -- embed "
            "must run (cytir.embed.stage.embed_stage) before compose")
    dims = {json.loads(p.read_text())["dim"] for p in manifests}
    if len(dims) != 1:
        raise ComposeError(
            f"{vectors_dir}: mixed vector dims {sorted(dims)} across "
            "shards -- a partial or mismatched embed run")
    return dims.pop()


def check_collection_dims(qdrant_url: str, collection: str, mrl: int) -> None:
    """`mrl` must equal the TARGET collection's served dense width.

    Reuses `promote_queue.fetch_collection_vector_config` rather than a
    second `/collections/<name>` read (engineering-doctrine #1). This is
    the exact bug task 641 found live ("the backlog pack's dense vectors
    were 128-dim against aw_e8's 256") -- caught here, at compose time,
    instead of surfacing as a Qdrant 400 mid-apply.
    """
    cfg = PQ.fetch_collection_vector_config(qdrant_url, collection)
    dense = (cfg or {}).get("dense") or {}
    served = dense.get("size")
    if served is None:
        raise ComposeError(
            f"{collection}: no 'dense' vector config found at {qdrant_url} "
            "-- cannot verify width before composing")
    if served != mrl:
        raise ComposeError(
            f"{collection} at {qdrant_url} serves dense width {served}, "
            f"but mrl={mrl} was requested -- refusing (task 641's "
            "128-vs-256 bug, caught here instead of at a live upsert 400)")


def _pack_row_count(pack_dir: Path, manifest: dict) -> int:
    """A queued pack's own row count for the family ceiling: the rows it will WRITE
    (`rows_to_write_max`, falling back to the legacy `rows` for a pack composed before that name
    existed, see `cytir.index.manifest_rows`), else a real count of its `pack_chunks` records (any
    pack enqueued before `rows` existed). Never estimated, never taken from a stale receipt."""
    rows = MR.rows_to_write_max(manifest)
    if rows is not None:
        return rows
    chunks_dir = manifest.get("pack_chunks")
    if not chunks_dir:
        return 0
    return sum(1 for _ in PSP.read_records(chunks_dir))


def family_running_total(queue_dir: str, family: str, *,
                         exclude_pack_id: str | None = None) -> dict:
    """Sum of rows across every pack in `queue_dir` whose manifest declares
    `family` -- REFUSED and PROVED-BLOCKED packs count too (a manifest in
    the queue is a commitment to enqueue that many rows, not only a landed
    one; a pack stuck on an unrelated gate still occupies its share of a
    reserve-draw ceiling). `exclude_pack_id` leaves out the pack currently
    being composed, so re-running compose for the SAME pack_id (a resume,
    or a re-seal after a refusal) does not double-count itself against the
    ceiling it is about to be checked against.

    Queue-state based, not a live Qdrant count: this is what `compose_pack`
    itself has authority over (whether to WRITE a manifest), and it must
    keep working even when twin/prod are unreachable -- the promotion gate
    downstream re-verifies everything against reality anyway."""
    total = 0
    packs: list[dict] = []
    qdir = Path(queue_dir)
    if not qdir.is_dir():
        return {"total_rows": 0, "packs": []}
    for pack_dir in sorted(p for p in qdir.iterdir() if p.is_dir()
                           and not p.name.startswith("_")):
        if pack_dir.name == exclude_pack_id:
            continue
        manifest_path = pack_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("family") != family:
            continue
        rows = _pack_row_count(pack_dir, manifest)
        total += rows
        packs.append({"pack_id": pack_dir.name, "rows": rows})
    return {"total_rows": total, "packs": packs}


def _backfill_lang_chunks(chunks_dir: str, vectors_dir: str, mrl: int) -> tuple[str, dict]:
    """Task 669/C: physically rewrite `chunks_dir` with an empty `lang`
    field filled from the tracked detector, via `normalize_sealed_pack_
    shape.merge(..., backfill_lang=True)` -- task 652's ALREADY-BUILT
    mechanism, reused rather than re-derived (engineering-doctrine: one
    truth, one home). A row's own DECLARED `lang` is never touched; a row
    the detector cannot confidently place in the product's language set is
    left EMPTY (dropped later, by `language_scope_filter`/the payload-
    carry gate, never by this function).

    VERIFIED before this was wired in (task 669): `lang` is NOT part of
    either the dense or sparse embedding surface for the
    `stage_structured_markdown_for_embedding.py` producer path -- `unit.
    dense_text`/`unit.sparse_text`/`canonical` (the text that is actually
    embedded, `generic_chunking.chunk_generic_records`) never reference
    `lang` anywhere. This is a payload-only rewrite: no re-embed, no
    change to any vector already sealed on disk, and `verify_identity`
    still reproduces every id unchanged (identity is minted from url +
    anchor + text, none of which this touches).

    Writes to a SIBLING directory (`<chunks_dir>-lang-backfilled`),
    leaving the original materialized chunks on disk untouched -- the same
    discipline `normalize_sealed_pack_shape.py` already uses for CC
    tranche 1 (a NEW `--out-dir`, never an in-place rewrite of raw
    materialization output)."""
    out_dir = Path(chunks_dir).parent / f"{Path(chunks_dir).name}-lang-backfilled"
    rows, report = NSP.merge(metadata_from=[chunks_dir], text_from=None,
                             vectors_dir=vectors_dir, mrl=mrl, backfill_lang=True)
    lb = report.get("lang_backfill") or {}
    blind = lb.get("left_gate_unavailable", 0)
    if blind:
        # An unavailable instrument is not an empty (verification-doctrine 6):
        # a detector that could not run leaves every row's `lang` empty, and
        # `--drop-undetected-lang` then drops ALL of them at the drain, which
        # sees "zero pages to check" hours later (ERIC 030000/040000, 2026-09-30:
        # 228,890 and 208,417 rows, `left_gate_unavailable` == rows). Refuse here,
        # before any manifest or READY exists.
        raise ComposeError(
            f"--backfill-lang: the language gate was unavailable for {blind:,} of "
            f"{lb.get('rows', 0):,} rows, so no `lang` could be detected. Run the "
            "compose step under the lang-gate interpreter (numpy<2 + fasttext, "
            "deploy/langgate/requirements.txt; AW_LANGGATE_PYTHON_BIN in "
            "the materialization current.env.example) with the "
            "release's import path. Nothing was written to the queue.")
    NSP.write_shard(rows, out_dir)
    return str(out_dir), report


def compose_pack(*, pack_id: str, family: str, chunks_dir: str,
                 vectors_dir: str, collection: str, mrl: int,
                 family_payload: dict | None = None,
                 family_payload_basis: str | None = None,
                 title_from: Iterable[str] = (),
                 license_from: str | None = None,
                 family_license: str | None = None,
                 language_scope_filter: bool = False,
                 backfill_lang: bool = False,
                 drop_undetected_lang: bool = False,
                 drop_undetected_lang_basis: str | None = None,
                 identity_reason_suffix: str = "",
                 qdrant_url: str | None = None,
                 anchor_fn: Callable[[dict], str] | None = None,
                 queue_dir: str | None = None,
                 identity_sample: int | None = None,
                 ceiling: int | None = None,
                 exclude_ids_file: str | None = None,
                 exclude_ids_reason: str | None = None,
                 served_targets: dict[str, tuple[str, str]] | None = None,
                 artifact_root: str | None = None,
                 consent_ctx=None,
                 supersede_url_group: str | None = None,
                 supersede: str | None = None,
                 allow_supersede: str | None = None,
                 copy_sealed_guard: bool = False,
                 line_coverage_old_texts: Iterable[str] = (),
                 line_coverage_wanted_ids: Iterable[str] = (),
                 ruled_skip: str | None = None) -> dict:
    """Verify identity, check widths, write `manifest.json` + `READY`.

    URL-GROUP SUPERSEDE (ruling 27 T6, ADR-114 am.2; `check_supersede_options`): `supersede=
    "url-group"`, `allow_supersede=<reason>` and `copy_sealed_guard=True` (all three, loda_exact
    only) write `supersede`, `allow_supersede` and the SEALED pack's own `PACK.json` guard block
    (`staged_text_guard`, plus `staged_text_guard_chunks_sha256`: the drain refuses a stage whose
    `PACK.json` names another chunks sha) into
    the manifest; the block is read and verified HERE (`url_group_supersede.sealed_guard_block`
    and `text_guard` against the URL groups of the chunk rows), so a pack the drain would refuse
    is not sealed. `line_coverage_old_texts` (stage dirs / shard files that hold the OLD rows'
    texts: for loda the task-417 and task-633 stages) and `line_coverage_wanted_ids` (files that
    name the served ids and urls: the task-705 export, the task-633 stage) make
    `cytir.index.line_coverage` write `line-coverage.v1.json` beside the manifest, bound by its
    sha256 (`line_coverage_file` / `line_coverage_sha256` / `line_coverage`). Without them no block
    is written and the drain keeps the snippet guard for every URL.

    `ruled_skip` (plan.md T24, `--ruled-skip FILE`): a JSON list of `{url, point_id,
    measured_coverage, ruling}` (`cytir.index.line_coverage.parse_ruled_skip`). Each entry is checked
    HERE against the block just measured (same url, coverage within 1e-6, below the floor); the
    file is copied beside the manifest as `ruled-skip.v1.json` and bound by `ruled_skip_file` /
    `ruled_skip_sha256` / `ruled_skip` (entries verbatim). The drain re-checks it.

    `supersede_url_group` (task 876): the reason (free text, non-empty) of an ADR-067 am.1 point 4
    URL-group supersede. The manifest then carries `supersede: "url-group"` and
    `allow_supersede: <reason>`, which the drain reads (`cytir.index.url_group_supersede`): the served
    rows of the pack's URL groups that the pack does not carry are replaced, export first. The
    drain itself refuses a family that is not on `URL_GROUP_FAMILIES` and a pack whose sealed stage
    holds no guard block. `None` (default): the manifest is exactly as before.

    CONSENT (task 684, ruling 20): every pack is stamped here. The chunk rows are re-written
    to a sibling `<chunks_dir>-consent-stamped` directory with `search_ok`, `train_ok`,
    `consent_checked_at` and `consent_basis` derived by `cytir.consent.derive` (the ADR-115 engine,
    the family being `family`), and `pack_chunks` points at it. A row the engine cannot
    stamp REFUSES the pack (`cytir.consent.stamp.ConsentStampRefused`); nothing is defaulted.
    `consent_ctx` is a `DeriveContext` (tests, a pinned `as_of`); absent it, the contracts
    file and the resolver's consent cache are read.

    `backfill_lang` (task 669/C, ERIC): before anything else, physically
    rewrites `chunks_dir` (to a sibling `-lang-backfilled` directory --
    see `_backfill_lang_chunks`'s docstring for why this is payload-only,
    no re-embed) and uses THAT as `chunks_dir` for identity verification
    and the manifest's own `pack_chunks` -- everything downstream
    (`promote_queue.py`'s drain, `language_scope_filter`) reads the
    backfilled rows exactly as it would any other pack, no separate
    consumption path.

    `drop_undetected_lang` (task 669/C): writes `manifest["lang_presence_
    waiver"]` -- the SAME opt-in mechanism CC tranche 1 used (task 652),
    consumed unchanged by `promote_queue.py`'s `language_scope_and_waiver_
    exclusions()` at drain time (`mechanical_content_check.lang_drop_
    detail` over the pack's rows, excluding every id still empty after
    `backfill_lang`). This is not a NEW waiver decision made here -- it
    mechanically implements Tim's own already-ratified, standing ruling
    for the whole `eric_official_pdf` family ("drop rows whose lang
    cannot be detected, as CC-t1", plan.md rule 12, ADR-113), never a
    per-pack judgment call. `drop_undetected_lang_basis` is a free-text
    citation of that ruling, written into the waiver's `rule` field for
    the record -- the same provenance discipline `family_payload_basis`
    already uses.

    Never touches Qdrant except a read-only collection-config GET (when
    `qdrant_url` is given) to check `mrl` against the served width --
    exactly the read-only discipline `cytir/index/promote_queue.py`'s own
    `dry_plan()` uses. This function writes files under `queue_dir` only;
    draining (twin, then prod) is `cytir/index/promote_queue.py`'s job, run
    separately (nightly timer, or `--only <pack_id>` for a dry check).

    `ceiling` (task 663, 2026-09-25, ADR-113): when given, refuses to WRITE
    this pack's manifest/READY at all if `family`'s running total already
    enqueued in `queue_dir` (via `family_running_total`, excluding this
    same `pack_id`) plus this pack's own row count would exceed it. This is
    the ADR-113 <=4.5M reserve-draw ceiling for `eric_official_pdf` --
    enforced at the one place new rows for the family enter the queue, not
    as a separate audit run after the fact. `None` (default) enforces no
    ceiling, for every family ADR-113 does not name.

    `license_from`/`family_license` (task 664, 2026-09-26): passed straight
    through into the manifest for `promote_queue.py`'s `plan_kwargs()` to
    read -- this door had NEITHER before, so any family whose rows carry no
    resolvable per-row `license` and no host-table entry (ADR-103
    amendment 2: Stack Exchange's licence is per-record, at
    `source_metadata.accepted_answer.license`, not host-uniform) refused
    forever with "unresolved license" the moment it was composed through
    here. `family_payload` cannot stand in for this: it fills
    `source_type`/`trust_tier`/`domain`, never `license`, which
    `aw_b1_projection.project_record` resolves before `family_payload` is
    even consulted.

    `family_payload_basis` (task 664, 2026-09-26): a free-text citation for
    WHERE `family_payload`'s values came from -- never fed to
    `project_record`, `apply_family_payload`, or any row's own payload,
    purely a manifest-level provenance note so a promotion receipt states
    its derivation instead of a bare number nobody can trace back (the
    coordinator's ruling on this exact question, task 664: derive
    source_type/trust_tier from a ratified rule or a served precedent,
    never invent one, and RECORD the derivation in the manifest).

    `exclude_ids_file` / `exclude_ids_reason` (task 680): a `.jsonl[.gz]` of
    `{"point_id": ...}` the admission floor does not admit (the OpenAlex ADR-102
    ladder trim is the first). Loaded here once to fail early, then written to
    the manifest with its sha256 so the drain excludes exactly those ids on the
    dry plan and the write -- see `cytir/index/exclude_file.py`.

    `served_targets` (`{"twin": (url, collection), "prod": (url, collection)}`, ruling 23.5, task
    702): the pack's ids that BOTH targets already hold are written to `already-served-ids.jsonl`
    in the pack directory and declared in the manifest as `exclude_ids_*` (the drain's existing
    path) plus `already_served_excluded` / `_file` / `_sha256`. A pack whose ids are all served
    is recorded in `<queue>/_composed-to-nothing/` and never enters the queue (the result has
    `all_served: True`). A half-applied pack is never shrunk -- see `cytir/index/served_exclusion.py`. `artifact_root`: where the drain
    keeps its twin-write-state artifacts (default `<queue_dir>/_artifacts`, the drain's own).
    """
    queue_dir = queue_dir or default_queue_dir()
    # a generator would yield nothing the second time: read each option ONCE
    old_texts = [str(x) for x in (line_coverage_old_texts or ())]
    wanted_files = [str(x) for x in (line_coverage_wanted_ids or ())]
    url_group = check_supersede_options(
        family, supersede=supersede, allow_supersede=allow_supersede,
        copy_sealed_guard=copy_sealed_guard, line_coverage_old_texts=old_texts,
        line_coverage_wanted_ids=wanted_files, served_targets=served_targets, ruled_skip=ruled_skip,
        supersede_url_group=supersede_url_group, exclude_ids_file=exclude_ids_file)
    sealed_guard = sealed_chunks_sha = None
    if url_group:
        try:
            sealed_guard = UGS.sealed_guard_block({"pack_chunks": chunks_dir})
            groups = {LC.url_key(r["url"]) for r in PSP.read_records(chunks_dir)}
            UGS.text_guard(sealed_guard, url_groups=len(groups))
            sealed_chunks_sha = UGS.sealed_chunks_sha256({"pack_chunks": chunks_dir})
        except PQ.QueueError as exc:
            raise ComposeError(f"{pack_id}: refusing to seal a url-group pack the drain would "
                               f"refuse: {exc}") from exc
    # A caller may pass anchor_fn explicitly (tests, a deliberate override);
    # absent that, resolve it from `family` (task 664, 2026-09-26 -- see
    # anchor_fn_for_family's docstring for the KeyError this closes).
    anchor_fn = anchor_fn or anchor_fn_for_family(family)
    original_chunks_dir = None
    lang_backfill_report = None
    if backfill_lang:
        original_chunks_dir = chunks_dir
        chunks_dir, backfill_report = _backfill_lang_chunks(
            chunks_dir, vectors_dir, mrl)
        lang_backfill_report = backfill_report.get("lang_backfill")
    if consent_ctx is None:
        consent_ctx = CONSENT_STAMP.load_context()
    stamped_dir = f"{chunks_dir.rstrip('/')}-consent-stamped"
    consent_report = CONSENT_STAMP.stamp_chunks_dir(
        chunks_dir, stamped_dir, consent_ctx, family=family,
        license_from=license_from, family_license=family_license)
    consent_stamped_from, chunks_dir = chunks_dir, stamped_dir
    scheme = id_scheme_for_family(family)
    identity = verify_identity(chunks_dir, anchor_fn=anchor_fn,
                               sample=identity_sample, scheme=scheme)
    native_dim = read_vector_native_dim(vectors_dir)
    if native_dim != PSP.SEALED_DIM:
        raise ComposeError(
            f"{vectors_dir}: vectors are {native_dim}-dim, but "
            f"promote_sealed_pack.py reads the sealed {PSP.SEALED_DIM}-dim "
            "artefact only")
    if qdrant_url:
        check_collection_dims(qdrant_url, collection, mrl)

    this_pack_rows = identity["checked"]
    running = {"total_rows": 0, "packs": []}
    if ceiling is not None:
        running = family_running_total(queue_dir, family,
                                       exclude_pack_id=pack_id)
        projected = running["total_rows"] + this_pack_rows
        if projected > ceiling:
            raise ComposeError(
                f"{pack_id}: refusing to enqueue -- family {family!r} "
                f"already has {running['total_rows']:,} row(s) queued "
                f"across {len(running['packs'])} pack(s), this pack adds "
                f"{this_pack_rows:,} more, projected total "
                f"{projected:,} exceeds the ceiling of {ceiling:,} "
                "(ADR-113 reserve-draw cap). Not written -- no manifest, "
                "no READY.")

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "pack_id": pack_id,
        "family": family,
        "pack_chunks": chunks_dir,
        "vectors": vectors_dir,
        "mrl": mrl,
        "collection": collection,
        "rows": this_pack_rows,         # legacy alias, see cytir.index.manifest_rows
        "rows_staged": this_pack_rows,
        "identity_check": "pack-declared",
        "identity_reason": (
            (_DECLARED_IDENTITY_REASON[scheme].format(**identity) if scheme != ID_SCHEME_LEGACY
             else _identity_reason(anchor_fn, identity) + FAMILY_IDENTITY_NOTE.get(family, ""))
            + identity_reason_suffix
        ),
    }
    if supersede_url_group is not None:
        if not str(supersede_url_group).strip():
            raise ComposeError("--supersede-url-group needs a reason: the drain refuses a "
                               "`supersede` pack without `allow_supersede`")
        manifest["supersede"] = "url-group"
        manifest["allow_supersede"] = str(supersede_url_group).strip()
    manifest["consent_stamp"] = {
        "from": consent_stamped_from,
        **{k: v for k, v in consent_report.items() if k != "stamped_from"}}
    if family_payload is not None:
        manifest["family_payload"] = family_payload
    if family_payload_basis:
        manifest["family_payload_basis"] = family_payload_basis
    if title_from:
        manifest["title_from"] = list(title_from)
    if license_from:
        manifest["license_from"] = license_from
    if family_license:
        manifest["family_license"] = family_license
    if exclude_ids_file:
        if not exclude_ids_reason:
            raise ComposeError(
                "--exclude-ids-file needs --exclude-ids-reason: every row the "
                "drain drops is reported with one (ADR-086)")
        exclude_sha = XF.sha256_file(exclude_ids_file)
        exclude_map = XF.load_exclude_ids(exclude_ids_file, sha256=exclude_sha,
                                          reason=exclude_ids_reason)  # refuses an empty file
        manifest["exclude_ids_file"] = exclude_ids_file
        manifest["exclude_ids_sha256"] = exclude_sha
        manifest["exclude_ids_reason"] = exclude_ids_reason
    if language_scope_filter:
        # Coordinator/Tim ruling, task 664, 2026-09-28 (ERIC): a row whose
        # language cannot be detected confidently is DROPPED and RECORDED,
        # never silently admitted -- the same opt-in manifest field CC
        # tranche 1 used (orchestrator ruling, 2026-09-23). Consumed by
        # promote_queue.py's plan_kwargs_with_scope() at drain time
        # (mechanical_content_check.language_scope_filter over the pack's
        # RAW rows, excluded via exclude_ids before payload-carry scoring
        # ever sees them) -- this flag is the ONLY thing compose needs to
        # write; the drop mechanism itself is not re-derived here.
        manifest["language_scope_filter"] = True
    if backfill_lang:
        # task 669/C: a manifest-level accountability record of the
        # rewrite `_backfill_lang_chunks` already did BEFORE identity
        # verification ran -- `pack_chunks` above already points at the
        # backfilled directory; this is provenance (what it was rewritten
        # FROM, and what the detector actually did), never consumed by
        # any downstream gate.
        manifest["lang_backfilled_from"] = original_chunks_dir
        manifest["lang_backfill_report"] = lang_backfill_report
    if drop_undetected_lang:
        manifest["lang_presence_waiver"] = {
            "issuer": "Tim",
            "at": "2026-09-28",
            "scope": family,
            "rule": drop_undetected_lang_basis or (
                "lang presence floor 99% (plan.md rule 12 / ADR-113: drop "
                "rows whose lang cannot be detected, as CC-t1, task 652)"),
        }
    if ceiling is not None:
        manifest["family_ceiling"] = {
            "ceiling": ceiling,
            "prior_family_total": running["total_rows"],
            "this_pack_rows": this_pack_rows,
            "projected_family_total": running["total_rows"] + this_pack_rows,
            "prior_packs_counted": [p["pack_id"] for p in running["packs"]],
        }

    if url_group:
        manifest["supersede"] = UGS.URL_GROUP
        manifest["allow_supersede"] = allow_supersede.strip()
        manifest["staged_text_guard"] = sealed_guard
        manifest["staged_text_guard_chunks_sha256"] = sealed_chunks_sha

    # UPPER BOUND on what the drain writes (never compared for equality with a written count; the
    # drain also drops language-scope, waiver, undated and projection rows): the stage less the caller's own exclusion ids that are in the pack.
    # (The served check below replaces this with its own count, which includes those ids.)
    to_write = this_pack_rows
    if exclude_ids_file:
        to_write -= sum(1 for r in PSP.read_records(chunks_dir) if str(r["point_id"]) in exclude_map)
    pack_dir = Path(queue_dir) / pack_id
    if served_targets:
        art = Path(artifact_root or (Path(queue_dir) / "_artifacts"))
        explicit = ((exclude_ids_file, exclude_ids_reason) if exclude_ids_file else None)
        twin_url, twin_coll = served_targets["twin"]
        prod_url, prod_coll = served_targets["prod"]
        # task 2218: ask the twin and prod for the ids the DRAIN will write (the target's id_rule),
        # not the pack's declared ids; hits come back keyed by pack id, as the exclusion file is
        # task 2220: the rows the projection refuses go in the same file by their declared id
        mint_map, rejected = _minted_and_rejected(
            chunks_dir, [str(r["point_id"]) for r in PSP.read_records(chunks_dir)],
            PQ.plan_kwargs(manifest), IR.declared_id_rule(twin_coll))
        decl = SE.declare_already_served(
            pack_dir=pack_dir, pack_id=pack_id,
            ids=[r["point_id"] for r in PSP.read_records(chunks_dir)],
            targets=served_targets,
            # the twin first, with the payload fields; prod only about the twin's hits, at the
            # drain's own prod width (one source of truth, task 689) and through the same `_api`
            # (the drain's timeouts and retries)
            twin_rows=_ask_by_minted(lambda i: RD.reader(twin_coll).decode_map(PSP.retrieve_payloads(
                twin_url, twin_coll, i, SE.PAYLOAD_COMPARE_FIELDS,
                workers=PQ.LOOKUP_WORKERS["twin"])), mint_map),    # canonical rows, any profile (task 855)
            prod_present=_ask_by_minted(lambda i: PSP.already_present(
                prod_url, prod_coll, i, workers=PQ.LOOKUP_WORKERS["prod"]), mint_map),
            pack_payloads=lambda i: _pack_payloads(chunks_dir, i, PQ.plan_kwargs(manifest)),
            resume_artifact_exists=(PQ.twin_written_ids_path(art, pack_id).exists()
                                    or PQ.content_pack_twin_state_path(art, pack_id).exists()),
            explicit_exclude=explicit, projection_rejected=rejected)
        if rejected:
            print(f"[compose] {pack_id}: {len(rejected):,} row(s) the projection refuses excluded "
                  f"by declared id: {decl['fields'].get('projection_rejected', {})}", file=sys.stderr)
        if decl["all_served"]:
            SE.write_composed_to_nothing(
                queue_dir, pack_id=pack_id, rows=this_pack_rows, fields=decl["fields"],
                excluded_ids=decl["excluded_ids"], payload_differs=decl["payload_differs"],
                family=family, pack_chunks=chunks_dir, vectors=vectors_dir,
                explicit_exclude_file=exclude_ids_file, projection_rejected=rejected)
            return {"pack_dir": None, "manifest": None, "identity": identity,
                    "ready_sha": None, "all_served": True, "served": decl["fields"]}
        manifest.update(decl["fields"])
        dropped = len(set(decl["excluded_ids"]) | set(decl.get("explicit_in_pack", ())))
        if "rows_to_write_max" in decl["fields"]:           # a re-compose carries what it declared
            to_write = decl["fields"]["rows_to_write_max"]
        elif decl.get("carried_over"):                  # declared before the new names: as it was
            to_write = decl["fields"].get("rows", to_write)
        elif dropped:
            to_write = this_pack_rows - dropped
        if dropped:
            # the legacy alias `rows` keeps its old value here: the rows the drain will WRITE;
            # the stage's full count stays on the record, the excluded count is
            # `already_served_excluded` and the caller's own exclusion file
            manifest["rows_before_exclusion"] = this_pack_rows
            manifest["rows"] = this_pack_rows - dropped
    # Task 710, guard 1: before the pack is sealed, its German rows must carry the parts the rule
    # predicts, in THIS interpreter. `warn` (default) records the count and prints a loud line;
    # `refuse` raises here, so no manifest and no READY exist. The drain checks again, in its own.
    manifest["german_parts"], gp_reason = _german_parts_report(chunks_dir, manifest, pack_id)
    if gp_reason:
        raise ComposeError(f"{pack_id}: refusing to enqueue -- {gp_reason}. Not written -- no "
                           "manifest, no READY.")
    # Both numbers, always, under their own names (`cytir.index.manifest_rows`). The alias `rows` is
    # NOT unified: a plain `--exclude-ids-file` compose (and `reseal`) keeps `rows` at the stage
    # count, as task 680 pinned ("the pack stays whole"), so a tool that reads `rows` sees no change.
    manifest["rows_staged"] = this_pack_rows
    manifest["rows_to_write_max"] = to_write
    pack_dir.mkdir(parents=True, exist_ok=True)
    if old_texts:
        wanted = LC.read_wanted(wanted_files) if wanted_files else None
        try:
            block = LC.build_block(
                new_rows=PSP.read_records(chunks_dir), new_chunks_sha256=sealed_chunks_sha,
                old_sources=old_texts, wanted=wanted)
        except LC.LineCoverageError as exc:
            raise ComposeError(f"{pack_id}: line coverage block not written: {exc}") from exc
        if not block.get("entries"):
            raise ComposeError(f"{pack_id}: old texts were given but the line coverage block is "
                               "empty; refusing to write a pack that silently measures nothing")
        manifest.update(write_line_coverage_block(block, pack_dir))
        if ruled_skip:
            data = Path(ruled_skip).read_bytes()
            try:
                ruled = LC.parse_ruled_skip(data)
                LC.check_ruled_skip(ruled, block["entries"])
            except LC.LineCoverageError as exc:
                raise ComposeError(f"{pack_id}: ruled skip refused: {exc}") from exc
            (pack_dir / LC.RULED_SKIP_FILE_NAME).write_bytes(data)
            manifest.update(LC.ruled_skip_fields(data, pack_dir / LC.RULED_SKIP_FILE_NAME, ruled))
    manifest_path = pack_dir / "manifest.json"
    manifest_bytes = json.dumps(manifest, indent=1, sort_keys=True).encode()
    manifest_path.write_bytes(manifest_bytes)
    ready_sha = hashlib.sha256(manifest_bytes).hexdigest()
    (pack_dir / "READY").write_text(ready_sha + "\n")
    return {"pack_dir": str(pack_dir), "manifest": manifest,
           "identity": identity, "ready_sha": ready_sha, "all_served": False}


_PROJECT_ROWS_KWARGS = frozenset((
    "consent", "attribution", "license_version", "identity_check", "identity_reason",
    "published_from", "published_status_from", "published_status_requires", "require_published",
    "family_payload", "title_from", "license_from", "family_license"))


def _german_parts_report(chunks_dir: str, manifest: dict, pack_id: str) -> tuple[dict, str | None]:
    """The stage-side guard: a seeded sample of the pack's German rows, projected exactly as the
    drain will project them (`_pack_payloads`' own way), judged by `cytir.index.german_parts_guard`.

    For a SEALED pack this measures the COMPOSE interpreter (the project venv, which has a
    splitter), not the one the drain writes with: it read tranche 12 as PASS although the drain
    wrote it without parts. The block says who measured it (`measured_by`); the DRAIN-side check
    (`promote_queue.drain`, `GPG.check_joined`) is the authority for sealed packs."""
    use = {"consent": None, "attribution": None, "license_version": None,
           **{k: v for k, v in PQ.plan_kwargs(manifest).items() if k in _PROJECT_ROWS_KWARGS}}
    sample, counts = GPG.sample_german(PSP.read_records(chunks_dir), lambda r: r.get("lang"))
    texts, unprojectable = [], 0
    for rec in sample:
        try:
            rows = PSP.project_rows(iter([rec]), **use)[0]
        except Exception:  # noqa: BLE001 -- unprojectable here = unchecked, the drain decides
            unprojectable += 1
            continue
        texts.extend(sparse for _dense, sparse, _payload, _rec in rows.values())
    ext = GPG.external_predictor()
    report = GPG.evaluate_texts(texts, counts, predictor=ext)
    if report["german_rows"]:       # a pack with no German row keeps the box-independent block
        report["measured_by"] = {"interpreter": sys.executable, "predictor_found": ext is not None,
                                 "authority": "advisory: the drain-side check in the drain's own "
                                              "interpreter is the authority for a sealed pack"}
    if unprojectable:
        report["unprojectable_sampled"] = unprojectable
    reason, lines = GPG.decide(report, pack_id)
    for line in lines:
        print(line)
    return report, reason


def _pack_payloads(chunks_dir: str, ids: list[str], kwargs: dict) -> dict[str, dict]:
    """The payload the drain would write for each of `ids`: the pack's own record run through the
    same projection (`PSP.project_rows`, the manifest's own kwargs). A row the projection refuses
    is left out, and the caller counts it as unchecked, never as equal."""
    want, out = set(ids), {}
    use = {"consent": None, "attribution": None, "license_version": None,
           **{k: v for k, v in kwargs.items() if k in _PROJECT_ROWS_KWARGS}}
    for rec in PSP.read_records(chunks_dir):
        pid = str(rec.get("point_id"))
        if pid not in want:
            continue
        try:
            rows = PSP.project_rows(iter([rec]), **use)[0]
        except Exception:  # noqa: BLE001 -- unprojectable here = unchecked, not a compose failure
            continue
        for _dense, _sparse, payload, _rec in rows.values():
            out[pid] = payload
    return out


def _minted_and_rejected(chunks_dir: str, ids: list[str], kwargs: dict,
                         id_rule: str | None) -> tuple[dict[str, str], dict[str, str]]:
    """`(pack id -> the id the drain WRITES, declared id -> reason)` in ONE pass (tasks 2218, 2220).

    The pack's record is run through `PSP.project_rows` with the target's `id_rule` (I1 on
    aw_e10), never a second minting. A row merged into an earlier row under one I1 id maps to the
    survivor's id. A row the projection refuses (`project_record` returned None: `skipped` counts
    exactly those; undated and out-of-scope rows are counted elsewhere) is absent from the map and
    present in the second dict: its vector would reach the drain's join with no row, so the
    composer excludes it by its DECLARED id. Reason: `projection-rejected: search-result-url` for a
    site-search URL (task 2009), else `projection-rejected: other`."""
    want, out, rejected, accepted = set(ids), {}, {}, set()
    use = {"consent": None, "attribution": None, "license_version": None,
           **{k: v for k, v in kwargs.items() if k in _PROJECT_ROWS_KWARGS}}
    for rec in PSP.read_records(chunks_dir):
        pid = str(rec.get("point_id"))
        if pid not in want:
            continue
        try:
            rows, skipped, _modes, vector_key, _undated = PSP.project_rows(
                iter([rec]), **use, id_rule=id_rule)
        except Exception:  # noqa: BLE001 -- unprojectable here = asked by its pack id, the drain decides
            continue
        for minted in rows:
            out[str(vector_key.get(minted, pid))] = minted
        if skipped and not rows:
            rejected[pid] = rejected_reason(rec.get("url"))
        elif rows:
            accepted.add(pid)
    shared = sorted(set(rejected) & accepted)
    if shared:
        # task 2222: the exclusion is by declared id, so it would drop the accepted row too
        raise ComposeError(
            f"refusing to compose: {len(shared)} declared point_id(s) are shared by a record the "
            f"projection accepts and a record it rejects (the declared-id exclusion would silently "
            f"drop the accepted row too); first: {shared[0]}. Not written -- no manifest, no READY.")
    return out, rejected


def _minted_id_map(chunks_dir: str, ids: list[str], kwargs: dict, id_rule: str | None) -> dict[str, str]:
    """pack id -> the id the drain WRITES for that row (task 2218); see `_minted_and_rejected`.
    A row the projection refuses is absent, and the caller then asks about its pack id as before."""
    return _minted_and_rejected(chunks_dir, ids, kwargs, id_rule)[0]


def _ask_by_minted(fn, mint_map: dict[str, str]):
    """Wrap an id-set lookup `fn(ids) -> keys-or-mapping` so it is asked the MINTED ids and
    answers in PACK ids (`served_exclusion` and the exclusion file speak pack ids)."""
    back: dict[str, list[str]] = {}
    for pid, minted in mint_map.items():
        back.setdefault(minted, []).append(pid)

    def asked(ids):
        ids = list(ids)
        asked_ids = sorted({mint_map.get(i, i) for i in ids})
        res = fn(asked_ids)
        hit = {m: (res[m] if isinstance(res, Mapping) else None) for m in res}
        out = {}
        for m, v in hit.items():
            for pid in back.get(m, [m]):
                if pid in set(ids):
                    out[pid] = v
        return out if isinstance(res, Mapping) else list(out)
    return asked


def embed_compose_enqueue(*, pack_id: str, family: str, chunks_dir: str,
                          vectors_dir: str, collection: str, mrl: int,
                          qdrant_url: str | None = None,
                          family_payload: dict | None = None,
                          family_payload_basis: str | None = None,
                          title_from: Iterable[str] = (),
                          license_from: str | None = None,
                          family_license: str | None = None,
                          language_scope_filter: bool = False,
                          backfill_lang: bool = False,
                          drop_undetected_lang: bool = False,
                          drop_undetected_lang_basis: str | None = None,
                          queue_dir: str | None = None,
                          skip_embed: bool = False,
                          ceiling: int | None = None,
                          exclude_ids_file: str | None = None,
                          exclude_ids_reason: str | None = None,
                          served_targets: dict[str, tuple[str, str]] | None = None,
                          artifact_root: str | None = None,
                          consent_ctx=None,
                          supersede_url_group: str | None = None,
                          supersede: str | None = None,
                          allow_supersede: str | None = None,
                          copy_sealed_guard: bool = False,
                          line_coverage_old_texts: Iterable[str] = (),
                          line_coverage_wanted_ids: Iterable[str] = (),
                          ruled_skip: str | None = None,
                          log=lambda msg: print(msg, flush=True)) -> dict:
    """The full hook: embed (GPU-locked) -> verify identity -> compose ->
    enqueue, for one sealed chunk stage. This is what the 656 chain calls
    once a tranche clears `chain_and_smoke_test.sh`.

    `ceiling`: see `compose_pack`. Checked (and may refuse) AFTER embedding
    -- the ceiling gates what gets ENQUEUED, not what gets embedded; a
    tranche this refuses still has its vectors sealed and on disk, so
    re-running compose later (once the ceiling is raised by a fresh ADR, or
    a differently-sized tranche frees headroom) never re-embeds it.
    """
    queue_dir = queue_dir or default_queue_dir()
    line_coverage_old_texts = [str(x) for x in (line_coverage_old_texts or ())]   # read ONCE (a
    line_coverage_wanted_ids = [str(x) for x in (line_coverage_wanted_ids or ())]  # generator is spent)
    check_supersede_options(      # ruling 27 T6: a refused option set never takes the GPU
        family, supersede=supersede, allow_supersede=allow_supersede,
        copy_sealed_guard=copy_sealed_guard, line_coverage_old_texts=line_coverage_old_texts,
        line_coverage_wanted_ids=line_coverage_wanted_ids, served_targets=served_targets,
        supersede_url_group=supersede_url_group, exclude_ids_file=exclude_ids_file,
        ruled_skip=ruled_skip)
    # task 684: the consent stamp is a pure function of the rows, so a pack it will refuse
    # is refused HERE, before the GPU is taken (compose_pack stamps again and writes).
    if consent_ctx is None:
        consent_ctx = CONSENT_STAMP.load_context()
    pre = CONSENT_STAMP.stamp_chunks_dir(
        chunks_dir, "", consent_ctx, family=family, license_from=license_from,
        family_license=family_license, dry=True)
    log(f"[compose] consent pre-check: {pre['rows_in']:,} rows stampable {pre['rules']}")
    if not skip_embed:
        log(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
            f"[compose] embedding {chunks_dir} -> {vectors_dir}")
        rc = embed_stage(chunks_dir, vectors_dir, log=log)
        if rc != 0:
            raise ComposeError(f"embed_stage failed (rc={rc}) for "
                               f"{chunks_dir}")
    log(f"[compose] composing pack {pack_id}")
    result = compose_pack(
        pack_id=pack_id, family=family, chunks_dir=chunks_dir,
        vectors_dir=vectors_dir, collection=collection, mrl=mrl,
        qdrant_url=qdrant_url, family_payload=family_payload,
        family_payload_basis=family_payload_basis,
        title_from=title_from, license_from=license_from,
        family_license=family_license,
        language_scope_filter=language_scope_filter,
        backfill_lang=backfill_lang,
        drop_undetected_lang=drop_undetected_lang,
        drop_undetected_lang_basis=drop_undetected_lang_basis,
        queue_dir=queue_dir, ceiling=ceiling,
        exclude_ids_file=exclude_ids_file,
        exclude_ids_reason=exclude_ids_reason, served_targets=served_targets,
        artifact_root=artifact_root, consent_ctx=consent_ctx,
        supersede_url_group=supersede_url_group, supersede=supersede,
        allow_supersede=allow_supersede, copy_sealed_guard=copy_sealed_guard,
        line_coverage_old_texts=line_coverage_old_texts,
        line_coverage_wanted_ids=line_coverage_wanted_ids, ruled_skip=ruled_skip)
    if result.get("all_served"):
        log(f"[compose] {pack_id}: every id is already served by the twin and prod -- composes "
            f"to nothing (recorded in {SE.settled_path(queue_dir, pack_id)}); not an error")
        return result
    if result["manifest"].get("already_served_excluded"):
        log(f"[compose] {pack_id}: {result['manifest']['already_served_excluded']:,} ids already "
            "served by the twin and prod, excluded; "
            f"{result['manifest'].get('already_served_payload_differs', 0):,} of them hold a "
            "different payload on the twin (left as they are, listed)")
    log(f"[compose] enqueued {result['pack_dir']} "
        f"(identity {result['identity']['matched']}/"
        f"{result['identity']['checked']}, READY {result['ready_sha'][:12]})")
    if ceiling is not None:
        fc = result["manifest"]["family_ceiling"]
        log(f"[compose] family {family!r} running total "
            f"{fc['projected_family_total']:,}/{fc['ceiling']:,}")
    return result


def served_targets_from_args(twin_url, twin_collection, prod_url, prod_collection):
    """The `served_targets` of `compose_pack`, or None when no prod URL is given. A prod URL
    without a twin URL refuses: half a check is not a check."""
    if not prod_url:
        return None
    if not twin_url:
        raise ComposeError("--served-prod-url needs --qdrant-url (the twin): the already-served "
                           "check asks both targets")
    return {"twin": (twin_url, twin_collection), "prod": (prod_url, prod_collection)}


def build_parser():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pack-id", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--chunks-dir", required=True)
    ap.add_argument("--vectors-dir", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--mrl", type=int, required=True)
    ap.add_argument("--qdrant-url", default=None,
                    help="twin URL to check mrl against the served dense "
                         "width before enqueueing (default: skip the check)")
    ap.add_argument("--queue-dir", default=None,
                    help="promotion queue (default: the operator setting queue_dir)")
    ap.add_argument("--skip-embed", action="store_true",
                    help="vectors_dir already holds a complete sealed "
                         "embed; compose + enqueue only")
    ap.add_argument("--ceiling", type=int, default=None,
                    help="refuse to enqueue this pack (no manifest, no "
                         "READY written) if --family's running total "
                         "already queued in --queue-dir plus this pack's "
                         "own rows would exceed it (ADR-113: 4500000 for "
                         "eric_official_pdf). Default: no ceiling.")
    ap.add_argument("--license-from", default=None,
                    help="dotted path to a per-row licence nested off the "
                         "top level (e.g. "
                         "source_metadata.accepted_answer.license) -- for "
                         "a family whose rows carry no top-level `license` "
                         "and no host-table entry (ADR-103 amendment 2).")
    ap.add_argument("--family-license", default=None,
                    help="a family-wide licence grant, used only when a "
                         "row has neither a top-level `license` nor "
                         "--license-from resolves one.")
    ap.add_argument("--family-payload", default=None,
                    help="a JSON object (source_type/trust_tier/etc) of "
                         "fields to fill for a row that lacks them -- never "
                         "overwrites a value the row already carries. This "
                         "door had no way to set family_payload/title_from "
                         "at all until task 664, 2026-09-26: ERIC and "
                         "stackoverflow_qa rows both carry no top-level "
                         "source_type/trust_tier/title, so every pack "
                         "composed through here refused at promotion with "
                         "PayloadCarryError (0 percent carry on all three) "
                         "-- the underlying compose_pack()/"
                         "embed_compose_enqueue() always accepted this "
                         "kwarg, the CLI simply never exposed it.")
    ap.add_argument("--title-from", action="append", default=[],
                    help="repeatable: a dotted path to lift a title from "
                         "(e.g. source_metadata.title) when the row's own "
                         "top-level `title` is absent.")
    ap.add_argument("--family-payload-basis", default=None,
                    help="free-text citation for where --family-payload's "
                         "values came from (e.g. a ratified rule module, or "
                         "a served precedent) -- written to the manifest "
                         "for the record, never fed to any row's payload.")
    ap.add_argument("--language-scope-filter", action="store_true",
                    help="opt in to dropping (and recording) a row whose "
                         "language cannot be detected confidently in the "
                         "product's set (EN/DE/FR/IT) -- writes manifest."
                         "language_scope_filter=true, consumed by "
                         "promote_queue.py's plan_kwargs_with_scope() at "
                         "drain time. Same mechanism CC tranche 1 used "
                         "(orchestrator ruling 2026-09-23); wired for ERIC "
                         "by Tim's ruling, task 664, 2026-09-28.")
    ap.add_argument("--backfill-lang", action="store_true",
                    help="opt in to physically rewriting --chunks-dir "
                         "(to a sibling -lang-backfilled directory) with "
                         "an empty `lang` field filled from the tracked "
                         "detector before identity verification -- "
                         "payload-only, no re-embed (verified: `lang` is "
                         "not part of the dense or sparse embedding "
                         "surface for this producer path). Same mechanism "
                         "`normalize_sealed_pack_shape.py --backfill-lang` "
                         "already uses for CC tranche 1 (task 652); wired "
                         "into the compose door for ERIC (task 669/C, "
                         "ERIC rows carry lang=null with nothing to fill "
                         "it before this).")
    ap.add_argument("--drop-undetected-lang", action="store_true",
                    help="opt in to writing manifest.lang_presence_waiver "
                         "-- excludes (at drain time) every row whose lang "
                         "is STILL empty after --backfill-lang, i.e. the "
                         "tracked detector could not place it confidently "
                         "in the product's language set at all. This "
                         "mechanically implements Tim's own standing ruling "
                         "for eric_official_pdf (plan.md rule 12, ADR-113: "
                         "'drop rows whose lang cannot be detected, as "
                         "CC-t1') -- not a new waiver decided here.")
    ap.add_argument("--consent-check-only", action="store_true",
                    help="run only the consent stamp's dry check (task 684) and exit 0, or "
                         "refuse non-zero. A caller that embeds in a separate step runs this "
                         "BEFORE the embed so a pack compose will refuse never burns the GPU.")
    ap.add_argument("--exclude-ids-file", default=None,
                    help="a .jsonl[.gz] of {\"point_id\": ...} the drain must "
                         "not write (rows below an admission floor, e.g. the "
                         "OpenAlex ADR-102 ladder trim). Its sha256 is pinned "
                         "in the manifest (task 680).")
    ap.add_argument("--exclude-ids-reason", default=None,
                    help="the reason every excluded row is reported with "
                         "(required with --exclude-ids-file).")
    ap.add_argument("--served-prod-url", default=None,
                    help="prod URL (e.g. the tunnel). With --qdrant-url (the twin) this declares, "
                         "at compose, the ids BOTH already serve as the drain's exclusion "
                         "(ruling 23.5). Omit it and nothing is checked.")
    ap.add_argument("--served-prod-collection", default="aw_prod")
    ap.add_argument("--supersede-url-group", default=None, metavar="REASON",
                    help="write `supersede: url-group` and `allow_supersede: REASON` into the "
                         "manifest (ADR-067 am.1 point 4, task 876): the drain replaces the served "
                         "rows of the pack's URL groups. Only for a family on "
                         "`url_group_supersede.URL_GROUP_FAMILIES`; the drain refuses others.")
    ap.add_argument("--supersede", default=None, choices=[UGS.URL_GROUP],
                    help="url-group: the drain replaces, per URL group of this pack, the served "
                         "rows the pack does not carry (loda_exact only; ruling 27 T6). Needs "
                         "--allow-supersede and --copy-sealed-guard; refused with "
                         "--served-prod-url.")
    ap.add_argument("--allow-supersede", default=None, metavar="REASON",
                    help="the reason written to manifest.allow_supersede (it lands in the receipt)")
    ap.add_argument("--copy-sealed-guard", action="store_true",
                    help="copy the sealed pack's own PACK.json stage-guard block into the "
                         "manifest, checked against the pack's URL groups (the drain re-reads it "
                         "from PACK.json and refuses a differing one)")
    ap.add_argument("--line-coverage-old-texts", action="append", default=[], metavar="PATH",
                    help="repeatable: a stage dir or shard file holding OLD rows' texts "
                         "({point_id, url, text}); with --supersede, writes the per-id line "
                         "coverage block (cytir/index/line_coverage.py) beside the manifest")
    ap.add_argument("--ruled-skip", default=None, metavar="FILE",
                    help="plan.md T24: a JSON list of {url, point_id, measured_coverage, ruling} "
                         "naming individual below-floor ids Tim waived; checked against the block, "
                         "copied beside the manifest and bound by sha256 (needs --supersede)")
    ap.add_argument("--line-coverage-wanted-ids", action="append", default=[], metavar="FILE",
                    help="repeatable: a .jsonl[.gz] of the served ids+urls the block should "
                         "cover (the old-generation export, the task-633 stage); ids with no "
                         "found text are counted unmeasured and fall back to the snippet guard")
    ap.add_argument("--drop-undetected-lang-basis", default=None,
                    help="free-text citation for --drop-undetected-lang's "
                         "ruling, written into the waiver's `rule` field "
                         "(default: cites plan.md rule 12 / ADR-113).")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    # task 696 slice 4: say where the queue and the write locks are, and why (never raises).
    print(PATHS.location_line("queue_dir", args.queue_dir, "queue_dir"), flush=True)
    print(PATHS.lock_location_line(), flush=True)
    if args.consent_check_only:
        rep = CONSENT_STAMP.stamp_chunks_dir(
            args.chunks_dir, "", CONSENT_STAMP.load_context(), family=args.family,
            license_from=args.license_from, family_license=args.family_license, dry=True)
        print(f"[compose] consent check: {rep['rows_in']:,} rows stampable {rep['rules']}")
        return 0
    family_payload = (json.loads(args.family_payload)
                      if args.family_payload else None)
    embed_compose_enqueue(
        pack_id=args.pack_id, family=args.family, chunks_dir=args.chunks_dir,
        vectors_dir=args.vectors_dir, collection=args.collection,
        mrl=args.mrl, qdrant_url=args.qdrant_url, queue_dir=args.queue_dir,
        skip_embed=args.skip_embed, ceiling=args.ceiling,
        license_from=args.license_from, family_license=args.family_license,
        family_payload=family_payload, title_from=args.title_from,
        family_payload_basis=args.family_payload_basis,
        language_scope_filter=args.language_scope_filter,
        backfill_lang=args.backfill_lang,
        drop_undetected_lang=args.drop_undetected_lang,
        drop_undetected_lang_basis=args.drop_undetected_lang_basis,
        exclude_ids_file=args.exclude_ids_file,
        exclude_ids_reason=args.exclude_ids_reason,
        supersede_url_group=args.supersede_url_group,
        served_targets=served_targets_from_args(
            args.qdrant_url, args.collection, args.served_prod_url,
            args.served_prod_collection),
        supersede=args.supersede, allow_supersede=args.allow_supersede,
        copy_sealed_guard=args.copy_sealed_guard,
        line_coverage_old_texts=args.line_coverage_old_texts,
        line_coverage_wanted_ids=args.line_coverage_wanted_ids,
        ruled_skip=args.ruled_skip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

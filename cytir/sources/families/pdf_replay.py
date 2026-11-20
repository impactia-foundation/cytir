"""PDF replay route of the family Protocol (task 782): sealed L0 PDF bytes -> `cytir.extract.pdf` -> rows.

THE GAP. The sitemap families (weko, bazl, bav, elcom) EXCLUDE every PDF url (`unsupported_kind:pdf`):
the PDF harvest was never ported. 435 PDF urls are served as raw bytes read as text (task 767); task
781 re-extracted all 435 from L0 and found no door to stage them through. This module is that door.

ONE IMPLEMENTATION, NOT FOUR. The four adapters stay the five-line `SitemapFamily` subclasses. Each
Protocol method of `SitemapFamily` starts with `if pdf_replay.active(ctx): return pdf_replay.<method>
(self, ...)`. Everything the route does lives below, once. `replay_l0 = True` on a family class opts
it in; any other family is refused.

THE INPUT. `cyt source run <family> --from LISTING --out STAGE`, where LISTING is a sealed L0 listing:
a JSONL file (or a directory holding `listing.jsonl`), one object per url,

    {"url": "https://www.weko.admin.ch/dam/de/sd-web/.../x.pdf",
     "sha256": "<hex of the PDF bytes>",
     "path": "pdf/<sha>.pdf",              # the bytes; relative to the listing's directory
     "lang": "de",                          # optional; else the pack's language rule for the url
     "provenance": "weko_docs"}             # optional; the served rows' lane label (see below)

The listing is the unit list: no sitemap is read, no page is fetched. Each url's bytes are checked
against their sha256 before extraction (sealed means the hash holds).

WHAT STAYS THE FRAMEWORK'S.
  * the contract: the route reads chunk policy, representation, id rule and licence from the family's
    row at run time. A row with any HELD (`unknown`) value is refused BEFORE a byte is extracted,
    with the contract's own message and each field named. This is the same refusal
    `family_stage.check_pack` makes at the door; it is made early so a 10-minute extraction is not
    spent on a pack the door will refuse. The door's refusal is not replaced.
  * the consent: the host of every url is resolved by the ONE resolver, live, as for the HTML route
    (the contract says "live per origin, at stage time"). It reads the host's robots.txt/TDM policy;
    it fetches no page. A decline EXCLUDES the unit; an unreadable policy QUARANTINES it. The pack
    is `consent.checked`, so the door accepts it.
  * the prose guard: the door's `binary_content` hold-out and the staged-text guard (word 5-gram
    coverage of the extracted markdown by the staged rows; image placeholders and heading markers are
    the DECLARED transformations, removed from the source text before it is compared).
  * the L0 seal (the replayed bytes are re-sealed content-addressed in the run's own l0/), the
    terminals, the pack.

WHAT THE ROUTE DECIDES (and the contract may overrule): the extractor is `cytir.extract.pdf.extract_pdf`
(liteparse by default, Docling for short complex documents, exactly as task 781 ran it); image-only
lines are dropped (`![](img_p1_1.png)` from liteparse, `<!-- image -->` from Docling); the chunk
policy is the one the contract names (`WebPageMixin.preflight` refuses a dispatcher selection that
differs from the declared sha256); ids are minted by the contract's `identity.minted_by`:

    text-anchor-v0   uuid5(NAMESPACE_URL, url#anchor:sha1(text))   what `Passage` mints (ADR-067 am.2)
    mint_point_id    uuid5(AW_NAMESPACE, canonical_url#digest)     `cytir.core.ids.mint_point_id`, Tier B

Any other `minted_by` text is refused: the route runs the two rules that exist, and does not guess.

LANDING. The rows are new ids at urls whose old rows (`aw_b1`, `mint_point_id` ids, raw bytes) are
still served. They land as URL-group supersede packs (`cytir.index.url_group_supersede`): the old rows
leave by URL, whatever their id scheme, and the new rows arrive in the same drain. A group needs the
rows' `provenance` to equal the served rows' lane label, so the listing carries it.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlsplit

from cytir.sources import contract as contract_mod
from cytir.sources.family import Captured, Doc, RunContext, Unit, UnitExcluded

REPLAY_ROUTE = "replay-l0"
LISTING_OPTION = "l0_listing"
LISTING_NAME = "listing.jsonl"
#: the staged-text guard's floor for this route (the guard's own default, now DECLARED so a pack
#: that lost text is refused at the door and `url_group_supersede` sees a threshold)
MIN_COVERAGE = 0.98
#: The word-coverage ruling for the replacement packs of this route (Tim, ruling 31, 2026-10-12; plan.md
#: T9(h)/T19): SKIP PER URL under ADR-114 am.2. The snippet guard (floor 0.50) is skipped for one url
#: only when the sealed stage-guard block passes (threshold = MIN_COVERAGE above, shortfall 0,
#: docs_unchecked 0, docs_checked = the url groups) AND the manifest's per-id block measures the old
#: row's line coverage by the new rows at 0.8 or more; an id the block did not measure falls back to
#: the snippet guard. A declaration only: the skip itself is the drain's code (am.2 "later PR").
WORD_COVERAGE_RULING = {"mode": "skip-per-url", "adr": "ADR-114 am.2", "ruling": 31,
                        "stage_guard_floor": MIN_COVERAGE, "per_id_line_coverage_floor": 0.8,
                        "fallback": "snippet-guard-0.50"}
TEXT_ANCHOR_V0 = "text-anchor-v0"
MINT_POINT_ID = "mint_point_id"

#: A line that holds only an image placeholder: liteparse `![](img_p1_1.png)`, Docling
#: `<!-- image -->`. No text was in it, so dropping it removes no word.
IMAGE_LINE = re.compile(r"^\s*(?:!\[[^\]]*\]\([^)]*\)|<!--\s*image\s*-->)\s*$")
_BLANKS = re.compile(r"\n{3,}")

#: Replaceable in tests: `(Path) -> cytir.extract.pdf.ExtractResult`. None = the real router.
EXTRACTOR: Callable | None = None


class ReplayError(RuntimeError):
    """The listing or the route's own preconditions are wrong. Nothing is staged."""


@dataclass(frozen=True)
class ListingRow:
    url: str
    sha256: str
    path: Path
    lang: str = ""
    provenance: str = ""


# ------------------------------------------------------------------------------ activation

def active(ctx: RunContext) -> bool:
    """True when this run replays a sealed L0 listing instead of reading a sitemap."""
    return bool(ctx.options.get(LISTING_OPTION))


def listing_file(value: str | Path) -> Path:
    """`--from` for a replay family: the listing file, or the directory that holds `listing.jsonl`."""
    p = Path(value)
    return p / LISTING_NAME if p.is_dir() else p


def read_listing(value: str | Path) -> list[ListingRow]:
    path = listing_file(value)
    if not path.is_file():
        raise ReplayError(f"{path}: the sealed L0 listing is not a file")
    rows: list[ListingRow] = []
    seen: set[str] = set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplayError(f"{path} line {n}: not JSON ({exc})") from exc
        url, sha, rel = d.get("url"), d.get("sha256"), d.get("path")
        if not (isinstance(url, str) and url and isinstance(rel, str) and rel):
            raise ReplayError(f"{path} line {n}: needs `url` and `path`")
        if not (isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)):
            raise ReplayError(f"{path} line {n} ({url}): `sha256` is not 64 lowercase hex")
        if url in seen:
            raise ReplayError(f"{path} line {n}: url listed twice: {url}")
        seen.add(url)
        p = Path(rel)
        rows.append(ListingRow(url=url, sha256=sha, path=p if p.is_absolute() else path.parent / p,
                               lang=str(d.get("lang") or ""),
                               provenance=str(d.get("provenance") or "")))
    if not rows:
        raise ReplayError(f"{path}: the listing holds no url")
    return rows


# ------------------------------------------------------------------------------ the contract

def contract_refusal(row: dict | None, family_id: str) -> str | None:
    """Why the family's declaration cannot run this route, or None. The first reason is the
    contract's own strict message (it names every HELD field); the fields are listed again, one
    per line, so a refusal says exactly what Tim's POLICY PR must fill."""
    if not row:
        return f"{family_id}: no contract row"
    problems = contract_mod.validate_family(row, strict=True)
    if not problems:
        return None
    held = sorted(row.get("unknowns") or contract_mod.actual_unknowns(row))
    return (f"{family_id}: the replay-l0 route needs a complete contract row and this one is HELD. "
            + " | ".join(problems)
            + (" -- fields: " + ", ".join(held) if held else "")
            + ". Fill them in a POLICY PR (Tim's merge); the route never reads a default.")


def id_rule(row: dict) -> str:
    """`text-anchor-v0` or `mint_point_id`, from `identity.minted_by`; anything else is refused."""
    by = str((row.get("identity") or {}).get("minted_by") or "")
    a, b = TEXT_ANCHOR_V0 in by, MINT_POINT_ID in by
    if a == b:
        raise ReplayError(
            f"{row.get('family_id')}: identity.minted_by {by[:120]!r} names "
            + ("both id rules" if a else "neither id rule")
            + f" ({TEXT_ANCHOR_V0}, {MINT_POINT_ID}); the replay route runs those two and does "
              "not guess a third")
    return TEXT_ANCHOR_V0 if a else MINT_POINT_ID


def contract_licence(row: dict) -> str:
    """The per-record `license` the contract declares: `licence.spdx`, else `licence.instrument`.
    Never the pack's value and never the host's (licence-basis-from-the-instrument)."""
    lic = row.get("licence") or {}
    for k in ("spdx", "instrument"):
        v = lic.get(k)
        if isinstance(v, str) and v.strip() and v != contract_mod.UNKNOWN:
            return v.strip()
    raise ReplayError(f"{row.get('family_id')}: the contract declares no licence.spdx/instrument")


# ------------------------------------------------------------------------------ text

def drop_image_lines(md: str) -> tuple[str, int]:
    """`(markdown without image-only lines, how many were dropped)`. A paragraph that held only an
    image leaves no empty gap."""
    kept, dropped = [], 0
    for ln in md.splitlines():
        if IMAGE_LINE.match(ln):
            dropped += 1
        else:
            kept.append(ln)
    return _BLANKS.sub("\n\n", "\n".join(kept)).strip(), dropped


_HEADING_MARK = re.compile(r"(?m)^#{1,6}[ \t]+")


def guard_text(md: str) -> str:
    """The text the staged-text guard compares the rows with (`Doc.source_text`): the markdown with
    its heading markers removed. The structured chunker consumes `## ` (it carries the heading as
    context, not as a token of the chunk), so a marker is a declared transformation, not lost text;
    measured on the WEKO Merkblatt the markers alone cost 3 points of 5-gram coverage (0.9696 -> 1.0).
    Every WORD of the document still has to be in a row."""
    return _HEADING_MARK.sub("", md)


def pdf_title(md: str, url: str) -> str:
    """A display title for a PDF page: the file name when it reads as words (three or more),
    else the first heading of three or more words, else the file name, else the url. PROVISIONAL:
    the contract's `title_shape` (HELD) decides the shape; this only keeps the percent-encoded file
    name (`Swisscom%20WAN...`, what the legacy rows show) out of the title."""
    stem = Path(unquote(urlsplit(url).path)).stem
    stem = re.sub(r"[_\s]+", " ", stem).strip()
    if len(stem.split()) >= 3:
        return stem
    for ln in md.splitlines():
        m = re.match(r"^#{1,6}\s+(.+?)\s*$", ln)
        if m and len(m.group(1).split()) >= 3:
            return m.group(1)[:200]
    return stem or url


# ------------------------------------------------------------------------------ the methods

def preflight(family, ctx: RunContext) -> None:
    """Refuse a HELD contract, an id rule it cannot run and an unusable listing, before any unit."""
    from cytir.sources.registry import FamilyRefused

    if not getattr(family, "replay_l0", False):
        raise FamilyRefused(f"{family.family_id}: this family has no replay-l0 route")
    why = contract_refusal(ctx.contract, family.family_id)
    if why:
        raise FamilyRefused(why)
    try:
        ctx.notes["id_rule"] = id_rule(ctx.contract)
        contract_licence(ctx.contract)
    except ReplayError as exc:
        raise FamilyRefused(str(exc)) from exc
    # this run's facts, on the per-run adapter instance (`registry.admit` builds one per run)
    family.route = REPLAY_ROUTE
    family.staged_text_min_coverage = MIN_COVERAGE


def discover(family, ctx: RunContext) -> list[Unit]:
    listing = read_listing(ctx.options[LISTING_OPTION])
    lang_of = family.plan().lang_fn()
    units = []
    for r in listing:
        host = (urlsplit(r.url).hostname or "").lower()
        units.append(Unit(
            key=r.url, url=r.url,
            meta={"sha256": r.sha256, "path": str(r.path), "provenance": r.provenance,
                  "lang": r.lang or lang_of(r.url) or "", "host_ok": host == family.host},
            consent_hosts=(host or family.host,)))
    ctx.notes.update({"route": REPLAY_ROUTE, "host": family.host, "units": len(units),
                      "listing": str(listing_file(ctx.options[LISTING_OPTION])),
                      "pdf_extract": {}})
    return units


def acquire(family, unit: Unit, ctx: RunContext):
    if not unit.meta["host_ok"]:
        # a listing that mixes hosts: counted, never silently dropped, never a foreign url under
        # this family's contract
        raise UnitExcluded(f"host_not_this_family:{urlsplit(unit.url).hostname}")
    body = Path(unit.meta["path"]).read_bytes()
    got = hashlib.sha256(body).hexdigest()
    if got != unit.meta["sha256"]:
        raise ReplayError(f"{unit.url}: bytes hash {got[:12]} != listing {unit.meta['sha256'][:12]}")
    if b"%PDF-" not in body[:1024]:
        raise UnitExcluded("not_a_pdf")
    return [Captured(unit, body, {"status": 200, "content_type": "application/pdf",
                                  "source": "l0-replay", "l0_sha256": got})]


def _extract(path: Path):
    if EXTRACTOR is not None:
        return EXTRACTOR(path)
    from cytir.extract import pdf
    return pdf.extract_pdf(path)


def extract(family, cap: Captured, ctx: RunContext):
    work = ctx.out_dir / "pdf-work"
    work.mkdir(parents=True, exist_ok=True)
    tmp = work / f"{cap.meta['l0_sha256']}.pdf"
    tmp.write_bytes(cap.body)
    try:
        res = _extract(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    md, images = drop_image_lines(res.markdown or "")
    ctx.notes["pdf_extract"][cap.unit.url] = {
        **res.provenance(), "l0_sha256": cap.meta["l0_sha256"],
        "words": len(md.split()), "image_lines_dropped": images}
    if not md.strip():
        return []
    common = family.passage_base(ctx, cap.unit)
    common["license"] = contract_licence(ctx.contract)
    return [Doc(url=cap.unit.url, text=md, title=pdf_title(md, cap.unit.url),
                lang=cap.unit.meta.get("lang") or "",
                extra={"passage_common": common, "provenance": cap.unit.meta.get("provenance")},
                source_text=guard_text(md))]


def mint_ids(rule: str, passage_row: dict) -> str:
    """The point id of one chunk row under the contract's rule."""
    from cytir.core import canon, common as C, ids
    from cytir.extract import html_extract as _html_extract

    if rule == TEXT_ANCHOR_V0:
        return C.mint_passage_point_id(passage_row["url"], passage_row.get("anchor"),
                                       passage_row["text"])
    digest = ids.content_digest(passage_row["text"])
    tier, comp = ids.classify_tier(passage_row.get("source_type"), passage_row.get("anchor"),
                                   True, digest, title=passage_row.get("title"))
    return ids.mint_point_id(canon.canonical_url(passage_row["url"]), comp)


def chunk(family, doc: Doc, ctx: RunContext) -> list[dict]:
    from cytir.core import common as C
    from cytir.extract import html_extract as _html_extract

    passages = _html_extract.passages_from_markdown(
        doc.url, doc.text, title=doc.title, site_name=None, common=doc.extra["passage_common"],
        lang_default=doc.lang or "en")
    rule = ctx.notes["id_rule"]
    rows = []
    for p in passages:
        row = C.passage_source_record(p)
        row["point_id"] = mint_ids(rule, row)
        if doc.extra.get("provenance"):
            row["provenance"] = doc.extra["provenance"]
        rows.append(row)
    return rows

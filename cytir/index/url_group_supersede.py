#!/usr/bin/env python3
"""URL-group supersede for a sealed pack (ADR-067 am.1 point 4; task 712).

THE DEFECT. A sealed pack with `allow_supersede` only lifts the "already served" refusal. A
changed law article has a new id (the id hashes the text), so the drain writes the new row BESIDE
the old one at the same URL and nothing deletes the old row (task 705, the id-lineage study: the loda
replacement of 2026-09-23 added 222,499 points and deleted none).

THE RULE. A manifest that declares `"supersede": "url-group"` (plus `allow_supersede`, the reason)
replaces, per URL group of the pack's rows, the served rows the pack does not carry:

  * GROUP = `supersede.payload_group_key` (canonical url, fragment kept, so `...#art_41` and
    `...#art_42` are two groups). Rows of one URL that differ only in `anchor` (the chunk rows of
    one article) are ONE group.
  * ALLOWLISTED FAMILY ONLY (`URL_GROUP_FAMILIES`: manifest `family` -> the `provenance` value its
    rows are SERVED with; today loda_exact -> `loda`). `provenance` is the family only for such a
    family (CC rows carry the discovery-lane label; fedlex and Wikipedia rows carry none), so any
    other family is REFUSED before a read or a write. For a listed family a served row is
    superseded when it is in a group the pack writes, is NOT one of the pack's ids, is not a card,
    and carries that `provenance`. A row of another provenance at the same URL stays (counted in
    the receipt). A row with no `provenance` cannot be classified: the pack is refused.
  * CARDS ARE NEVER TOUCHED. A pack card row opens no group (its key has the `|card` suffix) and a
    served card is never a victim (`supersede.is_card`, one test).
  * A group exists only because the pack writes at least one row at it, so a supersede replaces a
    page and never empties one. The new rows are written first, the old ones deleted after.

BEFORE ANY WRITE (`plan_for_drain`, run by the drain before the locks):
  (a) the restore copy: every superseded row, with every named vector and its full payload, goes
      to ONE file per pack in the directory the operator setting `removed_rows_export_dir` names
      (no default; it must exist; never the queue). The file is written aside, fsynced, renamed,
      hashed, and checked by `promote_queue.verify_removed_rows_export` (the same function the
      `superseded_generation` pack uses). The path, sha256 and row count go in the receipt. If
      the copy cannot be written and verified, the pack is refused.
  (b) the word-coverage guard of the content packs (`validate_tombstone_word_coverage`, the floor
      of `data/promotion-gate-contract.v1.json`): a URL whose new rows carry clearly fewer words
      than the rows they replace is SKIPPED, not forced (task 876, ADR-114 am.2): its old rows
      stay, and none of its pack rows are written. The skipped URLs are counted and named with
      their ratios in the receipt (`word_coverage_skipped_urls`). The rest of the pack lands. When
      the skipped URLs are more than `SKIP_SHARE_MAX` (20 %) of the URLs checked, the whole pack is
      refused, as the content packs do (`COVERAGE_VIOLATION_SHARE_MAX`): a pack that loses a
      fifth of its URLs is a wrong pack, not a few odd documents. A skip needs the drain to hold
      the URL's pack rows out of the write (`skipped_pack_ids` of the plan, see `plan_for_drain`).
      WHAT (b) CAN SEE. The index stores a 25-word snippet per row and no body text. So (b)
      compares snippet words: it sees a replacement that has fewer or shorter rows. It CANNOT see
      text cut INSIDE a row (the loda `[:8000]` cut left the snippet intact).
      SKIP PER URL (ruling 27 T6, ADR-114 am.2): (b) is skipped for a URL when (i) condition (c)
      below holds for the pack AND (ii) the manifest binds a per-id LINE-COVERAGE block
      (`cytir/index/line_coverage.py`: the measure is defined there) and every superseded row of the
      URL is measured at that URL at >= 0.8. Any URL with an unmeasured or lower row keeps (b)
      as it is. A law's changed OLD text may thus be deleted: the new rows are the complete current
      article (c), and the old lines they do not hold are the law's own changes. Blind families
      never reach this (c). Therefore:
  (c) the SEALED pack's own stage-time block (PR #631): `PACK.json` of the stage dir, key
      `pack_staged_text_guard`, bound to the pack by the sha256 of its chunks file. A block typed
      into the queue manifest never counts (a differing one refuses the pack). It must declare a
      threshold, with `docs_shortfall` 0, `docs_unchecked` 0 and `docs_checked` equal to the number
      of URL groups the pack writes. A family that declares no threshold saw no source text: it is
      BLIND and never uses url-group supersede.

THE WRITE. Twin first (new rows, then the superseded rows deleted and verified gone by id, then
the count bound), then prod the same. Before each delete the export file is re-verified.
PROD HAS ITS OWN EXPORT: prod's payloads are normalised and its BM25 vectors pruned, so the twin
export is not an exact restore source for prod. After the prod upsert and before the first prod
delete the superseded ids are read from PROD by id (batches <= 500, a pause, never a scroll), prod's
own rows (all vectors, full payload) go to a second file beside the twin export and are verified
against prod's vector names and sizes. Only the ids prod HOLDS are deleted and counted: an id the
twin had and prod does not is `absent`, not `deleted`. Counts and the file are in the receipt and
the state; a resume re-verifies the prod export before the prod leg. A state
file `<artifact_root>/<pack_id>-url-group-supersede.json` records the ids and the export, written
BEFORE the first delete: a re-run after a kill (twin done, prod owed) finds the twin already
without the old rows, so it reads the ids from the state, re-verifies the export's sha256, and
finishes prod. A mismatch stops the drain (`QueueError`, state kept).

A pack WITHOUT the `supersede` field never reaches this module. `allow_supersede` alone keeps its
old meaning (lifts the refusal, deletes nothing).
"""
from __future__ import annotations

from cytir.acquire.gzip_pinned import deterministic_gzip_open
import hashlib
import json
import os
import pathlib
import time

from cytir.core import paths as PATHS
from cytir.core.canon import document_url_key
from cytir.index import dry_run as DRY
from cytir.index import line_coverage as LC
from cytir.index import point_delete as PD
from cytir.index import promote_sealed_pack as PSP
from cytir.index import supersede as US
from cytir.index import row_decode as RD  # canonical stored rows, any profile (task 855)
from cytir.sources.families import fedlex_contract as FC  # dependency-light: re only (task 1038)

#: The manifest field that declares the behaviour, and its one value.
FIELD = "supersede"
URL_GROUP = "url-group"
STATE_SCHEMA = "aw.url-group-supersede-state.v1"
#: `cytir.sources.family_stage.STAGE_SCHEMA`, copied: importing that module would pull its file
#: writers into the drain's import closure (a test pins the two equal).
SEALED_STAGE_SCHEMA = "aw.family-stage.v1"
#: Rows per retrieve of the export read (vectors make a row heavy).
EXPORT_BATCH = 256
#: Urls named in a refusal line.
NAMED = 20
#: Reads of PROD (gentle: a 30 GB serving box). Rows per retrieve, and the pause after each.
PROD_PRESENCE_BATCH = 500
PROD_EXPORT_BATCH = 200
PROD_READ_SLEEP_S = 0.25
class ScopeRule:
    """A family whose SERVED rows carry no `provenance` (task 1038, fedlex): the family is told by
    the URL scope of the row, and the unit a refresh replaces is the whole scope, not one url.

    A scope is the group: `group_key(url)` (None = the url is not in the family at all). The drain
    asks the twin for rows by KEYED url only (the `url` payload index is a keyword index: there is
    no prefix query), so the rule names the urls to ask for: `candidates(pack_urls)` first, then
    `chase(found_urls, asked)` until it names none (contiguous windows `-c1..-cK`)."""

    label = "scope"

    def group_key(self, url: str | None) -> str | None:
        raise NotImplementedError

    def candidates(self, pack_urls) -> list[str]:
        raise NotImplementedError

    def chase(self, found_urls, asked) -> list[str]:
        return []

    def coverage_url(self, url: str) -> str:
        return url

    def __repr__(self) -> str:
        return self.label

    __str__ = __repr__


class FedlexScope(ScopeRule):
    """fedlex_exact: scope = host `fedlex.admin.ch` + WHOLE path `/eli/<eli>/<lang>` (the act in one
    language; fragment, `www.` and scheme ignored). Victims = every served non-card row whose url is
    in a scope the pack writes: the bare `#art_N`, the windows `#art_N-cK`, whatever spelling the
    anchor has. Victim candidates are asked for by keyed url: per pack row its own url, the base
    anchor `#art_N` and the first window `#art_N-c1`; each served window found makes the next
    `-c<K+1>` a candidate (windows are contiguous), so a stale tail window is found too."""

    label = "scope:fedlex.admin.ch/eli/<eli>/<lang>"

    def group_key(self, url):
        sc = FC.act_scope(url)
        return None if sc is None else f"fedlex:{sc[0]}/{sc[1]}"

    def candidates(self, pack_urls):
        out: set[str] = set()
        for u in pack_urls:
            base, anchor, _ = FC.split_anchor(u)
            out.add(u)
            if anchor:
                out.update((f"{base}#{anchor}", f"{base}#{anchor}-c1"))
        return sorted(out)

    def chase(self, found_urls, asked):
        nxt: set[str] = set()
        for u in found_urls:
            base, anchor, k = FC.split_anchor(u)
            if anchor and k is not None:
                cand = f"{base}#{anchor}-c{k + 1}"
                if cand not in asked:
                    nxt.add(cand)
        return sorted(nxt)

    def coverage_url(self, url):
        return FC.coverage_url(url)


FEDLEX_SCOPE = FedlexScope()

#: Families that may declare `supersede: url-group`: manifest `family` -> the `provenance` value
#: their rows are SERVED with. `provenance` is a family key only here: CC rows carry the discovery
#: lane label, fedlex and Wikipedia rows carry none. Adding a family is a reviewed change to this
#: constant, with the served value measured on prod and the twin.
URL_GROUP_FAMILIES: dict[str, "str | ScopeRule"] = {
    "loda_exact": "loda",
    # Task 876: the four regulator PDF families of the replay door (task 782). The labels are the
    # `provenance` of the SERVED rows, read on the twin (`aw_e8`, through `row_decode`) on
    # 2026-10-12: every `official` row at the 200 door-covered urls of the four hosts carries
    # exactly this label (weko 4,908 rows, bav 3,782, bazl 2,613, elcom 913). Other lanes at the
    # same hosts (`l1_backfill_refetch`, `endorsement-discovery`, the entity card) are not the
    # family: a url-group supersede keeps them.
    "scratch_weko_docs_exact": "weko_docs",
    "scratch_bav_docs_exact": "bav_docs",
    "scratch_bazl_docs_exact": "bazl_docs",
    "scratch_elcom_docs_exact": "elcom_docs",
    # Task 1038 (RULING 52): fedlex. Its served rows carry NO `provenance` (only `prov_id`, a
    # chunk-policy enum; read on the twin 2026-10-21), so the value is a ScopeRule: the family is
    # told by the url scope, and a refresh replaces the whole act-language.
    "fedlex_exact": FEDLEX_SCOPE,
}
#: The most URLs a pack may SKIP at guard (b), as a share of the URLs checked. Over it the whole
#: pack is refused. Equals `promote_queue.COVERAGE_VIOLATION_SHARE_MAX` (a test pins the two).
SKIP_SHARE_MAX = 0.20


def _q():
    """`promote_queue`, imported at call time: it imports this module at load."""
    from cytir.index import promote_queue as Q  # noqa: PLC0415
    return Q


# ------------------------------------------------------------------ declaration
def declared(manifest: dict) -> bool:
    """True when the manifest declares `supersede: url-group`. A different value, or the field
    without `allow_supersede` (the plan would then refuse the rows it must write beside), is a
    `QueueError`; the field is for sealed packs."""
    mode = manifest.get(FIELD)
    if mode is None:
        return False
    Q = _q()
    if mode != URL_GROUP:
        raise Q.QueueError(f"manifest `{FIELD}` is {mode!r}; the only value is {URL_GROUP!r}")
    if manifest.get("kind") == "content_pack":
        raise Q.QueueError(f"`{FIELD}: {URL_GROUP}` is for sealed packs; a content_pack carries "
                           "its own tombstones")
    if not manifest.get("allow_supersede"):
        raise Q.QueueError(f"`{FIELD}: {URL_GROUP}` needs `allow_supersede` (the reason): the "
                           "plan refuses a pack whose pages are served already without it")
    if manifest.get("already_served_excluded_file"):
        raise Q.QueueError(f"`{FIELD}: {URL_GROUP}` with `already_served_excluded_file` (task 888): "
                           "the already-served exclusion drops the unchanged rows that open their "
                           "URL groups, so the drain would delete their served rows and not write "
                           "them again")
    return True


def family_provenance(manifest: dict) -> "str | ScopeRule":
    """The `provenance` value the rows of the manifest's family are served with (or, for a family
    whose served rows carry none, its `ScopeRule`), from the
    allowlist `URL_GROUP_FAMILIES`. Any other family is refused (one line, before a read or a
    write): `provenance` is not a family key for it, so no victim set could be named."""
    Q = _q()
    family = manifest.get("family")
    served = URL_GROUP_FAMILIES.get(family) if isinstance(family, str) else None
    if not served:
        raise Q.QueueError(
            f"url-group supersede is not permitted for family {family!r}: only "
            + ", ".join(f"{f} (served provenance {p!r})" for f, p in sorted(URL_GROUP_FAMILIES.items()))
            + " may use it (URL_GROUP_FAMILIES; adding a family is a reviewed change)")
    return served


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _stage_dir(manifest: dict) -> pathlib.Path:
    """The sealed stage dir (the one that holds `PACK.json` and `chunks/`) of the manifest's
    `pack_chunks`: the `chunks` dir, a `chunks-*` sibling a compose step wrote beside it (the
    consent-stamped copy), or the stage dir itself."""
    named = manifest.get("pack_chunks")
    if not named:
        raise _q().QueueError("url-group supersede: the manifest names no `pack_chunks`")
    d = pathlib.Path(named)
    return d if (d / "PACK.json").is_file() else d.parent


def sealed_chunks_sha256(manifest: dict) -> str:
    """The `chunks_sha256` of the sealed pack's `PACK.json` (the sha of its one chunks file). Call
    it after `sealed_guard_block`, which verifies that the file hashes to it."""
    return json.loads((_stage_dir(manifest) / "PACK.json").read_text())["chunks_sha256"]


def sealed_guard_block(manifest: dict) -> dict:
    """The stage-time guard block of the SEALED pack, read from the pack's own `PACK.json` (the
    stage dir that holds `chunks/`; `pack_chunks` names the `chunks` dir or the stage dir) and bound
    to the pack: the file `chunks/shard-0000.texts.jsonl.gz` must hash to `PACK.json`'s
    `chunks_sha256` and be the only chunks file. A `staged_text_guard` typed into the queue manifest
    is not read; if there is one and it differs from the sealed block, the pack is refused."""
    Q = _q()
    stage = _stage_dir(manifest)
    pj = stage / "PACK.json"
    if not pj.is_file():
        raise Q.QueueError(
            f"`{FIELD}: {URL_GROUP}` needs the SEALED pack's own stage manifest ({pj} is missing): "
            "a pack staged before the stage-time guard (PR #631) cannot show it kept the source "
            "text, and the old rows would be deleted on a snippet comparison alone. Re-stage the "
            "pack")
    try:
        meta = json.loads(pj.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Q.QueueError(f"{pj} is unreadable ({exc})") from exc
    if meta.get("schema") != SEALED_STAGE_SCHEMA:
        raise Q.QueueError(f"{pj}: schema {meta.get('schema')!r}, expected {SEALED_STAGE_SCHEMA!r}")
    chunks = stage / "chunks"
    shards = sorted(chunks.glob("*.texts.jsonl.gz")) if chunks.is_dir() else []
    if [x.name for x in shards] != ["shard-0000.texts.jsonl.gz"] or not meta.get("chunks_sha256"):
        raise Q.QueueError(f"{pj}: the stage holds {[x.name for x in shards]} and declares "
                           f"chunks_sha256 {meta.get('chunks_sha256')!r}; one shard-0000 file "
                           "bound by its sha256 is required")
    got = _sha256_file(shards[0])
    if got != meta["chunks_sha256"]:
        raise Q.QueueError(f"{pj}: chunks file sha256 {got[:12]} != declared "
                           f"{str(meta['chunks_sha256'])[:12]}; the guard block does not belong to "
                           "this pack's rows")
    block = meta.get("pack_staged_text_guard")
    if not isinstance(block, dict):
        raise Q.QueueError(f"{pj} carries no `pack_staged_text_guard`: the pack was staged before "
                           "the guard (PR #631); re-stage it")
    copied = manifest.get("staged_text_guard_chunks_sha256")
    if copied is not None and copied != meta["chunks_sha256"]:
        raise Q.QueueError(
            f"the queue manifest names chunks sha256 {str(copied)[:12]} for its copied stage guard; "
            f"the sealed pack's chunks file is {str(meta['chunks_sha256'])[:12]}: the manifest was "
            "composed against other rows than the ones this stage holds")
    typed = manifest.get("staged_text_guard")
    if typed is not None and typed != block:
        raise Q.QueueError("the queue manifest carries a `staged_text_guard` that differs from the "
                           f"sealed pack's own block in {pj}; only the sealed block counts -- "
                           "remove the typed one")
    return block


def text_guard(block: dict, *, url_groups: int) -> dict:
    """(c): the sealed pack's guard block. Refuses when a doc fell short, when the family declares
    no threshold (no source text was seen: BLIND), when any doc was left unchecked, or when
    `docs_checked` is not the number of URL groups the pack writes. Returns what the guard saw."""
    Q = _q()
    short = block.get("docs_shortfall")
    unchecked = block.get("docs_unchecked")
    checked = block.get("docs_checked")
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (short, unchecked, checked)):
        raise Q.QueueError("`pack_staged_text_guard` lacks integer docs_checked / docs_unchecked / "
                           f"docs_shortfall (got {sorted(block)})")
    if short > 0:
        raise Q.QueueError(
            f"staged-text guard: {short} doc(s) fell below the coverage "
            f"{block.get('min_coverage')} (lowest {block.get('lowest_coverage')}, examples "
            f"{block.get('shortfall_examples')!r}) -- a pack that lost text is not a replacement")
    if not block.get("threshold_declared"):
        raise Q.QueueError(
            "staged-text guard: the family declares no coverage threshold, so no source text was "
            "compared (BLIND); url-group supersede refuses a blind family")
    if unchecked > 0:
        raise Q.QueueError(f"staged-text guard: {unchecked} doc(s) were left unchecked -- refusing")
    if checked != url_groups:
        raise Q.QueueError(
            f"staged-text guard: it checked {checked} doc(s) and the pack writes {url_groups} "
            "URL group(s); the block does not describe this pack's documents -- refusing")
    return {"docs_checked": checked, "docs_unchecked": unchecked, "docs_shortfall": short,
            "threshold_declared": True, "min_coverage": block.get("min_coverage"),
            "sees": "source text (word 5-grams)"}


#: Ruling 27 T6 c: the stage guard threshold that proves "the new rows are the complete article".
REQUIRED_STAGE_COVERAGE = 1.0


def line_coverage_plan(manifest: dict, urls: dict[str, str], *,
                       pack_dir: pathlib.Path | None = None) -> dict:
    """Ruling 27 T6 (ADR-114 am.2): which URLs may skip the snippet guard. `urls` = `{superseded id:
    its served url}`. Call it ONLY after `text_guard` passed on the sealed pack's own block (the
    caller does: condition (i)). Condition (ii): the manifest binds a per-id block
    (`line_coverage.load`: sha256 of its file, the sealed chunks sha, consistent entries) and EVERY
    superseded row of a URL has an entry at that URL with coverage >= `line_coverage.MIN_COVERAGE`.
    A URL with a row that is unmeasured, below the floor or measured at another URL keeps the
    snippet guard (all its rows). A block that cannot be trusted refuses the pack: it was declared.
    Ruled skip (plan.md T24): a manifest-bound file (`line_coverage.load_ruled_skip`) may name
    individual below-floor ids as covered; `check_ruled_skip` refuses the pack when an entry is not
    a superseded id of its url or its measure drifted. Nothing else about the URL changes.
    The block file is read from `pack_dir` (the pack's own dir in the queue).

    Condition (iii), ruling 27 T6 c: a skip means "the new rows are the COMPLETE current article",
    and only a stage guard at threshold 1.0 proves that. The sealed guard's `min_coverage` must be
    >= 1.0 (a number); a family that declared less (the drain used to accept any declared
    threshold) skips nothing: every URL keeps the snippet guard, and the summary says why
    (`skip_refused`).

    Returns `{skip_ids, summary}`; `skip_ids` is the set of superseded ids whose URL skips."""
    Q = _q()
    try:
        entries = LC.load(manifest, sealed_chunks_sha256=sealed_chunks_sha256(manifest),
                          pack_dir=pack_dir)
    except LC.LineCoverageError as exc:
        raise Q.QueueError(f"url-group supersede: {exc}") from exc
    try:
        ruled = LC.load_ruled_skip(manifest, pack_dir=pack_dir)
        if ruled is not None and entries is None:
            raise LC.LineCoverageError("ruled skip: the manifest has no line coverage block")
        ruled_ids = (LC.check_ruled_skip(ruled, entries, urls) if ruled is not None else {})
    except LC.LineCoverageError as exc:
        raise Q.QueueError(f"url-group supersede: {exc}") from exc
    ruled_summary = (None if ruled is None else
                     {"file": manifest["ruled_skip_file"], "sha256": manifest["ruled_skip_sha256"],
                      "entries": ruled})
    base = {"min_coverage": LC.MIN_COVERAGE, "block": None, "urls_skipped": 0, "ids_skipped": 0,
            "ruled_skip": ruled_summary, "ids_ruled_skip": 0,
            "urls_snippet_guard": 0, "ids_unmeasured": 0, "ids_below_min": 0,
            "ids_measured_at_other_url": 0}
    if entries is None:
        urls_n = len({LC.url_key(u) for u in urls.values()})
        return {"skip_ids": frozenset(), "summary": {**base, "urls_snippet_guard": urls_n,
                                                     "ids_unmeasured": len(urls)}}
    stage_min = sealed_guard_block(manifest).get("min_coverage")
    if (not isinstance(stage_min, (int, float)) or isinstance(stage_min, bool)
            or stage_min < REQUIRED_STAGE_COVERAGE):
        urls_n = len({LC.url_key(u) for u in urls.values()})
        return {"skip_ids": frozenset(), "summary": {
            **base, "block": {"file": manifest["line_coverage_file"],
                              "sha256": manifest["line_coverage_sha256"], "ids": len(entries)},
            "urls_snippet_guard": urls_n, "stage_min_coverage": stage_min,
            "skip_refused": f"the sealed stage guard's min_coverage is {stage_min!r}, below "
                            f"{REQUIRED_STAGE_COVERAGE}: only a complete stage copy lets the "
                            "snippet guard be skipped"}}
    by_url: dict[str, list[str]] = {}
    for i, u in urls.items():
        by_url.setdefault(LC.url_key(u), []).append(i)
    skip: set[str] = set()
    unmeasured = below = other = ruled_used = 0
    fallback_examples: list[str] = []
    for key, ids in by_url.items():
        ok = True
        for i in ids:
            e = entries.get(i)
            if e is None:
                unmeasured += 1
                ok = False
            elif LC.url_key(e["url"]) != key:
                other += 1
                ok = False
            elif e["coverage"] < LC.MIN_COVERAGE and i in ruled_ids:
                ruled_used += 1                  # T24: this one id is ruled covered
            elif e["coverage"] < LC.MIN_COVERAGE:
                below += 1
                ok = False
        if ok:
            skip.update(ids)
        elif len(fallback_examples) < NAMED:
            fallback_examples.append(key)
    summary = {**base, "block": {"file": manifest["line_coverage_file"],
                                 "sha256": manifest["line_coverage_sha256"],
                                 "ids": len(entries)},
               "urls_skipped": len({LC.url_key(urls[i]) for i in skip}), "ids_skipped": len(skip),
               "urls_snippet_guard": len(by_url) - len({LC.url_key(urls[i]) for i in skip}),
               "ids_unmeasured": unmeasured, "ids_below_min": below, "ids_ruled_skip": ruled_used,
               "ids_measured_at_other_url": other, "snippet_guard_examples": fallback_examples}
    return {"skip_ids": frozenset(skip), "summary": summary}


def unwritten_url_keys(manifest: dict, written_ids) -> set[str]:
    """Task 888 (D2): the URL keys of the sealed stage rows the drain does NOT write (dropped or
    excluded before `rows`). Line coverage is measured against every stage row, so a URL in this
    set cannot skip the snippet guard."""
    written = {str(i) for i in written_ids}
    named = pathlib.Path(manifest["pack_chunks"])
    chunks = named if PSP._text_shards(str(named)) else _stage_dir(manifest) / "chunks"
    return {LC.url_key(r["url"]) for r in PSP.read_records(str(chunks))
            if str(r["point_id"]) not in written}


# ------------------------------------------------------------------ the groups
def pack_groups(rows: dict[str, dict], family_provenance: str) -> tuple[dict, int]:
    """The URL groups the pack WRITES: `({group key: {url, rows}}, cards skipped)`. A pack card
    opens no group. Raises `QueueError` (the whole pack) for a row that names no document or whose
    `provenance` is not the family's served value (`URL_GROUP_FAMILIES`). Reads nothing."""
    Q = _q()
    groups: dict[str, dict] = {}
    problems: list[str] = []
    cards = 0
    for pid, pl in rows.items():
        pl = RD.as_canonical_row(pl)
        if Q._is_card_row(pl):
            cards += 1
            continue
        if document_url_key(pl.get("url")) is None:
            problems.append(f"{pid}: url {pl.get('url')!r} names no document")
            continue
        if isinstance(family_provenance, ScopeRule):   # task 1038: the group is the url scope
            gk = family_provenance.group_key(pl.get("url"))
            if gk is None:
                problems.append(f"{pid}: url {pl.get('url')!r} is outside the family scope "
                                f"{family_provenance}")
                continue
            g = groups.setdefault(gk, {"url": pl["url"], "urls": set(), "rows": 0})
            g["urls"].add(pl["url"])
            g["rows"] += 1
            continue
        prov = (pl.get("provenance") or "").strip()  # canonical row
        if prov != family_provenance:
            problems.append(f"{pid}: provenance {prov!r}, the family is served as "
                            f"{family_provenance!r}")
            continue
        g = groups.setdefault(US.payload_group_key(pl), {"url": pl["url"], "rows": 0})
        g["rows"] += 1
    if problems:
        raise Q.QueueError(f"url-group supersede: {len(problems)} pack row(s) cannot be grouped: "
                           + "; ".join(problems[:NAMED]))
    return groups, cards


def compute(rows: dict[str, dict], groups: dict, cards: int, family_provenance: str, *,
            twin_url: str, collection: str) -> dict:
    """`rows` = `{point id: payload}` the pack WRITES, `groups`/`cards` = `pack_groups` of them.
    Returns the superseded set: `{ids, urls, groups, pack_rows, pack_cards_skipped,
    superseded_groups, kept_other_family}`. A served row at a pack URL is a victim when it carries
    the family's `provenance`; one with none cannot be classified and refuses the pack (a
    `QueueError`). Reads the twin only."""
    Q = _q()
    pack_ids = {Q._canonical_point_id(i) for i in rows}
    if isinstance(family_provenance, ScopeRule):
        return _compute_scope(rows, groups, cards, family_provenance, pack_ids,
                              twin_url=twin_url, collection=collection)
    members = Q.served_group_members(twin_url, collection,
                                     sorted({g["url"] for g in groups.values()}))
    ids: list[str] = []
    urls: dict[str, str] = {}
    unclassified: list[str] = []
    other_family = 0
    hit_groups = 0
    for gkey, g in groups.items():
        hit = False
        for mid, pl in members.get(gkey, []):
            pl = RD.as_canonical_row(pl)          # served_group_members decodes already; idempotent
            if mid in pack_ids or Q._is_card_row(pl):
                continue
            prov = (pl.get("provenance") or "").strip()  # canonical row
            if not prov:
                unclassified.append(f"{mid} at {pl.get('url')!r}")
            elif prov == family_provenance:
                ids.append(mid)
                urls[mid] = pl.get("url") or g["url"]
                hit = True
            else:
                other_family += 1
        hit_groups += hit
    if unclassified:
        raise Q.QueueError(
            f"url-group supersede: {len(unclassified)} served row(s) at the pack's urls carry no "
            "`provenance`, so they cannot be called the same family or another one (a refresh "
            "would keep both or delete another family's row): " + "; ".join(unclassified[:NAMED]))
    return {"ids": sorted(ids), "urls": urls, "groups": len(groups),
            "pack_rows": sum(g["rows"] for g in groups.values()), "pack_cards_skipped": cards,
            "superseded_groups": hit_groups, "kept_other_family": other_family}


def _compute_scope(rows: dict[str, dict], groups: dict, cards: int, rule: ScopeRule,
                   pack_ids: set, *, twin_url: str, collection: str) -> dict:
    """`compute` for a `ScopeRule` family: the victim set is every served non-card row whose url is
    in a scope the pack writes and that is not one of the pack's ids. Rows are asked for by keyed
    url (`rule.candidates`, then `rule.chase` until it names none); a row found at a candidate url
    whose scope the pack does not write is counted in `kept_other_family`, never deleted. Reads the
    twin only."""
    Q = _q()
    asked: set[str] = set()
    frontier = rule.candidates({u for g in groups.values() for u in g["urls"]})
    seen: dict[str, dict] = {}
    while frontier:
        asked.update(frontier)
        got = Q.served_group_members(twin_url, collection, frontier)
        found: list[str] = []
        for items in got.values():
            for mid, pl in items:
                pl = RD.as_canonical_row(pl)
                if mid not in seen:
                    seen[mid] = pl
                    found.append(pl.get("url") or "")
        frontier = [u for u in rule.chase(found, asked) if u not in asked]
    ids: list[str] = []
    urls: dict[str, str] = {}
    other = 0
    hit: set[str] = set()
    for mid, pl in seen.items():
        if mid in pack_ids or Q._is_card_row(pl):
            continue
        gk = rule.group_key(pl.get("url"))
        if gk is not None and gk in groups:
            ids.append(mid)
            urls[mid] = pl.get("url") or ""
            hit.add(gk)
        else:
            other += 1
    return {"ids": sorted(ids), "urls": urls, "groups": len(groups),
            "pack_rows": sum(g["rows"] for g in groups.values()), "pack_cards_skipped": cards,
            "superseded_groups": len(hit), "kept_other_family": other,
            "asked_urls": len(asked)}


def word_coverage(rows: dict[str, dict], urls: dict[str, str], *, twin_url: str,
                  collection: str, contract_path: pathlib.Path,
                  coverage_url=None) -> dict:
    """(b): the content packs' word-coverage guard (`validate_tombstone_word_coverage`) over the
    pack's non-card rows against the superseded rows, SKIP PER URL (task 876, ADR-114 am.2).

    A url whose new rows carry fewer snippet words than the floor of the rows they replace is
    listed in `skipped` (`url`, `doc_key`, `new_words`, `new_rows`, `old_words`, `ratio`, `floor`);
    the caller leaves its old rows in place and writes none of its pack rows. When the skipped
    urls are more than `SKIP_SHARE_MAX` of the urls checked the whole pack is refused (a
    `QueueError` that names the urls): a minority of odd documents is skipped, a majority is a
    wrong pack. Returns `{urls_checked, floor, violations (the number skipped), skipped,
    skip_share, skip_share_max, sees}`."""
    Q = _q()
    if coverage_url is not None:     # task 1038: compare per BASE anchor (windows vs the bare row)
        rows = {pid: {**pl, "url": coverage_url(pl["url"])} if pl.get("url") else pl
                for pid, pl in rows.items()}
        urls = {i: coverage_url(u) for i, u in urls.items()}
    floor = Q.DEFAULT_TOMBSTONE_WORD_COVERAGE_FLOOR
    if contract_path.exists():
        floor = json.loads(contract_path.read_text()).get("tombstone_word_coverage_floor", floor)
    points = [{"payload": pl} for pl in rows.values() if not Q._is_card_row(pl)]
    tombs = [{"id": i, "url": u, "reason": "superseded"} for i, u in sorted(urls.items())]
    rep = Q.validate_tombstone_word_coverage(points, tombs, target_url=twin_url,
                                             collection=collection, floor=floor)
    skipped = [{"url": v["url"], "doc_key": v["doc_key"], "new_words": v["kept_words"],
                "new_rows": v["kept_rows"], "old_words": v["old_words"], "ratio": v["ratio"],
                "floor": floor} for v in rep["violations"]]
    share = (len(skipped) / rep["urls_checked"]) if rep["urls_checked"] else 0.0
    if share > SKIP_SHARE_MAX:
        named = [f"{v['url']} (new {v['new_words']} words in {v['new_rows']} row(s) vs old "
                 f"{v['old_words']}, ratio {v['ratio']})" for v in skipped[:NAMED]]
        raise Q.QueueError(
            f"url-group supersede: {len(skipped)} of {rep['urls_checked']} url(s) = {share:.1%} "
            f"would lose text (snippet words below {floor:.0%} of the rows they replace), over the "
            f"{SKIP_SHARE_MAX:.0%} skip ceiling -- refusing the whole pack rather than skipping a "
            "fifth of it: " + "; ".join(named))
    return {"urls_checked": rep["urls_checked"], "floor": floor, "violations": len(skipped),
            "skipped": skipped, "skip_share": round(share, 4), "skip_share_max": SKIP_SHARE_MAX,
            "sees": "snippet words only (<= 25 per row); not text cut inside a row"}


def apply_skips(cov: dict, rows: dict[str, dict], comp: dict, holds_skips: bool,
                coverage_url=None) -> dict:
    """Turn the skipped urls of `word_coverage` into the plan: the OLD rows of a skipped url are
    removed from `comp` (`ids`, `urls`), so none is deleted, and the PACK rows of it are named
    (`skipped_pack_ids`). Those must not be written: a pack row beside an old row is the doubled
    url the supersede exists to prevent. The caller says `holds_skips=True` only when its write
    path leaves exactly those ids out (the drain adds `skipped_pack_ids` to its `drop_ids`). A
    caller that does not (the default) gets a refusal here, before any write, that names the urls
    and says to rebuild the pack without them: failing closed, never a doubled url. Returns
    `{skipped_urls, skipped_old_rows, skipped_pack_ids}`."""
    Q = _q()
    keys = {s["doc_key"] for s in cov.get("skipped") or ()}
    if not keys:
        return {"skipped_urls": 0, "skipped_old_rows": 0, "skipped_pack_ids": []}

    def _key(u):          # the key word_coverage reported: the (coverage-mapped) document key
        return document_url_key(coverage_url(u) if coverage_url is not None else u)

    pack_ids = sorted(pid for pid, pl in rows.items()
                      if not Q._is_card_row(pl) and _key(pl.get("url")) in keys)
    if not holds_skips:
        raise Q.QueueError(
            f"url-group supersede: {len(keys)} url(s) fall below the word-coverage floor and are "
            f"to be skipped, but this write path does not hold their {len(pack_ids)} pack row(s) "
            "out of the write (the drain writes every row that is not in `drop_ids`); writing "
            "them would leave both generations at the url. Rebuild the pack without these url(s): "
            + "; ".join(f"{s['url']} (ratio {s['ratio']})" for s in cov["skipped"][:NAMED]))
    old = [i for i in comp["ids"] if _key(comp["urls"][i]) in keys]
    for i in old:
        del comp["urls"][i]
    comp["ids"] = [i for i in comp["ids"] if i not in set(old)]
    return {"skipped_urls": len(keys), "skipped_old_rows": len(old), "skipped_pack_ids": pack_ids}


# ------------------------------------------------------------------ the export
#: The repo checkout this module lives in: an export dir inside it is not durable storage.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
#: Filesystems that do not survive a reboot.
VOLATILE_FSTYPES = ("tmpfs", "ramfs", "devtmpfs")


def _fstype_of(path: pathlib.Path) -> str | None:
    """The filesystem type of the mount that holds `path` (longest mount-point prefix in
    `/proc/self/mountinfo`), or None when it cannot be read."""
    best, fstype = -1, None
    try:
        with open("/proc/self/mountinfo", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                mount = parts[4].replace("\\040", " ")
                kind = parts[parts.index("-") + 1]
                prefix = mount.rstrip("/")
                if (str(path) == mount or str(path).startswith(prefix + "/")) and len(prefix) > best:
                    best, fstype = len(prefix), kind
    except (OSError, ValueError, IndexError):
        return None
    return fstype


def volatile_reason(d: pathlib.Path) -> str | None:
    """Why `d` is not durable storage, or None: under /tmp, on tmpfs/ramfs, or inside the repo
    checkout (a clean or a re-clone would take the restore copy with it)."""
    d = d.resolve()
    tmp = pathlib.Path("/tmp").resolve()
    if d == tmp or tmp in d.parents:
        return "lies under /tmp"
    fs = _fstype_of(d)
    if fs in VOLATILE_FSTYPES:
        return f"is on a {fs} filesystem"
    if d == REPO_ROOT or REPO_ROOT in d.parents:
        return "lies inside the repo checkout"
    return None


def export_dir(queue_dir: pathlib.Path | None = None) -> pathlib.Path:
    """The operator setting `removed_rows_export_dir`: set, existing, durable (not under /tmp, not
    on tmpfs, not in the repo checkout) and not inside a promotion queue: neither the configured
    one nor `queue_dir`, the queue THIS drain reads (`--queue-dir`)."""
    Q = _q()
    configured = PATHS.operator_settings().removed_rows_export_dir
    if not configured:
        raise Q.QueueError("url-group supersede needs a durable export directory: set "
                           "AW_REMOVED_ROWS_EXPORT_DIR (or `removed_rows_export_dir` in the operator "
                           "overlay) to an existing directory on durable storage; no default")
    d = pathlib.Path(configured)
    if not d.is_dir():
        raise Q.QueueError(f"removed_rows_export_dir {d} is not an existing directory (create it "
                           "on durable storage; the drain does not invent one)")
    real = d.resolve()
    queues = [q.resolve() for q in (Q.live_queue_dir(), queue_dir) if q is not None]
    for queue in queues:
        if real == queue or queue in real.parents:
            raise Q.QueueError(f"removed_rows_export_dir {d} lies inside the promotion queue "
                               f"{queue}")
    why = volatile_reason(real)
    if why:
        raise Q.QueueError(f"removed_rows_export_dir {d} {why}: the restore copy must be on "
                           "durable storage")
    return d


def _encode_vectors(vec: dict) -> dict:
    """Named vectors as `supersede.encode_point_vectors` writes them (a sparse leg tagged)."""
    out = {}
    for name, v in (vec or {}).items():
        out[name] = ({"__sparse__": True, **v} if isinstance(v, dict) and "indices" in v else v)
    return out


def _write_rows(src_url: str, collection: str, ids: list[str], urls: dict[str, str],
                path: pathlib.Path, *, batch: int, sleep_s: float, allow_missing: bool
                ) -> tuple[dict | None, list[str]]:
    """The restore copy of the rows `ids` that `src_url`/`collection` HOLDS: one JSON line per row
    (`id`, `payload`, `vector`), gzip, written aside, fsynced, renamed, then hashed (raw `.gz`
    bytes) and verified by `verify_removed_rows_export` against that collection's own vector
    names and sizes. Reads by id in batches (a pause after each), never a scroll. An id the source
    does not hold refuses the export, unless `allow_missing` (prod: the twin may hold what prod
    does not). Returns `(export, ids written)`; `export` is None when nothing was held."""
    Q = _q()
    DRY.refuse(f"url-group supersede export {path}")
    tmp = path.with_name(path.name + ".part")
    present: list[str] = []
    with open(tmp, "wb") as raw:
        with deterministic_gzip_open(raw, compresslevel=9) as gz:
            for i in range(0, len(ids), batch):
                chunk = ids[i:i + batch]
                got = PSP._api(src_url, f"/collections/{collection}/points",
                               {"ids": chunk, "with_payload": True, "with_vector": True}) or []
                by_id = {Q._canonical_point_id(r["id"]): r for r in got}
                for want in chunk:
                    r = by_id.get(Q._canonical_point_id(want))
                    if r is None:
                        if allow_missing:
                            continue
                        raise Q.QueueError(f"url-group supersede export: {want} is no longer on "
                                           f"{collection} -- the plan is stale, refusing")
                    line = {"id": r["id"], "payload": r.get("payload") or {},
                            "vector": _encode_vectors(r.get("vector"))}
                    gz.write((json.dumps(line, sort_keys=True) + "\n").encode("utf-8"))
                    present.append(want)
                if sleep_s:
                    time.sleep(sleep_s)
        raw.flush()
        os.fsync(raw.fileno())
    if not present:
        os.remove(tmp)
        return None, []
    os.replace(tmp, path)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    export = {"path": str(path), "sha256": _sha256_file(path), "rows": len(present)}
    verified = _verify({"export": export}, present, urls, url=src_url, collection=collection)
    return {**export, "tombstoned_ids_covered": verified["tombstoned_ids_covered"]}, present


def write_export(twin_url: str, collection: str, ids: list[str], urls: dict[str, str],
                 path: pathlib.Path) -> dict:
    """The twin's restore copy (every id must be on the twin; a vanished row refuses)."""
    export, _present = _write_rows(twin_url, collection, ids, urls, path, batch=EXPORT_BATCH,
                                   sleep_s=0, allow_missing=False)
    return export


def write_prod_export(prod_url: str, collection: str, ids: list[str], urls: dict[str, str],
                      path: pathlib.Path) -> "tuple[dict | None, list[str]]":
    """PROD's own restore copy of the rows `ids` it HOLDS (task 773): read BY ID with the retrieve
    endpoint (`PROD_EXPORT_BATCH` per call, `PROD_READ_SLEEP_S` pause), never a scroll, verified
    against prod's own vector names and sizes. Prod payloads are normalised and its BM25 vectors
    pruned, so the twin export is not an exact restore source for prod. An id prod does not hold is
    skipped (goal state). Returns `(export or None, ids written)`."""
    return _write_rows(prod_url, collection, ids, urls, path, batch=PROD_EXPORT_BATCH,
                       sleep_s=PROD_READ_SLEEP_S, allow_missing=True)


def _verify(manifest_like: dict, ids: list[str], urls: dict[str, str], *, url: str,
            collection: str) -> dict:
    """`verify_removed_rows_export` over `ids` (the one verifier), against the vector names and
    sizes of `collection` at `url`."""
    Q = _q()
    tombs = [{"id": i, "url": urls[i]} for i in ids]
    out = Q.verify_removed_rows_export(
        manifest_like, tombs, dense_config=Q.fetch_collection_vector_config(url, collection),
        sparse_names=Q.fetch_collection_sparse_names(url, collection))
    out.pop("ids", None)
    out.pop("identities", None)
    return out


def reverify_export(plan: dict, *, twin_url: str, twin_collection: str) -> dict:
    """The twin export of `plan` must still be the file the plan recorded: sha256, rows, every id
    a complete restore row. Raises `QueueError`."""
    return _verify({"export": plan["export"]}, plan["ids"], plan["urls"], url=twin_url,
                   collection=twin_collection)


def reverify_before_write(plan: dict, *, twin_url: str, twin_collection: str) -> dict | None:
    """`reverify_export` for the moment before a side is written; None when nothing is to be
    deleted (no ids, so no export)."""
    if not plan["ids"]:
        return None
    return reverify_export(plan, twin_url=twin_url, twin_collection=twin_collection)


# ------------------------------------------------------------------ prod: presence and export
def prod_export_path(plan: dict) -> pathlib.Path:
    """The prod export, beside the twin export."""
    twin = pathlib.Path(plan["export"]["path"])
    return twin.with_name(twin.name.replace("-removed-rows", "-prod-removed-rows"))


def present_on(url: str, collection: str, ids: list[str]) -> list[str]:
    """The `ids` the collection holds, read by id (no payload, no vector; batches of
    `PROD_PRESENCE_BATCH` with a pause; never a scroll or a filter)."""
    Q = _q()
    held: set[str] = set()
    for i in range(0, len(ids), PROD_PRESENCE_BATCH):
        chunk = ids[i:i + PROD_PRESENCE_BATCH]
        got = PSP._api(url, f"/collections/{collection}/points",
                       {"ids": chunk, "with_payload": False, "with_vector": False})
        # A 200 without a `result` is NOT "the collection holds none" (task 2364 review, finding 1)
        if not isinstance(got, list):
            raise Q.QueueError(f"{collection}: the id lookup answered {type(got).__name__}, not a "
                               "list -- refusing to read that as 'none present'")
        asked = {Q._canonical_point_id(x) for x in chunk}
        back = {Q._canonical_point_id(r["id"]) for r in got}
        if back - asked:
            raise Q.QueueError(f"{collection}: the id lookup returned {len(back - asked)} id(s) "
                               "that were not asked for -- refusing")
        held |= back
        if PROD_READ_SLEEP_S:
            time.sleep(PROD_READ_SLEEP_S)
    return [i for i in ids if Q._canonical_point_id(i) in held]


def reverify_prod_export(plan: dict, *, prod_url: str, prod_collection: str) -> dict | None:
    """The prod export recorded in `plan["prod"]`: sha256, rows, every id it covers a complete
    restore row by PROD's vector names and sizes. None when prod held none of the ids."""
    prod = plan.get("prod")
    if not prod or not prod.get("export"):
        return None
    return _verify({"export": prod["export"]}, prod["present_ids"], plan["urls"], url=prod_url,
                   collection=prod_collection)


def prepare_prod(plan: dict, *, prod_url: str, prod_collection: str) -> dict:
    """Before the first prod delete: read what PROD holds of the superseded ids and export prod's
    own rows. First time: one pass by id (<= `PROD_EXPORT_BATCH` per call, a pause between), the
    export verified, counts and file written to the plan, the summary and the state. On a resume
    (`plan["prod"]` is set): the recorded export is re-verified, prod is read again by id and may
    hold only ids that were exported (anything else refuses). Sets `plan["prod"]["to_delete"]`,
    the ids to delete on prod: those it holds, measured."""
    Q = _q()
    prod = plan.get("prod")
    if prod is None:
        export, present = _write_rows(prod_url, prod_collection, plan["ids"], plan["urls"],
                                      prod_export_path(plan), batch=PROD_EXPORT_BATCH,
                                      sleep_s=PROD_READ_SLEEP_S, allow_missing=True)
        prod = {"requested": len(plan["ids"]), "present": len(present),
                "absent": len(plan["ids"]) - len(present), "export": export,
                "present_ids": present, "to_delete": present}
        plan["prod"] = prod
        plan["summary"]["prod"] = _prod_summary(prod)
        write_state(pathlib.Path(plan["artifact_root"]), plan, stage="deleting")
        return prod
    reverify_prod_export(plan, prod_url=prod_url, prod_collection=prod_collection)
    now = present_on(prod_url, prod_collection, plan["ids"])
    unexported = sorted(set(now) - set(prod["present_ids"]))
    if unexported:
        raise Q.QueueError(f"{plan['pack_id']}: prod holds {len(unexported)} superseded id(s) its "
                           f"export does not cover (e.g. {unexported[:3]}) -- refusing")
    prod["to_delete"] = now
    return prod


def _prod_summary(prod: dict) -> dict:
    """What the receipt and the state say about prod (no id lists)."""
    return {"requested": prod["requested"], "present": prod["present"], "absent": prod["absent"],
            "export": prod["export"]}


def reverify_generation_export(manifest: dict, tombstones: list[dict], *, twin_url: str,
                               twin_collection: str) -> dict | None:
    """Change 2: the pack deletes rows under `superseded_generation`, so its declared export
    (`manifest["export"]`) is re-verified (sha256, rows, every id a complete restore row)
    immediately before its PROD leg. None when no tombstone has that reason. Raises `QueueError`;
    the drain stops and the twin-write state stays."""
    Q = _q()
    if not any(t.get("reason") in Q.EXPORT_BACKED_REASONS for t in tombstones):
        return None
    out = Q.verify_removed_rows_export(
        manifest, tombstones,
        dense_config=Q.fetch_collection_vector_config(twin_url, twin_collection),
        sparse_names=Q.fetch_collection_sparse_names(twin_url, twin_collection))
    out.pop("ids", None)
    out.pop("identities", None)
    return out


# ------------------------------------------------------------------ state
def state_path(artifact_root: pathlib.Path, pack_id: str) -> pathlib.Path:
    return artifact_root / f"{pack_id}-url-group-supersede.json"


def write_state(artifact_root: pathlib.Path, plan: dict, *, stage: str) -> pathlib.Path:
    Q = _q()
    path = state_path(artifact_root, plan["pack_id"])
    Q.atomic_write_text(path, json.dumps({
        "schema": STATE_SCHEMA, "pack_id": plan["pack_id"], "manifest_sha": plan["manifest_sha"],
        "stage": stage, "at": Q.now(), "ids": plan["ids"], "urls": plan["urls"],
        "export": plan["export"], "prod": plan.get("prod"), "summary": plan["summary"],
        "skipped_pack_ids": plan.get("skipped_pack_ids", [])},
        indent=1) + "\n")
    return path


def read_state(artifact_root: pathlib.Path, pack_id: str, manifest_sha: str | None) -> dict | None:
    """The state a re-run must reuse, or None. Stages: `export-verified` (the export exists,
    nothing is deleted yet: a re-run plans afresh), `deleting` (written before the first delete:
    the twin may already have lost rows prod still holds, so a re-run of THIS manifest reuses the
    ids) and `applied`. A `deleting` state of ANOTHER manifest is a supersede that stopped halfway
    and a resealed manifest must not forget it: `QueueError`."""
    Q = _q()
    path = state_path(artifact_root, pack_id)
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Q.QueueError(f"{path} is unreadable ({exc}): a url-group supersede may have "
                           "stopped halfway; resolve by hand") from exc
    if rec.get("manifest_sha") == manifest_sha:
        return rec if rec.get("stage") in ("deleting", "applied") else None
    if rec.get("stage") != "deleting":
        return None
    raise Q.QueueError(
        f"{path} records a url-group supersede of an earlier manifest "
        f"({str(rec.get('manifest_sha'))[:12]}) that began deleting; the pack was resealed. "
        "Its old rows may be deleted on the twin and not on prod: resolve by hand")


def mark_applied(artifact_root: pathlib.Path, plan: dict) -> None:
    write_state(artifact_root, plan, stage="applied")


# ------------------------------------------------------------------ the drain's entry point
def plan_for_drain(manifest: dict, *, manifest_sha: str | None, joined: dict, drop_ids,
                   twin_url: str, artifact_root: pathlib.Path, write: bool,
                   contract_path: pathlib.Path, queue_dir: pathlib.Path | None = None,
                   holds_skips: bool = False) -> dict:
    """Everything before the locks: declaration, the family allowlist, the groups, the sealed
    pack's text guard (c), the word guard (b), the export (a) and the state. `write=False` (a dry
    run, a blocked gate) computes and checks everything and writes nothing: the summary then says
    what it would write. `queue_dir` = the queue this drain reads (the export dir may not lie in
    it). `holds_skips`: the caller leaves `plan["skipped_pack_ids"]` out of its write (default
    False: a url that fails guard (b) then refuses the pack, see `apply_skips`). Returns
    `{pack_id, manifest_sha, ids, urls, export, prod, summary, skipped_pack_ids}`; raises
    `QueueError` to refuse."""
    Q = _q()
    declared(manifest)
    served_provenance = family_provenance(manifest)       # before any read or write
    pack_id = manifest["pack_id"]
    collection = manifest["collection"]
    rows = {pid: v[2] for pid, v in joined.items() if pid not in drop_ids}
    groups, cards = pack_groups(rows, served_provenance)
    guard = text_guard(sealed_guard_block(manifest), url_groups=len(groups))
    state = read_state(artifact_root, pack_id, manifest_sha)
    if state is not None:
        plan = {"pack_id": pack_id, "manifest_sha": manifest_sha, "ids": state["ids"],
                "urls": state["urls"], "export": state["export"], "prod": state.get("prod"),
                "skipped_pack_ids": state.get("skipped_pack_ids", []),
                "artifact_root": str(artifact_root)}
        if plan["skipped_pack_ids"] and not holds_skips:   # same rows held out as the first run
            raise Q.QueueError(f"url-group supersede resume: {len(plan['skipped_pack_ids'])} pack "
                               "row(s) of skipped url(s) would be written; this write path does "
                               "not hold them out")
        if plan["ids"]:
            reverify_export(plan, twin_url=twin_url, twin_collection=collection)
        plan["summary"] = {**state["summary"], "resumed_from_state": state.get("stage"),
                           "export": state["export"]}
        return plan
    comp = compute(rows, groups, cards, served_provenance, twin_url=twin_url,
                   collection=collection)
    qdir = queue_dir if queue_dir is not None else Q.live_queue_dir()
    lcp = (line_coverage_plan(manifest, comp["urls"],
                              pack_dir=None if qdir is None else pathlib.Path(qdir) / pack_id)
           if comp["ids"]
           else {"skip_ids": frozenset(), "summary": None})
    if lcp["skip_ids"]:       # task 888 D2: coverage saw rows this drain does not write
        partial = unwritten_url_keys(manifest, rows)
        kept = frozenset(i for i in lcp["skip_ids"] if LC.url_key(comp["urls"][i]) not in partial)
        if kept != lcp["skip_ids"] and lcp["summary"] is not None:
            lcp["summary"] = {**lcp["summary"], "ids_skip_withdrawn_unwritten_rows":
                              len(lcp["skip_ids"]) - len(kept)}
        lcp = {**lcp, "skip_ids": kept}
    snippet_urls = {i: u for i, u in comp["urls"].items() if i not in lcp["skip_ids"]}
    cov_url = served_provenance.coverage_url if isinstance(served_provenance, ScopeRule) else None
    cov = (word_coverage(rows, snippet_urls, twin_url=twin_url, collection=collection,
                         contract_path=contract_path, coverage_url=cov_url)
           if snippet_urls else {"urls_checked": 0, "violations": 0})
    skips = apply_skips(cov, rows, comp, holds_skips, cov_url)
    summary = {"mode": URL_GROUP, "family": manifest["family"],
               "served_provenance": str(served_provenance), "pack_rows": comp["pack_rows"],
               "pack_urls": comp["groups"], "pack_cards_skipped": comp["pack_cards_skipped"],
               "superseded_rows": len(comp["ids"]), "superseded_urls": comp["superseded_groups"] - skips["skipped_urls"],
               "kept_other_family": comp["kept_other_family"], "text_guard": guard,
               "word_coverage": cov, "line_coverage": lcp["summary"],
               "word_coverage_skipped_urls": skips["skipped_urls"],
               "word_coverage_skipped_old_rows": skips["skipped_old_rows"],
               "word_coverage_skipped_pack_rows": len(skips["skipped_pack_ids"]),
               "word_coverage_skip_ceiling": SKIP_SHARE_MAX, "export": None}
    plan = {"pack_id": pack_id, "manifest_sha": manifest_sha, "ids": comp["ids"],
            "urls": comp["urls"], "export": None, "prod": None, "summary": summary,
            "skipped_pack_ids": skips["skipped_pack_ids"]}
    if not comp["ids"]:
        return plan
    path = export_dir(queue_dir) / f"{pack_id}-{(manifest_sha or 'nosha')[:12]}-removed-rows.jsonl.gz"
    if not write or DRY.active():
        summary["export"] = {"would_write": str(path), "rows": len(comp["ids"])}
        return plan
    plan["export"] = write_export(twin_url, collection, comp["ids"], comp["urls"], path)
    summary["export"] = plan["export"]
    plan["artifact_root"] = str(artifact_root)
    write_state(artifact_root, plan, stage="export-verified")
    return plan


# ------------------------------------------------------------------ the write
def delete_superseded(role: str, url: str, collection: str, plan: dict, *, twin_url: str,
                      twin_collection: str, batch: int) -> int:
    """Delete the superseded rows of one side, each batch verified gone, and return the number
    DELETED. The twin export is re-verified FIRST, on both sides: no verified restore copy, no
    delete. Twin: every id (all were read for the export). Prod: `prepare_prod` first (presence by
    id, prod's own export verified, state written), then only the ids prod holds; an id prod never
    held is `absent`, not counted as deleted. Idempotent (an id already gone is a no-op), which
    is what lets a re-run finish prod."""
    ids = plan["ids"]
    if not ids:
        return 0
    Q = _q()
    if not plan.get("export"):
        raise Q.QueueError(f"{plan['pack_id']}/{role}: superseded rows to delete and no export "
                           "recorded -- refusing")
    reverify_export(plan, twin_url=twin_url, twin_collection=twin_collection)
    if role == "twin":     # BEFORE the first delete: a re-run must know the twin may be ahead
        write_state(pathlib.Path(plan["artifact_root"]), plan, stage="deleting")
        return PD.delete_points_verified(url, collection, ids, batch=batch)
    prod = prepare_prod(plan, prod_url=url, prod_collection=collection)
    return PD.delete_points_verified(url, collection, prod["to_delete"], batch=batch)

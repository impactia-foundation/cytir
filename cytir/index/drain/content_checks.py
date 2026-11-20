"""Drain content checks: row, tombstone, card, soft-404, binary, duplicate and generation checks.

Task 2893 (RULING 80, drain split step 6): moved verbatim out of `cytir/index/promote_queue.py`, which
re-exports every name here. Reads only; no Qdrant write. A moved body that calls a name a test
patches on the facade reads it at call time through `_Q` (the facade module).
"""
from __future__ import annotations

import datetime
import gzip
import hashlib
import json
import os
import pathlib
import re
import socket
import urllib.error
import urllib.parse
import uuid

from cytir.chunk import registry as CHREG
from cytir.core.canon import document_url_key, strip_tracking_url
from cytir.core.prose_guard import binary_content
from cytir.index import card_waivers as CW
from cytir.index import change_log as CL
from cytir.index import news_window_filter as NWF
from cytir.index import point_delete as PD
from cytir.index import promote_sealed_pack as PSP
from cytir.index import row_decode as RD
from cytir.index import supersede as US
from cytir.index.drain.common import QueueError, TransientReadError
from cytir.index.drain.queue_scope import GATE_CONTRACT_PATH
from cytir.index.drain.receipts import HELD_CARD_OVERWRITE
from cytir.sources import soft404 as SOFT404


# ----------------------------------------------------- content packs + tombstones
#: Task 641 (2026-09-23, coordinator's ruling): "packs declare their own
#: deletes" -- generic, and the same shape ADR-112/#310 needs for public
#: daily update packs. A `content_pack`-kind manifest carries ALREADY-
#: PROJECTED qdrant points (the producing lane resolved identity/licence/
#: embedding itself -- news's `materialize_news_pack.py` in particular) plus
#: an OPTIONAL `tombstones` file: point id + url + reason
#: (`superseded`|`aged_out`|`takedown`), one per line. Applied as ONE
#: transaction per side: upsert the points, then delete the tombstones,
#: both twin first then prod, both verified.
#: `superseded_generation` (ADR-114 am.1, ruling 26 item 21a): a PURE tombstone pack
#: that removes an OLD generation of rows whose new rows are already served at the
#: same URLs. Not a takedown (it never enters the public dump's takedown list) and
#: not `superseded` (that needs replacement points in the pack). Its checks are
#: `check_generation_siblings` (per id, on the target) and
#: `check_superseded_generation_pack` (the pack's shape and its export).
SUPERSEDED_GENERATION = "superseded_generation"
#: Task 767 (index-text defects of 2026-10-08). Two more PURE tombstone reasons, each re-derived
#: mechanically from the STORED row (never trusted from the pack), each with the same restore
#: export as `superseded_generation`:
#:   * `canonical_duplicate` -- the row's url differs from `strip_tracking_url(url)` (a
#:     `?authuser=N` copy of a page) and a row of the same family is served at the canonical url.
#:   * `binary_content` -- the stored snippet fails `cytir.core.prose_guard.binary_content` (a binary
#:     file read as text).
CANONICAL_DUPLICATE = "canonical_duplicate"
BINARY_CONTENT = "binary_content"
#: Task 1363 (ADR-123 am.1, ruling 61 Q2): `soft_404` -- the stored title and snippet match the
#: declared soft-404 detector (`cytir.sources.soft404.detect`), a federal error page served as content.
SOFT_404 = "soft_404"
#: Task 842 (ADR-123, ruling 29 T12): `restore_rollback` undoes a RESTORE pack (rows the
#: operator put back from an export). OPERATOR-ONLY (no composer in `cytir.ops` offers it), PURE and
#: export-backed like the three above, never in the takedown list, and the ONLY export-backed
#: reason that may delete a CARD. Its per-id rule is `check_restore_rollback`.
RESTORE_ROLLBACK = "restore_rollback"
#: The reasons that make a PURE, export-backed tombstone pack: one reason per pack.
EXPORT_BACKED_REASONS = (SUPERSEDED_GENERATION, CANONICAL_DUPLICATE, BINARY_CONTENT,
                         RESTORE_ROLLBACK, SOFT_404)
TOMBSTONE_REASONS = ("superseded", "aged_out", "takedown") + EXPORT_BACKED_REASONS
#: Same env var `news_refresh_delta.sh`/`news_refresh_cron.sh` read
#: for the purge/aged-out horizon (Tim's 2026-08-11 ruling) -- imported by
#: NAME, not a second copy of the default.
RETAIN_MONTHS_ENV = "AW_NEWS_RETAIN_MONTHS"
DEFAULT_RETAIN_MONTHS = 36


def load_content_pack_points(path: str) -> list[dict]:
    """The pack's own already-built qdrant points ({id, payload, vector}),
    read as-is -- no projection, no re-minting; the producing lane already
    did that and this driver trusts its receipt the same way it trusts a
    sealed pack's vectors."""
    opener = gzip.open if path.endswith(".gz") else open
    points = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                points.append(json.loads(line))
    return points


def load_tombstones(path: str) -> list[dict]:
    with open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _verify_pack_file(path: str, expected_sha256: str | None, label: str) -> None:
    """Bug found live, task 641, 2026-09-23: this hashed the file's RAW
    on-disk bytes, but every content_pack producer (task 640's news
    composer, checked against three real packs) declares `*_sha256` as the
    sha256 of the DECOMPRESSED content -- a `.gz`'s compressed bytes differ
    from that on every run (gzip's own timestamp/OS/mtime header fields),
    so this refused three genuinely-correct, byte-verified packs as
    'does not match what was declared ready'. Decompress `.gz` first, same
    as `load_content_pack_points` already does when it actually reads the
    file -- the check must hash what gets USED, not what sits on disk."""
    if not expected_sha256:
        return
    h = hashlib.sha256()
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    got = h.hexdigest()
    if got != expected_sha256:
        raise QueueError(
            f"{label} {path}: sha256 {got[:12]} != manifest's "
            f"{expected_sha256[:12]} -- refusing a file that does not match "
            "what was declared ready")


#: An id a purge means to delete that is ALREADY GONE has reached the goal
#: state, not failed to reach it (coordinator's ruling, task 664,
#: 2026-09-28, news-aged-out-weekly-2026W39: 38 of 349 ids missing --
#: "for an aged-out purge, an id that is already gone is the goal state").
#: Missing ids are dropped and recorded (never silently vanished --
#: `missing_ids`/`missing_share` in the report, same shape `coverage_
#: skipped` above already uses), and the SURVIVING (present) tombstones
#: are still reason-validated and applied normally. The whole pack still
#: refuses when MORE than this share of ITS tombstones are missing -- that
#: shape is evidence something is systemically wrong with what this pack
#: thinks it should be deleting (see task 668
#: tombstone-mismatch.md: a 99.95%-missing pack is NOT this case and stays
#: refused), not an ordinary "the purge ran twice" idempotency gap.
TOMBSTONE_MISSING_SHARE_MAX = 0.20


def _norm_source_type(value) -> str:
    return (value or "").strip().lower() if isinstance(value, str) else ""


def _is_card_row(payload: dict | None) -> bool:
    """`supersede.is_card` on a payload: type `entity_card`, or carrying
    `entity_kind`/`qid` (a wikidata entity row typed `reference`, `official`,
    `news`... -- task 698 third review R2)."""
    pl = payload or {}
    return US.is_card(pl.get("source_type"), pl.get("entity_kind"), pl.get("qid"))


#: Payload fields the sibling read returns for every row of a url group.
GENERATION_SIBLING_FIELDS = ("url", "source_type", "entity_kind", "qid", "provenance", "content_digest")
#: Urls per filtered scroll of the sibling read (each url is matched under its
#: stored spellings, `supersede.url_variants`).
GENERATION_SIBLING_URL_BATCH = 64


def served_group_members(url: str, collection: str, urls: list[str]
                         ) -> dict[str, list[tuple[str, dict]]]:
    """`{group key: [(point id, payload), ...]}` of the rows `collection` serves under
    any spelling of `urls`, keyed by `supersede.payload_group_key` (canonical url,
    `|card` suffix for a card). A filtered scroll on the indexed `url` keyword
    (`supersede.served_payloads_by_url` is the same lookup but drops the ids). TWIN ONLY:
    the drain validates on the twin, and no scroll ever runs on prod. Every payload is the
    CANONICAL row (`row_decode`, the collection's declared profile): a coded twin reads like a raw
    one (task 855)."""
    decode = RD.reader(collection)
    variants: dict[str, set[str]] = {}
    for u in urls:
        variants[u] = US.url_variants(u)
    ordered = sorted(variants)
    out: dict[str, list[tuple[str, dict]]] = {}
    for i in range(0, len(ordered), GENERATION_SIBLING_URL_BATCH):
        vals = sorted(set().union(*(variants[u] for u in
                                    ordered[i:i + GENERATION_SIBLING_URL_BATCH])))
        offset = None
        while True:
            body = {"limit": 1024, "with_payload": list(GENERATION_SIBLING_FIELDS),
                    "with_vector": False,
                    "filter": {"must": [{"key": "url", "match": {"any": vals}}]}}
            if offset is not None:
                body["offset"] = offset
            res = PSP._api(url, f"/collections/{collection}/points/scroll", body) or {}
            for p in res.get("points", []):
                pl = decode(p.get("payload"))
                out.setdefault(US.payload_group_key(pl), []).append(
                    (_canonical_point_id(p["id"]), pl))
            offset = res.get("next_page_offset")
            if offset is None:
                break
    return out


def check_generation_siblings(tombstones: list[dict], stored_by_id: dict[str, dict], *,
                              target_url: str, collection: str,
                              export_ids: set[str]) -> list[str]:
    """`superseded_generation`, the per-id rule (ADR-114 am.1). Returns the problems, one
    line per failing id; the caller refuses the WHOLE pack on any (never a partial apply).
    Every tombstoned id must: be a row of the document its tombstone names (same
    `document_url_key` as the stored row); not be a card (`supersede.is_card`); carry a
    `provenance` (its family); and have, on the target, at least one OTHER served row in the
    same canonical url group (`supersede.payload_group_key`, so never a card) with the same
    `provenance`, that is not in the tombstone set and not in `export_ids`. The export holds
    EVERY old-generation row (`verify_removed_rows_export`), so a row outside it is the newer
    generation; another old row of the same family would otherwise pass as a "survivor".
    `stored_by_id` is the ids' stored payloads, already read by the caller (present ids only),
    read as canonical rows (`row_decode.as_canonical_row`: a coded `provenance` is its family, task 855)."""
    from cytir.index import promote_queue as _Q  # noqa: PLC0415 -- patched names resolve at call time
    stored_by_id = RD.as_canonical_rows(stored_by_id)
    problems: list[str] = []
    tomb_ids = {_canonical_point_id(t["id"]) for t in tombstones}
    checked: list[dict] = []
    for t in tombstones:
        stored = stored_by_id.get(t["id"]) or {}
        tkey = document_url_key(t.get("url"))
        if tkey is None:
            problems.append(f"{t['id']}: superseded_generation url {t.get('url')!r} names no "
                            "document (empty or host-less)")
        elif document_url_key(stored.get("url")) != tkey:
            problems.append(f"{t['id']}: the stored row's url {stored.get('url')!r} is not the "
                            f"tombstone's url {t.get('url')!r} (same-document key differs)")
        elif _is_card_row(stored):
            problems.append(f"{t['id']}: the stored row is an entity card -- a card is never "
                            "tombstoned under superseded_generation")
        elif not (stored.get("provenance") or "").strip():  # canonical row
            problems.append(f"{t['id']}: the stored row carries no `provenance` (family), so no "
                            "same-family survivor can be named")
        else:
            checked.append(t)
    if checked:
        members = _Q.served_group_members(
            target_url, collection, sorted({stored_by_id[t["id"]]["url"] for t in checked}))
        for t in checked:
            stored = stored_by_id[t["id"]]
            gkey = US.payload_group_key(stored)
            if not any(mid not in tomb_ids and mid not in export_ids and not _is_card_row(pl)
                       and pl.get("provenance") == stored.get("provenance")  # canonical row
                       for mid, pl in members.get(gkey, [])):
                problems.append(
                    f"{t['id']}: no surviving row at {t.get('url')!r} with provenance "
                    f"{stored.get('provenance')!r} outside the tombstone set and outside the removed-rows "  # canonical row
                    "export (a newer-generation row) -- deleting it would empty the page, not "
                    "retire an old generation")
    return problems


def check_canonical_duplicates(tombstones: list[dict], stored_by_id: dict[str, dict], *,
                               target_url: str, collection: str,
                               export_ids: set[str]) -> list[str]:
    """`canonical_duplicate`, the per-id rule (task 767). One line per failing id; the caller
    refuses the WHOLE pack on any. Every tombstoned id must: be a row of the document its
    tombstone names; not be a card; carry a `provenance`; have a stored url that is a VARIANT
    (`strip_tracking_url(url) != url`, re-derived here, never trusted from the pack); and have,
    on the target, a SURVIVOR: another served row, not a card, of the same `provenance`, outside
    the tombstone set and outside the export, in the same canonical group, whose url is either the
    CANONICAL spelling or the spelling the tombstone declares as `survivor_url` (a page that was
    only ever captured as variants keeps ONE spelling; the declared one is read and must serve),
    AND that survivor must hold THIS row's text: a survivor row with the same `content_digest`
    (task 773: for 631 of the first 11,441 rows the survivor did not hold the row's text, and on 17
    pages it held under half the words). A row with no `content_digest` is refused: nothing proves
    its text is kept. A page is never emptied and never thinned: no survivor of the text, no delete.
    Rows are read as canonical rows (`row_decode`): the digest compare is on the 32-hex key, the
    same on a raw and on a coded twin (task 855)."""
    from cytir.index import promote_queue as _Q  # noqa: PLC0415 -- patched names resolve at call time
    stored_by_id = RD.as_canonical_rows(stored_by_id)
    problems: list[str] = []
    tomb_ids = {_canonical_point_id(t["id"]) for t in tombstones}
    checked: list[dict] = []
    for t in tombstones:
        stored = stored_by_id.get(t["id"]) or {}
        surl = str(stored.get("url") or "")
        if document_url_key(t.get("url")) is None or document_url_key(surl) != document_url_key(t.get("url")):
            problems.append(f"{t['id']}: the stored row's url {surl!r} is not the tombstone's url "
                            f"{t.get('url')!r} (same-document key differs)")
        elif _is_card_row(stored):
            problems.append(f"{t['id']}: the stored row is an entity card -- never tombstoned")
        elif not (stored.get("provenance") or "").strip():  # canonical row
            problems.append(f"{t['id']}: the stored row carries no `provenance` (family)")
        elif strip_tracking_url(surl) == surl:
            problems.append(f"{t['id']}: the stored url {surl!r} is already canonical -- it is the "
                            "page, not a variant of it")
        elif not str(stored.get("content_digest") or "").strip():  # canonical row
            problems.append(f"{t['id']}: the stored row carries no `content_digest`, so no survivor "
                            "can be shown to hold its text")
        elif t.get("survivor_url") and t["survivor_url"] == surl:
            problems.append(f"{t['id']}: the declared survivor_url is the tombstoned url itself")
        else:
            checked.append(t)
    if checked:
        want: set[str] = set()
        for t in checked:
            want.add(strip_tracking_url(stored_by_id[t["id"]]["url"]))
            if t.get("survivor_url"):
                want.add(str(t["survivor_url"]))
        members = _Q.served_group_members(target_url, collection, sorted(want))
        for t in checked:
            stored = stored_by_id[t["id"]]
            canon = strip_tracking_url(stored["url"])
            ok_urls = {canon} | ({str(t["survivor_url"])} if t.get("survivor_url") else set())
            gkey = US.payload_group_key(stored)
            if not any(mid not in tomb_ids and mid not in export_ids and not _is_card_row(pl)
                       and pl.get("provenance") == stored.get("provenance")  # canonical row
                       and str(pl.get("url") or "") in ok_urls
                       and pl.get("content_digest") == stored.get("content_digest")  # canonical row
                       for mid, pl in members.get(gkey, [])):
                problems.append(
                    f"{t['id']}: no surviving row holding this row's text (same content_digest) at "
                    f"the canonical url {canon!r}"
                    + (f" or the declared survivor_url {t['survivor_url']!r}" if t.get("survivor_url") else "")
                    + f" with provenance {stored.get('provenance')!r} outside the tombstone set -- "  # canonical row
                    "deleting it would lose text or empty the page, not remove a duplicate")
    return problems


#: Task 773: `binary_content` removes a row ONLY on the url of a binary FILE: a `.pdf` url, or a host
#: that serves its PDFs at other paths (the three vocational-education sites). Text in the wrong
#: charset, spam and prose the test misreads are other defects with other reasons.
BINARY_FILE_URL_RE = re.compile(r"(?i)\.pdf(?:[?#]|$)")
BINARY_FILE_HOSTS = frozenset({"formazioneprof.ch", "berufsbildung.ch", "formationprof.ch"})


def is_binary_file_url(url: "str | None") -> bool:
    """True for a `.pdf` url or any url on `BINARY_FILE_HOSTS` (`www.` folded)."""
    u = str(url or "")
    host = (urllib.parse.urlsplit(u).hostname or "").lower()
    host = host[4:] if host.startswith("www.") else host
    return bool(BINARY_FILE_URL_RE.search(u)) or host in BINARY_FILE_HOSTS


def check_binary_content(tombstones: list[dict], stored_by_id: dict[str, dict]) -> list[str]:
    """`binary_content`, the per-id rule (task 767): the STORED row is the tombstone's document,
    is not a card, and its stored snippet fails the prose test (`cytir.core.prose_guard`), re-run
    here on the target's own text. A row that reads as prose is never deleted under this reason."""
    problems: list[str] = []
    for t in tombstones:
        stored = stored_by_id.get(t["id"]) or {}
        tkey = document_url_key(t.get("url"))
        if tkey is None or document_url_key(stored.get("url")) != tkey:
            problems.append(f"{t['id']}: the stored row's url {stored.get('url')!r} is not the "
                            f"tombstone's url {t.get('url')!r} (same-document key differs)")
        elif _is_card_row(stored):
            problems.append(f"{t['id']}: the stored row is an entity card -- never tombstoned")
        elif not is_binary_file_url(stored.get("url")):
            problems.append(f"{t['id']}: the stored url {stored.get('url')!r} is not a binary file "
                            "(.pdf or a PDF host); binary_content removes rows of binary files only")
        elif not binary_content(stored.get("snippet")):
            problems.append(f"{t['id']}: the stored snippet reads as prose "
                            f"({str(stored.get('snippet'))[:60]!r}); binary_content refuses it")
    return problems


def soft_404_second_gate(stored: dict) -> bool:
    """The independent gate of `soft_404` (as `is_binary_file_url` is for `binary_content`): the
    stored url is an error-page path, OR the title and the snippet each match the detector alone.
    The detector's length guard cannot apply to a short stored snippet, so a title that merely
    discusses errors would pass the first gate alone."""
    title, snippet = str(stored.get("title") or ""), str(stored.get("snippet") or "")
    return SOFT404.error_path(stored.get("url")) or bool(
        SOFT404.detect(title, "") and SOFT404.detect("", snippet))


def check_soft_404(tombstones: list[dict], stored_by_id: dict[str, dict]) -> list[str]:
    """`soft_404`, the per-id rule (task 1363): the STORED row is the tombstone's document, is not a
    card, its stored title and snippet match the declared detector
    (`cytir.sources.soft404.detect`), AND the second gate holds (`soft_404_second_gate`), all re-run
    here on the target's own text. A row the detector does not
    match is never deleted under this reason; the pack's own claim is not read."""
    problems: list[str] = []
    for t in tombstones:
        stored = stored_by_id.get(t["id"]) or {}
        tkey = document_url_key(t.get("url"))
        if tkey is None or document_url_key(stored.get("url")) != tkey:
            problems.append(f"{t['id']}: the stored row's url {stored.get('url')!r} is not the "
                            f"tombstone's url {t.get('url')!r} (same-document key differs)")
        elif _is_card_row(stored):
            problems.append(f"{t['id']}: the stored row is an entity card -- never tombstoned")
        elif SOFT404.detect(str(stored.get("title") or ""), str(stored.get("snippet") or "")) is None:
            problems.append(f"{t['id']}: the stored title/snippet "
                            f"({str(stored.get('title'))[:40]!r}, {str(stored.get('snippet'))[:40]!r}) "
                            "does not match the soft-404 detector; soft_404 refuses it")
        elif not soft_404_second_gate(stored):
            problems.append(f"{t['id']}: the stored url {stored.get('url')!r} is not an error-page path "
                            "and the title and the snippet do not each match the detector alone; "
                            "soft_404 refuses a page that only discusses errors")
    return problems


#: The payload keys that say WHICH row a stored row is. `check_restore_rollback` compares them.
ROLLBACK_IDENTITY_KEYS = ("source_type", "entity_kind", "qid", "register_id", "lei",
                          "content_digest")
#: The keys that say what ENTITY a row is. A row with none of them and no `content_digest` is its
#: url and nothing else: the identity check cannot tell it from any rewrite, so it is refused.
ROLLBACK_ENTITY_KEYS = ("entity_kind", "qid", "register_id", "lei")


def row_identity(payload: dict | None) -> tuple:
    """(same-document url key, then `ROLLBACK_IDENTITY_KEYS` in order) of a payload; an absent key
    reads as ''. `source_type` is normalized like `_norm_source_type`; `content_digest` is its
    canonical 32-hex key (`row_decode.digest_key`), so a pack row (64 hex) and a coded stored row
    (32 hex) of the same text agree (task 855). Export rows and stored rows go through this one
    function."""
    pl = payload or {}
    return (document_url_key(pl.get("url")),) + tuple(
        _norm_source_type(pl.get(k)) if k == "source_type"
        else RD.digest_key(str(pl.get(k) or "").strip()) if k == "content_digest"
        else str(pl.get(k) or "").strip()
        for k in ROLLBACK_IDENTITY_KEYS)


def identity_is_weak(identity: tuple) -> bool:
    """True when `row_identity` says nothing beyond the url: no `content_digest` and no entity key."""
    named = dict(zip(("url",) + ROLLBACK_IDENTITY_KEYS, identity))
    return not named["content_digest"] and not any(named[k] for k in ROLLBACK_ENTITY_KEYS)  # canonical row


def check_rolled_back_pack(manifest: dict, tombstones: list[dict],
                           export_identities: "dict[str, tuple]") -> list[str]:
    """`restore_rollback` against the pack it undoes (review 850 M1). `manifest["rollback_of"]` is
    resolved as a pack directory NEXT TO this pack's own (the same queue; the tombstones file's
    parent is the pack directory). That pack must: exist with a readable manifest and a verified
    points file; carry a `receipt.json` whose status is LANDED; hold every tombstoned id among its
    points; and its point row for the id must have the same `row_identity` as the export row. So
    the export is the restore pack's own row, never a copy taken just before the rollback.
    One line per failing id (the first three non-id lines end the check). Reads files only."""
    from cytir.index import promote_queue as _Q  # noqa: PLC0415 -- patched names resolve at call time
    name = str(manifest.get("rollback_of") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) or name in (".", ".."):
        return [f"rollback_of {name!r} is not a plain pack id"]
    if name == manifest.get("pack_id"):
        return ["rollback_of names this pack itself"]
    queue = pathlib.Path(manifest["tombstones"]).resolve().parent.parent
    pdir = queue / name
    if not (pdir / "manifest.json").is_file():
        return [f"rollback_of {name!r}: no pack directory with a manifest.json in {queue}"]
    try:
        status = json.loads((pdir / "receipt.json").read_text()).get("status")
    except (OSError, ValueError):
        status = None
    if status != "LANDED":
        return [f"rollback_of {name!r}: its receipt status is {status!r}, not LANDED -- there "
                "is nothing to roll back"]
    try:
        pm = json.loads((pdir / "manifest.json").read_text())
        _verify_pack_file(pm["pack_points"], pm.get("pack_points_sha256"), "rolled-back pack_points")
        rows = {_canonical_point_id(p["id"]): row_identity(p.get("payload"))
                for p in _Q.load_content_pack_points(pm["pack_points"])}
    except (OSError, ValueError, KeyError, QueueError) as exc:
        return [f"rollback_of {name!r}: its points cannot be read ({type(exc).__name__}: {exc})"]
    problems: list[str] = []
    for t in tombstones:
        cid = _canonical_point_id(t["id"])
        if cid not in rows:
            problems.append(f"{t['id']}: not a row of {name!r} -- a rollback deletes only the "
                            "rows its restore pack wrote")
        elif export_identities.get(cid) != rows[cid]:
            problems.append(f"{t['id']}: the export row is not {name!r}'s row for this id "
                            f"(export {export_identities.get(cid)!r} vs pack {rows[cid]!r})")
    return problems


def check_restore_rollback(tombstones: list[dict], stored_by_id: dict[str, dict],
                           export_identities: "dict[str, tuple]") -> list[str]:
    """`restore_rollback`, the per-id rule (task 842, ADR-123). One line per failing id; the caller
    refuses the WHOLE pack on any. The pack's export (`verify_removed_rows_export`) holds the rows
    the restore wrote; a rollback deletes them, and the export is what puts them back. So every
    tombstoned id must (1) be in that export (checked there, with a complete restore row), (2) name
    the export row's document in its tombstone url, and (3) be STILL that row on the target: the
    stored row's identity (`row_identity`) must equal the export row's. A row rewritten since the
    restore is another row; deleting it under this reason would lose what the export does not hold.
    A card is allowed here, and only here among the export-backed reasons."""
    stored_by_id = RD.as_canonical_rows(stored_by_id)
    problems: list[str] = []
    for t in tombstones:
        want = export_identities.get(_canonical_point_id(t["id"]))
        stored = stored_by_id.get(t["id"]) or {}
        if want is None:
            problems.append(f"{t['id']}: the export holds no prior row of this id -- no rollback "
                            "without the row it would delete")
        elif identity_is_weak(want) or identity_is_weak(row_identity(stored)):
            problems.append(f"{t['id']}: the row has no `content_digest` and no entity key "
                            "(entity_kind, qid, register_id, lei): its identity is its url "
                            "alone, which proves nothing -- refusing")
        elif document_url_key(t.get("url")) != want[0]:
            problems.append(f"{t['id']}: the tombstone url {t.get('url')!r} is not the export row's "
                            "document (same-document key differs)")
        elif row_identity(stored) != want:
            problems.append(f"{t['id']}: the stored row is not the export's row any more "
                            f"(stored {row_identity(stored)!r} vs export {want!r}) -- rewritten "
                            "since the restore; refusing")
    return problems


def check_superseded_generation_pack(manifest: dict, points: list[dict],
                                     tombstones: list[dict]) -> None:
    """An export-backed pack's shape (`EXPORT_BACKED_REASONS`): PURE (no points) and every
    tombstone of ONE reason. Raises `QueueError`; reads nothing."""
    reasons = sorted({str(t.get("reason")) for t in tombstones})
    if points:
        raise QueueError(
            f"a {'/'.join(reasons)} pack carries no points (found {len(points)}): it is a "
            "pure tombstone pack; replacements are served already")
    if len(reasons) != 1 or reasons[0] not in EXPORT_BACKED_REASONS:
        raise QueueError(
            f"an export-backed tombstone pack holds only that reason (exactly one of "
            f"{list(EXPORT_BACKED_REASONS)}); found {reasons} -- refusing the whole pack")
    if reasons[0] == RESTORE_ROLLBACK and not str(manifest.get("rollback_of") or "").strip():
        raise QueueError("a restore_rollback pack names the pack it undoes (`rollback_of`, a pack "
                         "id) in the manifest -- none found")


def verify_removed_rows_export(manifest: dict, tombstones: list[dict], *,
                               dense_config: dict, sparse_names: "list[str] | tuple[str, ...]"
                               ) -> dict:
    """The manifest's `export` -- `{path, sha256, rows}` -- is the restore copy of the rows a
    `superseded_generation` pack removes. Before any delete: the file exists, its raw bytes
    hash to the declared sha256, every line parses with an `id`, a `payload` and `vector`s,
    and EVERY tombstoned id is in it (task 574's lesson: an export that misses ids is not a
    rollback). For every tombstoned id the row must also be a COMPLETE restore source: each
    named vector of the collection (`dense_config`: name -> {size}; `sparse_names`), a dense
    vector of the declared size, a sparse vector whose `indices` and `values` arrays have the
    same length, a non-empty payload, and a payload url whose same-document key
    (`document_url_key`, the one the drain uses) equals the tombstone's. Any failing id
    refuses the pack, named. Raises `QueueError`; reads no collection.

    Note the export's sha256 hashes the RAW `.gz` BYTES as stored, unlike the other `*_sha256`
    fields of a pack, which hash the decompressed content (`pack.sha256_file`).

    Returns `{path, sha256, rows, tombstoned_ids_covered, ids}`; `ids` is the set of every
    canonical id in the export (the caller pops it: it is not for a receipt). A `restore_rollback`
    pack adds `identities` (tombstoned canonical id -> `row_identity` of its export row); the
    caller pops that too."""
    export = manifest.get("export")
    if not isinstance(export, dict) or not export.get("path") or not export.get("sha256"):
        raise QueueError("an export-backed tombstone pack declares its `export` "
                         "({path, sha256, rows}) in the manifest -- none found")
    path = pathlib.Path(export["path"])
    if not path.is_file():
        raise QueueError(f"declared export {path} does not exist")
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != str(export["sha256"]).lower():
        raise QueueError(f"declared export {path}: sha256 {h.hexdigest()[:12]} != manifest's "
                         f"{str(export['sha256'])[:12]}")
    wanted = {_canonical_point_id(t["id"]): document_url_key(t.get("url")) for t in tombstones}
    opener = gzip.open if str(path).endswith(".gz") else open
    ids: set[str] = set()
    identities: dict[str, tuple] = {}
    bad: dict[str, str] = {}
    rows = 0
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("id") or not row.get("vector") or "payload" not in row:
                raise QueueError(f"declared export {path}: row {rows + 1} lacks id/payload/vector")
            rid = _canonical_point_id(row["id"])
            ids.add(rid)
            rows += 1
            if rid in wanted:
                identities[rid] = row_identity(row.get("payload"))
                why = _export_row_problem(row, wanted[rid], dense_config, sparse_names)
                if why:
                    bad[rid] = why
    if export.get("rows") is not None and int(export["rows"]) != rows:
        raise QueueError(f"declared export {path}: manifest says rows={export['rows']}, "
                         f"file has {rows}")
    absent = [t["id"] for t in tombstones if _canonical_point_id(t["id"]) not in ids]
    if absent:
        raise QueueError(f"declared export {path} lacks {len(absent)} of {len(tombstones)} "
                         f"tombstoned id(s) (e.g. {absent[:5]}) -- no restore copy, refusing")
    if bad:
        first = sorted(bad)[:5]
        raise QueueError(
            f"declared export {path}: {len(bad)} of {len(tombstones)} tombstoned id(s) are not a "
            "complete restore row -- refusing; first "
            + "; ".join(f"{i}: {bad[i]}" for i in first))
    out = {"path": str(path), "sha256": h.hexdigest(), "rows": rows,
           "tombstoned_ids_covered": len(tombstones), "ids": ids}
    if any(t.get("reason") == RESTORE_ROLLBACK for t in tombstones):
        out["identities"] = identities      # for check_restore_rollback; the caller pops it
    return out


def _export_row_problem(row: dict, tomb_key: "str | None", dense_config: dict,
                        sparse_names) -> "str | None":
    """Why one export row cannot restore its point, or None. See `verify_removed_rows_export`."""
    vec = row.get("vector") or {}
    if not isinstance(vec, dict):
        return "vector is not a name -> vector map"
    for name, cfg in dense_config.items():
        v = vec.get(name)
        size = (cfg or {}).get("size")
        if not isinstance(v, list) or not v:
            return f"dense vector {name!r} missing"
        if size is not None and len(v) != size:
            return f"dense vector {name!r} has {len(v)} dims, collection says {size}"
    for name in sparse_names:
        v = vec.get(name)
        if not isinstance(v, dict) or not isinstance(v.get("indices"), list) \
                or not isinstance(v.get("values"), list):
            return f"sparse vector {name!r} missing or without indices/values arrays"
        if len(v["indices"]) != len(v["values"]):
            return (f"sparse vector {name!r}: {len(v['indices'])} indices vs "
                    f"{len(v['values'])} values")
    payload = row.get("payload")
    if not isinstance(payload, dict) or not payload:
        return "payload empty"
    if tomb_key is None or document_url_key(payload.get("url")) != tomb_key:
        return (f"payload url {payload.get('url')!r} is not the tombstone's document "
                "(same-document key differs)")
    return None


def validate_tombstones(tombstones: list[dict], *, pack_urls: set[str],
                        target_url: str, collection: str,
                        retain_months: int | None = None,
                        now_ref: "datetime.date | None" = None,
                        pack_source_types: "dict[str, set[str]] | None" = None,
                        pack_card_flags: "dict[str, set[bool]] | None" = None,
                        pack_pairs: "dict[str, set[tuple[str, bool]]] | None" = None,
                        pack_cards: "dict[str, list[dict]] | None" = None,
                        export_ids: "set[str] | None" = None,
                        export_identities: "dict[str, tuple] | None" = None
                        ) -> dict:
    """The mechanical checks (coordinator's ruling, 2026-09-23,
    amended 2026-09-28 for missing ids): every id exists on the target
    before the delete -- an already-missing id is DROPPED and recorded,
    not a refusal, unless more than `TOMBSTONE_MISSING_SHARE_MAX` of the
    pack's tombstones are missing (a present id whose stored row cannot be
    read -- a race with a delete -- counts as missing too); a `superseded` id's url must be in
    THIS pack's own url set (proof the pack actually replaces it, not an
    arbitrary deletion), compared with `document_url_key`, and the STORED
    row's own `url` must have the same key and its `source_type` must be one
    of the replacing pack rows' (`pack_source_types`, task 698); an `aged_out` id's current `published` must be
    older than the retention window (`news_window_filter.in_window`,
    imported, never re-implemented). Any other problem on a PRESENT id --
    an unknown reason, a superseded id with no matching url, an aged_out
    id that is not actually aged out -- still REFUSES THE WHOLE PACK.
    Returns `{"count", "by_reason", "missing_ids", "missing_share",
    "_tombstones"}` on PASS -- `_tombstones` is the reduced (present-only)
    list the caller must actually apply, exactly the `_points_data`/
    `_tombstones_data` convention `compose_content_pack` already uses for
    a coverage-skip.
    """
    if not tombstones:
        return {"count": 0, "by_reason": {}, "missing_ids": [],
               "unreadable_ids": [], "missing_share": 0.0, "_tombstones": []}
    ids = [t["id"] for t in tombstones]
    present = set(PSP.already_present(target_url, collection, ids))
    # ONE id retrieval (indexed/id access) for the present ids of both
    # reasons: `published` for aged_out, the STORED `url` + `source_type` for
    # superseded (task 698 review item 4: the tombstone's own url is
    # self-declared; the row it deletes must really be that document).
    need_read = [t["id"] for t in tombstones
                 if t["id"] in present
                 and t.get("reason") in ("aged_out", "superseded") + EXPORT_BACKED_REASONS]
    stored_by_id = (PD.fetch_payload_fields(
        target_url, collection, need_read,
        ["published", "url", "source_type", "entity_kind", "qid",
         "register_id", "register_scheme", "lei", "provenance", "snippet", "content_digest",
         "title"])
        if need_read else {})
    # A present id whose stored row cannot be read is a race with a delete
    # between the two calls (task 698 review 2, F4): it has reached its goal
    # state exactly like an absent id, so it is a MISSING id (counted under
    # the ceiling), never a refusal -- a refusal here would be cached until
    # the pack is resealed.
    unreadable = {i for i in need_read if i not in stored_by_id}
    # `missing_ids` = absent on the target; `unreadable_ids` = present but the
    # stored row could not be read. Both are dropped and both count under the
    # ceiling, but they are different facts and the receipt names them apart
    # (task 698 third review R3).
    absent = [i for i in ids if i not in present]
    unreadable_ids = [i for i in ids if i in unreadable]
    missing = absent + unreadable_ids
    missing_share = len(missing) / len(ids)
    if missing_share > TOMBSTONE_MISSING_SHARE_MAX:
        raise QueueError(
            f"{len(missing)} of {len(ids)} tombstone id(s) ({missing_share:.1%}) "
            f"do not exist (or could not be read) on {target_url}/{collection} "
            f"(e.g. {missing[:5]}) -- "
            f"over the {TOMBSTONE_MISSING_SHARE_MAX:.0%} ceiling, a sign "
            "something is systemically wrong with what this pack thinks it "
            "should be deleting, not an ordinary idempotent re-run -- "
            "refusing the whole pack")
    missing_set = set(missing)
    tombstones = [t for t in tombstones if t["id"] not in missing_set]

    retain_months = retain_months or int(
        os.environ.get(RETAIN_MONTHS_ENV, DEFAULT_RETAIN_MONTHS))
    now_ref = now_ref or datetime.datetime.now(datetime.timezone.utc).date()
    cutoff = NWF.window_cutoff(now_ref, months=retain_months)

    problems: list[str] = []
    by_reason: dict[str, int] = {}
    # The supersede selector groups rows by `canonical_url` (scheme and
    # `www.` folded), so every tombstone-vs-pack url comparison here uses
    # `document_url_key` too: a stored `https://swissinfo.ch/x` row is
    # replaced by a kept `https://www.swissinfo.ch/x` row (task 698). A url
    # in no spelling of the pack's own set, or with no host, still refuses.
    pack_keys = {k for k in (document_url_key(u) for u in pack_urls) if k}
    for t in tombstones:
        reason = t.get("reason")
        by_reason[reason] = by_reason.get(reason, 0) + 1
        if reason not in TOMBSTONE_REASONS:
            problems.append(f"{t.get('id')}: unknown reason {reason!r}")
        elif reason not in CL.QUEUE_TO_PUBLIC_REASON:
            # task 2497: the change log maps every queue reason to a public one at landing time; a
            # reason with no row would fail AFTER both legs wrote and strand the pack. Refuse it here.
            problems.append(f"{t.get('id')}: reason {reason!r} has no public change-log mapping")
        elif reason == "superseded":
            tkey = document_url_key(t.get("url"))
            if tkey is None:
                problems.append(
                    f"{t['id']}: superseded url {t.get('url')!r} names no "
                    "document (empty or host-less) -- refusing")
                continue
            if tkey not in pack_keys:
                problems.append(
                    f"{t['id']}: superseded url {t.get('url')!r} is not in "
                    "this pack's own url set -- an orphan tombstone, not a "
                    "genuine replacement")
                continue
            stored = stored_by_id[t["id"]] or {}
            skey = document_url_key(stored.get("url"))
            if skey != tkey:
                problems.append(
                    f"{t['id']}: the stored row's url {stored.get('url')!r} "
                    f"is not the tombstone's url {t.get('url')!r} (same-"
                    "document key differs) -- the tombstone would delete a "
                    "different document")
            elif pack_source_types is not None or pack_pairs is not None:
                # The PAIR (raw source_type, is_card) of the stored row must be
                # among the pairs of the pack rows replacing that key: the raw
                # type is compose's exact rule (an `entity_card` pack row can
                # never delete a stored `official`/`reference` row), the card
                # flag keeps a page row from deleting a row carrying
                # entity_kind/qid. Two separate sets would let a pack holding
                # an `entity_card` row AND a plain `official` row at one key
                # delete a stored `official`+marker row (task 698, Opus
                # re-check of #551, N1).
                st = _norm_source_type(stored.get("source_type"))
                stored_card = _is_card_row(stored)
                if pack_pairs is not None:
                    pairs = pack_pairs.get(tkey, set())
                else:   # legacy caller: separate sets; a card is a raw entity_card by default
                    types = pack_source_types.get(tkey, set())
                    flags = (pack_card_flags.get(tkey, set()) if pack_card_flags is not None
                             else {x == "entity_card" for x in types})
                    pairs = {(x, f) for x in types for f in flags}
                if (st, stored_card) not in pairs:
                    problems.append(
                        f"{t['id']}: stored row source_type {st!r} is not "
                        "among the source_types of the pack rows replacing "
                        f"{t.get('url')!r} "
                        f"({sorted({x for x, _ in pairs})}; card flags {sorted({f for _, f in pairs})}) -- "
                        "exact match required (compose's filter_cross_family_"
                        "stale rule; a row with entity_kind/qid counts as an "
                        "entity_card); e.g. a page row must never supersede "
                        "an entity card, or the reverse")
                elif stored_card and pack_cards is not None and not any(
                        US.card_replaces(pc, stored)
                        for pc in pack_cards.get(tkey, [])):
                    # The pair matched (a card for a card), but the pack's cards
                    # at this key are other ENTITIES: `supersede.card_replaces`
                    # is the one rule (same canonical url, same raw type,
                    # overlapping identity; an anonymous stored card is
                    # replaced by any card of its type).
                    problems.append(
                        f"{t['id']}: stored card identity "
                        f"{sorted(US.card_identity(stored))} is not replaced by "
                        "any card of this pack at "
                        f"{t.get('url')!r} (pack card identities "
                        f"{[sorted(US.card_identity(pc)) for pc in pack_cards.get(tkey, [])]}) "
                        "-- a card tombstone needs a pack card of an overlapping "
                        "entity identity (register id, LEI, QID)")
        elif reason in (SUPERSEDED_GENERATION, CANONICAL_DUPLICATE, RESTORE_ROLLBACK):
            pass   # per-id checks need a group read or the export: batched below, after this loop
        elif reason == BINARY_CONTENT:
            problems.extend(check_binary_content([t], stored_by_id))
        elif reason == SOFT_404:
            problems.extend(check_soft_404([t], stored_by_id))
        elif reason == "aged_out":
            pub = (stored_by_id[t["id"]] or {}).get("published")
            in_win = NWF.in_window(pub, cutoff)
            if in_win is not False:
                problems.append(
                    f"{t['id']}: aged_out claimed but published={pub!r} is "
                    f"not older than the {retain_months}-month retention "
                    f"window (in_window={in_win})")
        # 'takedown': no additional mechanical check named by the ruling --
        # accepted as declared (a publisher/legal takedown is a human fact,
        # not something this driver re-derives).
    generation = [t for t in tombstones if t.get("reason") == SUPERSEDED_GENERATION]
    if generation:
        if export_ids is None:
            raise QueueError("superseded_generation needs the removed-rows export's ids "
                             "(`export_ids`) to tell a survivor from another old row")
        problems.extend(check_generation_siblings(
            generation, {t["id"]: stored_by_id.get(t["id"]) or {} for t in generation},
            target_url=target_url, collection=collection, export_ids=export_ids))
    duplicates = [t for t in tombstones if t.get("reason") == CANONICAL_DUPLICATE]
    if duplicates:
        if export_ids is None:
            raise QueueError("canonical_duplicate needs the removed-rows export's ids "
                             "(`export_ids`) to tell a survivor from another variant")
        problems.extend(check_canonical_duplicates(
            duplicates, {t["id"]: stored_by_id.get(t["id"]) or {} for t in duplicates},
            target_url=target_url, collection=collection, export_ids=export_ids))
    rollbacks = [t for t in tombstones if t.get("reason") == RESTORE_ROLLBACK]
    if rollbacks:
        if export_identities is None:
            raise QueueError("restore_rollback needs the removed-rows export's row identities "
                             "(`export_identities`) to prove each id is still the row it restored")
        problems.extend(check_restore_rollback(
            rollbacks, {t["id"]: stored_by_id.get(t["id"]) or {} for t in rollbacks},
            export_identities))
    if problems:
        raise QueueError(
            f"tombstone validation refused the whole pack ({len(problems)} "
            "of " + f"{len(tombstones)} problem(s)):\n  - "
            + "\n  - ".join(problems[:50]))
    return {"count": len(tombstones), "by_reason": by_reason,
           "missing_ids": absent, "unreadable_ids": unreadable_ids,
           "missing_share": round(missing_share, 4),
           "_tombstones": tombstones}


#: Default only used when the tracked gate contract carries no explicit
#: value -- the contract is the one home for this number (coordinator's
#: ruling, 2026-09-23); this is a fallback, not a second copy meant to win.
DEFAULT_TOMBSTONE_WORD_COVERAGE_FLOOR = 0.50


def _snippet_words(payload: dict | None) -> int:
    return len(((payload or {}).get("snippet") or "").split())


def _chunk_epoch(payload: dict | None) -> int | None:
    """`payload.rule_epochs.chunk` as a plain int, else None (absent or malformed). A stored row
    must be decoded first (`row_decode`); a pack row is raw by construction."""
    v = ((payload or {}).get("rule_epochs") or {}).get("chunk")  # canonical row
    return v if isinstance(v, int) and not isinstance(v, bool) else None


#: Orchestrator ruling, 2026-09-25 (task 660, live incident: news-daily-
#: 20260924 refused WHOLE on 2 of its 40 superseded urls, both a Swiss news
#: wire's "Meldung" stub -- a genuinely thin source page, not a composer
#: defect). A word-coverage violation is scored PER URL, not per pack: a
#: url whose replacement collapsed below the floor keeps its EXISTING prod
#: rows untouched (its tombstone AND its own new rows are dropped from
#: THIS apply, named `coverage_skipped` in the receipt so nothing is
#: silently lost) while every other url in the pack still lands. The whole
#: pack still refuses when MORE than this share of its superseded urls
#: violate -- that shape is evidence of a systemic extraction failure
#: (task 640's original incident: every url the composer touched
#: collapsed), which is categorically different from a handful of
#: genuinely thin pages and must not be silently worked around one url at
#: a time.
COVERAGE_VIOLATION_SHARE_MAX = 0.20


def validate_tombstone_word_coverage(points: list[dict], tombstones: list[dict],
                                    *, target_url: str, collection: str,
                                    floor: float = DEFAULT_TOMBSTONE_WORD_COVERAGE_FLOOR,
                                    content_words_by_url: dict | None = None
                                    ) -> dict:
    """Task 640's real defect, 2026-09-23: a recomposition collapsed
    multi-chunk articles to one row per url, and the mechanical checks that
    already existed (id exists, superseded url is in the pack) could not
    see it -- both were still true of the shrunken pack. This one can: for
    every url a `superseded` tombstone names, the pack must (a) keep at
    least one row for that url, and (b) carry total words (on `payload.
    snippet`, the only text a content_pack payload has -- ADR-039) at least
    `floor` times the words being tombstoned for that url.

    Pure measurement -- never raises, never filters. What a violation MEANS
    (drop the url from this apply, or refuse the whole pack) is the
    caller's decision (`content_pack_dry_check`, `COVERAGE_VIOLATION_
    SHARE_MAX`), because that decision needs the violation share across the
    WHOLE pack, which only the caller can see.

    TWO BASES (task 984, RULING 43). `snippet` (the default above) counts
    snippet words, and the snippet is capped at 25 words per row (ADR-039),
    so across a chunk-policy change (180-word -> 768-word chunks) the ratio
    is really new rows / old rows and fails by construction. A pack whose
    composer declared `content_words_by_url` (the words of the chunk text it
    wrote, per document key) is compared on `content_words` instead, for a
    url that meets ALL of: a declaration for it; the pack rows carry ONE
    `rule_epochs.chunk`; every superseded old row carries a `rule_epochs.chunk`
    that differs from it; and every old epoch has a known word width
    (`registry.epoch_word_width`). Then new = the declared words, old = the
    old rows x the width of their epoch. The old side is an UPPER BOUND (a
    short last chunk is counted at full width), which is the conservative
    direction: it can only make a violation likelier, never hide one. Any
    other url keeps the snippet rule; a declared url that fell back says why
    in `basis_fallback`. Every violation entry carries `basis`.
    """
    superseded = [t for t in tombstones if t.get("reason") == "superseded"]
    if not superseded:
        return {"urls_checked": 0, "violations": []}

    def _key(u):
        return document_url_key(u) or u

    declared: dict[str, int] = {}
    for k, v in (content_words_by_url or {}).items():
        if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
            declared[_key(k)] = v
    new_words_by_url: dict[str, int] = {}
    new_rows_by_url: dict[str, int] = {}
    new_epochs_by_url: dict[str, set] = {}
    for p in points:
        u = (p.get("payload") or {}).get("url")
        if not u:
            continue
        u = _key(u)
        new_rows_by_url[u] = new_rows_by_url.get(u, 0) + 1
        new_epochs_by_url.setdefault(u, set()).add(_chunk_epoch(p.get("payload")))
        new_words_by_url[u] = new_words_by_url.get(u, 0) + _snippet_words(p.get("payload"))

    old_ids_by_url: dict[str, list[str]] = {}
    sample_url: dict[str, str] = {}
    for t in superseded:
        old_ids_by_url.setdefault(_key(t["url"]), []).append(t["id"])
        sample_url.setdefault(_key(t["url"]), t["url"])
    all_old_ids = [i for ids in old_ids_by_url.values() for i in ids]
    old_payloads = PD.fetch_payload_fields(target_url, collection, all_old_ids,
                                          ["snippet", "url", "rule_epochs"])
    decode = RD.reader(collection)   # a coded twin reads like a raw one (task 855)
    old_payloads = {i: decode(pl) for i, pl in old_payloads.items()}
    tomb_url = {t["id"]: t["url"] for t in superseded}

    violations = []
    for u, old_ids in old_ids_by_url.items():
        # Old words of ONE document = the words of its best stored COPY, not
        # the sum over copies (task 698 review 2, F1). A "copy" is the rows
        # stored under one exact url spelling (a multi-chunk article is
        # several rows of ONE spelling: those add up). The twin may hold the
        # same document under two spellings (apex + `www.`); a re-capture
        # tombstones both, but the replacement only has to carry the text
        # of the best copy, not of both -- summing would count the same
        # article twice and refuse a pack that loses nothing. Two genuinely
        # different copies are still covered: the kept rows are compared
        # with the LARGEST copy, so a text loss against it still violates.
        words_by_spelling: dict[str, int] = {}
        content_by_spelling: dict[str, int] = {}
        old_epochs: set = set()
        for i in old_ids:
            pl = old_payloads.get(i)
            spelling = (pl or {}).get("url") or tomb_url[i]
            words_by_spelling[spelling] = (words_by_spelling.get(spelling, 0)
                                           + _snippet_words(pl))
            ep = _chunk_epoch(pl)
            old_epochs.add(ep)
            width = CHREG.epoch_word_width(ep) if ep is not None else None
            content_by_spelling[spelling] = (content_by_spelling.get(spelling, 0)
                                             + (width or 0))
        old_words = max(words_by_spelling.values(), default=0)
        kept_rows = new_rows_by_url.get(u, 0)
        kept_words = new_words_by_url.get(u, 0)
        basis, fallback = "snippet", None
        if u in declared:
            new_eps = new_epochs_by_url.get(u, set())
            if len(new_eps) != 1 or None in new_eps:
                fallback = "pack rows do not carry one rule_epochs.chunk"
            elif None in old_epochs:
                fallback = "an old row has no rule_epochs.chunk"
            elif old_epochs & new_eps:
                fallback = "chunk epoch is unchanged"
            elif any(CHREG.epoch_word_width(e) is None for e in old_epochs):
                fallback = "an old chunk epoch has no known word width"
            else:
                basis = "content_words"
                old_words = max(content_by_spelling.values(), default=0)
                kept_words = declared[u]
        if old_words == 0:
            continue  # nothing measurable to shrink from
        ratio = kept_words / old_words
        if kept_rows < 1 or ratio < floor:
            entry = {
                "url": sample_url[u], "doc_key": u, "kept_rows": kept_rows, "kept_words": kept_words,
                "old_words": old_words, "ratio": round(ratio, 3),
                "floor": floor, "basis": basis}
            if fallback and basis == "snippet":
                entry["basis_fallback"] = fallback
            violations.append(entry)
    return {"urls_checked": len(old_ids_by_url), "violations": violations}


def fetch_collection_vector_config(url: str, collection: str) -> dict:
    """`{"dense": {"size": 256, "distance": "Cosine", ...}, ...}` -- the
    target collection's NAMED DENSE vector configs, read straight off
    Qdrant. Sparse vectors (e.g. `bm25`) live under a separate
    `sparse_vectors` config key with no fixed `size` to check against, so
    this deliberately only covers dense."""
    info = PSP._api(url, f"/collections/{collection}")
    return (info or {}).get("config", {}).get("params", {}).get("vectors", {}) or {}


def fetch_collection_sparse_names(url: str, collection: str) -> list[str]:
    """Names of the collection's SPARSE vectors (`sparse_vectors` config), read off Qdrant."""
    info = PSP._api(url, f"/collections/{collection}")
    return sorted(((info or {}).get("config", {}).get("params", {})
                   .get("sparse_vectors", {}) or {}))


def validate_vector_shapes(points: list[dict], vector_config: dict) -> list[str]:
    """Refuse-worthy vector name/dimension problems, checked against EVERY
    point (not a sample): a partial re-embed can leave only some rows at
    the wrong width. Bug found live, task 641, 2026-09-23: the backlog
    pack's dense vectors were 128-dim against aw_e8's 256, and dry_plan let
    it all the way to a live twin upsert before Qdrant's own 400 caught it
    -- this runs at dry-run time, before any write, against every row."""
    problems: list[str] = []
    if not points or not vector_config:
        return problems
    seen: set[tuple[str, int | None]] = set()
    for p in points:
        vec = p.get("vector") or {}
        for name, cfg in vector_config.items():
            expected = cfg.get("size")
            if expected is None or name not in vec:
                continue
            got = vec[name]
            got_dim = len(got) if hasattr(got, "__len__") else None
            if got_dim != expected and (name, got_dim) not in seen:
                seen.add((name, got_dim))
                problems.append(
                    f"vector {name!r}: expected dim {expected}, got {got_dim}")
    return problems


#: Payload fields the card-overwrite guard reads off a stored row.
CARD_GUARD_FIELDS = ("url", "source_type", "entity_kind", "qid", "register_id",
                     "register_scheme", "lei")


def _canonical_point_id(pid) -> str:
    """The form Qdrant returns an id in: a UUID in any spelling (upper case, no
    hyphens, braces) folds to the lower-case hyphenated form (`uuid.UUID`, as
    `cytir.core.ids.record_point_id` and `bm25_saturate._uuid_ge` do; the project
    has no other id normalizer, `common.require_qdrant_point_id` only refuses a
    non-canonical spelling). An integer id or any other string is kept as
    written."""
    text = str(pid)
    try:
        return str(uuid.UUID(text))
    except ValueError:
        return text


def check_card_overwrites(points: list[dict], url: str, collection: str, *,
                          hold_out: bool = False, waivers: dict | None = None) -> dict:
    """Refuse a pack whose upsert would OVERWRITE BY ID a stored entity card with
    a row that is not a card that replaces it (`supersede.card_replaces`: both
    are cards, same canonical url, same raw type, overlapping identity by
    precedence register id > LEI > QID).

    EVERY stored card is covered (`supersede.payload_is_card`). A stored card
    WITH an identity is overwritten only by a card of an overlapping identity.
    A stored card with NO identity keeps today's rule: a card of the same raw
    type at the same canonical url may replace it; a non-card row may not. A
    stored non-card row is not this guard's business.

    The L1-backfill failure: six cards were overwritten in place with shop text
    (same Tier C id).

    ONE read on `url`: a batched id retrieve of the pack's ids with a short
    payload projection (`PSP.retrieve_payloads`, indexed, 500 ids per request,
    the same shape as `already_present`). No scroll, no per-point request, no
    prod request (the dry check runs against the twin only). Ids are compared
    in canonical UUID form on both sides (Qdrant answers in that form whatever
    the pack spells); an id Qdrant returned that no pack row matches is a
    per-pack refusal, never an exception.

    Task 728 (ruling 26 item 19). (a) `hold_out=True`: the bad rows are RETURNED
    under `"held"` (`{"id", "url", "reason"}`, the reasons below, nothing new) instead
    of refusing the pack; the caller writes the rest. `hold_out=False` (default) refuses
    the whole pack exactly as before. (b) A stored card whose id, canonical url and
    whole identity-token set a waiver row names (`card_waivers`; `waivers` = the rows of the
    PACK'S OWN sealed file, `_content_pack_dry_check` loads it from the manifest), overwritten by
    a non-card row at the same canonical url, is admitted and listed under `"waived"` (id, pr,
    evidence sha); any other overwrite is a bad row as before."""
    ids = [str(p["id"]) for p in points]
    stored = PSP.retrieve_payloads(url, collection, ids, CARD_GUARD_FIELDS)
    by_id = {_canonical_point_id(p["id"]): (p.get("payload") or {}) for p in points}
    stored_cards = 0
    anonymous = 0
    problems: list[str] = []
    held: list[dict] = []
    waived: list[str] = []
    waivers = waivers or {}      # the pack's own sealed file, loaded by the dry check (ADR-114 am.1)
    for raw_pid, old in stored.items():
        pid = _canonical_point_id(raw_pid)
        if pid not in by_id:
            raise QueueError(
                f"card overwrite guard: the twin returned id {raw_pid!r}, which "
                "matches no id of this pack even in canonical UUID form -- "
                "refusing the pack")
        if not US.payload_is_card(old):
            continue
        stored_cards += 1
        if not US.card_identity(old):
            anonymous += 1
        new = by_id[pid]
        if not US.card_replaces(new, old):
            wrow = (CW.admits(waivers, pid, old, US.card_identity(old), new)
                    if not US.payload_is_card(new) else None)
            if wrow is not None:
                waived.append({"id": pid, "pr": wrow["pr"], "evidence_sha256": wrow["evidence_sha256"]})
                continue
            reason = (
                f"{pid}: stored card {sorted(US.card_identity(old))} at "
                f"{old.get('url')!r} would be overwritten by a "
                f"{'card' if US.payload_is_card(new) else 'non-card'} row "
                f"{new.get('source_type')!r} at {new.get('url')!r} with identity "
                f"{sorted(US.card_identity(new))}")
            problems.append(reason)
            held.append({"id": pid, "url": new.get("url"), "reason": reason})
    if problems and not hold_out:
        raise QueueError(
            f"card overwrite guard refused the whole pack ({len(problems)} stored "
            "card(s) would be overwritten by id with a row that is not a card "
            "of an overlapping identity):\n  - "
            + "\n  - ".join(problems[:50]))
    return {"ids_read": len(stored), "stored_cards": stored_cards,
            "stored_cards_with_identity": stored_cards - anonymous,
            "stored_cards_anonymous": anonymous, "refused": 0,
            "waived": waived, "held": held if hold_out else []}


def content_pack_dry_check(manifest: dict, url: str,
                          contract_path: pathlib.Path = GATE_CONTRACT_PATH) -> dict:
    """`_content_pack_dry_check` with a transport failure of any twin read re-raised as
    `TransientReadError`. A 4xx (on either read helper) or a malformed answer stays a cached refusal."""
    try:
        return _content_pack_dry_check(manifest, url, contract_path)
    except PSP.PromotionError as exc:
        if getattr(exc, "http_status", 0) >= 500:
            raise TransientReadError(f"{type(exc).__name__}: {exc}") from exc
        raise
    except urllib.error.HTTPError as exc:
        # `PD._api` does not wrap: it raises the raw HTTPError (a URLError subclass), so classify
        # by status BEFORE the generic clause, as the PromotionError branch does for `PSP._api`.
        # 5xx = the twin is down (retry); anything else is an answer (cached refusal).
        if exc.code >= 500:
            raise TransientReadError(f"{type(exc).__name__}: {exc}") from exc
        raise QueueError(
            f"twin read refused with HTTP {exc.code} on {urllib.parse.urlsplit(exc.url or '').path}"
            f" -- refusing the pack ({exc})") from exc
    except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
        raise TransientReadError(f"{type(exc).__name__}: {exc}") from exc


def _content_pack_dry_check(manifest: dict, url: str,
                           contract_path: pathlib.Path = GATE_CONTRACT_PATH) -> dict:
    """Read-only: load the points + tombstones, verify file shas, validate
    tombstones against `url` (twin first, always). Raises on any problem
    EXCEPT a minority tombstone word-coverage violation (task 660,
    `COVERAGE_VIOLATION_SHARE_MAX`), where the violating urls are silently
    dropped from the RETURNED `_points_data`/`_tombstones_data` -- so a
    "PASS" here can still mean "some urls were skipped"; check
    `tombstone_word_coverage.coverage_skipped` in the result. No write."""
    from cytir.index import promote_queue as _Q  # noqa: PLC0415 -- patched names resolve at call time
    points = _Q.load_content_pack_points(manifest["pack_points"])
    _verify_pack_file(manifest["pack_points"], manifest.get("pack_points_sha256"),
                      "pack_points")
    rows_count = manifest.get("rows")
    if rows_count is not None and len(points) != rows_count:
        raise QueueError(
            f"manifest says rows={rows_count} but pack_points has "
            f"{len(points)} -- refusing a count mismatch")
    vector_config = _Q.fetch_collection_vector_config(url, manifest["collection"])
    vector_problems = _Q.validate_vector_shapes(points, vector_config)
    if vector_problems:
        raise QueueError(
            f"vector shape mismatch against {manifest['collection']}: "
            + "; ".join(vector_problems))
    # Task 728 (ruling 26 item 19a): a bad card row is HELD OUT, the rest of the pack is judged
    # and written. The WHOLE URL is held (ADR-067 am.1 item 4: never keep both generations): every
    # pack row at the url of a bad card row, and every tombstone there, as `coverage_skipped` does.
    # Held BEFORE the pack's url/pair/card sets are built. Over `COVERAGE_VIOLATION_SHARE_MAX` of
    # the rows, or no row left: the whole pack is refused.
    waivers, waiver_report = load_pack_waivers(manifest)
    card_guard = _Q.check_card_overwrites(points, url, manifest["collection"], hold_out=True,
                                       waivers=waivers)
    card_guard["waiver_file"] = waiver_report
    bad_rows = card_guard.pop("held")
    all_pack_ids = {p["id"] for p in points}     # S3: the overlap check below sees the whole pack
    held_rows: list[dict] = []
    held_keys: set[str] = set()
    if bad_rows:
        bad = {_canonical_point_id(h["id"]): h["reason"] for h in bad_rows}
        held_keys = {k for k in (document_url_key((p.get("payload") or {}).get("url"))
                                 for p in points if _canonical_point_id(p["id"]) in bad) if k}
        kept = []
        for p in points:
            pid = _canonical_point_id(p["id"])
            key = document_url_key((p.get("payload") or {}).get("url"))
            if pid in bad:
                held_rows.append({"point": p, "code": HELD_CARD_OVERWRITE, "reason": bad[pid]})
            elif key in held_keys:
                held_rows.append({"point": p, "code": HELD_SAME_URL,
                                  "reason": f"same url as a held card row ({key}): the url's "
                                            "generation is held whole"})
            else:
                kept.append(p)
        share = len(held_rows) / len(points)
        if not kept or share > COVERAGE_VIOLATION_SHARE_MAX:
            raise QueueError(
                f"card overwrite guard refused the whole pack ({len(bad_rows)} stored card(s) "
                f"would be overwritten by id with a row that is not a card of an overlapping "
                f"identity; {len(held_rows)} of {len(points)} row(s) = {share:.1%} would be held, "
                f"over the {COVERAGE_VIOLATION_SHARE_MAX:.0%} ceiling or no row left):\n  - "
                + "\n  - ".join(h["reason"] for h in bad_rows[:50]))
        points = kept
    pack_urls = {(p.get("payload") or {}).get("url") for p in points}
    pack_pairs: dict[str, set[tuple[str, bool]]] = {}
    pack_cards: dict[str, list[dict]] = {}
    for p in points:
        pl = p.get("payload") or {}
        k = document_url_key(pl.get("url"))
        if k:
            pack_pairs.setdefault(k, set()).add(
                (_norm_source_type(pl.get("source_type")), _is_card_row(pl)))
            if _is_card_row(pl):
                pack_cards.setdefault(k, []).append(pl)
    held_tombstone_ids: list[str] = []
    tombstones: list[dict] = []
    tombstone_report = {"count": 0, "by_reason": {}}
    coverage_report = {"urls_checked": 0, "violations": []}
    if manifest.get("tombstones"):
        tombstones = load_tombstones(manifest["tombstones"])
        _verify_pack_file(manifest["tombstones"], manifest.get("tombstones_sha256"),
                          "tombstones")
        tc = manifest.get("tombstones_count")
        if tc is not None and len(tombstones) != tc:
            raise QueueError(
                f"manifest says tombstones_count={tc} but tombstones file "
                f"has {len(tombstones)} -- refusing a count mismatch")
        tomb_ids = {t["id"] for t in tombstones}
        overlap = tomb_ids & all_pack_ids
        if held_keys:
            # a tombstone at the url of a held row has no replacement: the stored row stays
            held_tombstone_ids = [t["id"] for t in tombstones
                                  if document_url_key(t.get("url")) in held_keys]
            tombstones = [t for t in tombstones if t["id"] not in set(held_tombstone_ids)]
        if overlap:
            raise QueueError(
                f"{len(overlap)} tombstone id(s) also appear in this "
                "pack's own points -- a pack cannot both write and delete "
                f"the same id: {sorted(overlap)[:20]}")
        export_ids = None
        generation_export = None
        if any(t.get("reason") in EXPORT_BACKED_REASONS for t in tombstones):
            check_superseded_generation_pack(manifest, points, tombstones)
            # the restore copy first: it must be complete for every id, and its ids tell the
            # sibling check which rows are the OLD generation
            generation_export = verify_removed_rows_export(
                manifest, tombstones, dense_config=vector_config,
                sparse_names=_Q.fetch_collection_sparse_names(url, manifest["collection"]))
            export_ids = generation_export.pop("ids")
        export_identities = (generation_export.pop("identities", None)
                             if generation_export is not None else None)
        if export_identities is not None:
            rollback_problems = check_rolled_back_pack(
                manifest, [t for t in tombstones if t.get("reason") == RESTORE_ROLLBACK],
                export_identities)
            if rollback_problems:
                raise QueueError(
                    f"restore_rollback refused the whole pack ({len(rollback_problems)} "
                    "problem(s)):\n  - " + "\n  - ".join(rollback_problems[:50]))
        tombstone_report = validate_tombstones(
            tombstones, pack_urls=pack_urls, target_url=url,
            collection=manifest["collection"],
            pack_pairs=pack_pairs, pack_cards=pack_cards, export_ids=export_ids,
            export_identities=export_identities)
        # An already-missing tombstone id has reached its goal state, not
        # failed to reach it (coordinator's ruling, task 664, 2026-09-28)
        # -- validate_tombstones() drops it and returns the reduced,
        # present-only list here; everything downstream (word-coverage
        # check, the actual delete) applies to THAT list, never the
        # original one, so a dropped id is never re-checked or re-deleted.
        tombstones = tombstone_report.pop("_tombstones")
        if generation_export is not None:
            tombstone_report["export"] = generation_export
        floor = DEFAULT_TOMBSTONE_WORD_COVERAGE_FLOOR
        if contract_path.exists():
            doc = json.loads(contract_path.read_text())
            floor = doc.get("tombstone_word_coverage_floor", floor)
        declared_words = manifest.get("content_words_by_url")
        if declared_words is not None and not isinstance(declared_words, dict):
            raise QueueError("manifest content_words_by_url must be a {doc_key: int} map "
                             "-- refusing the pack")
        coverage_report = validate_tombstone_word_coverage(
            points, tombstones, target_url=url,
            collection=manifest["collection"], floor=floor,
            content_words_by_url=declared_words)
        if coverage_report["violations"]:
            share = (len(coverage_report["violations"])
                    / coverage_report["urls_checked"])
            if share > COVERAGE_VIOLATION_SHARE_MAX:
                raise QueueError(
                    f"tombstone word-coverage refused the whole pack "
                    f"({len(coverage_report['violations'])} of "
                    f"{coverage_report['urls_checked']} url(s) = {share:.1%} "
                    f"below the {floor:.0%} floor, over the "
                    f"{COVERAGE_VIOLATION_SHARE_MAX:.0%} systemic-failure "
                    "share -- refusing the whole pack rather than skipping "
                    f"a majority of it): "
                    + json.dumps(coverage_report["violations"][:20]))
            # Below the systemic-failure share: skip just these urls (drop
            # their tombstone AND their own new rows from THIS apply,
            # leaving prod's existing rows for them untouched) and land
            # everything else.
            skipped_keys = {v["doc_key"] for v in coverage_report["violations"]}

            def _skipped(u):
                return (document_url_key(u) or u) in skipped_keys
            points = [p for p in points
                     if not _skipped((p.get("payload") or {}).get("url"))]
            n_declared = len(tombstones)
            tombstones = [t for t in tombstones if not _skipped(t.get("url"))]
            coverage_report["coverage_skipped"] = coverage_report["violations"]
            # Task 1467: `tombstones.count` is what the pack DECLARES; `twin/prod.tombstoned` is what the
            # drain DELETED. The gap was invisible (hf-models-20261015: 38 declared, 19 deleted, the 19
            # kept rows sat in the 3 coverage-skipped urls). Say it in the receipt, by name.
            tombstone_report["withheld_by_coverage"] = n_declared - len(tombstones)
    return {"points": len(points), "pack_urls": len(pack_urls),
           "tombstones": tombstone_report,
           "card_overwrite_guard": card_guard,
           "tombstone_word_coverage": coverage_report,
           "_points_data": points, "_tombstones_data": tombstones,
           "_held_out_cards": {"rows": held_rows, "tombstone_ids": held_tombstone_ids}}


HELD_SAME_URL = "same_url_as_held_card"    # a pack row at the url of a bad card row


def load_pack_waivers(manifest: dict) -> tuple[dict, dict | None]:
    """The waiver rows THIS pack carries (ADR-114 am.1 item 3): the manifest's `card_waivers`
    `{path, sha256}` names a sealed file. No field, no waivers. A sha mismatch, a missing or an
    unsealed or malformed file refuses the pack (a fact about the pack: it is re-judged when the
    manifest, and so its sha, changes). Never reads the repo's source file. An I/O error that is
    not "no such file" is a `TransientReadError` (not cached)."""
    spec = manifest.get("card_waivers")
    if spec is None:
        return {}, None
    if not (isinstance(spec, dict) and all(isinstance(spec.get(k), str) and spec[k]
                                           for k in ("path", "sha256"))):
        raise QueueError("manifest card_waivers must be {path, sha256} -- refusing the pack")
    try:
        rows = CW.load_waivers(pathlib.Path(spec["path"]), expected_sha256=spec["sha256"])
    except CW.CardWaiverError as exc:
        raise QueueError(f"card waiver file refused: {exc}") from exc
    except OSError as exc:
        raise TransientReadError(f"card waiver file unreadable: {exc}") from exc
    return rows, {"path": spec["path"], "sha256": spec["sha256"], "rows": len(rows)}

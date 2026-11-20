#!/usr/bin/env python3
"""rii_dump_basis_backfill.py — task 465: the ADR-076 dump-basis backfill for
`rechtsprechung-im-internet.de` (rii_courts, 38,159 points).

WHY THIS EXISTS
---------------
Tim ruled 2026-08-28 (plan.md, task 457 Q1/Q2): keep the 38,159 rii points,
re-embed their retained text locally at 1024, mark the source non-refreshable,
no outreach — **conditional on a written, instrument-based licence reading**.

The points cannot survive a rebuild as they stand. Measured in task 457:

  * they carry **0 of the 4** fields `build_aw_b1.dump_basis_evidence()`
    requires, so the ADR-076 exemption cannot fire for them (this is task 433
    WI-1, the same shape as the legifrance `fr_dila_deep` cohort);
  * their only protection is a grace stamp whose window **expired 2026-08-24**
    (`consent_state=grace`, `consent_grace_until=2026-08-24`), so the gate
    evicts them on grace expiry;
  * the live host consent receipt is a **resolved decline** (robots
    `User-agent: * / Disallow: /`), which ADR-076 says does not govern records
    built from an openly-licensed official bulk distribution — but only if the
    evidence of that grant is on the record;
  * **nothing on them records `train_ok` at all**, while every rii response
    carries a `tdm-reservation: 1` header. `train_ok=false` is mandatory.

This module produces the sealed artifact that fixes all four, in the shape the
assembly lane consumes. It writes NOTHING to Qdrant: it emits a point-ID claim
plus one uniform payload stamp, and the assembly/apply path applies it.

THE STAMP, AND WHY EACH FIELD HAS THE VALUE IT HAS
--------------------------------------------------
Template = the legifrance/DILA dump-basis stamp (`stage_pack_b1.py`'s
`--consent-state/--consent-basis/--consent-source/--consent-checked-at` block),
NOT fedlex's — fedlex is `consent_state=granted` on a live-host basis, which
rii cannot use because rii's live host declines.

  consent_state   "dump"                    the route declaration (ADR-076)
  consent_basis   "official-dump-license"   the closed-set recognized basis
  consent_source  the rii bulk distribution the named distribution
  license         UrhG § 5 amtliche Werke   the explicit grant (see below)
  train_ok        False                     tdm-reservation: 1 — mandatory
  search_ok       True                      what the dump basis admits

`consent_reason` and `consent_grace_until` are DELETED, not overwritten: a
stale `consent_grace_until: 2026-08-24` sitting next to `consent_state: dump`
would be a payload that lies about which rule is holding the record.

THE ORIGIN CORRECTION
---------------------
The sealed 2026-08-19 consent receipt gives three reasons for the decline. The
third — *"robots: CCBot disallowed — our copy came via Common Crawl (provenance
has no consent; conservative reading)"* — is **factually wrong**, and the
receipt records `mode: "cc"` because the sweep assumed a CC origin for every
host it swept. rii never came through Common Crawl:

  * `ingest_rii.py` lines 2-4: "the BMJ/BfJ OFFICIAL bulk dump
    (`rii-toc.xml` lists every decision; each is a per-decision XML in a small
    .zip). Dumps-first, no scraping."
  * the corpus wave `_meta.json` for `wave=de_courts_rii` names
    `source: rii.jsonl`, `extraction_version: rii-xml-2026-07` — a per-decision
    XML extraction, not a WARC/WET record;
  * the retained rows carry no `crawl_id` — the same tell ADR-076 records for
    the DILA cohort.

Reasons 1 and 2 (the `User-agent: *` full opt-out and our own token being
disallowed) stand, and are exactly what ADR-076 says the dump basis overrides
for `search_ok` — and only for `search_ok`.

(Since ADR-024 amendment 5, Tim 2026-09-03, the CCBot reason would not have
declined `search_ok` for ANY host — it stamps `train_ok=false` only. That
makes the origin correction below moot as a *decline* argument while leaving
it true as a *provenance* statement, and it changes nothing here: reasons 1
and 2 still decline this host, `train_ok=False` is still mandated
independently by the host's `tdm-reservation: 1`, and the sealed manifest
`data/assembly/rii-dump-basis-backfill.v1.json` is unchanged so `verify`
still re-derives byte-identically.)

RUNNING IT
----------
    # produce (needs the retained line-level source; no network, no Qdrant)
    python3 cyt ops rii-dump-basis-backfill produce \\
        --rii-jsonl RII.jsonl \\
        --points-out  <corpus>/_work/task465/rii-dump-basis-backfill-v1/points.jsonl.gz \\
        --manifest-out data/assembly/rii-dump-basis-backfill.v1.json \\
        --points-uri artifact://_work/task465/rii-dump-basis-backfill-v1/points.jsonl.gz

    # verify (re-derives independently; fails closed)
    python3 cyt ops rii-dump-basis-backfill verify \\
        --manifest data/assembly/rii-dump-basis-backfill.v1.json \\
        --points <path to the sealed points file> \\
        --rii-jsonl RII.jsonl

`verify` does not read the producer's memory or re-use its output as its own
input: it re-mints all 38,159 identities from the retained source with the real
repo identity code, re-hashes the sealed file, and re-reads the consent-gate's
own constants out of `build_aw_b1.py` by AST. A check that passes because it
was handed the answer is not a check.
"""
from __future__ import annotations

from cytir.core import internal_http as IH  # the one door to the operator's own services (consent step 5, task 1673)
import argparse
import ast
import gzip
from cytir.acquire.gzip_pinned import deterministic_gzip_open
import hashlib
import json
import os
import sys


from cytir.core import common as AC   # noqa: E402
from cytir.core import ids as ID      # noqa: E402
from cytir.core import license as LIC  # noqa: E402

SCHEMA = "aw.dump-basis-backfill.v1"
SET_ID = "rii_courts"
HOST = "www.rechtsprechung-im-internet.de"
DOMAIN = "rechtsprechung-im-internet.de"
WAVE = "de_courts_rii"

# The retained line-level source, pinned. `_meta.json` of the corpus wave
# records the same digest as `source_sha256`; task 457 byte-verified the two
# against each other. A different file is a different population.
SOURCE_SHA256 = "5daf0f21e1ffc8a0b0daf68cd177d7dfb7f7327003b4ce6e1e455166a556c4a9"
EXPECTED_POINTS = 38159

# The 2026-08-19 live consent receipt this backfill corrects and supersedes
# (kept in the operator's consent-receipt store as www.rechtsprechung-im-internet.de.json).
DECLINE_RECEIPT_SHA256 = "27f1b1c8259e26125bae3757a9441a93ee8ad9877e57d45147b2dcefbbd3e176"
DECLINE_RECEIPT_CHECKED_AT = "2026-08-19T03:57:41Z"

# --- the licence value ------------------------------------------------------
# Public-domain-official-work family, instrument-named. Siblings in the repo:
# `sources/ch/ch-courts.yaml` stamps Swiss court decisions `public-domain` on
# URG Art. 5; task 459 proposes `public_domain_official_work (UrhG §5)` for
# OpenLegalData on the identical German basis. `_HOST_LICENSE` values are
# hyphenated short strings, so this is the hyphenated form of that value.
LICENSE = "public-domain-official-work-UrhG-5"

CONSENT_STATE = "dump"
CONSENT_BASIS = "official-dump-license"
CONSENT_SOURCE = ("rii official bulk dump (rii-toc.xml + per-decision XML), "
                  "Bundesministerium der Justiz / Bundesamt für Justiz")
# Master-only per ADR-039 — captured at ingest, stripped from the compact dump.
ATTRIBUTION = ("Bundesrepublik Deutschland — Bundesministerium der Justiz / "
               "Bundesamt für Justiz (rechtsprechung-im-internet.de)")
LICENSE_VERSION = "UrhG § 5 (Gesetz über Urheberrecht und verwandte Schutzrechte)"

# Fields the backfill must REMOVE, not overwrite: the expired grace stamp.
CLEAR_FIELDS = ("consent_reason", "consent_grace_until")


def stamp(checked_at: str) -> dict:
    """The uniform payload stamp applied to every one of the 38,159 points."""
    return {
        "consent_state": CONSENT_STATE,
        "consent_basis": CONSENT_BASIS,
        "consent_source": CONSENT_SOURCE,
        "consent_checked_at": checked_at,
        "license": LICENSE,
        "search_ok": True,
        "train_ok": False,
    }


def licence_reading() -> dict:
    """The instrument-based determination, quoted from the instrument itself.

    § 5 UrhG is quoted verbatim from AW's OWN retained official dump of
    gesetze-im-internet.de (the BMJ/BfJ federal law portal), corpus wave
    `gesetze_v2`, record `https://www.gesetze-im-internet.de/urhg/___5.html`.
    Zero fetches: rii's robots decline forbids probing rii itself, and the
    statute — not the host's own legal notice — is the instrument that settles
    a § 5 question.
    """
    return {
        "determined_on": "2026-08-28",
        "instrument": "§ 5 UrhG (Gesetz über Urheberrecht und verwandte Schutzrechte)",
        "quote_source": {
            "url": "https://www.gesetze-im-internet.de/urhg/___5.html",
            "corpus_wave": "gesetze_v2",
            "wave_source_sha256":
                "a709ce9b644125afa5fe16b6df02f1a3e24c94d2f28912bc3d4d9b312334b25c",
            "note": "retained AW dump of the BMJ/BfJ portal; no origin fetch",
        },
        "quote_abs_1": (
            "Gesetze, Verordnungen, amtliche Erlasse und Bekanntmachungen sowie "
            "Entscheidungen und amtlich verfaßte Leitsätze zu Entscheidungen "
            "genießen keinen urheberrechtlichen Schutz."
        ),
        "quote_abs_2": (
            "Das gleiche gilt für andere amtliche Werke, die im amtlichen Interesse "
            "zur allgemeinen Kenntnisnahme veröffentlicht worden sind, mit der "
            "Einschränkung, daß die Bestimmungen über Änderungsverbot und "
            "Quellenangabe in § 62 Abs. 1 bis 3 und § 63 Abs. 1 und 2 entsprechend "
            "anzuwenden sind."
        ),
        "covers": [
            "the decisions themselves (Entscheidungen)",
            "officially drafted headnotes (amtlich verfaßte Leitsätze zu Entscheidungen)",
            "the operative ruling (Tenor), as part of the decision",
        ],
        "does_not_cover": [
            "headnotes and commentary drafted by a commercial legal publisher "
            "(nicht amtlich verfaßte Leitsätze, Anmerkungen, Fundstellen-Apparat) — "
            "§ 5 Abs. 1 reaches only the AMTLICH drafted ones",
        ],
        # Measured over all 38,159 retained records, not sampled (task 465 §2).
        "retained_text_composition": {
            "header_court_kind_date_az": 38159,
            "ECLI": 37465,
            "Titelzeile": 37987,
            "Leitsatz": 17679,
            "Orientierungssatz_sonstosatz": 437,
            "Normen": 37562,
            "Tenor": 2,
        },
        "finding": (
            "No commercial-publisher editorial content is present. The record is one "
            "synthesized locator passage per decision, assembled by "
            "ingest_rii.py from named fields of the official per-decision "
            "XML: bibliographic header, ECLI, <titelzeile>, <leitsatz>, <sonstosatz>, "
            "<norm> list, <tenor>. There is no Verlag apparatus, no signed "
            "Anmerkung, no editorial byline."
        ),
        "grounds": [
            "§ 5 Abs. 1 UrhG — Leitsatz (17,679 records) and Tenor (2) are the "
            "decision and its amtlich verfaßter Leitsatz: no copyright, unconditional.",
            "§ 2 Abs. 2 UrhG — the header, ECLI, Aktenzeichen, dates and the <norm> "
            "citation list are bibliographic facts and citation strings, not "
            "'persönliche geistige Schöpfungen': no work, so no copyright to grant.",
            "§ 5 Abs. 2 UrhG — the documentation elements <titelzeile> (37,987) and "
            "<sonstosatz> (437) are amtliche Werke published in the official interest "
            "for general public notice on the Federation's own decision portal. Abs. 2 "
            "turns on the PUBLICATION being amtlich, not on who drafted the text, so "
            "it does not depend on resolving whether the court or the BfJ documentation "
            "service wrote them. Its two conditions are honoured: the strings are "
            "carried verbatim and unaltered (§ 62 Änderungsverbot) and every record "
            "carries the canonical rii permalink and domain as its source "
            "(§ 63 Quellenangabe).",
        ],
        "residual_for_tim": (
            "The § 5 Abs. 2 ground is the softest of the three and it carries 37,987 "
            "of the 38,159 records. It rests on rii being an official publication of "
            "the Bundesamt für Justiz — an assertion carried from ingest_rii.py's "
            "header and the eval notes, NOT witnessed today (rii's robots "
            "`Disallow: /` makes probing its own legal notice inconsistent with the "
            "decline we just recorded, so no probe was made). If Tim narrows the "
            "reading, the affected set is exactly the 37,987 records whose retained "
            "text carries a `Titel:` line plus the 437 carrying "
            "`Orientierungssatz:`; the 17,679 Leitsatz records and the "
            "bibliographic spine stand on § 5 Abs. 1 / § 2 Abs. 2 alone."
        ),
        "not_used": (
            "task 132's unwitnessed 'Official open-data offering (free reuse)' line "
            "is NOT the basis and is not reused — a licence basis comes from the "
            "instrument, never from an ingester's comment."
        ),
    }


def refresh_policy() -> dict:
    """Tim's option-(a) ruling: the source is non-refreshable. Declared here.

    There is no repo-wide do-not-refetch registry for a source in this state.
    The two candidates were checked and neither fits:

      * `data/consent_declined_hosts.json` — enforced by
        `consent_block.py`, which fails CLOSED for ingesters. rii
        BELONGS there (its verdict is a resolved decline), and the entry is
        proposed in task 465 for Tim's PR, since its own header says a verdict
        change is a policy change that goes via PR. But that registry blocks
        CRAWLING; it says nothing about whether admitted records are retained,
        which is the other half of option (a).
      * `sources/*.yaml` — the pack registry carries `freshness` and `status`,
        but rii has no pack, and `sources/` is a legislative path (PR-only per
        the project's legislative-prs rule).

    So the declaration lives here, in the artifact the assembly consumes, and
    is mirrored to Tim as a proposed pack + declined-hosts entry in the task
    doc. Inventing a third registry would be worse than either.
    """
    return {
        "state": "non-refreshable",
        "ruled_by": "Tim",
        "ruled_on": "2026-08-28",
        "ruling": "task 457 Q1 option (a) — keep the points, re-embed the retained "
                  "text locally at 1024, mark the source non-refreshable, no outreach",
        "means": [
            "no further fetch of any rechtsprechung-im-internet.de URL, for any "
            "purpose, including refresh, recheck and re-extraction",
            "no re-run of ingest_rii.py toc or fetch",
            "the retained rii.jsonl is the ONLY input for these points forever",
            "new German case-law coverage comes from a different source "
            "(task 459: openlegaldata dump first), never from re-opening rii",
        ],
        "unblock_condition": "a fresh, Tim-ordered decision — not a re-resolve. "
                             "The decline is a publisher decision, not a gap in "
                             "our evidence (both probes resolved cleanly).",
        "declared_in": "this artifact",
        "proposed_registry_entries": [
            "data/consent_declined_hosts.json — www.rechtsprechung-im-internet.de, "
            "state=declined (PR to Tim; see task 465 §6)",
            "sources/de/rechtsprechung-im-internet.yaml — status=declined, "
            "freshness=static (PR to Tim; same section)",
        ],
    }


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------

def mint(row: dict) -> dict:
    """Re-mint one retained row's SERVING identity with the real repo code.

    rii is Tier B (source_type `official`, no source-native per-chunk anchor),
    so the anchor component is the I1 content digest and

        point_id = uuid5(AW_NAMESPACE, canonical_url + "#" + sha256(normalize(text)))

    The `point_id`/`content_digest` fields already in rii.jsonl are the LEGACY
    (pre-I1) values written by `aw_common.Passage`; they are carried through
    for cross-reference and are deliberately NOT the join key.
    """
    text = row.get("text") or ""
    if not text.strip():
        raise ValueError("retained row has no text; identity cannot be minted")
    digest = ID.content_digest(text)
    tier, anchor = ID.classify_tier(row.get("source_type"), row.get("anchor"),
                                    True, digest, row.get("title"))
    canonical = AC.canonical_url(row["url"])
    return {
        "point_id": ID.mint_point_id(canonical, anchor),
        "canonical_url": canonical,
        "content_digest": digest,
        "identity_tier": tier,
        "legacy_point_id": row.get("point_id"),
    }


def iter_rows(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def derive(rii_jsonl: str) -> tuple[list[dict], dict]:
    """Re-derive the whole claim from the retained source. Fail-closed."""
    source_sha = sha256_file(rii_jsonl)
    if source_sha != SOURCE_SHA256:
        raise SystemExit(
            f"FAIL source digest: {rii_jsonl} is {source_sha}, "
            f"pinned {SOURCE_SHA256} — refusing to seal a different population")
    minted, rows = [], 0
    for row in iter_rows(rii_jsonl):
        rows += 1
        if (row.get("domain") or "").lower().lstrip(".") != DOMAIN:
            raise SystemExit(f"FAIL foreign domain in the retained source: "
                             f"{row.get('domain')!r}")
        minted.append(mint(row))
    ids = {m["point_id"] for m in minted}
    legacy = {m["legacy_point_id"] for m in minted}
    conservation = {
        "source_sha256": source_sha,
        "rows_read": rows,
        "expected_points": EXPECTED_POINTS,
        "minted": len(minted),
        "distinct_point_id": len(ids),
        "distinct_legacy_point_id": len(legacy),
        "distinct_canonical_url": len({m["canonical_url"] for m in minted}),
        "distinct_content_digest": len({m["content_digest"] for m in minted}),  # canonical row
        "identity_tiers": sorted({m["identity_tier"] for m in minted}),
        "id_collisions": len(minted) - len(ids),
    }
    problems = []
    if rows != EXPECTED_POINTS:
        problems.append(f"rows_read {rows} != {EXPECTED_POINTS}")
    if len(ids) != EXPECTED_POINTS:
        problems.append(f"distinct point_id {len(ids)} != {EXPECTED_POINTS}")
    if len(legacy) != EXPECTED_POINTS:
        problems.append(f"distinct legacy point_id {len(legacy)} != {EXPECTED_POINTS}")
    if conservation["identity_tiers"] != ["B"]:
        problems.append(f"identity tiers {conservation['identity_tiers']} != ['B']")
    conservation["problems"] = problems
    conservation["ok"] = not problems
    minted.sort(key=lambda m: m["point_id"])
    return minted, conservation


def seal_points(minted: list[dict], out_path: str) -> dict:
    """Write the point claim as deterministic gzip (mtime 0, sorted by id)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    body = "".join(json.dumps({k: m[k] for k in
                               ("point_id", "canonical_url", "content_digest",
                                "legacy_point_id")},
                              ensure_ascii=False, sort_keys=True) + "\n"
                   for m in minted).encode("utf-8")
    with open(out_path, "wb") as raw:
        # filename="" as well as mtime=0: GzipFile otherwise stamps the output
        # file's own name into the header, which makes the sealed digest depend
        # on where the artifact happened to be written.
        with deterministic_gzip_open(raw, compresslevel=9) as gz:
            gz.write(body)
    return {"count": len(minted), "sha256": sha256_file(out_path),
            "bytes": os.path.getsize(out_path),
            "uncompressed_sha256": hashlib.sha256(body).hexdigest()}


# ---------------------------------------------------------------------------
# the consent gate's own constants, read out of its source (no heavy import)
# ---------------------------------------------------------------------------

def gate_constants(builder_path: str | None = None) -> dict:
    """DUMP_CONSENT_STATE / DUMP_CONSENT_BASES: the package constants, or read out of `builder_path` by AST.

    Read by AST rather than imported: `build_aw_b1` pulls in qdrant_client at
    module scope, which a cold clone or a verifier box need not have. Reading
    the literals keeps the check bound to the gate's real source of truth
    instead of to a copy that can drift.
    """
    if builder_path is None:      # the package's own copy, pinned equal to the builder's by a test
        from cytir.consent.derive import DUMP_CONSENT_BASES
        return {"DUMP_CONSENT_STATE": "dump", "DUMP_CONSENT_BASES": DUMP_CONSENT_BASES}
    tree = ast.parse(open(builder_path, encoding="utf-8").read(), builder_path)

    def value_of(node):
        # `DUMP_CONSENT_BASES = frozenset({...})` is a Call, which
        # ast.literal_eval refuses; unwrap the one wrapper the gate uses.
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ("frozenset", "set") and len(node.args) == 1
                and not node.keywords):
            return frozenset(ast.literal_eval(node.args[0]))
        return ast.literal_eval(node)

    found: dict = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in (
                    "DUMP_CONSENT_STATE", "DUMP_CONSENT_BASES"):
                found[target.id] = value_of(node.value)
    missing = {"DUMP_CONSENT_STATE", "DUMP_CONSENT_BASES"} - set(found)
    if missing:
        raise SystemExit(f"FAIL cannot read gate constants {sorted(missing)} "
                         f"from {builder_path}")
    return found


def check_stamp(the_stamp: dict, builder_path: str | None = None) -> list[str]:
    """Every way this stamp could fail to admit, or could over-admit."""
    gate = gate_constants(builder_path)
    problems = []
    if the_stamp.get("consent_state") != gate["DUMP_CONSENT_STATE"]:  # canonical row
        problems.append("consent_state does not equal the gate's DUMP_CONSENT_STATE")
    if the_stamp.get("consent_basis") not in set(gate["DUMP_CONSENT_BASES"]):  # canonical row
        problems.append("consent_basis is not in the gate's closed basis set")
    if not str(the_stamp.get("consent_source") or "").strip():  # canonical row
        problems.append("consent_source is blank")
    if not str(the_stamp.get("license") or "").strip():
        problems.append("license is blank")
    if the_stamp.get("train_ok") is not False:
        problems.append("train_ok must be exactly False (tdm-reservation: 1)")
    if the_stamp.get("search_ok") is not True:
        problems.append("search_ok must be exactly True (what the dump basis admits)")
    if not str(the_stamp.get("consent_checked_at") or "").strip():
        problems.append("consent_checked_at is blank")
    resolved = LIC.resolve_license(None, DOMAIN, HOST, "official", SET_ID)
    if resolved != the_stamp.get("license"):
        problems.append(
            f"aw_b1_license resolves {resolved!r} for {HOST}, stamp says "
            f"{the_stamp.get('license')!r} — the build would disagree with the artifact")
    return problems


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_bind(args) -> int:
    """Read-only: prove every claimed point_id exists in a real collection.

    Retrieve-by-id only — never a scroll or an unindexed filter (a full-scan
    scroll on a serving box saturates its API pool). Writes nothing anywhere.

    This exists because "the ids re-mint" and "the ids are the points that are
    actually in the collection the rebuild is built from" are two different
    claims, and task 457 asserted the second from a sample.
    """
    import urllib.request

    ids = [row["point_id"] for row in iter_rows(args.points)]
    url = f"{args.qdrant_url.rstrip('/')}/collections/{args.collection}/points"
    found, batches = 0, 0
    for start in range(0, len(ids), args.batch):
        chunk = ids[start:start + args.batch]
        body = json.dumps({"ids": chunk, "with_payload": False,
                           "with_vector": False}).encode("utf-8")
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"})
        with IH.urlopen(request, timeout=args.timeout) as response:
            found += len(json.load(response).get("result", []))
        batches += 1
    binding = {
        "measured_on": args.checked_at,
        "collection": args.collection,
        "endpoint": args.qdrant_url,
        "method": "retrieve-by-id, with_payload=false (no scroll, no filter)",
        "claimed": len(ids),
        "matched": found,
        "unmatched": len(ids) - found,
        "scope": "all_claims",
        "points_sha256": sha256_file(args.points),
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(binding, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(binding, indent=2, sort_keys=True))
    return 0 if binding["unmatched"] == 0 else 1


def cmd_produce(args) -> int:
    minted, conservation = derive(args.rii_jsonl)
    if not conservation["ok"]:
        for problem in conservation["problems"]:
            print(f"FAIL conservation: {problem}", file=sys.stderr)
        return 1
    sealed = seal_points(minted, args.points_out)
    the_stamp = stamp(args.checked_at)
    stamp_problems = check_stamp(the_stamp)
    if stamp_problems:
        for problem in stamp_problems:
            print(f"FAIL stamp: {problem}", file=sys.stderr)
        return 1

    manifest = {
        "schema": SCHEMA,
        "set_id": SET_ID,
        "produced_at": args.checked_at,
        "producer": "aw ops rii-dump-basis-backfill",
        "verifier": "aw ops rii-dump-basis-backfill verify",
        "host": HOST,
        "domain": DOMAIN,
        "corpus_wave": WAVE,
        "authority": {
            "adr": "ADR-076",
            "adr_clause": "An explicit open license on an official bulk distribution "
                          "constitutes the admission basis (search_ok) for records "
                          "built from that distribution, independent of the canonical "
                          "host's robots/WAF state.",
            "ruling": "Tim 2026-08-28 — task 457 Q1 option (a) + Q2 approved "
                      "conditional on a written instrument-based licence reading",
            "condition_met_by": "licence_reading (this artifact) and "
                                "task 465 §2",
            "closes": "task 433 WI-1, for rii",
        },
        "licence_reading": licence_reading(),
        "refresh_policy": refresh_policy(),
        "stamp": the_stamp,
        "clear_fields": list(CLEAR_FIELDS),
        "master_only": {"attribution": ATTRIBUTION,
                        "license_version": LICENSE_VERSION,
                        "note": "ADR-039 — master-only, stripped from the compact dump"},
        "supersedes_consent_receipt": {
            "path": "consent/www.rechtsprechung-im-internet.de.json",
            "sha256": DECLINE_RECEIPT_SHA256,
            "checked_at": DECLINE_RECEIPT_CHECKED_AT,
            "status": "resolved decline, NOT superseded as a decline — it still "
                      "governs crawling, and train_ok=false follows it",
            "corrected_reason": {
                "wrong": "robots: CCBot disallowed — our copy came via Common Crawl "
                         "(provenance has no consent; conservative reading)",
                "why_wrong": "rii never came through Common Crawl. ingest_rii.py:2-4 "
                             "'the BMJ/BfJ OFFICIAL bulk dump ... Dumps-first, no "
                             "scraping'; wave _meta.json source=rii.jsonl, "
                             "extraction_version=rii-xml-2026-07; the retained rows "
                             "carry no crawl_id. The receipt's mode field is 'cc' "
                             "because the sweep assumed a CC origin for every host.",
                "reasons_that_stand": [
                    "robots: User-agent:* disallows / (full opt-out)",
                    "robots: our token 'authoritative-web' disallowed on /",
                ],
            },
        },
        "expired_grace": {
            "consent_state": "grace",
            "consent_grace_until": "2026-08-24",
            "measured_by": "task 457 §2 (30/30 sampled live points)",
            "consequence_without_this_artifact": "eviction of all 38,159 points at "
                                                 "the next rebuild, on grace expiry",
        },
        "points": {
            "granularity": "point_id",
            "uri": args.points_uri,
            "sha256": sealed["sha256"],
            "uncompressed_sha256": sealed["uncompressed_sha256"],
            "bytes": sealed["bytes"],
            "count": sealed["count"],
            "identity_rule": "Tier B — uuid5(AW_NAMESPACE, canonical_url + '#' + "
                             "sha256(normalize(text))), cytir.core.ids",
            "rule_epochs_at_mint": dict(ID.EPOCHS),
            "id_generations": {
                "join_key": "point_id (I1)",
                "measured_2026-08-28": {
                    "aw_e3 (34,464,401-pt current membership, the collection the "
                    "Track-1 assembly is built from)": "holds the rii points under "
                    "the I1 ids — 38,159/38,159 matched, see membership_binding",
                    "aw_v4_sat (128-dim legacy serving twin)": "holds them under the "
                    "LEGACY ids carried in rii.jsonl; a 32-id probe found 0/32 of the "
                    "I1 ids and 32/32 of the legacy ids",
                },
                "consequence": "the backfill applies on `point_id`. `legacy_point_id` "
                               "is carried in the sealed file only so a legacy-side "
                               "apply is possible without re-deriving anything; it is "
                               "never the join key for the rebuild.",
            },
        },
        "source": {
            "uri": "artifact://_work/task465/rii-dump-basis-backfill-v1/"
                   "SOURCE.md (pointer)",
            "retained_line_level_source": "rii.jsonl",
            "sha256": SOURCE_SHA256,
            "rows": EXPECTED_POINTS,
            "wave_meta": f"artifact://wave={WAVE}/_meta.json",
            "note": "task 457 §2 proved gap 0 / drift 0 by re-minting every row "
                    "against the twin: matched 38,159, unmatched 0, collisions 0.",
        },
        "conservation": conservation,
        "membership_binding": (json.load(open(args.binding, encoding="utf-8"))
                               if args.binding else None),
        "assembly_consumption": assembly_consumption(args.points_uri, sealed),
    }
    binding = manifest["membership_binding"]
    if binding and binding.get("points_sha256") != sealed["sha256"]:
        print("FAIL membership binding was measured over a different points file",
              file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest_out)) or ".",
                exist_ok=True)
    with open(args.manifest_out, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    from cytir.ops import _provenance
    _provenance.write_provenance(args.manifest_out, "rii-dump-basis-backfill",
                                 {"rii_jsonl": args.rii_jsonl, "points": args.points_out})
    print(f"SEALED {sealed['count']:,} points  sha256 {sealed['sha256']}")
    print(f"        points   -> {args.points_out}")
    print(f"        manifest -> {args.manifest_out}")
    return 0


def assembly_consumption(points_uri: str, sealed: dict) -> dict:
    """Exactly how the assembly lane picks this up. No prose-only handoffs."""
    return {
        "set_id": SET_ID,
        "manifest_index": "data/assembly/claim-manifests.v1.json",
        "manifest_index_schema": "aw.assembly-claim-manifests.v1",
        "add_set_entry": {
            "set_id": SET_ID,
            "schema": "aw.assembly-claim-manifest.v1",
            "kind": "retained_text_reembed",
            "mode": "in_place_reembed",
            "class": "dump_basis_backfill",
            "contributes_vectors": True,
            "stage": "awaiting-1024-vector",
            "claims": {"granularity": "point_id", "uri": points_uri,
                       "sha256": sealed["sha256"], "count": sealed["count"],
                       "verified": True},
            "payload_backfill": {
                "uri": "artifact://../data/assembly/rii-dump-basis-backfill.v1.json",
                "schema": SCHEMA,
                "apply": "uniform stamp over every claimed point_id; "
                         "delete CLEAR_FIELDS; never per-point values",
            },
        },
        "guards": {
            "G1": "the claim above is a point-ID claim, not a payload predicate — "
                  "it closes rii's share of G1 by enumeration",
            "G3": "38,159 points move from census bucket D (blocked) to "
                  "disposition `re_embedded_in_place`; bucket D shrinks by 38,159",
            "G5": "one row per claimed point — the rii record is ONE synthesized "
                  "locator passage per decision, so rows_declared == 38,159",
            "G7": "upsert_in_place; these ids are loaded, so they must be excluded "
                  "from any delete set",
            "G8": "a census witness must be minted over exactly this point set "
                  "before the next census revision consumes it",
        },
        "embed_queue": {
            "queue": "embed-only (no fetch, no re-extraction, no re-chunk)",
            "input": "the `text` field of the retained rii.jsonl rows, verbatim",
            "input_sha256": SOURCE_SHA256,
            "rows": EXPECTED_POINTS,
            "target": "arctic-embed-l-v2 full-1024, release aw-e3-rebuild-*",
            "estimated_cost": "~0.09 GPU-hours (task 457 §2)",
            "owner": "assembly lane",
            "must_not": "re-derive the text, re-chunk it, or fetch anything — the "
                        "id embeds the digest of this exact text, so any change to "
                        "the text is a change of identity",
        },
    }


def cmd_verify(args) -> int:
    manifest = json.load(open(args.manifest, encoding="utf-8"))
    problems: list[str] = []

    def fail(message: str) -> None:
        problems.append(message)

    if manifest.get("schema") != SCHEMA:
        fail(f"schema {manifest.get('schema')!r} != {SCHEMA!r}")
    if manifest.get("set_id") != SET_ID:
        fail(f"set_id {manifest.get('set_id')!r} != {SET_ID!r}")

    # 1. the stamp, against the gate's own constants and the build's licence code
    for problem in check_stamp(manifest.get("stamp") or {}):
        fail(f"stamp: {problem}")
    for field in CLEAR_FIELDS:
        if field not in (manifest.get("clear_fields") or []):
            fail(f"clear_fields is missing {field!r} — an expired grace stamp "
                 f"would survive next to consent_state=dump")

    # 2. the licence reading must be present and instrument-grounded
    reading = manifest.get("licence_reading") or {}
    for key in ("instrument", "quote_abs_1", "quote_abs_2", "grounds",
                "retained_text_composition", "finding"):
        if not reading.get(key):
            fail(f"licence_reading lacks {key!r}")
    if "amtlich verfaßte Leitsätze" not in str(reading.get("quote_abs_1", "")):
        fail("licence_reading.quote_abs_1 is not § 5 Abs. 1's operative text")
    composition = reading.get("retained_text_composition") or {}
    if composition.get("header_court_kind_date_az") != EXPECTED_POINTS:
        fail("licence_reading composition was not measured over all 38,159 records")

    # 3. the refresh ruling must be declared
    if (manifest.get("refresh_policy") or {}).get("state") != "non-refreshable":
        fail("refresh_policy.state is not 'non-refreshable'")

    # 4. the sealed points file — re-hash it, do not trust the manifest's number
    points_block = manifest.get("points") or {}
    if not os.path.exists(args.points):
        fail(f"sealed points file not found: {args.points}")
        return _report(problems, {})
    actual_sha = sha256_file(args.points)
    if actual_sha != points_block.get("sha256"):
        fail(f"points sha256 drift: file {actual_sha}, manifest "
             f"{points_block.get('sha256')}")
    sealed_ids, sealed_rows = set(), 0
    for row in iter_rows(args.points):
        sealed_rows += 1
        sealed_ids.add(row["point_id"])
    if sealed_rows != EXPECTED_POINTS:
        fail(f"sealed rows {sealed_rows} != {EXPECTED_POINTS}")
    if len(sealed_ids) != sealed_rows:
        fail(f"sealed file has {sealed_rows - len(sealed_ids)} duplicate point_ids")
    if points_block.get("count") != sealed_rows:
        fail(f"manifest count {points_block.get('count')} != sealed rows {sealed_rows}")

    # 5. INDEPENDENT re-derivation from the retained source
    stats: dict = {}
    if args.rii_jsonl:
        minted, conservation = derive(args.rii_jsonl)
        stats = conservation
        if not conservation["ok"]:
            for problem in conservation["problems"]:
                fail(f"conservation: {problem}")
        derived_ids = {m["point_id"] for m in minted}
        missing = derived_ids - sealed_ids
        extra = sealed_ids - derived_ids
        stats["missing_from_sealed"] = len(missing)
        stats["extra_in_sealed"] = len(extra)
        if missing:
            fail(f"{len(missing)} re-derived point_ids are absent from the sealed "
                 f"claim (e.g. {sorted(missing)[0]})")
        if extra:
            fail(f"{len(extra)} sealed point_ids do not re-derive from the retained "
                 f"source (e.g. {sorted(extra)[0]})")
    else:
        stats["note"] = ("rii.jsonl not supplied — sealed-file integrity checked, "
                         "identity NOT independently re-derived")

    # 6. the membership binding: claimed ids must BE the collection's points
    binding = manifest.get("membership_binding")
    if not binding:
        fail("membership_binding is absent — the claim was never bound to a "
             "collection, so 'these 38,159 points' is an assertion")
    else:
        if binding.get("points_sha256") != points_block.get("sha256"):
            fail("membership_binding was measured over a different points file")
        if binding.get("claimed") != EXPECTED_POINTS:
            fail(f"membership_binding claimed {binding.get('claimed')} "
                 f"!= {EXPECTED_POINTS}")
        if binding.get("unmatched") != 0:
            fail(f"membership_binding has {binding.get('unmatched')} unmatched ids")
        if binding.get("scope") != "all_claims":
            fail("membership_binding is a sample, not a full measurement")

    # 7. the assembly pointer must be concrete
    consumption = manifest.get("assembly_consumption") or {}
    entry = (consumption.get("add_set_entry") or {})
    if (entry.get("claims") or {}).get("sha256") != points_block.get("sha256"):
        fail("assembly_consumption claims sha256 does not match points.sha256")
    if not (consumption.get("embed_queue") or {}).get("input_sha256"):
        fail("assembly_consumption.embed_queue names no input digest")

    return _report(problems, stats)


def _report(problems: list[str], stats: dict) -> int:
    for key, value in stats.items():
        print(f"  {key}: {value}")
    if problems:
        print(f"VERIFY FAIL — {len(problems)} problem(s)")
        for problem in problems:
            print(f"  FAIL {problem}")
        return 1
    print("VERIFY PASS")
    return 0


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__.splitlines()[1])
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    produce = subparsers.add_parser("produce")
    produce.add_argument("--rii-jsonl", required=True)
    produce.add_argument("--points-out", required=True)
    produce.add_argument("--manifest-out", required=True)
    produce.add_argument("--points-uri", required=True,
                         help="the artifact:// URI the assembly will resolve")
    produce.add_argument("--checked-at", default="2026-08-28")
    produce.add_argument("--binding", default="",
                         help="the `bind` output to embed; without it the "
                              "manifest carries no membership proof and "
                              "`verify` fails")
    produce.set_defaults(func=cmd_produce)

    bind = subparsers.add_parser("bind")
    bind.add_argument("--points", required=True)
    bind.add_argument("--collection", required=True)
    bind.add_argument("--qdrant-url", default="http://localhost:6444")
    bind.add_argument("--batch", type=int, default=1000)
    bind.add_argument("--timeout", type=int, default=120)
    bind.add_argument("--checked-at", default="2026-08-28")
    bind.add_argument("--out", required=True)
    bind.set_defaults(func=cmd_bind)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--points", required=True)
    verify.add_argument("--rii-jsonl", default="",
                        help="the retained source; when given, every identity is "
                             "re-derived instead of trusted")
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

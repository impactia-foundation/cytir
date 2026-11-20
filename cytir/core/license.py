#!/usr/bin/env python3
# Moved from the flat module `aw_b1_license` (task 677, task 643 S5); the old path is an alias shim.
"""aw_b1_license.py — per-record `license` resolution for the aw_b1 build.

ADR-039 requires a per-record SPDX-style `license` in the compact redistributable
dump. The preflight audit (GAP #8) found it is NOT uniform across staging:
wave-2 (the 1.85M pack) samples lack it, and no per-wave assertion exists. The
build MUST NOT silently pass an absent license — it either resolves one from
wave/source metadata or records the record as a licensing EXCEPTION in the
MANIFEST (counted, never hidden).

This module is a best-effort resolver over the well-known sources plus an
explicit `None -> exception` fall-through. It is deliberately conservative: an
unknown source yields `None` (an exception, surfaced in the manifest), never a
guessed permissive license.

**Task 629 P8 + task 636 (both 2026-09-22, same day, same defect, found
independently and reconciled here).** The table had drifted from the rule it
states: `canada.ca` carried `OGL-Canada-2.0` INHERITED from its
`laws-lois.justice.gc.ca` / `bankofcanada.ca` siblings, never read from
canada.ca's own terms — which forbid commercial redistribution without
permission. Task 629 found and corrected `canada.ca` itself (178,417 live rows
over-claimed; `task 629 pool-a-licence.v1.json`) and escalated the
two untouched siblings; task 636 read those two instruments, found `admin.ch`
sharing the same uncited-entry shape, and found a second, unrelated mechanism
stamping `license` from a domain's topic class with no instrument read at all
(escalated further, not fixed here — see task 636 §6).
Every entry now carries a `citation`: a URL + operative-sentence quote (fetched
from the build box, never summarized — memory `webfetch-summarizer-confabulates`) or,
where the grant is statutory rather than a terms page, the instrument's legal
citation. A citation is *load-bearing*, not documentation:
`test_aw_b1_license.py` fails the build if any entry lacks one (memory
`licence-basis-from-the-instrument`).

Resolution order (first hit wins):
  1. an already-present, non-empty, non-NOASSERTION `license` on the record
     (trust the ingester)
  2. exact host match, then the LONGEST matching host *suffix* in `_HOST_LICENSE`
     (so `open.canada.ca` — a real Open Government Licence grant — resolves
     independently of, and is never shadowed by, the generic `canada.ca` entry;
     see task 636 §4 for why the previous registrable-domain shortcut could not
     tell the two apart)
  3. source_type rules (entity_card synthesized -> CC0; reference/docs unknown)
  4. None  -> LICENSING EXCEPTION (manifest.sources[].license_exceptions++)
"""
from __future__ import annotations

from typing import NamedTuple


class _Entry(NamedTuple):
    license: str
    citation: str  # URL + operative-sentence quote, or a statutory citation


# host (exact or suffix, matched host-first then longest-suffix-wins) -> _Entry.
# Every value's basis is the DECLARING INSTRUMENT — never the fact that the
# content happens to be hosted on a government domain (memory
# `licence-basis-from-the-instrument`). Where the instrument is a live terms
# page, the citation carries the fetch date so staleness is visible.
_HOST_LICENSE: dict[str, _Entry] = {
    # --- Switzerland -----------------------------------------------------
    # Fedlex serves federal law and is the AW SPA/JS-rendered — curl cannot
    # read it directly (same limitation noted elsewhere in this repo for
    # claude.ai shares). The grant is not a terms-page courtesy, it is
    # statutory: Art. 5 para. 1 let. c of the Federal Act on Copyright and
    # Related Rights (URG/LDA, SR 231.1) excludes "Erlasse, Beschlüsse und
    # Urteile" of public authorities from copyright altogether — the same
    # doctrine already accepted below for rechtsprechung-im-internet.de's
    # German equivalent (§ 5 UrhG). LOWER CONFIDENCE: verify with a
    # JS-capable fetch before the next dump; recheck_after 2026-12-01.
    # RULING 53 (Tim 2026-10-22, task 1040): the value is `public-domain`, the one the fedlex door
    # stamps and `sources/ch/fedlex.yaml` declares; the old `SwissGov-Fedlex-Terms` string named no
    # licence the statute grants (the basis below is the same and is unchanged).
    "fedlex.admin.ch": _Entry(
        "public-domain",
        "Art. 5 para. 1 let. c URG/LDA (SR 231.1) — official enactments, "
        "decisions and their grounds are excluded from copyright protection. "
        "Page itself is a JS-rendered SPA (fetched 2026-09-22, no server-"
        "rendered text available to curl); statutory basis not a live quote.",
    ),
    # swisstopo's geodata has been open government data since 2021 — a real,
    # verified grant, DISTINCT from bare admin.ch (below).
    "swisstopo.admin.ch": _Entry(
        "swisstopo-OGD-2021",
        "https://www.swisstopo.admin.ch/en/terms-and-conditions (fetched "
        "2026-09-22): \"Since March 1, 2021, swisstopo's standard digital "
        "products, such as digital maps, aerial photographs, and landscape "
        "models, have been made available free of charge as open government "
        "data and can be used freely.\"",
    ),
    # Bare admin.ch is NOT swisstopo and NOT fedlex — task 636 found it was
    # sharing swisstopo's old "SwissGov-Terms" value with no citation, and
    # the site's own terms say the opposite of "reuse allowed": general
    # admin.ch content (federal-agency pages, media releases) requires
    # written consent to reproduce. Downgraded to the same fair-use/
    # snippet-pointer posture as the Swiss news hosts below. `ch-gov-open-
    # cite`, stamped directly at ingest by the admin.ch media-release route
    # (sources/ch/admin-ch.yaml, task 293) from a ROBOTS reading rather than
    # this copyright clause, is the same defect one layer up the pipeline —
    # flagged, not fixed here (this table cannot override an already-
    # present `license`; see task 636 §6).
    "admin.ch": _Entry(
        "copyright-fairuse-snippet-pointer",
        "https://www.admin.ch/gov/en/start/terms-and-conditions.html "
        "(fetched 2026-09-22): \"Copyright and any other rights relating to "
        "texts, illustrations, photos or any other data available on the "
        "Federal authorities' websites are the exclusive property of the "
        "federal authorities... Any reproduction requires the prior written "
        "consent of the copyright holder.\"",
    ),
    # --- Wikipedia / Wikimedia / Wikidata ---------------------------------
    "wikipedia.org": _Entry(
        "CC-BY-SA-4.0",
        "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en "
        "(fetched 2026-09-22): content is made available \"under a free "
        "license or in the public domain\"; Wikipedia article text is "
        "CC BY-SA 4.0 per the Terms of Use §7.",
    ),
    "wikimedia.org": _Entry(
        "CC-BY-SA-4.0",
        "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en "
        "(fetched 2026-09-22) — same grant as wikipedia.org above.",
    ),
    "wikidata.org": _Entry(
        "CC0-1.0",
        "https://www.wikidata.org/wiki/Wikidata:Licensing (fetched "
        "2026-09-22): \"All structured data in the main, property and "
        "lexeme namespaces is made available under the Creative Commons "
        "CC0 License (Public domain).\"",
    ),
    # --- Germany -----------------------------------------------------------
    "gesetze-im-internet.de": _Entry(
        "GermanGov-Terms",
        "§ 5 Abs. 1 UrhG (Gesetz über Urheberrecht und verwandte "
        "Schutzrechte) — Gesetze, Verordnungen, amtliche Erlasse and "
        "Entscheidungen are outside copyright protection; the portal is the "
        "Bundesministerium der Justiz's own official publication of that "
        "text (spot-checked 2026-09-22, page carries no usable terms text of "
        "its own to quote). Same statutory family as "
        "rechtsprechung-im-internet.de below.",
    ),
    # German federal court decisions (BMJ/BfJ official bulk dump). NOT inferred
    # from the host: § 5 Abs. 1 UrhG puts "Entscheidungen und amtlich verfaßte
    # Leitsätze zu Entscheidungen" outside copyright altogether, and the
    # documentation elements the retained record also carries (Titelzeile,
    # Orientierungssatz) are amtliche Werke under § 5 Abs. 2 — carried verbatim,
    # with the canonical permalink as the § 63 Quellenangabe. The written
    # determination is task 465 §2 and the
    # `licence_reading` block of data/assembly/rii-dump-basis-backfill.v1.json.
    # Same family as sources/ch/ch-courts.yaml's `public-domain` (URG Art. 5).
    "rechtsprechung-im-internet.de": _Entry(
        "public-domain-official-work-UrhG-5",
        "§ 5 Abs. 1 + Abs. 2 UrhG, applied in "
        "task 465 RII dump basis determination §2 and "
        "the RII dump-basis backfill record's `licence_reading` "
        "block — the written determination, not re-derived here.",
    ),
    # task 641 (2026-09-23): the OLD (`de_openlegaldata_v1`) family's sealed
    # packs (34,323 rows) refused with `unresolved license` because this
    # table had no entry for their host — NOT because the basis was
    # undeclared. `sources/de/openlegaldata.yaml` (admitted, one commit,
    # task 460 — c52cc5bc) already carries the ratified determination:
    # `license.default: "public_domain_official_work (UrhG §5)"`, quoting
    # OpenLegalData's own HuggingFace dataset card verbatim ("acts,
    # statutory instruments... and decisions (including court decisions)...
    # do not enjoy copyright protection (UrhG § 5)") and explicitly warning
    # NEVER to stamp the sibling `ODbL` value — that licence covers only
    # OpenLegalData's own database compilation/metadata, never the
    # underlying public-domain decision text (same discipline as
    # `rechtsprechung-im-internet.de` above; both are the same UrhG § 5
    # family, different official/aggregator hosts). Encoding the DECLARED
    # basis here, not re-deriving it.
    "de.openlegaldata.io": _Entry(
        "public-domain-official-work-UrhG-5",
        "sources/de/openlegaldata.yaml `license.default` + `license.notes` "
        "(admitted task 460, commit c52cc5bc) — the written determination, "
        "not re-derived here. Same statutory family as "
        "rechtsprechung-im-internet.de (§ 5 Abs. 1 UrhG); NEVER stamp this "
        "host's sibling `ODbL` value, which the same source file confines "
        "to OpenLegalData's own compilation/metadata.",
    ),
    # --- France (DILA / Etalab) ---------------------------------------------
    "legifrance.gouv.fr": _Entry(
        "Etalab-2.0",
        "Direct fetch WAF-blocked from the operator box 2026-09-22 (HTTP 403; memory "
        "`consent-probe-vantage` — record as unreadable, not declined). "
        "Corroborated by insee.fr's identical grant below (same Licence "
        "Ouverte / Etalab instrument, same DINUM open-data programme) and by "
        "this repo's own DILA full-ingest (PR #236, task 633). "
        "recheck_after 2026-10-01: refetch legifrance.gouv.fr's own "
        "\"Réutilisation des données\" page directly.",
    ),
    "insee.fr": _Entry(
        "Etalab-2.0",
        "https://www.insee.fr/fr/information/2008466 (fetched 2026-09-22): "
        "\"sous la Licence Ouverte / Open Licence version 2.0 (Etalab). "
        "Cette licence autorise la réutilisation libre, y compris à des "
        "fins commerciales.\"",
    ),
    # French DILA / service-public.fr — Licence Ouverte / Etalab 2.0 (task 629
    # P8, verified 2026-09-22 two ways): the live footer on both
    # www.service-public.gouv.fr and entreprendre.service-public.gouv.fr reads
    # "Sauf mention contraire, tous les contenus de ce site sont sous licence
    # etalab-2.0"; and the DILA XML dataset itself (task 629's actual pack
    # source) is catalogued on data.gouv.fr with license="fr-lo" (Licence
    # Ouverte) for both the particuliers and entreprendre fiche datasets.
    # Suffix-matches entreprendre.service-public.gouv.fr.
    "service-public.gouv.fr": _Entry(
        "Etalab-2.0",
        "https://www.service-public.gouv.fr and "
        "https://entreprendre.service-public.gouv.fr footers (fetched "
        "2026-09-22): \"Sauf mention contraire, tous les contenus de ce site "
        "sont sous licence etalab-2.0.\" Corroborated by the DILA dataset's "
        "own data.gouv.fr catalogue entry, license=\"fr-lo\".",
    ),
    # --- Canada --------------------------------------------------------------
    # `open.canada.ca` is the actual Open Government Licence – Canada
    # catalogue. It is a DIFFERENT site from bare canada.ca (below), and must
    # win on an exact/suffix match before the generic canada.ca entry can —
    # see the longest-suffix resolution order in `resolve_license`.
    "open.canada.ca": _Entry(
        "OGL-Canada-2.0",
        "https://open.canada.ca/en/open-government-licence-canada (fetched "
        "2026-09-22): \"The Information Provider grants you a worldwide, "
        "royalty-free, perpetual, non-exclusive licence to use the "
        "Information, including for commercial purposes...\" Version 2.0, "
        "\"Date modified: 2022-12-02\".",
    ),
    # Bare canada.ca — CORRECTED 2026-09-22 (task 629 P8, cross-verified by
    # task 636). General Government of Canada informational pages (news
    # releases, department pages) are Crown copyright with a NON-commercial-
    # only default grant, NOT the Open Government Licence, which only covers
    # material catalogued on open.canada.ca — a separate instrument. Verified
    # on both the general Terms page and a live news-release page whose own
    # footer links only to the Terms page, never to the OGL. 178,417 live rows
    # carried the inherited `OGL-Canada-2.0` value before this fix; see
    # task 629 pool-a-licence.v1.json and
    # task 636 A task-636 twin sample of
    # 178,417/178,417 matching rows found 0 under `open.canada.ca` — the whole
    # bucket was the general site.
    "canada.ca": _Entry(
        "CanadaGov-Terms-NC-Attribution",
        "https://www.canada.ca/en/transparency/terms.html (fetched "
        "2026-09-22): \"Unless otherwise specified you may reproduce the "
        "materials in whole or in part for non-commercial purposes, and in "
        "any format, without charge or further permission, provided you ... "
        "indicate that the reproduction is a copy of the version available "
        "at [URL]\" and \"Unless otherwise specified, you may not reproduce "
        "materials on this site ... for the purposes of commercial "
        "redistribution without prior written permission from the copyright "
        "administrator.\" Non-commercial + attribution grant only — does not "
        "cover AW's redistributable compact dump.",
    ),
    # Federal statutes/regulations/case law (task 636 — task 629 explicitly
    # left this sibling host untouched/out of scope, so this was still
    # carrying the same inherited OGL-Canada-2.0 defect as canada.ca until
    # now): a real, and BROADER, grant than OGL-Canada-2.0 — the Reproduction
    # of Federal Law Order applies with no attribution requirement (due
    # diligence + not-official-version only).
    "laws-lois.justice.gc.ca": _Entry(
        "SI/97-5",
        "https://laws-lois.justice.gc.ca/eng/regulations/SI-97-5/page-1.html "
        "(Reproduction of Federal Law Order, SI/97-5, fetched 2026-09-22): "
        "\"Anyone may, without charge or request for permission, reproduce "
        "enactments and consolidations of enactments of the Government of "
        "Canada, and decisions and reasons for decisions of "
        "federally-constituted courts and administrative tribunals, provided "
        "due diligence is exercised in ensuring the accuracy of the "
        "materials reproduced and the reproduction is not represented as an "
        "official version.\" Same instrument already used correctly "
        "elsewhere in this corpus for decisions.scc-csc.ca.",
    ),
    # UK legislation (task 2196b): the declaring instrument is the Open Government Licence v3.0,
    # the one the uk_legislation family contract row declares (licence.instrument_url, read 200
    # on 2026-10-02). EU-origin items carried in UK law sit under Commission Decision 2011/833/EU;
    # both permit reuse with attribution, and the per-record field carries the OGL instrument.
    "legislation.gov.uk": _Entry(
        "OGL-UK-3.0",
        "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/ "
        "(Open Government Licence v3.0, Crown copyright, instrument URL read 200 on 2026-10-02; declared by the uk_legislation "
        "family contract row, licence.instrument_url).",
    ),
    # Bank of Canada runs its OWN terms, not OGL-Canada-2.0 (task 636 — also
    # left untouched by task 629): broadly permissive (attribution + accuracy
    # + a notice-if-sold condition), with named carve-outs for bank-note
    # images and the logo/wordmark.
    "bankofcanada.ca": _Entry(
        "BankOfCanada-Terms",
        "https://www.bankofcanada.ca/terms/ (fetched 2026-09-22), §1 "
        "\"Copyright / Permission to Reproduce\": \"the Bank permits you to "
        "freely use, copy, distribute and transmit its website content\" "
        "subject to §1.1 attribution, §1.2 accuracy and §1.3 a "
        "notice-if-sold condition; §2 excepts bank-note images and the "
        "Bank's logo/wordmark.",
    ),
    "statcan.gc.ca": _Entry(
        "StatCan-OGL",
        "https://www.statcan.gc.ca/en/terms-conditions/open-licence "
        "(fetched 2026-09-22, redirected from /en/reference/licence): "
        "\"Statistics Canada grants you a worldwide, royalty-free, "
        "non-exclusive licence to: use, reproduce, publish, freely "
        "distribute, or sell the Information.\"",
    ),
    # RFC Editor / IETF documents (task 629 P8) — governed by BCP 78 + the
    # IETF Trust Legal Provisions (TLP) 5.0, NOT "public domain". Every RFC's
    # own Copyright Notice reads: "This document is subject to BCP 78 and the
    # IETF Trust's Legal Provisions Relating to IETF Documents
    # (https://trustee.ietf.org/license-info) in effect on the date of
    # publication". TLP 5.0 §3.c.iii is the operative grant for exactly AW's
    # use: copying unmodified PORTIONS, attributed to IETF and identifying the
    # RFC — a portion + attribution + source identification, never full-text,
    # never modified. That is what ADR-039's minimum-content record already is.
    "rfc-editor.org": _Entry(
        "IETF-TLP-5.0-portion-BCP78",
        "RFC Copyright Notice (verified verbatim on rfc9110.txt, fetched "
        "2026-09-22): \"subject to BCP 78 and the IETF Trust's Legal "
        "Provisions Relating to IETF Documents\"; TLP 5.0 §3.c.iii "
        "(trustee.ietf.org) grants \"to copy, publish, display and "
        "distribute unmodified portions of IETF Contributions and IETF "
        "Documents ... provided that ... each such portion is clearly "
        "attributed to IETF and identifies the RFC ... from which it is "
        "taken.\"",
    ),
    # --- US NLM / PubMed -----------------------------------------------------
    "pubmed.ncbi.nlm.nih.gov": _Entry(
        "NLM-PubMed-Terms",
        "https://www.nlm.nih.gov/web_policies.html (fetched 2026-09-22): "
        "\"Works produced by the U.S. government are not subject to "
        "copyright protection in the United States. Any such works found on "
        "National Library of Medicine (NLM) Web sites may be freely used or "
        "reproduced without permission in the U.S.\"",
    ),
    # --- arXiv metadata --------------------------------------------------------
    "arxiv.org": _Entry(
        "arXiv-metadata-CC0",
        "https://info.arxiv.org/help/license/index.html (fetched "
        "2026-09-22): \"A Creative Commons CC0 1.0 Universal Public Domain "
        "Dedication will apply to all metadata.\" (Governs metadata only — "
        "not the papers themselves, which carry per-submission licenses.)",
    ),
    # --- MDN -------------------------------------------------------------------
    "developer.mozilla.org": _Entry(
        "CC-BY-SA-2.5",
        "https://developer.mozilla.org/en-US/docs/MDN/Writing_guidelines/"
        "Attrib_copyright_license (fetched 2026-09-22): \"Unless otherwise "
        "indicated, the content is available under the terms of the "
        "Creative Commons Attribution-ShareAlike license (CC-BY-SA), v2.5 "
        "or any later version.\"",
    ),
    # --- Swiss public-service broadcasters (SRG SSR) ----------------------
    # NOT inferred from the host: each string below is the `license.default`
    # of an already-ratified source pack, transcribed verbatim — the licence
    # basis comes from the declaring instrument, never from where the
    # content happens to live.
    #   swissinfo.ch         -> sources/ch/swissinfo.yaml  license.default
    #   srf/rts/rsi.ch       -> sources/ch/ch-media.yaml   license.default
    # Both packs are ADR-039 minimum-content pointers (url + anchor + <=25-word
    # snippet + embedding + lang), never a mirror. Task 448.
    "swissinfo.ch": _Entry(
        "permissive-or-fairuse-snippet",
        "The ratified source pack `sources/ch/swissinfo.yaml`'s "
        "`license.default` (task 448) — an Act of the pack's own ratified "
        "admission, not re-derived here.",
    ),
    "srf.ch": _Entry(
        "copyright-fairuse-snippet-pointer",
        "The ratified source pack `sources/ch/ch-media.yaml`'s "
        "`license.default` (task 448) — an Act of the pack's own ratified "
        "admission, not re-derived here.",
    ),
    "rts.ch": _Entry(
        "copyright-fairuse-snippet-pointer",
        "The ratified source pack `sources/ch/ch-media.yaml`'s "
        "`license.default` (task 448) — an Act of the pack's own ratified "
        "admission, not re-derived here.",
    ),
    "rsi.ch": _Entry(
        "copyright-fairuse-snippet-pointer",
        "The ratified source pack `sources/ch/ch-media.yaml`'s "
        "`license.default` (task 448) — an Act of the pack's own ratified "
        "admission, not re-derived here.",
    ),
    # watson.ch (CH Media online news; admitted by the Board 2026-10-03, POLICY §4.7). The value is
    # `license.default` of the ratified pack `sources/ch/watson.yaml`, the same ADR-039 minimum-content
    # pointer posture as the SRG publishers above. NOT a grant: the site's terms reserve all rights and
    # allow automated retrieval only with CH Media's consent (the pack's notes say so).
    "watson.ch": _Entry(
        "copyright-fairuse-snippet-pointer",
        "The ratified source pack `sources/ch/watson.yaml`'s `license.default` "
        "(POLICY §4.7, Board admission 2026-10-03). Its notes record the terms of use "
        "(https://www.watson.ch/o/agb, Ziff. 4, read 2026-10-03): all rights reserved; automated retrieval "
        "\"only with CH Media's express consent\". The string is the pointer posture, not a grant.",
    ),
    # technical_manufacturer_catalog (task 629 P8, verified 2026-09-22). Each
    # site's own Terms give a different basis — read individually, none
    # inferred from hosting:
    #   sunfounder.com — Terms of Service (Section 2, /pages/terms-of-service)
    #     restricts reproduction generally and its separate Intellectual-
    #     Property-Rights page is a dead embed (9-bill.com/index/legal returns
    #     HTTP 500) — no open grant found. Same ADR-039 minimum-content
    #     fair-use pointer posture as the SRG SSR broadcasters above.
    "sunfounder.com": _Entry(
        "copyright-fairuse-snippet-pointer",
        "https://www.sunfounder.com/pages/terms-of-service (fetched "
        "2026-09-22): \"You agree not to reproduce, duplicate, copy, sell, "
        "resell or exploit any portion of the Service ... without express "
        "written permission by us.\"",
    ),
    #   upesy.com — Terms of Sale Article 14.2 (/policies/terms-of-sale) is an
    #     affirmative CONTRACTUAL grant, not just fair use, distinct from
    #     Article 14.1's Œuvre (tutorials) CC-BY-NC-ND grant, a different
    #     section of the site and not this one.
    "upesy.com": _Entry(
        "upesy-tos-indexing-permitted",
        "https://www.upesy.com/policies/terms-of-sale, Art. 14.2 (fetched "
        "2026-09-22): \"Cette interdiction n'est pas applicable aux robots "
        "d'indexation ayant pour seul objet de scanner le contenu du Site "
        "aux fins d'indexation\" (the reproduction ban does not apply to "
        "indexing robots whose sole purpose is scanning the Site's content "
        "for indexing) — exactly AW's use.",
    ),
    #   toybox.com — Terms of Service ("Certain Restrictions", /pages/terms-
    #     of-service) forbids scraping/crawling generally but carves out
    #     public-search-engine indexing; matches ADR-039's minimum-content-
    #     index-not-mirror posture exactly; revocable.
    "toybox.com": _Entry(
        "toybox-tos-search-index-permission-revocable",
        "https://www.toybox.com/pages/terms-of-service (fetched 2026-09-22): "
        "\"we grant the operators of public search engines revocable "
        "permission to use spiders to copy materials from the Services for "
        "the sole purpose of and solely to the extent necessary for creating "
        "publicly available searchable indices of the materials, but not "
        "caches or archives of such materials.\"",
    ),
    # github.com — deliberately NOT a host rule (task 629 P8). A GitHub-hosted
    # file's licence is the REPOSITORY's, so it is per-record, never
    # host-inferred (memory licence-basis-from-the-instrument). Resolution
    # must come from `existing_license` alone; a "NOASSERTION" value there is
    # rejected below, never treated as a grant. Verified: `"github.com" not
    # in _HOST_LICENSE` and no suffix in this table catches it either
    # (test_pool_a_licence.py::test_github_origins_do_not_resolve_via_a_host_rule).
}

# source_type -> license (only for types with a defensible blanket license).
_SOURCE_TYPE_LICENSE = {
    # Entity/site-search cards are AW-synthesized descriptions (grounded on
    # Wikidata/register data) — CC0 for the synthesized text; the underlying
    # register facts are gov public data.
    "entity_card": "CC0-1.0",
}

# SPDX defines NOASSERTION as "no assertion was made about the license" — it is
# explicitly NOT a grant. Found independently by task 629 (github_agent_
# resources captured it verbatim from GitHub's repo-license API for 100% of a
# 3,962-receipt sample) and task 636 (the same latent bug, found while auditing
# this resolver generally). Trusting it as `existing_license` would admit rows
# under a string that means the opposite of admission. Treat it — and the
# equivalent textual "none"/"unknown"/"null"/empty — as absent, never as a
# value.
_NOT_A_GRANT = {"noassertion", "none", "unknown", "null", ""}


def resolve_license(existing_license, domain, host, source_type, provenance=None):
    """Return an SPDX-style license string or None (=> licensing exception).

    `existing_license`: value already on the record (trusted if truthy, unless
        it is one of SPDX's non-grant sentinels — see `_NOT_A_GRANT`).
    `domain`/`host`: the record's stored domain/host.
    `source_type`: the record's source_type.
    """
    if existing_license:
        ex = str(existing_license).strip()
        if ex and ex.lower() not in _NOT_A_GRANT:
            return ex

    h = (host or domain or "").lower().lstrip(".")
    if h.startswith("www."):
        h = h[4:]

    # Exact match, else the LONGEST matching suffix — never the naive
    # registrable-domain guess this table used before task 636. A naive
    # last-two-labels cut cannot distinguish `open.canada.ca` (a real OGL
    # grant) from `canada.ca` (no permissive grant); longest-suffix can,
    # because `open.canada.ca` is itself a table key.
    if h in _HOST_LICENSE:
        return _HOST_LICENSE[h].license
    best_suffix = None
    for suffix in _HOST_LICENSE:
        if h.endswith("." + suffix) and (best_suffix is None or len(suffix) > len(best_suffix)):
            best_suffix = suffix
    if best_suffix is not None:
        return _HOST_LICENSE[best_suffix].license

    st = (source_type or "").strip().lower()
    if st in _SOURCE_TYPE_LICENSE:
        return _SOURCE_TYPE_LICENSE[st]

    return None  # licensing exception — surfaced + counted in the MANIFEST

# AW Index Admission Policy

**Version:** v3.4 (2026-10-23: ruling 59, the corporate own-site paragraph in §4.9) — v3.3 (2026-10-07: ruling 27 clarifications to U2, §4.1, §4.2 and §4.8) — v3.2 (2026-10-05: ruling 26 clarifications to U1, §4.7 and §7, plus the
2026-10-03 admission records of PR #566 that were not versioned) — v1 RATIFIED by the board (Tim) 2026-07-22; v2 = the D1
vendor-content amendment (§4.4, 2026-07-27); **v3 = the register-published-
individuals amendment (§4.9.1, 2026-07-30, task 363, RATIFIED by board
merge 2026-07-30); v3.1 = same-day strengthening** — the aggregation
boundaries, and the privacy stance itself, made explicit rather than
implicit — lands as a versioned PR per §9; board merge = ratification.
Major version: v3 opened a wholly new admission category (natural-person
names/roles, not just organizations) — per Tim's framing, "making people
searchable is a governance level decision." v3.1 is a minor/clarifying
bump on top of it (§9): it does not change what is admissible, it makes
the constraints on that admission — and the values behind them — explicit
in the text instead of assumed. v3.2 is a minor bump (§9): it adds three
clarifications and changes no past decision. v3.3 is a minor bump (§9): it adds four clarifications and changes no past decision. Admission decisions cite
`POLICY.md@v3.3 §…`. Amendments per §9.
**Layer:** legislative (see `specs/index-governance.md` §2). Every admission,
density, and eviction decision cites the policy version and section it
applied, e.g. `POLICY.md@v1 §4.1`.
**Scope:** what may enter the AW index, at what density, and how it leaves.
Serving/ranking behavior is out of scope (that is code + evals, not law).

---

## §1 Principles

- **P1 — Inclusion is principled; density is earned.** Admission (§3–§4) is
  a rules question. Density (§5) is an evidence question. No decision may
  mix them: a source cannot be excluded for low demand, nor granted depth
  for mere admissibility.
- **P2 — Authority is structural before it is behavioral.** For most layers,
  what makes a source authoritative is *who it is* (the official publisher,
  the peer-review venue, the project's own docs) — verified once —
  not engagement metrics, which are gameable.
- **P3 — Index the origin.** Prefer the primary source over mirrors,
  aggregating republishers, or SEO derivatives. An aggregator is admissible
  only when it *is* the practical origin (e.g. entscheidsuche for cantonal
  court decisions) or the origin is inaccessible.
- **P4 — Consent is live and final.** The two flags (`search_ok`,
  `train_ok`) are re-resolved per build (architecture §5); revocation is
  honored at the next build, no exceptions, no grace for AW's convenience
  (the grace window in architecture §5 covers *unreachable* endpoints, not
  refusals).
- **P5 — Everything is logged; nothing is silent.** Admissions, rejections,
  escalations, demotions, evictions — each writes an admission-block entry
  (governance §3). An unexplainable index record is a policy violation.

## §2 Universal requirements (all layers)

A source is admissible only if ALL hold:

- **U1 Consent** — `search_ok` affirmatively resolvable (robots/TDM/ai.txt
  per the consent resolver). Absence of signal is governed by the
  resolver's published defaults, never by wishful reading.
  *Dump published by the source itself (Tim 2026-10-05, ruling 26, item 22).*
  When the source itself publishes a dump under an open licence, the licence
  is the consent instrument. The family takes the dump rule r1 of ADR-115
  (as ADR-103 am.2 ruled for Stack Exchange). A robots probe of the host is
  not the gate. The licence must be read from the publisher's own statement,
  never inferred from hosting.
- **U2 License compatibility** — the minimum-content record (ADR-039:
  url, title, snippet, embedding, terms, metadata) must be redistributable
  in the compact dump. Deep entity/source *cards* additionally follow
  ADR-049: permissive text → deep card; copyleft/unclear → pointer card.
  *Licence spelling (Tim 2026-10-07, ruling 27, T7).* A family that moves to the
  family door writes the licence as an SPDX identifier, for example
  `CC-BY-SA-3.0`. The rows served today keep the dump's own words (for example
  `CC BY-SA 3.0`) until the rebuild `aw_e10` writes the new spelling.
- **U3 No paywalled core** — content behind payment/subscription is
  inadmissible as content. *Existence pointers* (a DOI landing page, an
  article headline on a metered site) are admissible as pointers where the
  landing itself is open (§4.2, §4.7).
- **U4 Operational floor** — stable canonical URLs; fetchable politely
  (honors rate limits; no WAF-circumvention beyond documented, sanctioned
  routes); resolvable ownership (a registrant or organization exists).
- **U5 Primary-source preference (P3)** — if the candidate republishes an
  admissible origin, admit the origin instead; the candidate may still
  qualify as an aggregator under §4-layer rules.

## §3 Hard disqualifiers (auto-reject, band-exempt)

- **D1** Consent denied or revoked (P4).
- **D2** Paywalled/subscription-gated core content (U3).
- **D3** SEO parasites & content mills — machine-spun, affiliate-stuffed, or
  scraped-mirror content (the `parasite_domains` machinery is the executive
  arm of this rule).
- **D4** Impersonation — sites presenting as an authority they are not
  (fake gov/org lookalikes), regardless of content quality.
- **D5** Content illegal under Swiss law (operating jurisdiction).
- **D6** Structurally unfetchable/unstable — no canonical URLs, aggressive
  cloaking, or content that materially differs per requester.

D-rules are not scoreable and never enter the bands (§6): a D-hit rejects.

## §4 Per-layer admission criteria

Each layer states: what authority *is* there, the admission floor, and the
default form. (Density defaults live in §5.)

### §4.1 Law & government
Authority = being the official publisher of the jurisdiction: primary law,
gazettes, courts, agencies (fedlex, admin.ch family, cantonal portals,
Genesis/gesetze, DILA, CanLII-class). **Floor:** official status verified
(gov TLD, registry, or constitutional role). Aggregators admissible per P3
when they are the practical origin (entscheidsuche, lexfind). Default form:
deep — this layer is the grounding mission.

**Law rows: one article, split only when it must be (Tim 2026-10-07, ruling 27,
T6).** The row of a law family is the source's own article. When the article
fits the embedder window (2,560 tokens), it stays one row. When it is longer,
the family splits it at paragraph marks into the fewest balanced chunks that
each fit the window, with the law title and the article header on every chunk.
A row over the window is split, never silently cut. A row id of generation 0
follows the rule named `text-anchor-v0` (decisions.md ADR-067 am.2). The
contract declares both: `chunk_policy` `native` where the article always fits,
`native-chunked` where it may be split.

### §4.2 Scholarship
Authority = peer-review or moderated-preprint provenance, or being a
primary research registry (arXiv, PubMed, CT.gov, ERIC (education research);
OpenAlex/Crossref as CC0 metadata backbones). **Floor:** venue listed in the
accepted registries; predatory-publisher lists exclude. **Ceiling note
(ADR-113):** this layer's ratified ceiling (PORTFOLIO §3) does not by itself
exclude an admissible registry — ERIC is admitted per §5's demand-reserve
evidence path, the ceiling numbers and the draw ledger living in PORTFOLIO,
not here. Paywalled papers: DOI-landing pointers
only (U3). Default form: metadata + abstract records; deep only for open
full-text.
*OpenAlex fixture works keep their author fields (Tim 2026-10-07, ruling 27,
T7).* OpenAlex is CC0. The works of the public fixture keep their author fields.

### §4.3 Reference & encyclopedic
Authority = editorial process with public revision/correction machinery
(Wikipedia per language, established dictionaries, standards bodies'
glossaries/specs). **Floor:** the process, not the brand — a reference work
without visible editorial control is a website, not a reference. Default
form: deep for the four languages (EN/DE/FR/IT).

### §4.4 Technical documentation & agent-resources
Authority = self-documentation: a project's canonical docs are definitionally
authoritative *about that project* (P2). **Floor:** the project clears a
usage bar (package-registry downloads, dependents, endorsement mass — not
stars alone); docs are the canonical instance (not a docs-mirror). Includes
agent-resources (AGENTS.md, llms.txt, MCP registries, skills). License per
U2/ADR-049. Default form: deep for permissive-licensed canonical docs.

**Vendor-content boundary (v2, D1 2026-07-27).** Self-documentation
authority extends only to a project's *canonical documentation*. Commercial-
vendor content that is not that — vendor engineering blogs, product
marketing, glossary/checklist/SEO surfaces, and vendor tool listings —
carries the dedicated **vendor-content trust class (`trust_tier = 0.60`)**,
below the technical-documentation default (0.75), regardless of content
quality or title-match strength. Genuine product documentation (docs
subdomains, `/docs/`-class paths) keeps the documentation class even on an
otherwise-vendor host.

**Standing rule — discovery never inherits a mission default.**
Discovery-sourced admissions (endorsement/outlink-graph waves, agent-resource
registry sweeps) never inherit a mission-content class default. A
discovery-admitted host receives a mission-tier trust class only by
affirmative classification (canonical documentation per this section,
community per §4.8, or another §4 layer); absent that, it enters at the
vendor-content class (0.60) or below. *Incident basis:* the wave-2/tier-1
vendor-SEO flood (tasks 283/286 admissions at the generic 0.75 tech default;
gate evidence tasks 327–329; displacer analysis
task 328's `displacers328.out`). Ruled by the board (Tim, D1,
2026-07-27, in-session); implemented on the build collection by task 330.

### §4.5 Statistics & data
Authority = official statistical agencies, IGOs, central banks, and primary
data publishers (BFS, INSEE, StatCan, Eurostat-class, BoC/SNB). **Floor:**
official mandate. Prefer the API/dump route where offered (channel census).
Default form: deep for indicator/definition pages; datasets as cards.

### §4.6 Health
Authority = official health authorities, WHO-class IGOs, professional
societies, and evidence-synthesis bodies (Cochrane-class). **Floor:
deliberately the strictest** — harm asymmetry. No commercial wellness
content, no practitioner marketing, no supplement-adjacent media, even when
popular. Escalation band is narrower here: when in doubt, out.

### §4.7 News
Authority = editorial accountability: a masthead, a corrections policy, a
legal publisher. **Floor:** those three, verifiable. Rolling ~12-month
window; `published_date` capture is mandatory at ingest (a news record
without a date is inadmissible — it cannot age out). Metered mainstream
sites: headline/landing pointers per U3. Default form: rolling window per
the freshness policy.

**Admission records (board rulings, 2026-10-03).**
- **`watson.ch` is admitted** as a news publisher (Tim, plan.md ruling 23.13).
  `robots.txt` read from the AW crawl host on 2026-10-03: `User-agent: *` `Allow: /`,
  `Crawl-delay: 1`. The probe from the AW crawl host gave `search_ok=true` and `train_ok=true`.
  ADR-024 amendment 2 held `watson.ch` card-only because `*` was disallowed.
  That premise no longer holds (ADR-024 amendment 6). The site's terms of use
  reserve all rights and allow automated retrieval only with the publisher's
  consent; the consent resolver does not read terms, and the Board has not ruled
  on that sentence. Pack: `sources/ch/watson.yaml`.
- **`heise.de` is not admitted.** First check the share of its articles behind the
  heise+ paywall (U3). No fetch starts before that number exists.

**Consent stamp for news (board rulings, 2026-10-05, ruling 26, item 14).**
- **Q1.** When the current consent record would loosen a flag that the stage
  set to false (for example SRG `train_ok`), only those rows are left out.
  The rest of the day's pack lands.
- **Q2.** News is our own direct fetch. The stamp reads only the host's
  direct-fetch record, never the Common Crawl record.

### §4.8 Community & discussion
Authority = none — and the layer says so. Admissible venues (HN, Stack
Exchange class) earn inclusion by moderation quality and signal density,
carry a discussion-tier trust label, and are never ranked as factual
authority. **Floor:** established moderation + public archives. Density
strictly earned (§5); default form: thread/answer records, tightly capped.

**Stack Exchange attribution (Tim 2026-10-07, ruling 27, T7).** The author's
display name and profile URL are attribution under the licence (ADR-039). They are
master-only and are stripped from the dump. The licence version is read per
row, from the row's own licence field, not from a date boundary.

### §4.9 Commercial & transactional
Authority = being a legitimate operating business agents may need to *act
through* (retailers, booking, services). **Floor:** registry-verified
operating entity (Zefix-class evidence, per ADR-040-compliant resolution)
+ demand evidence. Default form: **cards only** (site/category/brand — the
task-299 pattern), never product-page crawls.

**Corporate own-site (v3.4, 2026-10-23, ruling 59).** The official website of an SMI-listed
company (PORTFOLIO 3i) may be read as pages, not as a card. It receives the vendor-content trust
class (`trust_tier = 0.60`, §4.4), never a mission tier: a corporate site is authoritative about
itself, and only about itself. The domain comes from open data confirmed by two sources, never
from a search-engine API. This paragraph admits no other commercial site.

#### §4.9.1 Register-published individuals (v3, 2026-07-30; v3.1 same day)

Commercial-register entity cards (§4.9) may additionally carry the natural
persons the register **itself** publishes in connection with the entity —
signatories, board members, management — as a searchable facet on the
company's own card. This subsection is the governance gate for that one
narrow extension; nothing else about §4.9 changes.

**Scope.** Natural persons ONLY, and ONLY in their register-published role
for the specific entity: signatories, board/management members, as
published by the commercial register itself (Zefix/SOGC — Swiss Official
Gazette of Commerce, *Schweizerisches Handelsamtsblatt*/SHAB). Nothing
aggregated beyond what the register's own publication already contains — no
cross-referencing a person across their other directorships, no biography,
no photo, no contact detail, no employment history. A person's name and
role-at-this-entity, on that entity's card, and nothing more. This is not a
general people-search feature and does not create person-entity cards of
their own; the person is discoverable only as a token on the company card
they are registered against. **No other private individuals** are made
searchable by this subsection — it does not extend to natural persons
appearing in any non-register source (news mentions, social profiles,
directories); that boundary is unchanged and reaffirmed below.

**Legal basis.** Swiss commercial-register entries are subject to statutory
publicity: registration and the identity of signatories/board members is a
condition of legal capacity to act for the entity (CO/OR Art. 931 ff.,
HRegV), and the register's own publication mandate (SHAB, Art. 9/35 ORC —
*"La mutation de l'entité juridique mentionnée a été effectuée dans le
registre du commerce"* / *"Die Mutation der aufgeführten Rechtseinheit wurde
im Handelsregister vorgenommen"*, cited verbatim on every SHAB publication)
exists precisely so that this information is public and citable by anyone.
AW is not disclosing anything the register does not itself already
disclose to any visitor of zefix.ch or shab.ch; AW is making that
already-public, already-intentionally-published fact findable by an AI
agent that is looking for it — squarely inside AW's mission (surfacing
authoritative facts, not creating new exposure).

**Safeguards.**
- **TAKEDOWN.md is explicitly extended** to a fast-path for person-name
  removal requests: a person named on a company card (or anyone on their
  behalf) may request their name be removed from AW's index by the same
  `bender@impactia.org` / `TAKEDOWN` channel, verified by a lighter bar than
  domain-ownership verification (a name + the company UID/name they are
  associated with is sufficient to identify the token to remove — no domain
  email or token page required, since the requester does not control a
  domain). Target: acknowledge within 3 business days, resolve within 7,
  same as the existing domain takedown SLA. **Honored globally, regardless
  of the requester's jurisdiction or nationality** — this is not a
  territorial-applicability gate; see the values note below.
- **Person tokens are removed WITHOUT deleting the company card.** A
  takedown against a person's name strips that name/token from the card's
  sparse and dense text and pins the removal (future register passes must
  not silently re-add it without a fresh review) — the company's own card
  (identity, purpose, site-search affordance) is untouched, since that is
  independently admissible under §4.9 on its own terms.
- **Currency follows the register.** If the register itself records a
  person's departure (an "Ausgeschiedene Personen" / departed-persons
  mutation), the next register pass removes that person automatically —
  AW's index should never lag the register's own public state in the
  privacy-protective direction.

**Boundaries.**
- **No enrichment of a private individual from any non-register source.**
  This subsection authorizes indexing exactly what the register publishes,
  nothing pulled from elsewhere and attached to a name found here.
- **Public-figure cards from other admission lanes are unaffected.** A
  notable person's own Wikidata/Wikipedia-backed entity card (if any exists
  under a different §4 layer, e.g. §4.3) is governed by that layer's rules,
  not this one; this subsection neither creates nor constrains those.
- **This is a card-facet, not a people-search product.** No dedicated
  person-search UI/endpoint is authorized by this subsection; the only
  surface is "this company card is now also findable by a signatory's
  name," the same way it is findable by its domain or UID.
- **No person entity records.** A register-published individual never gets
  their own card/record in the index. Their name exists exclusively as a
  search token on the company card(s) they are registered against. A
  person→companies view is not a curated object anywhere in AW's data —
  it can only arise ephemerally as a set of independent search results at
  query time (the same sense in which typing a name into any general
  search engine surfaces multiple unrelated pages about people who share
  it — *Google Spain* search-engine semantics, not a dossier AW compiles
  and stores).
- **No cross-company identity resolution.** AW never asserts that
  same-named individuals appearing on different company cards are the same
  person. No disambiguation, no linking, no "this person's other
  directorships" aggregation, no mandate-count logic — each name-token is
  scoped strictly to the one card it was extracted from. Building that kind
  of cross-register profile is exactly the enrichment the boundary above
  prohibits, and is treated as a D-rule-adjacent violation if ever proposed.

**Values and lawful basis.** AW follows the GDPR and the Swiss LPD/nDSG **by
conviction, as our own standard of conduct, irrespective of whether either
formally applies to a given user, request, or jurisdiction.** This is not a
compliance posture adopted where required and dropped where not — it is a
design constraint the way ADR-039 (honest per-record licensing) or ADR-040
(no competitor search) are design constraints: self-imposed, applied
uniformly, and not something we litigate our way around. Concretely for
this subsection:
- **Lawful basis is legitimate interest (GDPR Art. 6(1)(f) / LPD Art. 6
  para. 1), documented here, not left implicit.** The interest: enabling an
  AI agent to find the authoritative source for a statutory-publicity fact
  the register itself already publishes to the world. Balanced against the
  data subject's interest via: minimality (role-at-this-entity only, no
  aggregation — the boundaries above), the takedown fast-path (objection
  honored fast and globally, no jurisdictional gate), and register-tracked
  currency (departures removed automatically). *Manni* (C-398/15) supports
  register publicity as a legitimate aim; *Google Spain* (C-131/12)
  supplies the balancing/delisting frame the takedown fast-path mirrors;
  the EDÖB/Moneyhouse precedent is the opposite failure mode from this
  subsection's design — it condemned creditworthiness scoring and
  enrichment *beyond* the register's own content, which is precisely what
  the boundaries above forbid.
- **Data-subject rights are honored globally.** Objection/delisting via the
  TAKEDOWN fast-path is not gated on the requester being in the EU/EEA or
  Switzerland — anyone whose name appears this way gets the same fast
  removal, everywhere, because we hold the standard, not because a specific
  law compels it for them.

**Launch checklist (required before this section's held artifact is
exposed to public search traffic — not merge conditions, launch
conditions):**
- [ ] **Privacy notice on the serving surface.** websear.ch currently has
  none. Required before launch: controller identity, what is processed and
  why (including this section's register-role facet), and the rights
  routes (the TAKEDOWN fast-path, linked from the notice).
- [ ] **Records of processing** (GDPR Art. 30 / LPD-equivalent) covering
  this facet specifically: source (Zefix/SHAB), lawful basis (above),
  retention/currency mechanism (register-tracked), and the takedown
  procedure.
- [ ] **Counsel review** of this section (the pre-launch legal review
  already flagged) — scheduling is Tim's call, tracked as a launch
  blocker for this facet, not for §4.9 generally.

**Application note.** Any build artifact enriching register cards with this
facet is staged and HELD until this section is ratified (board merge to
`main` = ratification, per §9) — a held artifact activates automatically on
merge; it is not re-approved separately. **The launch checklist above gates
public exposure, separately from and in addition to the merge/ratification
gate** — a merged, applied artifact still should not serve public traffic
until the checklist is complete. Pre-ratification, register cards may
still gain the non-person facets already admissible under §4.9 as-is
(record-currency date, founding date, register status) — those carry no
governance gate; only the people facet is gated by this subsection.

#### §4.9.2 Public-figure own-website cards (v3.2 PROPOSED, 2026-08-07 — Tim ruled the lane open; ratified on merge)

**What this authorizes.** Entity cards for the **own public websites of
well-known natural persons** — researchers, authors, artists, officials,
founders — where the person has deliberately published a public web presence
(personal site, lab page, official author page). The card describes the SITE
and its public purpose; its content comes from the site itself and public
notability sources (Wikipedia/Wikidata), exactly like any other site card.

**Why this is a different lane from §4.9.1.** §4.9.1 concerns data the
*register* publishes about persons in a statutory role. This subsection
concerns what a person *chose to publish about themselves* on the open web —
the ordinary case of indexing a website, where the "entity" happens to be a
person. Origin: Tim's 2026-07-10 question ("what about well-known people with
personal websites?"), unanswered until audit 397; ruled open 2026-08-07.

**Admission bar.**
- **Notability-gated**: the person has a Wikipedia article or a Wikidata item
  with independent references (the same organic-backbone bar as other
  entity cards) — no long-tail person indexing.
- **Own-site only**: the card's site is the person's own published presence
  (verified the same way company sites are — self-reference, name-on-page);
  never a directory, profile page on a third-party platform, or aggregation.
- **Consent-gated like any site**: live `search_ok` on the person's own
  domain is the consent signal — the person controls it directly.
- **No cross-source aggregation**: the card carries the site's own
  description + notability identity; no register joins, no contact-detail
  harvesting, no linking to §4.9.1 register tokens. The §4.9.1
  no-dossier boundary is reaffirmed: a person's company roles and their
  personal site are never merged into one profile.

**Safeguards.** The TAKEDOWN person-name fast path applies to these cards
identically (and trivially: the person controls the domain, so domain-level
takedown also works). GDPR/LPD-by-conviction posture applies in full.

#### §4.9.3 Register-grounded foundations & associations of mission relevance (PROPOSED, 2026-09-19 — task 620; not ratified)

**What this clarifies.** §4.9's floor — "registry-verified operating entity
(Zefix-class evidence) + demand evidence" — and the entity-card build's own
admission gate, ratified 2026-07-10 (*"active company (any legal form) +
validated live site → card"*, `specs/entity-cards-phase2-companies.md` D7),
were never legal-form-restricted: a Swiss foundation (Stiftung, eCH-0097
0110) or association (Verein, 0109) with an active register entry and a
verified official website has always been mechanically admissible. Nothing
in POLICY has said so in a §4 heading, and §4.9's own title ("Commercial &
transactional") reads as excluding a non-profit foundation on first pass —
which is very likely why the two production website-discovery routes (the
CC .ch-domain sweep, TLD-scoped and therefore blind to non-`.ch` official
sites; the slug-guess resolver, which does try `.org`/`.swiss`/`.com` but has
only ever run as one-off task batches, not standing infrastructure) have
produced essentially zero of them: impactIA itself — the foundation behind
this project, tier A since ADR-104 — surfaced once, at `resolve_confidence
1.0`, and the batch containing it was never merged (task 620 §2). This subsection names the class explicitly so it is
onboarded on purpose rather than reached by accident or missed by an
engine's TLD scope.

**Scope.** Swiss foundations and associations **of mission relevance** —
social-contract institutions, AI-governance bodies, research foundations —
under the existing §4.9 floor: an active register entry (Zefix, `lf` code
0109 or 0110) + a verified, ownership-confirmed official website (ADR-041
verify contract: UID-on-page, or name-match plus at least one of
{street, NPA, city}). This is **not** a blanket foundation/association
ingest — mission relevance is a curated inclusion test, not a legal-form
gate; most of the register's 31,279 active 0109/0110 entries (measured
2026-09-19, the serving box file `/opt/aw/depth161/register_table.json`) are outside
scope (local sports clubs, condominium associations, private family
foundations). A naive AI/governance keyword pass over their register
`Zweck` text finds ~84 plausible candidates (noisy substring match — an
actual selector needs a curated term list and a manual pass); the real
population under a real selector is estimated in the low hundreds.

**Admission bar (identical to §4.9.1's underlying entity, since a
foundation's board is register-published the same way a company's is).**
- Active Zefix entry, `lf` 0109 or 0110, status not dissolved.
- Verified official website under the ADR-041 verify contract — the
  standing resolver must be extended to try `.org`/`.swiss`/`.com` for
  this class as a matter of course, not as a one-off grind (task 620's
  build script and `specs/register-card-build.md` §2 point 3 are the
  reusable method; wiring it into the standing pipeline is implementation,
  not this ruling).
- Mission-relevance curated list (AI governance, digital/fundamental
  rights, research foundations of the kind ADR-032 amendment 2 already
  treats as first-class content themes) — reviewed periodically, not
  algorithmically inferred from the Zweck text alone (the 84-hit keyword
  pass is a sizing bound, not the selector).
- Board/officer facet inherits §4.9.1 in full: register-published names
  only, same TAKEDOWN fast path, same no-dossier boundary.

**Why now.** Named directly by the founder as an authority (ADR-104) but
absent as an entity card since the register-card build began (2026-07) —
the gap this subsection closes is not hypothetical.
#### §4.9.4 Register-first card discovery (PROPOSED, task 621, revised task 623 — ratified on merge)

**Card eligibility is the company register, not a host TLD.** The roster of
candidate entities for a §4.9 card is the register's own bulk product for the
jurisdiction. A crawl inventory selected by host TLD may supply *evidence about a
candidate* — its homepage, its imprint, a checksum-valid register id printed on the
page — but may never supply the *candidate list*.

**Why this is a policy clause and not an implementation note.** A TLD-scoped roster
makes a whole class of entity undiscoverable in a way no count can reveal, because
nothing downstream knows the rows existed. Measured on production 2026-09-19:
**57,473 of the 194,255 register-id-bearing entity cards (29.6%) sit on non-`.ch`
domains** that the standing sweep's query cannot return — `.com` 37,001, `.org`
8,834, `.fr` 5,463, `.de` 1,646, `.swiss` 1,468. Every one of them was found by a
one-off script nobody scheduled. The foundation behind this project is the worked
example: impactIA is a Swiss foundation on `impactia.org`, resolvable at confidence
1.0 by the register-first resolver, and permanently invisible to the engine that
runs every month.

This clause changes nothing about *resolution*: ADR-041's verify contract and graded
confidence, ADR-042's one-entity-many-domains rule and ADR-040's no-competitor-search
prohibition all stand exactly as ratified. It changes what gets a chance to be
resolved.

**Consequences, stated so they are ratified together with the clause:**

- A per-country card lane is defined by its **register adapter**, not by a TLD. The
  candidate TLD set is an input to the slug-guess leg, never the population boundary.
- The card's identity is its **URL**, not its register id. One URL admits one card:
  where several register entities verify against one domain, the strictly-highest-
  confidence claimant wins and a tie excludes the cluster. (Measured need: 95 URLs in
  task 621's own shortlist were claimed by 2–12 different register entities.)
- A register id is not a join key until normalised. Production carries the Swiss
  scheme under three spellings (`CHE` 92,187 · `CHE-UID` 88,801 · `CH-UID` 587), and
  a subtraction on the raw field would have overwritten 367 live cards belonging to
  other companies.
- The TLD-scoped sweep is **retained as a site-crawl evidence collector**, with its
  TLD parameterised. It is the card-roster stage, and only that stage, that is gated.

**A register is not one permission — it is two, and a country's licence can grant
one without the other (task 623, Tim 2026-09-20).** A register's bulk product can
be used two ways, and this clause governs both:

- **`register_role: locator`.** The register extract is consulted **internally
  only** — to enumerate candidates, guess a domain, and prove identity under the
  ADR-041 §4 verify contract — and nothing register-derived is published. The
  card's content (name as displayed, site text, snippet) is **site-sourced**; no
  officers, no register-only payload field (legal form, register seat, register
  category) ever reaches the card. This is the eligible reading of a licence that
  grants a personal, non-exclusive, purpose-locked internal-use right but forbids
  redistribution — Tim's own framing: *"we're not redistributing the dump, we're
  using our own dump to locate company websites."*
- **`register_role: enrichment`.** The register's licence additionally permits
  publishing register-sourced fields — the current CH/UK/FR/CA shape. Officers
  are a **further, narrower permission within this mode**, ratified per
  jurisdiction (§4.9.1, CH-only today) and never assumed to transfer by analogy.

A country's `register_role` — and, where a country supports more than one
verification path (a register-free variant sourced entirely from a statutory
on-page disclosure duty, never from the register itself, carries its own named
role, e.g. `self-declared-verified`) — is a **declared field**, set in
`data/company-card-portfolio-allocation.v1.json`'s source-specific constraints and
in `data/representation-contract.v1.json`'s `entity_card` family. It is **never
inferred at build time** from which fields a card happens to carry: the producer
refuses to write a register-sourced field onto a card whose country is declared
`locator` (or a register-free role), the same way it already refuses an officer
facet outside CH. Mode does not relax ADR-041's verify contract, ADR-042's
one-entity-many-domains rule or ADR-040's no-competitor-search prohibition in
either reading — it only gates what a passing card may **publish**.

## §5 Density tiers

| tier | holds | granted by |
|---|---|---|
| **T0 pointer** | one card: identity + site-search affordance | admission alone |
| **T1 cards** | entity/section/category cards | structural role (official org, canonical project) |
| **T2 deep-sections** | full coverage of designated sections | evidence: sustained query-demand, endorsement mass, or eval lift |
| **T3 exhaustive** | the whole admissible corpus of the source | reserved: primary law, reference works, canonical docs of major projects |

- **T-rules:** admission grants T0. T1 by structural role at admission
  time. **T2+ requires evidence and is never granted at admission** (P1),
  except layers whose §4 entry names a deeper default (law, reference) —
  those defaults are policy, hence citable and revisable.
- Every tier assignment/upgrade cites its evidence in the admission block.
- **Demand reserve (ADR-113).** A layer already at or over its ratified
  PORTFOLIO ceiling may still admit a new member if the admission cites
  evidence under this section — query-miss/endorsement demand for the class,
  or (as for ERIC) a named strategic theme's measured demand share. The
  reserve draw itself is an allocation fact (points, ceiling, ledger),
  tracked in PORTFOLIO.md, not here; this section is the evidence bar the
  draw must clear, the same bar T2+ already clears.

## §6 Decision bands

Assessment scores each candidate against §2–§4 (mechanical checks +
authority evidence). Three bands, thresholds set per layer in the pack
tooling and tuned by the escalation-rate metric:

- **auto-admit** — clear pass; logged, no human.
- **escalate** — ambiguous; queued for board (Tim) with the evidence file.
  Target rate: rare. A rising rate means the policy needs an amendment, not
  more escalations.
- **auto-reject** — clear fail or any D-rule; logged with the failing rule.

Band decisions are reversible by appeal (§8) or policy amendment (§9) —
never by quiet re-scoring.

**Tranche admission (ADR-110, Tim 2026-09-23).** Landing more rows/hosts for
a family/host set that already has a PORTFOLIO row is an **auto-admit**
execution of this section, not a policy change — it lands per
`sources/README.md`'s change process, orchestrator-merged on green CI. What
this section admits is unchanged; only who signs the landing of an already-
admitted family moves. A NEW family/host, or anything that changes §2-§5
themselves, still runs this band and, if escalated, still goes to Tim.

## §7 Demotion & eviction

Triggers, all logged as ledger events:

- **E1 Consent revocation** → removal at next build (P4). Immediate
  out-of-band build if the source requests urgency.
- **E2 Death** — domain lapsed/parked/hijacked (the dns_maturity
  parked-penalty is a *detector* here, not a ranking feature).
- **E3 Earn-it failure** — quarterly review (governance §1.3): a T2+ pack
  whose demand/eval evidence no longer supports its tier demotes one tier;
  it may re-earn.
- **E4 Reclassification** — a source found to violate a D-rule
  (parasite/impersonation discovered late) → reject as if new, plus a
  ledger note on what the assessment missed.
- **E5 Aging** — layer-native expiry (news 12-month window) — mechanical,
  not a governance event.

**A policy-excluded URL keeps its entity card (Tim 2026-10-05, ruling 26,
item 14 Q3).** When policy excludes the pages at a URL (for example a
derivative of an admitted source, such as a Hugging Face derivative), the
entity card at that URL stays. The card is built from the register row, not
from the excluded page (ADR-103).

## §8 Appeals

- **Site owners:** consent opt-out is already self-service (U1/P4).
  Contest-my-assessment (admission, tier, or D3/D4 classification) via the
  public contact channel; resolved by curator, escalated to board on
  dispute; outcome logged.
- **Users/community:** propose sources via `sources/_proposed/` PRs — the
  same funnel the proposer loop uses; same policy, same bands.
- **The fork (governance §2):** this policy, the packs, and the ledger are
  permissively-licensed data. Ultimate disagreement is resolvable by
  forking the index and amending the policy in your copy. AW considers
  this outcome a success of the model, not a failure.

## §9 Amendment

- Policy changes land as versioned PRs (`v1 → v2 …`), ratified by the
  board. Semantic versioning of law: additions/clarifications = minor;
  changes that could flip past decisions = major, and trigger a re-audit of
  decisions that cited the amended section.
- The curator may propose amendments (with the escalation-rate and appeal
  data as evidence) but never self-ratify.
- Ratified versions are immutable; the ledger cites versions, so the
  editorial history of the index remains replayable forever.

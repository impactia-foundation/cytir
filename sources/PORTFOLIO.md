# AW Index Portfolio

**Version:** v1.1 — RATIFIED by the board (Tim) upon merge of PR ratify/portfolio-v1.1.
Campaign 1 (PubMed reshape) additionally awaits the ADR-074 citation-threshold ladder
selection (task 316) — ratifying this matrix authorizes the campaign, the ladder sets its knife. The allocation
instrument of `specs/index-governance.md` §1; admission legality per
`POLICY.md@v1`; density tiers per POLICY §5.
**Cadence:** reviewed quarterly against the manifest-derived gap map;
changed only by PR.

---

## 1. Horizons

- **Operative target: ~100M useful points** (the strategy directive).
  This document allocates it.
- **Design ceiling: ~1B pages / 10 yr** (C2, ADR-011). Not allocated here;
  the 100M shape is the seed the 1B grows from, re-ratified as real demand
  data accumulates.

## 2. Measured baseline (2026-07-22, twin facet — not lore)

47,824,966 points. Two numbers indict the current shape:

| finding | value | reading |
|---|---|---|
| `source_type=reference` | **43.2M = 90.3%** | one bucket is the index. Inside it: the ~28–30M PubMed over-share (63% of the original build) + Wikipedia + misc. The other 14 source types share 9.7%. |
| `lang=en` | **38.0M = 79.4%** | vs the mission's four languages: DE 8.7%, FR 8.1%, IT 3.7% |

Remainders worth noting: `official` 2.86M, `gov-*` ~0.7M, `docs` 355k,
`entity_card` 278k, `news` 61k, `health` 25k, `community` ~133k — and a
**26k Romansh remnant** (out of scope since 2026-06-21) plus 65 `nl` points:
eviction candidates at aw_b1 (POLICY §7 E-events, logged).

*Caveat (rule-vs-data-drift): `source_type` is fossil-stamped by old rules.
aw_b1 restamps under I2; from then on this table comes from MANIFESTs.*

## 3. Target matrix @100M (v1 proposal — board adjusts numbers, not shape)

| layer (POLICY §4) | current (est.) | target @100M | rationale |
|---|---|---|---|
| §4.1 law & government | ~3.5M | **12M** | the grounding core. CH exhaustive (T3); DE/FR/CA primary law + courts; cantonal complete |
| §4.2 scholarship | ~28–30M | **10M** + reserve draws | the reshape: PubMed capped ~5M (≈5%), arXiv cs-aware ~1.5M, OpenAlex **repriced ~5.42M** (Tim 2026-09-01, ruling "L2, reprice to keep" on task 490's measured trim curves — see §4.2a). Biggest single delta: **−~20M**. Plus **ERIC ≤4.5M**, demand-reserve draw (ADR-113, 2026-09-25 — see §3f) |
| §4.3 reference | ~13M | **15M** | Wikipedia 4 langs full + dictionaries + standards bodies |
| §4.4 tech docs + agent-resources | ~0.6M (incl. staged GitHub) | **15M** | strategy priorities #2/#3 (AI, software). Canonical docs T3 for major projects; agent-resources first-class |
| §4.5 statistics & data | ~0.25M | **3M** | BFS/INSEE/StatCan/Eurostat-class via API routes |
| §4.6 health | 25k | **3M** | Cochrane/WHO/authorities under the strictest floor |
| §4.7 news (rolling 12-mo) | ~0.1M | **2M rolling** | steady-state window, not accumulation |
| §4.8 community | ~133k | **1M cap** | HN/SE, discussion-tier, hard-capped |
| §4.9 commercial cards | ~50k | **0.3M** | cards only; registry-verified |
| entity/site-card backbone | 278k | **2M** | the Wikidata organic backbone (governance §4.4 `generated`) |
| **allocated** | | **~63M** | |
| **demand reserve** | | **~30.19M** (was ~34.69M) | deliberately unallocated: filled bottom-up by T2+ promotions on frontier evidence (query-miss, endorsement). The portfolio proposes; demand disposes. **−~2.31M drawn 2026-09-01** to fund the §4.2a OpenAlex repricing overage. **−≤4.5M drawn 2026-09-25** to fund ERIC's admission (ADR-113, §3f) — again the reserve's ratified purpose ("allocated from evidence"), not a silent budget slip. |

**Language shape @100M:** EN ~55–60% (from 79%), DE/FR/IT carrying the rest
— achieved by *where* the targets sit (law, gov, news are language-native),
never by quota-filtering sources (language follows source).

## 3a. §4.2a — OpenAlex repricing (Tim, 2026-09-01, ruling: "L2, reprice to keep")

**The ruling, verbatim:** *"L2, reprice to keep."* Adopts the L2 (moderate)
age-scaled citation ladder measured by task 490 as the OpenAlex component's
admission floor, and reprices the §4.2 scholarship-layer target line to the
resulting kept figure rather than the original ~3M placeholder.

**Basis (task 490, measurement only, zero mutations):** of the
**6,388,005** materialized OpenAlex **documents** (6,404,873 vector rows) on disk (10 sealed
tranches, cross-checked against the ledger's 6,386,355-document count),
the L2 ladder —

```
KEEP if  is_core = true
    OR   py <  2010            AND cited_by_count >= 10
    OR   2010 <= py <= 2014    AND cited_by_count >= 5
    OR   2015 <= py <= 2019    AND cited_by_count >= 3
    OR   py >= 2020                                        # unchanged: is_core /
                                                            # target-country carve-out
                                                            # already governs here
```

— trims **965,817** documents (**15.12%**; 966,649 points) and keeps **5,422,188** documents. `is_core` and
the target-country (`py>=2020`) carve-out are untouched by the ladder at every
severity tested (task 490 §2); the trim tracks each field's corpus share
almost exactly, so no field is disproportionately gutted (§2, "Trim by
field"). One named exception: **Law** loses 17–29% of its (small, ~117k-row)
slice under any tested ladder — flagged to Tim, no exemption ruled; Law
stays on the same L2 floor as every other field.

**Before / after the reprice:**

| | before (v1.1) | after (this ruling) |
|---|---:|---:|
| §4.2 OpenAlex component target | ~3M | **5,422,188** documents (~5.42M); 5,438,138 points served (49/49 drain receipts `promotion-queue/openalex-*/receipt.json`, 86 rows dropped by the mechanical-content rule) |
| OpenAlex materialized population (unpriced) | 6,388,005 documents (6,404,873 vector rows) | 6,388,005 documents (unchanged — the reprice sets the *admitted* figure, not a new acquisition) |
| OpenAlex ladder-trimmed (sealed, excluded) | — | 965,817 documents (966,649 points) |

**Layer arithmetic, stated honestly:** the scholarship layer does not fit
inside its 10M ceiling once OpenAlex is repriced to its real, measured
kept figure rather than the original ~3M placeholder.

| component | serving / kept (points; OpenAlex also in documents) | source |
|---|---:|---|
| PubMed | 4,969,092 | ADR-074 execution census (task 462 G1 assembly-guard enumeration, decisions.md ADR-074/§ADR-101 evidence; task 453 coverage census) |
| arXiv | 1,918,324 | same census (task 462 G1; decisions.md ADR-101 evidence table) |
| OpenAlex (repriced, L2) | 5,438,138 points (5,422,188 documents ratified) | served: 49 drain receipts `promotion-queue/openalex-*/receipt.json` (task 629); documents: task 490, this ruling |
| **scholarship layer total** | **12,325,554 (~12.33M)** points (12,309,604 with OpenAlex at the ratified document figure) | sum |
| **ratified §4.2 ceiling** | 10,000,000 | this document, `specs/next-100m-expansion-waves.md` |
| **overage** | **2,325,554 (~2.33M)** points (2,309,604 at the ratified document figure) | funded from the demand reserve (row above); +15,950 points = 16,036 extra chunk points of 10,741 multi-chunk documents, less 86 rows dropped |

Task 490 already flagged that no age-scaled citation floor closes the ~2.1×
§4.2 overallocation task 488 found (even L3-sharp only reaches 5,112,815,
still 1.7× the old ~3M target) — this ruling does not try to; it accepts
OpenAlex's real, measured shape at L2 and funds the resulting layer overage
from the demand reserve instead of forcing a sharper cut that would start
eating into `is_core`/target-country protections. **This PR legislates the
repriced target and the amended cap policy only** — it does not itself
delete, embed, or fold any row; identifying and sealing the exact ~965,817
ladder-trimmed documents (and folding the ~5.42M kept rows into production) is a
follow-up execution task against the now-ratified L2 floor (see decisions.md
ADR-102).

## 3f. ERIC admission from the demand reserve (ADR-113, Tim 2026-09-25, "F7b A")

Fork F7b (task 615 item 10): ERIC's ≈148,212-PDF backlog does not fit
§4.2's 10M ceiling on its own (already 12,325,554 served points after §3a; 12,309,604 at the ratified document figure). Resolved as a demand-reserve draw,
not a §4.2 amendment:

| component | value | source |
|---|---:|---|
| chunks/doc, current policy (768w, epoch 4) | 40.456 | E631-1, 250 real docs, tranche 010000 |
| chunks/doc, new policy (1024w `-generic-arm`, epoch 8) | 30.06 | E631-1, same 250 docs — 25.70% fewer chunks |
| ERIC backlog (PDFs never Docling'd) | 148,212 | `specs/portfolio-100m.md` §6.1 / task 631 |
| **reserve draw ceiling** | **≤ 4,455,253 (≤ 4.5M)** | 148,212 × 30.06, this ruling |
| demand evidence for the draw | 1.05% of measured in-mission agentic demand | task 395, "education & learning-science research" theme (§3b) |

The chunk policy is ratified **selectable for `eric_official_pdf` only** (`data/chunk-policy-
registry.v1.json`'s `admitted_for_families`) — it stays `proposed` for the rest of `chunk.generic`
pending a held-out retrieval eval; only the structural reduction is ratified here. Not done here:
no re-chunk/re-extraction — the growth lane re-queues ERIC's backlog under the new policy and
ceiling. See decisions.md ADR-113.

## 3g. Common Crawl authority-frontier tranches 7–12 (Tim 2026-10-05, ruling 26, item 18a)

Confirmed. The 59,141 hosts of the roster's "unallocated" cell (task 634 held the cell out of
scope on 2026-09-22 pending Tim) form tranches 7–12 of the Common Crawl authority frontier.
Admission is inside PORTFOLIO (ADR-110, tranche admission). The class is endorsed or keyword
edu, gov and reference hosts, at the trust level of POLICY §4.4 (≤ 0.60–0.85). Rows are tier C,
consent-stamped at stage time, and spam-screened. Landed so far: tranche 7 (902,518 rows) and
tranches 8–12 (manifest rows 765,324 / 487,654 / 597,322 / 567,149 / 591,969).
Nothing else about the cell is decided here.

## 3h. Supply rules for the Common Crawl frontier (Tim 2026-10-07, ruling 27, T2)

1. **Tranche 13 and later.** Hosts that are endorsed only by outlinks (no PORTFOLIO cell) may
   form tranche 13 and later. Each host first passes a class pre-screen and a spam pre-screen.
   Admission stays inside PORTFOLIO (as for §3g).
2. **Depth.** An endorsed edu, gov or reference host is read to depth 120 pages. The rule was 60.
3. **Edu rows of tranches 7–12 draw on the demand reserve.** The scholarship layer is over its
   ceiling (§3a). The edu-class rows of tranches 7–12 (§3g) are therefore counted as draws on
   the demand reserve, as ERIC was (§3f, ADR-113). The number is `N` = the served rows of
   tranches 7–12 whose host class is edu in the roster. The growth task 714 counts `N` from the
   drain receipts. The reserve falls by `N` (from ~30.19M). The count enters the draw ledger of
   §3 when task 714 reports it. This PR does not state `N`.

## 3i. SMI corporate sites (Tim 2026-10-23, ruling 59)

> RULING 59 (Tim 2026-10-23 ~04:50Z): "Yes, admit roche.com and the SMI corporate sites".

1. **Class.** `corporate`: the official website of a company that is a constituent of the SMI.
   A corporate site is authoritative about ITSELF (its results, reports, press releases, products,
   governance). It is not authoritative about its industry, and it is not a mission-tier layer.
2. **Roster source.** The constituent list is the one SIX publishes for the SMI (20 companies on
   2026-10-23; the list, URL and date are recorded in the task evidence). The roster is
   re-read from SIX at each quarterly review; a change of constituents is a roster change inside
   this class, not a new ruling. Domains come from open data only: Wikidata P856 (official
   website), confirmed by a second open source (Wikipedia external links). A domain that two
   open sources do not confirm is listed as `unresolved` and is not guessed. No search-engine API
   is used to resolve a company to a domain. A national `.ch` site of the same company is admitted
   only when two open sources name it (as roche.ch, zurich.ch).
3. **Trust class.** The vendor-content class of POLICY 4.4: `trust_tier = 0.60`. Never the
   mission tier: POLICY 4.4 gives a mission tier only by affirmative classification, and a
   corporate site is classified here as a corporate-communications source. `roche.ch` moves from
   `edu` to `corporate` in the authority table.
4. **Consent.** Probed with the package resolver before any fetch, and again at
   stage time. A host that answers 403 from two vantage points is `unreadable` (WAF), is not
   `declined`, and is held out of the roster until a probe reads it. A CCBot or AI-training
   disallow sets `train_ok = False` only (ADR-024 am.5).
5. **Depth and refresh.** Common Crawl outlink tranche 20, behind tranche 19, depth 120 pages per
   host (as for a ruling-named host in 3h). Refresh class: `static` (3c: re-verify about
   quarterly), with the quarterly roster review. Nothing is crawled live.
6. **Size.** 21 hosts at about 145 rows per host: about 3,000 rows. It does not touch the 100M
   matrix; it draws on the demand reserve at this size.
7. **Not decided here.** Companies outside the SMI (SMIM, SPI, foreign blue chips) are not in this
   class. Each needs its own ruling.

## 3b. Strategic themes (cross-cutting dimension)

Layers are rows; **themes cut across them** (mental health spans health +
scholarship + community; zero-employee ops spans law + finance + docs +
management scholarship). v1.0 flattened themes into layers and lost their
accounting — Tim's challenge, 2026-07-22. Fix: every pack carries `theme:`
tags; MANIFESTs roll up per theme; this table is the thematic ledger.

Verified holdings (staging-collection census + live probes, 2026-07-22):

| theme | components in index | status |
|---|---|---|
| **mental health** | d176 (38.8k) + CT.gov-psych (2.2k) + psy_csm; probe: *psychological safety* → wiki+PubMed ✓ | **strategic** — named target; §4.4-generated pilot; **research-frontier on weekly freshness (§3c)** |
| **zero-employee-company ops** (the dogfood theme — the org runs on it) | SME (13.1k) + business (42.6k) + fin-facets + FINMA + org-psych scholarship; probe: *créer une SARL seul* → kmu.admin.ch ✓ | **strategic** — named target; grows with law/gov + docs campaigns |
| **AI / agents** | skills (57.3k) + frameworks (14k) + HF cards + agent-resources + GitHub (staged 246k) | **strategic** — carried by the §4.4 15M docs target; **research-frontier (arXiv) on daily freshness (§3c)** |
| **social contract / AI governance** | twin census 2026-07-26 (task 323, 36-cell scorecard): law-spine cells green via eur-lex/fedlex/admin.ch/bger/ige; IGO normative layer near-absent (ilo 4 / oecd 4 / who 6 / unesco 14 pts); EDÖB 330 | **strategic** — accounting-only row, NO allocation change (Tim, PR #47, 2026-07-26); packs P1–P5 (~0.5M, `specs/social-contract-coverage.md`) draw on ratified layer headroom + demand reserve and carry `theme: [social-contract]`; tracked by re-running the census panel |
| **transition_competences** (FR-framed, ADR-032 am.2) | crawled; 17 eval golds | **strategic** — ratified 2026-07-12; row restored by audit 397 (was dropped in the v1.0 flattening) |
| **droits_fondamentaux** (FR-framed, ADR-032 am.2) | crawled; 18 eval golds | **strategic** — ratified 2026-07-12; row restored by audit 397 |
| DEI / rights (**explicitly incl. inclusive leadership** — Tim 2026-08-07: merged by ruling, named so it cannot silently vanish; its 16 eval golds track it) | 58.9k + 36.9k | covered — organic maintenance |
| philosophy | 133.2k | covered — organic maintenance |
| EU (institutional) | 78.1k | covered — organic maintenance |
| **energy & climate transition** (PROPOSED from demand evidence, task 395: 1.53% of in-mission agentic demand; ipcc.ch absent from index, PubMed floods the class) | CH leg excellent (bfe.admin.ch); intl authorities MISSING | proposed — add IEA/IPCC/IRENA-class sources; ratified on merge |
| **education & learning-science research** (task 395: 1.05%; PISA/OECD stats excellent, EdTech-research queries return vendor blogs only) | partial + ERIC ≤4.5M queued (ADR-113, §3f) | **strategic** — first draw ratified on merge; ERIC is the education-research authority this theme named as missing |
| **urban governance / public policy** (PROPOSED, task 395: 0.90%; PubMed-flood pattern) | missing | proposed — OECD-urban/UN-Habitat-class; ratified on merge |

Strategic themes get named targets and quarterly review; covered themes ride
the backbone + demand reserve (governance §4.4 — a theme deepens on
frontier evidence, not sentiment).

## 3d. Geographic expansion roadmap (v1 — Tim, 2026-08-07; stages gated, not dated)

Countries are a *treatment*, not just a source list: the "Switzerland
treatment" = primary law + courts + regulators + statistics + **company
register with verified website links** (Zefix/CC-join pattern, ADR-040/041) +
entity cards + native-language depth. Stages advance on evidence (gap analysis,
demand, index-impact measurement), each through its own admission gates.

| stage | scope | register treatment | status |
|---|---|---|---|
| 0 (done/ongoing) | **CH** exhaustive | Zefix — **194,255 register-id-bearing entity cards serving** (measured `aw_e8` 2026-09-19; the earlier "~110k" was one of five circulating CH counts — see `specs/register-first-cards-all-countries.md` §8.5), **29.6% of them on non-`.ch` domains** | live |
| 1a | **FR-speaking Belgium** — Justel/Moniteur legal spine, federal+Wallonia-Brussels gov | KBO/BCE register: **redistribution (mode B) DECLINED on licence** (task 435, 2026-08-20 — bespoke click-through contract, art. 11 forbids assigning the rights to a third party, exactly what a redistributable dump would need); **locator-only use (mode A) VIABLE** (task 623, 2026-09-20 — Tim: *"we're not redistributing the dump, we're using our own dump to locate company websites"*; the same licence's art. 2.6 personal/non-exclusive right, exercised under a registered purpose, permits the internal use even though it forbids publication). The legal spine is unaffected either way | legal spine queued; **register leg reopened in mode A**, 20,000 cards restored pending the registration-purpose ruling (`specs/register-first-cards-all-countries.md` §4.6); mode B (officers, register fields) stays blocked pending outreach |
| 1b | **France + Germany + Canada** (Tim 2026-08-07: honors the July ordering; FR-Canada also strengthens the French leg) = full CH treatment, **explicitly incl. the subnational layer** (Tim 2026-08-07): FR régions/départements/communes portals, DE Länder/Kreise/Kommunen, CA provinces/municipalities — mirroring CH's cantonal/communal depth | SIREN/SIRENE (number-join proven 12.9% of .fr, task 225) — **FR is GO**, Licence Ouverte 2.0, 5,783,560 cardable, parquet already on disk; CA federal **GO**, OGL-Canada, ~693,000 active, CSV not yet pulled; DE Handelsregister: **mode B (register-derived publication) DECLINED on statute** (task 435): Nutzungsordnung Ziff. 3 forbids systematic retrieval for building a register, § 52 HRV obliges the operator to make it technically impossible, and there is no stable per-company URL at all — not a scout gap. **Mode A (silent internal locator) does not apply either** — Ziff. 3 binds the retrieval itself, not its publication. **Mode A′ (register-FREE, site-first via the statutory Impressum + VIES) is PENDING Tim's ruling** (task 623, 2026-09-20, `specs/register-first-cards-all-countries.md` §4.7a) — if ratified, the 50,000 quota is restored without ever querying the Handelsregister; if declined, the only re-open for mode B remains EU HVD Reg. 2023/138 compliance | queued |
| 2 | **Italy, UK, US** | Registro Imprese / **Companies House — GO on the BULK product**, basis is **s.47(3) CDPA 1988** statutory permission, *not* OGL-3.0, and *not* the API (`api.company-information` robots 401 = DECLINED); 5,190,464 active rows, zip already on disk / US has **no federal company register**: SEC EDGAR is a disclosure system covering 39,002 operating filers against a 65,000 quota — the register is 50 state SoS, unscouted | **UK ready to build**; US quota unbacked (scope decision) |
| 3 † | **Smaller-language probe: Romania, North Macedonia, Kosovo** — deliberately small legs to MEASURE how low-resource-language expansion impacts index quality (embedder behavior, ranking, floods) before broader EU rollout | ONRC / CRM / ARBK (scout) | experiment, after stage 2 start |
| 4 † | remaining EU countries/languages | per-country scout | on stage-3 verdict |

† Stages 3–4 admit languages beyond the ratified EN/DE/FR/IT scope
(spec.md / CONSTITUTION). Ruling (Tim 2026-08-07): starting stage 3
**requires a language-scope amendment** to spec + CONSTITUTION first — the
probe is the evidence-gathering FOR that amendment; the ratified scope stays
EN/DE/FR/IT until it passes.

**Register-treatment addendum (task 621, PROPOSED — ratified on merge; revised task
623, 2026-09-20).** The "company register with verified website links" half of the
country treatment is **register-first**: the candidate roster is the register's
bulk product, and a host-TLD crawl supplies evidence about a candidate, never the
candidate list (`sources/POLICY.md` §4.9.4). Two consequences for this roadmap:
(a) a stage's register leg is scoped by its **register adapter**, not by a ccTLD,
so a country's companies on `.com`/`.org` are in scope from the start; (b) a
register is not one permission but two — **`register_role: locator`** (the bulk
product is consulted internally only, to find and verify a site; nothing
register-derived is published) and **`register_role: enrichment`** (the licence
additionally permits publishing register-sourced fields, the CH/UK/FR/CA shape)
— and a country's licence can grant one without the other. **BE is mode-A
viable now** (its licence permits the internal use it forbids redistributing);
**DE has a PENDING mode-A′ route** (register-free, site-first via the statutory
Impressum, never touching the Handelsregister — Ziff. 3/§ 52 HRV never engage
because the register is never queried) alongside its standing mode-B statute
block; **US's 65,000 has no measured source behind it regardless of mode.** If
DE's mode-A′ is ratified, all seven ratified quotas are deliverable; if declined,
70,000 (DE+BE, now down to DE alone once BE's mode-A is ratified) needs
reallocation. Per-country detail, licences and statutes quoted from the
instrument, and the recommended reallocation are in
`specs/register-first-cards-all-countries.md` (§1a defines the two modes, §4.6
is BE, §4.7/§4.7a is DE, §6 has both scenarios' arithmetic).

Rule for every stage: the subnational government layer (state/regional +
municipal portals) is part of the treatment, not an optional extra — CH's
cantonal/communal depth is the template.

Second rule (Tim 2026-08-07): each country leg includes its **flagship
industry verticals**, mirroring the FULL ratified CH industry matrix (17
verticals, not a subset — audit 397 §5): banking/FINMA, insurance,
watchmaking, pharma, food/FMCG, commodities, machinery, chemicals, medtech,
ICT, construction, logistics, energy, fintech, tourism, fiduciaire/
accounting, retail — each as sector regulators + industry associations +
canonical companies + sector documentation (+ VET where the country has an
equivalent system). Per-country vertical lists come from the stage scout
(e.g. FR luxury/aerospace/agrifood, DE automotive/Mittelstand machinery/
chemicals — scout proposes, board ratifies).

**DE courts sub-facet addendum (2026-08-28, task 460 PR):** stage 1b's DE
courts spine gains a second, refreshable channel — OpenLegalData
(`sources/de/openlegaldata.yaml`) — alongside the rechtsprechung-im-internet.de
(rii) 38,159 points, which plan.md's 2026-08-28 ruling freezes as
non-refreshable (rii's robots now carries a Disallow-all with a
DG_JUSTICE_CRAWLER-only carve-out plus `tdm-reservation: 1`). Counting
boundary, per `specs/next-100m-expansion-waves.md`'s discipline (source rows
are not points until measured): OpenLegalData's live API reported 423,944
decisions on 2026-05-21 (task 459), and its current HuggingFace dump
(`dump-20260520`, re-verified live by task 460) holds 423,941 — both are
SOURCE-SIDE counts, not a point projection. No ingester exists yet
(an `ingest_openlegaldata` script is a declared, not-yet-built path) and no
acquisition run has happened; this PR admits the source and its acquisition
routes only. Once ingested, the pack draws on §4.1's DE law-layer target
(12M @100M, §3 table); it does not itself move any number in that table
today. Federal-court light top-up (BGH/BFH/BSG/BAG) and a fresh NRW scout
are named future work, not part of this admission (task 459 §3 ranks 2/4).

Rules: language-follows-source (never quota-filter); every leg consent-gated
per host; registers only via native/open routes (ADR-040 — no competitor
search APIs); each stage's index impact measured before the next widens
(displacement-by-better vs eviction-by-volume, per the eval framework).

**Theme addendum (same ruling):** add to §3b — **AI startups worldwide**
(the Y Combinator universe as the seed set: all YC companies, canonical
sites/docs/blogs, registry-verified where carded) as a strategic theme under
priority #2 (AI). Theme gap-detection instrument: **mine the agentic search
datasets** (DRGym, search-arena, own probe corpus) for query themes the
portfolio doesn't cover — recurring analysis, feeds §3b revisions.

## 3c. Freshness classes (the third dimension)

Freshness is a mission pillar (secondary only to grounding) and applies far
beyond news — a stale library-docs page or superseded statute misleads an
agent as badly as old news. Every pack declares its class (architecture
§4.1); MANIFESTs report **staleness budget** per layer: % of records older
than their class SLO (`last_seen` / `published` make it computable).

| class | SLO | applies to | mechanism | status |
|---|---|---|---|---|
| daily | <2 days | news headlines, HN; **research-frontier: new arXiv (cs.AI/agents) + MEDLINE daily-update files for strategic themes** (Tim 2026-07-25) | sitemap/feed delta cron; arXiv OAI + PubMed dailies | built (task 294) for news; research-frontier cron to build post-b1 |
| weekly | <14 days | agent-resources, docs releases, stats releases; **CT.gov new trials + preprint sweeps (mental-health theme)** | feed/registry delta | partial (HN ingest built; docs-release + research watches to build) |
| rolling-12mo | window | news corpus | delta add + age-out delete | policy set; blocked on `published` backfill (POLICY §4.7) |
| amendment-driven | <7 days of change | law (fedlex SPARQL delta), standards | source-native change feeds | to build (fedlex SPARQL supports it) |
| static | re-verify ~quarterly | scholarship, philosophy, reference | digest re-check on refetch | default |

The reshape's corollary: freshness spend should follow the matrix — a 15M
docs layer with a weekly class costs more refresh than 28M static PubMed
ever did; the delta cadence (ADR-066) is what makes that affordable.

## 4. Campaigns (portfolio ops after aw_b1 — all delta, no rebuilds)

1. **The reshape** (ex-Scenario B): PubMed 28M→5M delta-delete + arXiv
   re-slice. Sheds ~23M points; instantly rebalances every share above.
   Embed cost ≈ zero (deletion); starts when board ratifies this matrix.
2. **Docs/agent-resources buildout** toward 15M: GitHub staged (246k) is
   the seed; package-registry-driven admission under §4.4.
3. **FR/DE law packs** (authored, M4 pilot): DILA + Genesis/gesetze paths
   already exist as extractors. **DILA/LEGI LODA family (`sources/fr/legi.yaml`,
   task 633): ceiling raised** from an undocumented nature filter (130,907
   rows currently served) to the full in-force TNC population (≤1.39M
   dump-measured members, gated by the standard quality floor, not by
   nature) — the family's first explicit admission record, under POLICY
   §4.1's "deep" default (§5 T-rules). Closes part of the §4.1 gap named in
   specs/portfolio-100m.md §6.1.
   **Task 695 (POLICY PR, 2026-10-02): seven more law-layer (§4.1) families
   declared** in `data/source-family-contracts.v1.json`, all official
   dump/API routes, no HTML crawl: `dila_juri` (JADE + CASS, **year floors
   lifted**, ~0.64M as one locator per decision), `dila_kali` (~0.15–0.2M),
   `dila_jorf` (0.3–0.6M, floor unchanged), `govuk_content` (OGL v3, 1.4–2.8M
   est), `us_federal_register_ecfr` (public domain, 2–4M est),
   `uk_legislation` (OGL v3, 1.5–4M est, robots Crawl-delay), and
   `it_senato_gazzetta` (CC BY 4.0 + L. 633/1941 art. 5, 0.3–0.8M est). They
   draw on the §4.1 line of the §3 matrix and do not raise it; the
   estimates are plan figures (task 691), the adapters are unbuilt, and the
   bytes-per-point lever gates the tail (box full at ~64M). Normattiva's
   API route (api.normattiva.it, consent HELD) is not declared and stays
   blocked. Evidence: task 695.
   **POLICY PR of 2026-10-03 (Tim's rulings, plan.md ruling 23):** the Federal
   Register family (`us_federal_register_ecfr`) may also use the govinfo bulk XML
   route, beside the per-document API (US Government work, 17 U.S.C. 105; GPO
   notice read on 2026-10-03). The consent basis of `fedlex_exact` (URG Art. 5) and
   of six Swiss sitemap families (ElCom, BAV, BAZL, WEKO, ASTRA html, Swissmedic
   html; live per origin) is declared. The stale "adapter not yet built" texts of
   five families are corrected. No ceiling moves.
4. **Backbone generation** (`generated` packs): Wikidata institution graph
   → cards; the mental-health organic-vs-curated diff is the calibration.
5. **News steady-state**: date-backfill the staged batches (POLICY §4.7
   compliance), then cron delta refresh.
   **2026-10-03:** `watson.ch` admitted (POLICY §4.7; pack `sources/ch/watson.yaml`);
   the daily news lane is declared as the family `news_daily`. `heise.de` is not
   admitted until its heise+ paywall share is checked.

## 5. Mechanics

- Quarterly: MANIFEST actuals vs this matrix → gap map → next waves.
  Off-cycle changes only for E-events or board direction.
- Every cell change cites evidence (demand data, endorsement mass, eval
  lift) in the PR — same discipline as admission blocks.
- The demand reserve is not a slush fund: T2+ promotions drawing on it cite
  POLICY §5 evidence, and the quarterly review reports reserve burn-down.

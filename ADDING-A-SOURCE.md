# ADDING A SOURCE — from a clean checkout to a staged ingest

Companion to `QUICKSTART.md`. That doc builds a tiny index from a source someone
already wired up. This one is about **adding a source that nobody has wired up**.

Until task 434 there was no path from a source pack to a running ingest. The
`sources/**/*.yaml` files were read by exactly one program — the schema
validator — and a human had to open the YAML, read the `extractor:` field, and
invoke the named script by hand with arguments recorded nowhere. Adding a source
therefore meant writing a script, and the repo accumulated ~86 of them.

Now a pack is executable:

```bash
cyt source plan sources/_examples/example-authority.yaml          # no network, no crawler identity
cyt source run  sources/_examples/example-authority.yaml          # consent preflight, then fetch
python3 run_source.py sources/_examples/example-authority.yaml   # the same door, older name (a forwarder)
```

---

## What you need

- The repo, `python3`, and `pip install -e .` (QUICKSTART step 0).
- For `--plan` and validation: **nothing else.** No network, no GPU, no
  crawler identity. You can inspect any pack on a laptop.
- To actually fetch: a configured crawler identity (below) and, per
  `data/acquisition-policy.v1.json`, a machine that is allowed to fetch.
  **The orchestrator host only orchestrates; retrieval runs on the crawl host.**

### Crawler identity (required before any fetch)

```bash
cp operator.example.json operator.json    # then edit it
# or:
export CYTIR_OPERATOR_URL=https://example.org/about-our-crawler
export CYTIR_OPERATOR_CONTACT=you@example.org
```

The pipeline **refuses to fetch** without this rather than sending a placeholder
UA. That is deliberate: every host you touch logs that string, every allow/deny
decision is applied to it, and our consent evidence is only worth something if it
names the crawler that actually made the request.

---

## The model in one paragraph

A pack declares **what** the source is (`adapter`, trust, licence, consent
posture, admission record) and **how it is acquired** (`acquire:`). Those are two
different questions and the schema keeps them separate on purpose:

- **`adapter`** — which *engine* parses the bytes: `cc`, `dump`, `sitemap`,
  `registry`, `api`, `cards`.
- **`acquire.route`** — *where the bytes come from* and its politeness rank,
  from the ranked list in `data/acquisition-policy.v1.json`:

  | rank | route | prefer it when |
  |---:|---|---|
  | 1 | `source_owned_incremental_feed` | there is an RSS/Atom/OAI-PMH/change feed |
  | 2 | `source_owned_bulk_dump` | the publisher ships XML/JSON/CSV/WARC |
  | 3 | `documented_source_api` | there is a documented REST/SPARQL API |
  | 4 | `common_crawl` | CC already has the pages |
  | 5 | `source_owned_sitemap_fetch` | the site declares a sitemap |
  | 6 | `direct_origin_crawl` | **last resort**, and you must justify it |

**Use the highest-ranked route the source actually supports.** Rank 6 requires a
`reason_receipt` recording which better routes you checked and why they failed.

### `acquire` goes on the ROUTE, not the pack

This is the part people get wrong. A pack often reaches different hosts
different ways. A Swiss media pack reaches srf.ch and rsi.ch through
Common Crawl and rts.ch through its sitemap — one pack, two shapes. So
`acquire:` belongs on each `routes[]` entry. A pack-level `acquire:` is only a
**default** that routes may override.

---

## Adding a source, end to end

### 1. Write the pack

Put it at `sources/<region-or-theme>/<name>.yaml`. Copy the nearest existing
pack; `sources/_examples/example-authority.yaml` is the template for a sitemap
source (a placeholder, not a real admission), and `sources/ch/watson.yaml` is a
real sitemap pack.

Minimal sitemap example:

```yaml
id: example-authority
name: "Example Authority"
country: CH
layer: law-government
adapter: sitemap
acquisition_policy: aw-acquisition-v1
outlink_policy: capture-and-receipt
preservation_profiles: [web_html]
provenance: authored
trust_tier: 0.9
freshness: static
license: {default: public-domain}
consent: {basis: live-resolve-open}
status: planned

admission:
  channel: official-dump
  evidence: ["why we believe this source is authoritative"]
  decision: auto-admit
  policy: "POLICY.md@v1 §4.1"
  date: "2026-08-20"

routes:
  - host: www.example.ch
    lang: [de]
    status: planned
    acquire:
      route: source_owned_sitemap_fetch
      params:
        urls: ["https://www.example.ch/sitemap.xml"]
      select:
        path_regex: "(?i)^/(publications|law)(/|$)"
        deny_preset: generic-apparatus
        min_path_segments: 2
      lang: {const: de}
      delay_s: 2.0
      cap: 2000
```

**The `admission:` block is governed.** Per `sources/README.md`, `POLICY.md`,
`PORTFOLIO.md` and pack `admission:` blocks change **only via a PR merged by the
board**. Everything else in a pack lands normally.

### 2. Selecting URLs without writing code

`select:` is a closed vocabulary — deliberately small, because a survey of the
97 hand-written predicates in the old scripts found they all reduce to these:

| key | meaning |
|---|---|
| `path_regex` | keep only paths matching this |
| `path_prefix` | keep only paths starting with one of these |
| `path_deny_regex` | drop paths matching this (checked first) |
| `segment1_in` | keep only these first path segments |
| `deny_preset` | a named preset from `data/url-select-presets.v1.json` |
| `min_path_segments` | drop section landing pages (default 2 with a preset) |

`lang:` is either `{const: de}` or `{by_path_segment: {de: de, fr: fr}, default: de}`.

Use `deny_preset: generic-apparatus` unless you have a reason not to — it drops
search pages, logins, feeds, calendars and legal boilerplate, which are never
authoritative documents.

If your source genuinely needs logic this cannot express, **do not widen the
vocabulary** — ship a thin extractor (step 4).

### 3. Validate, then look at the plan

```bash
python3 -m cytir.sources.schema sources/_examples/example-authority.yaml
python3 run_source.py example-authority --plan
```

`--plan` compiles the pack and prints one line per runnable route. It touches no
network. Routes whose `status` is `declined`, `not-viable` or `deferred` are
skipped — a deferred route is a decision on record, not a work item.

The validator checks more than shape. It reads
`data/acquisition-policy.v1.json` and rejects a route that is not in the ratified
enum, a declared `rank` that disagrees with policy, a missing required param, and
an unknown param key (so a typo fails loudly instead of being ignored).

A rank-6 route without a `reason_receipt` currently prints a **warning**. That
becomes an error once the existing offenders are fixed.

### 3b. Route params you are likely to need

Each route takes its own `params`. The validator rejects unknown keys, so a typo
fails at review instead of being silently ignored.

| route | required | useful optional |
|---|---|---|
| `source_owned_sitemap_fetch` | `urls` | `expand`, `ok_hosts` |
| `source_owned_bulk_dump` | `format` + one of `url` / `url_template` | `transport`, `ref`, `subpath`, `member_glob`, `checksum_url`, `index_endpoint`, `ids` |
| `documented_source_api` | `endpoint` | `dialect`, `paginate`, `query`, `headers` |
| `source_owned_incremental_feed` | `url` | `kind`, `since` |
| `common_crawl` | — | `path_prefixes`, `crawl`, `index_mode`, `host_scope` |
| `direct_origin_crawl` | one of `seeds` / `url_pattern` | `hub`, `hub_depth`, `date_from`, `date_to`, `date_format` |

Three things worth knowing, because each came from a real source:

- **A bulk dump is not always an HTTP GET.** Set `transport: git` (with `ref`
  and `subpath`) when the publisher distributes via a repository — the Italian
  Senate ships 1.71 GB of Akoma Ntoso that way. It is a dump; only its transport
  differs.
- **`url_pattern` for enumerable URL schemes.** Some publishers have no dump, no
  API and no sitemap, but a predictable path — Gazzetta Ufficiale is addressable
  by date. Use `url_pattern` with `date_from`/`date_to`/`date_format` instead of
  listing seeds. This is still rank 6, so it still needs a `reason_receipt`.
- **`dialect` for API flavour.** In use: `rest`, `openapi`, `sparql`, `oai-pmh`,
  `sdmx`, `s3-inventory`. Deliberately not an enum — `api` is the most diverse
  bucket in the codebase and a premature enum would be a wall, not a contract.

### 3c. Committing never to crawl a source

Some sources are admitted on the condition that we take the publisher's bulk
file and never touch their HTML pages. Declare it:

```yaml
acquire:
  route: source_owned_bulk_dump
  no_fallback: true
  params: {url: "https://example.gov/register.csv", format: csv}
```

`no_fallback: true` is a binding commitment, enforced in **two** places because
a promise a tool cannot check is just a comment:

1. **At validation** — a pack that sets `no_fallback` anywhere and declares a
   crawl route anywhere is rejected. The contradiction cannot reach a run.
2. **At run time** — `run_source.py` refuses to dispatch such a plan to a crawl
   engine, so a hand-built plan cannot bypass the promise either.

Declared at pack level, it binds every route.

### 4. If the source needs a bespoke dialect

Real sources are bespoke: DE Genesis, FR DILA and StatCan WDS each speak their
own API. A bespoke dialect is a **family adapter**: one class in
`cytir/sources/families/`, registered with a declaration in data (see "Writing the
adapter" below). `cyt source run <family_id>` drives it through the same stages.
What the framework standardises is everything *around* the dialect: consent,
politeness, staging, receipts. It does not standardise the dialect itself.

Your adapter must **import** identity, canonicalization, tokenization and
chunking from `cytir/core/ids.py` and `cytir/core/common.py`. A new `uuid5` outside
those modules is a defect by definition.

> **Private operator step.** The old pack field `extractor:` pointed at a
> standalone script from the operator's private build tree. Those scripts are not
> in this repository, and a pack that names one cannot run here. The packs for
> `source_owned_bulk_dump`, `documented_source_api`, `source_owned_incremental_feed`
> and `common_crawl` that still name an `extractor:` are an operator step until
> their family is ported into `cytir/sources/families/`.

### 5. Run it

```bash
# smoke test: 20 documents, see that selection and extraction behave
python3 run_source.py example-authority --out-dir build/example --limit 20

# dry run: resolve consent live, then stop before fetching
python3 run_source.py example-authority --dry-run
```

Consent is resolved **live, per host, before any fetch**, in this order:

1. `consent_block.assert_crawlable(host)` — recorded decline decisions
2. `consent_resolver.resolve_cached_host(...)` — live AIPREF / robots / TDMRep
3. `consent_policy.gate_bucket(rec, search_indexing=True)` — the index rubric

Hosts that do not come back `ok` or `ok_search_only` are skipped and recorded.
A `status: unknown` verdict is **transient and retryable** — it is never a
refusal, and must never be recorded as one.

The run writes a receipt to `<out-dir>/run_<pack-id>.json` with the consent
verdict for every host, the plans, and what each engine returned.

---

## Which engine actually runs

| route | engine |
|---|---|
| `source_owned_sitemap_fetch` | the `sitemap` family (`cytir/sources/families/sitemap.py`) |
| `direct_origin_crawl` | not ported to a family: an operator step |
| `source_owned_bulk_dump` | a registered family (`cyt source families`), else an operator step |
| `documented_source_api` | a registered family, else an operator step |
| `source_owned_incremental_feed` | a registered family, else an operator step |
| `common_crawl` | the `cc` family (`cytir/sources/families/cc.py`) |

`run_source.py` and `cyt source run` **dispatch; they do not fetch, extract, chunk, tokenize,
canonicalize or mint ids.** The family adapter and the framework around it do that.
A family whose declaration still holds an `unknown` (chunk policy, id rule, route) is
refused with a named reason. `fedlex_exact` is refused this way today (see QUICKSTART
section 1b for what is and is not on the path).

---

## Writing the adapter (the source-family interface)

A pack says WHAT a source is. A **family** is what the pipeline runs for it: a declaration in
data plus one class (task 683). `cyt source run <family_id>` drives any registered family through
the same stages, so a new country, topic or site is never a new script:

```bash
cyt source families                          # what is registered, and its declared consent basis
cyt source run loda_exact --out-dir build/loda --opt root=data/samples/legi-loda \
    --opt release=aw-sample-loda-v1                # offline, ends at the sealed pack (QUICKSTART 1b)
                                                   # needs the pinned tokenizer: QUICKSTART step 0b
```

| stage | owner | rule |
|---|---|---|
| discover | adapter | list the units; cheap, no content |
| consent | framework | every host through the ONE resolver before any byte; a decline EXCLUDES, an unreadable policy QUARANTINES |
| acquire | adapter | fetch one unit; the framework seals the bytes (content-addressed L0) before `extract` |
| extract / chunk | adapter | under the chunk policy the contract declares (`cytir.chunk` registry) |
| embed precondition | framework | `assert_embeddable`: refused for an unchecked pack, an `unknown` consent basis, or a recorded declined host. Vectors come from `cyt embed shards`; the index is `cyt index build` (slice 8 step 2). `--embed` is retired |
| compose | framework | `pack/records.jsonl` + `manifest.json` (contract digest, consent verdicts, terminals) |

Every unit ends `admitted`, `excluded(reason)` or `quarantined(reason)`, and the manifest prints
`in = admitted + excluded + quarantined`. **An error is never an empty**: a 404 is the origin's
answer (excluded), a timeout is ours (quarantined, re-probe).

To add a family:

1. **Declare it** in `data/source-family-contracts.v1.json` (consent basis, licence instrument,
   chunk policy, identity). This is a POLICY PR; an undeclared family is refused.
2. **Write one class** implementing `cytir.sources.family.SourceFamily` and decorate it with
   `@register_family`. A sitemap site is five lines (`cytir/sources/families/sitemap.py`,
   `ElcomDocsFamily`): the pack already holds the URLs, `select`, `lang`, `delay_s` and `cap`.
3. **Name the module** in `cytir.sources.registry.FAMILY_MODULES` (or `$CYTIR_FAMILY_MODULES` out of tree).
4. **Prove it.** A dump/API family gets a golden over a sealed tape (`cytir.acquire.tape`, the fedlex family golden test); a live-site family gets a selection-equivalence test
   against the script it replaces (the sitemap family test).

The three exemplars: `fedlex_exact` (dump: rows byte-equal to the pre-pack fedlex dump script on the
sealed #442 tape), `scratch_elcom_docs_exact` (sitemap pack: selection equal to
`cytir.sources.cov.engine.enum_sitemap` on the same inventory), `cc_authority_frontier_w7` (Common Crawl:
a sealed tranche, dry run; <= 5 requests/s, ADR-014).

---

## Verifying a migration

If you are moving an existing bespoke ingester onto a pack, prove it — a
migration that silently changes what gets indexed is the failure mode worth
catching.

- **Deterministic source (dump/API):** run both and compare bytes, **after pinning
  what the run date writes**. Every fedlex row carries `last_seen` = the day the dump
  ran, so the raw file hash changes daily. Compare with that field normalized:
  `sed 's/"last_seen": "[0-9-]*"/"last_seen": "X"/' out | sha256sum`.
  `ch-fedlex` with `--eli cc/1999/404 --langs de fr it` gives 732 passages and
  (consolidation 20240303, `representation=channel_split`, normalized)
  `sha256 1b6d7430f4c4234b385b4588216d3a52c459d69fcc899c4a67567309a1e3efeb`.
  The raw hashes are the same bytes on their own day: `98c4def6…` on 2026-08-20
  (task 434) and `0134f0b5…` on 2026-09-30 (task 677). Task 677's first note called
  `98c4def6…` irreproducible; that was wrong. The output with `last_seen` set back to
  2026-08-20 hashes to exactly `98c4def6…`, so the extractor did not change between
  them. Re-baseline on a deliberate change to the fedlex extractor (`cytir/sources/families/fedlex.py`), and record the
  consolidation date, because this source is amendment-driven. The same check runs
  **offline in CI**: the fedlex offline golden test replays sealed
  SPARQL and XML replies for one small act through the real fedlex extractor's `main()`.
- **Live source (sitemap/crawl):** you cannot byte-compare two crawls of a
  moving site. Compare the thing the migration changed — the **selection
  decision** — by running the old predicates and the new compiled ones over the
  same inventory. the task 434 equivalence script (kept in the maintainers' task record) does this;
  it agreed on all 18,399 URLs for the education pack.

---

## What this does not cover

- **Embedding and indexing.** Once you have a sealed pack, vectors come from
  `cyt embed shards` and the index from `cyt index build` (slice 8 step 2; the old `cyt embed
  index` door is retired), and prod updates go through the blue/green door, never by hand.
- **Rewriting the big dump ingesters.** PubMed, Wikipedia and arXiv produced
  ~88% of the corpus through genuinely bespoke code. They are deliberately out
  of scope: automating them would buy nothing. This tool exists for the long
  tail, where the per-source effort actually goes. Its success is measured in
  **scripts not written**.

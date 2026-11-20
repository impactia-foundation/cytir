"""Builder projection: `project_record`, the dense/sparse surface helpers and
the payload-contract evaluators (moved from the b1 builder script, task 2103).

The code below is the original block, unchanged. It is NOT merged with
`cytir.core.projection` (a later dedupe). Payloads and surfaces are byte-identical."""
from __future__ import annotations

from collections import defaultdict

from qdrant_client import models as qm

import cytir.core.common as AC
import cytir.core.ids as ID
import cytir.core.license as LIC
import cytir.core.projection as PROJ
from cytir.core import country as CT
from cytir.index import serving_metadata as SM
from cytir.index.snippet_surface import strip_provenance_header
from cytir.sources.families import fedlex_akn as FEDLEX_AKN
from cytir.consent.build_gate import (
    ConsentGate, DUMP_CONSENT_STATE, dump_basis_evidence, _host_of, _is_gov_host)


FEDLEX_DOMAIN = "fedlex.admin.ch"
NEWS_PACKS = {"aw_dswissinfo_cov", "aw_dchmedia_cov"}
# ch-media undated-gov exclusion (task 321, orchestrator ruling; preflight
# T8/M21): 1,389 of 1,810 aw_dchmedia_cov rows are UNDATED admin.ch pages —
# government press releases the ch-media ingest mislabeled as news. A row from
# this pack on admin.ch (or any subdomain) with no resolvable published date
# (payload or sidecar) is EXCLUDED from the merge; dated rows and non-admin.ch
# hosts merge as before. Counted as staged_excluded_undated_gov.
CHMEDIA_PACK = "aw_dchmedia_cov"

# --- task 358 R1: payload-contract assertions ------------------------------
# The class of bug this exists to kill: a field the LIVE SERVING SCORER keys
# on (`cytir/serve/retrieval.py`, `pl.get(...)`) silently drops out of
# `project_record`'s passthrough whitelist, so the code path that reads it
# keeps running — no exception, no error, just a permanently no-op ranking
# signal (rule-vs-data-drift, memory note). Found this way: `canton` (task
# 123/144 cantonal-scope penalty) was omitted from the passthrough tuple for
# the entire aw_b1 rebuild — 0/336,723 ch_cantonal_law points carried it,
# the penalty ran as a silent no-op index-wide, discovered only by an
# unrelated reachability audit (task 358a), not by any build gate. Each
# entry: a POPULATION predicate over the projected payload, a FIELD that
# population must carry, and the minimum presence rate. Same fail-loud
# pattern as check (h)/(k)'s consent counters — asserted every build, not
# just when someone remembers to look.
PAYLOAD_CONTRACT = [
    {"name": "canton_on_cantonal_law",
     "population": lambda pl: pl.get("provenance") == "ch_cantonal_law",
     "field": "canton", "min_rate": 0.999},
    {"name": "names_on_entity_card",
     "population": lambda pl: pl.get("source_type") == "entity_card",
     "field": "names", "min_rate": 0.999},
]


def evaluate_payload_contract(points: list[dict],
                              rules: list[dict] | None = None,
                              min_sample: int = 20) -> tuple[bool, dict]:
    """Pure, standalone (unit-testable independent of a full build run):
    for each rule, count the population + how many carry `field`, compare the
    rate to `min_rate`. A population under `min_sample` is reported but can't
    fail/pass meaningfully (matches acceptance-check (g)'s convention) — ok in
    that case, so a build with zero of some rare class doesn't spuriously
    block. `points` are aw_b1 point-dicts (`{"payload": {...}, ...}`, the same
    shape `run_acceptance_checks` operates on)."""
    rules = PAYLOAD_CONTRACT if rules is None else rules
    overall_ok = True
    detail = {}
    for rule in rules:
        pop = present = 0
        for p in points:
            pl = p["payload"]
            if rule["population"](pl):
                pop += 1
                if pl.get(rule["field"]) is not None:
                    present += 1
        rate = present / pop if pop else None
        ok = rate is None or pop < min_sample or rate >= rule["min_rate"]
        detail[rule["name"]] = {"population": pop, "present": present,
                                "rate": round(rate, 4) if rate is not None else None,
                                "min_rate": rule["min_rate"], "ok": ok}
        if pop >= min_sample and not ok:
            overall_ok = False
    return overall_ok, detail


def evaluate_dump_basis_emission(points: list[dict]) -> tuple[bool, dict]:
    """Pure: every point ADMITTED on ADR-076 dump evidence must EMIT it.

    Task 433 WI-2 / task 468. This is deliberately not a PAYLOAD_CONTRACT rule:
    those key their population on the projected payload, so a rule that both
    selects and asserts on `consent_state` would pass vacuously at population
    zero — exactly the tautology that lets a stamp-strip look green. The
    population here comes from the SOURCE payload (`point["dump_basis"]`, set
    by project_record from the same `dump_basis_evidence` predicate the gate
    admits on), the assertion from the emitted payload. Exact, not a rate: a
    single unstamped dump-route point is a point the next rebuild evicts.
    """
    population = missing = 0
    invented: list[str] = []
    for point in points:
        payload = point["payload"]
        stamped = payload.get("consent_state") == DUMP_CONSENT_STATE
        if point.get("dump_basis"):
            population += 1
            if not (stamped and dump_basis_evidence(payload)):
                missing += 1
        elif stamped:
            # A dump stamp on a record whose source carried no recognized
            # evidence: the laundering shape ADR-076 closes. Named, not
            # counted away.
            invented.append(payload.get("url") or payload.get("point_id") or "?")
    ok = missing == 0 and not invented
    return ok, {"population": population, "unstamped": missing,
                "stamped_without_source_evidence": len(invented),
                "first_unsourced": invented[:3], "ok": ok}


def compose_index_text(rec: dict) -> str:
    """Shared title/site composition for both full-build channels.

    This used to be a vendored copy for an old build-host checkout. That creates
    a mirror contract between the normal pack path and full rebuild precisely
    where channel-specific representations need byte-identical behavior. A
    build must now run from the repo-current pipeline rather than silently use
    a stale local fork.
    """
    return AC.compose_index_text(rec)





# --------------------------------------------------------------------------
# per-record projection
# --------------------------------------------------------------------------
def _to_sparse(vec_entry):
    """Normalize a scrolled bm25 sparse vector into (indices, values) lists."""
    if vec_entry is None:
        return None
    if isinstance(vec_entry, dict):
        return list(vec_entry.get("indices", [])), list(vec_entry.get("values", []))
    idx = getattr(vec_entry, "indices", None)
    val = getattr(vec_entry, "values", None)
    if idx is None:
        return None
    return list(idx), list(val)


def _match_chunk_text(texts: list[str], snippet, stats: defaultdict):
    """Resolve an AMBIGUOUS corpus-text join (anchor-less multi-chunk page: the
    KV key (canonical_url, anchor=∅) maps to several chunk texts) to THIS
    record's own chunk text, using the record's stored snippet.

    The stored snippet is `extractive_snippet(chunk_text)` — a whitespace-
    normalized PREFIX of the chunk's own text (optionally sentence-trimmed,
    optionally "…"-suffixed) — so prefix-match is the discriminator. Returns
    the matched text or None (caller falls back to the honest no-text tail;
    task 319 fix 3 / GAP A33 — never guess a chunk's text).
    """
    snip = ID.normalize_digest_text(snippet or "")
    if snip.endswith("…"):
        snip = snip[:-1].rstrip()
    if len(snip) < 12:   # too short to discriminate between sibling chunks
        stats["ambig_snippet_too_short"] += 1
        return None
    norm = [(t, ID.normalize_digest_text(t)) for t in texts]
    pref = [t for t, n in norm if n.startswith(snip)]
    if pref:
        if len(pref) > 1:
            stats["ambig_multi_prefix_match"] += 1
        stats["ambig_matched_prefix"] += 1
        return pref[0]
    cont = [t for t, n in norm if snip in n]
    if cont:
        stats["ambig_matched_contained"] += 1
        return cont[0]
    stats["ambig_unmatched"] += 1
    return None


def _retokenize(text: str, lang, rec_for_compose: dict):
    """compose_index_text(title/site_name+body) -> folded bm25_tokens -> saturate.
    Returns (indices, values)."""
    composed = compose_index_text(rec_for_compose)
    # Shared projection owns both the historical whole-record path and the
    # explicit entity-event segmented path.  Keeping the full rebuild on the
    # same function prevents a future rebuild from silently flattening the
    # event boundaries restored by a source-scoped pack.
    return PROJ.saturated_bm25(composed, lang)


def _channel_surfaces(corpus_row: dict | None, text: str) -> tuple[str, str, bool]:
    """Return (dense_raw, sparse_raw, reembed_dense) for a corpus join.

    Normal corpus rows have one text surface and preserve the frozen dense
    vector. Fedlex's explicit R2 channels are atomic: accepting only one side
    would silently turn a representation experiment into an unmeasured hybrid,
    so it fails loud. An explicit dense channel means the source vector was
    made from a different document surface and must not be reused.
    """
    if not corpus_row:
        return text, text, False
    dense = corpus_row.get("dense_text")
    sparse = corpus_row.get("sparse_text")
    if dense is None and sparse is None:
        return text, text, False
    if not dense or not sparse:
        raise ValueError("incomplete corpus dense/sparse text channels")
    return dense, sparse, True


def projected_snippet(domain: str, text: str) -> str:
    """The snippet `project_record` serves for a row with pack/corpus TEXT (<= 25 words).

    One function for the drain and for `cyt ops snippet-repatch` (task 2734): a re-derived
    snippet is never a second implementation of the rule.

    Task 2733: the epoch-4 fedlex pack's `text` carries Akoma Ntoso eId scaffolding
    (`Art. 336 c {#art_336_c} ... paragraph {#art_336_c/para_1} 1 ...`). The INDEXED surfaces
    are stripped (task 564); the snippet is cut from the same text and must be too. One
    function, `fedlex_akn.strip_anchor_markup`, never a second regex. Gated on the serving
    domain, so no other family's brace-hash text is touched.
    Task 2737: skip a leading `## Record identity` provenance block (bfs/swisstopo/pubmed).
    """
    src = FEDLEX_AKN.strip_anchor_markup(text) if domain == FEDLEX_DOMAIN else text
    return AC.extractive_snippet(strip_provenance_header(src), max_words=25)


def project_record(src, origin: str, corpus_row: dict | None,
                   published_map: dict, conf_map: dict, gate: ConsentGate,
                   stats: defaultdict,
                   seen_ids: set | None = None,
                   metadata_registry=None) -> tuple[dict | None, dict]:
    """Project ONE source record (twin base point or staging point) into an
    aw_b1 PointStruct-dict. Returns (point_or_None, meta). None => excluded.

    `src` is a qdrant Record (has .id, .payload, .vector). `origin` is the source
    collection name (base collection or staging pack). `corpus_row` is the joined
    corpus text row (or None => no-text tail).

    `metadata_registry` (task 515) is an optional
    `aw_serving_metadata.ServingMetadataRegistry`. This function DERIVES
    identity, country, licence resolution and the snippet, and PASSES THROUGH
    the fields the ranker weighs -- `title`, `trust_tier`, `source_type`,
    `lang`, `site_name`, `last_seen`, `provenance`. That is correct when the
    input is what this signature says it is, a point out of a source
    collection. It is silently wrong when the input is a sealed
    embedding-stage record, whose contract is identity + text: task 514 shipped
    a collection with `trust_tier` and `source_type` blank on 11,273,954 rows
    (36.5%) and no count check could see it, because the projection returns a
    point either way. Passing a registry makes the absences resolvable and,
    crucially, COUNTED. Omitting it reproduces the historical behaviour
    byte-for-byte -- the full-rebuild path really does receive payloads.
    """
    payload = dict(src.payload or {})
    url = payload.get("url") or ""
    if not url:
        stats["no_url"] += 1
        return None, {}
    # The REQUEST host, read before hydration and never rewritten by it. The
    # consent gate, the cache key and the robots verdict are all properties of
    # the host the fetch actually addressed; the SERVING `domain` below is a
    # different thing that happens to look the same, and conflating them would
    # let a metadata fill move a consent verdict.
    host = _host_of(payload)
    if metadata_registry is not None:
        payload = SM.hydrate(payload, AC.canonical_url(url),
                             metadata_registry, stats)
    native_anchor = payload.get("anchor")
    # `_host_of` returns the URL netloc VERBATIM, `www.` included, and every
    # production ingester stamped the CANONICALIZED host instead -- the same
    # fold `aw_common.canonical_url` applies. The two agree on any record that
    # carries its own `domain`, and diverge on exactly the sealed epoch-4 /
    # pack record that carries none. Task 515 counted the damage on the staged
    # tree: 2,638,905 rows under a `www.` name, 2,568,264 of which production
    # wrote bare -- and `retrieval.py` looks this field up against
    # authority_hosts.json (18,312 hosts, not one `www.`-prefixed), the
    # hub-floor table (legifrance.gouv.fr, bfs.admin.ch), parasite_domains and
    # the official/country rules. A `www.` payload misses every one of them in
    # silence: fedlex lost its authority boost while bger kept its, and 14 of
    # the 34 golds the build lost to the twin were fedlex.
    domain = (payload.get("domain") or SM.serving_domain(url) or host
              or "").lower()
    source_type = payload.get("source_type")
    lang = payload.get("lang")

    # --- ch-media undated-gov exclusion (task 321; preflight T8/M21) ---
    # BEFORE the consent gate so an excluded row never inflates the consent
    # counters (grace/declined stay truthful). Ruling: aw_dchmedia_cov ×
    # admin.ch-or-subdomain × no published date anywhere => a gov press
    # release mislabeled as news — excluded from the merge, counted loudly.
    if origin == CHMEDIA_PACK and _is_gov_host(host):
        if not (payload.get("published")
                or published_map.get(AC.canonical_url(url))):
            stats["staged_excluded_undated_gov"] += 1
            return None, {"reason": "undated_gov:" + host}

    # --- consent gate ---
    admit, creason = gate.verdict(host, corpus_row, payload)
    if not admit:
        stats["consent_excluded"] += 1
        return None, {"reason": creason}
    grace = creason if creason.startswith("grace:") else None
    if grace:
        stats["consent_grace_retained"] += 1

    # --- text: corpus join wins; else staging payload text (rare); else tail ---
    # An anchor-less multi-chunk page yields SEVERAL texts under one join key
    # (task 319 / GAP A33): resolve to THIS chunk's text via its snippet, or
    # honestly take the no-text tail — never let sibling chunks share a text
    # (that is the 318 chunk-collapse: one digest -> one id -> one survivor).
    text = None
    if corpus_row is not None:
        texts = corpus_row.get("texts")
        if texts is None and corpus_row.get("text"):
            texts = [corpus_row["text"]]
        if texts:
            if len(texts) == 1:
                text = texts[0]
            else:
                stats["ambig_join"] += 1
                text = _match_chunk_text(texts, payload.get("snippet"), stats)
    if text is None and payload.get("text"):
        text = payload["text"]
    has_text = bool(text)

    # --- digest (always; per-chunk — the Tier-B identity component) ---
    if has_text:
        digest = ID.content_digest(text)
    else:
        digest_input, dq = ID.tier_b_fallback_digest_input(
            payload.get("snippet"), payload.get("title"), url)
        stats[f"digest_{dq}"] += 1   # degenerate_* classes collapse chunks — loud
        digest = ID.content_digest(digest_input)

    # --- identity (I1) ---
    # news packs carry the article title in `anchor` for every chunk (a per-page
    # label, not per-chunk) -> force Tier B so chunks are digest-addressed and not
    # collapsed to one-per-article (identity.md §2/§3 data-loss trap).
    is_news = origin in NEWS_PACKS
    tier, anchor_component = ID.classify_tier(
        source_type, native_anchor, has_text, digest,
        title=payload.get("title"), force_tier_b=is_news)
    canon = AC.canonical_url(url)
    point_id = ID.mint_point_id(canon, anchor_component)

    # --- id-collision demotion + persisted salt (task 325) ---
    # 322-ignition aborted on its own tripwire: 893,740 upserts (1.89%) minted
    # ids already taken by a sibling record — degenerate-title collapse
    # (w3/rfc-editor/ietf/bger: no snippet, no KV text), single-PAGE-text KV
    # keys joined by every chunk of the URL (finma 93%), identical snippets.
    # Rule: the FIRST record minting an id keeps it. A later collider is
    # demoted down the honest fallback chain — it never keeps the disputed
    # text (that text demonstrably describes a sibling or the whole page, not
    # this chunk). If the fallback id also collides, an EXPLICIT ordinal salt
    # is appended and PERSISTED in the payload (`id_salt`), so the id re-mints
    # from stored fields in every future build. content_digest stays unsalted.
    id_salt = 0
    if seen_ids is not None and point_id in seen_ids:
        if has_text and tier == "B":
            stats["id_collision_demoted_text"] += 1
            text = None
            has_text = False
            digest_input, dq = ID.tier_b_fallback_digest_input(
                payload.get("snippet"), payload.get("title"), url)
            stats[f"digest_{dq}"] += 1
            digest = ID.content_digest(digest_input)
            tier, anchor_component = ID.classify_tier(
                source_type, native_anchor, has_text, digest,
                title=payload.get("title"), force_tier_b=is_news)
            point_id = ID.mint_point_id(canon, anchor_component)
        while point_id in seen_ids:
            id_salt += 1
            point_id = ID.mint_point_id(
                canon, ID.salted_anchor_component(anchor_component, id_salt))
        if id_salt:
            stats["id_collision_salted"] += 1
    if seen_ids is not None:
        seen_ids.add(point_id)

    # --- vectors: explicit dense channel re-embeds; sparse re-tokenized/kept ---
    raw_vec = src.vector or {}
    dense = raw_vec.get("dense") if isinstance(raw_vec, dict) else raw_vec
    if dense is None:
        stats["no_dense"] += 1
        return None, {"reason": "no dense vector on source"}

    tok_epoch = ID.EPOCHS["tok"]
    is_card = (source_type == "entity_card")
    # ADR-060's SINGLE-SURFACE card variant (task 516's ruling; task 515 sec.9.1
    # raised it). The paired-channel + bundle_digest contract below describes
    # the full-rebuild corpus join, where a card arrives with an explicit
    # dense/sparse pair. A sealed 1024 re-embed of an existing card arrives
    # with the ONE surface its producer sealed and the recipe that made it, and
    # production itself ships 361,468 of its 370,432 cards with no
    # bundle_digest -- so refusing this shape does not enforce ADR-060, it just
    # strips `source_type` off 336,254 rows the ranker then cannot weigh. The
    # variant is recognised from the record's own recipe, never from a set id,
    # and it is COUNTED so the open half of ADR-060 stays visible.
    single_surface_card = bool(
        is_card and SM.is_recipe_owned_card(payload)
        and not (corpus_row or {}).get("dense_text")
        and not (corpus_row or {}).get("sparse_text"))
    if single_surface_card:
        stats["card_single_surface_no_bundle_digest"] += 1
    if is_card and not has_text:
        # Card sparse is recipe-owned content. Preserving its old posting would
        # sever bundle_digest from BM25 and repeat the task-365 loss class.
        raise ValueError(f"entity_card lacks corpus text for sparse regeneration: {url}")
    if (is_card and not single_surface_card
            and (not corpus_row or not corpus_row.get("dense_text")
                 or not corpus_row.get("sparse_text"))):
        raise ValueError(
            f"entity_card lacks canonical paired corpus channels: {url}")
    dense_raw, sparse_raw, reembed_dense = _channel_surfaces(corpus_row, text)
    if has_text:
        rec_for_compose = {"text": sparse_raw, "title": payload.get("title"),
                           "site_name": payload.get("site_name"),
                           "source_type": source_type}
        sidx, sval = _retokenize(sparse_raw, lang, rec_for_compose)
        stats["retokenized"] += 1
    else:
        # Passage no-text tail only: keep existing, stamp tok=1.
        kept = _to_sparse(raw_vec.get("bm25") if isinstance(raw_vec, dict) else None)
        if kept is None:
            sidx, sval = [], []
            stats["tail_no_sparse"] += 1
        else:
            sidx, sval = kept
        # A CARRIED posting keeps the epoch it was BUILT at. Hardcoding 1 was
        # right when the only carried postings predated the tok@2 accent fold;
        # stamping tok@1 onto a posting the source payload declares as tok@2 is
        # a false statement about the row, and `rule_epochs` is the fossil
        # record the NEXT rebuild reads to decide what to redo (task 515;
        # rule-vs-data-drift). Absent declaration still means 1.
        tok_epoch = ((payload.get("rule_epochs") or {}).get("tok")
                     if kept is not None else None) or 1
        stats[f"tail_tok{tok_epoch}"] += 1
        stats["tail_notext"] += 1

    # --- country@2 ---
    country = CT.country_for_record(domain, source_type,
                                    payload.get("register_id"), payload.get("lei"))
    if country == CT.KEEP_EXISTING:
        country = payload.get("country")

    # --- license (assert-or-exception per wave) ---
    lic = LIC.resolve_license(payload.get("license"), domain, host, source_type,
                              payload.get("provenance"))
    lic_exception = lic is None

    # --- snippet <=25 words ---
    if has_text:
        snippet = projected_snippet(domain, text)
    else:
        snippet = AC.extractive_snippet(payload.get("snippet") or "", max_words=25)

    # --- published (news source_types only) ---
    published = None
    if origin in NEWS_PACKS or payload.get("published"):
        published = payload.get("published") or published_map.get(canon)

    # --- chunk-generation instrument (task 536, closing task 535 sec.2.2) ---
    # `rule_epochs["chunk"]` below used to be hardcoded from `ID.EPOCHS`, the
    # BASELINE this build was budgeted for -- never what actually produced
    # THIS row. A re-materialised row (task427/472 epoch-4 chunker, 8 RE-CHUNK
    # + 74 MIXED hosts of the 82 measured) declares `rule_epochs.chunk: 4` and
    # `chunk_n_words` on its OWN sealed stage record -- confirmed on disk,
    # `_work/task427/chunk4-run-0001/ca_canadaca_cc/*.jsonl.gz` -- and this
    # projection dropped both, so the SERVED payload reads `chunk: 1` /
    # `build_id: aw_b1` on every row regardless of which chunker actually ran.
    # That is memory `rule-vs-data-drift` one field further: after this fix,
    # `rule_epochs` and `chunk_n_words` are the fossil record they were meant
    # to be, not a constant. Record wins, same as every field `hydrate()`
    # touches; absent on the incoming record means the baseline, never a
    # guess.
    chunk_epoch = ((payload.get("rule_epochs") or {}).get("chunk")
                   if isinstance(payload.get("rule_epochs"), dict) else None)
    if chunk_epoch is None:
        chunk_epoch = ID.EPOCHS["chunk"]
    chunk_n_words = payload.get("chunk_n_words")
    chunk_rule = payload.get("chunk_rule")
    prov_id = payload.get("prov_id")

    # --- resolve_confidence (register cards) ---
    resolve_conf = payload.get("resolve_confidence")
    if is_card and resolve_conf is None:
        resolve_conf = conf_map.get(domain)

    # --- assemble aw_b1 payload ---
    out_payload = {
        "url": url,
        "title": payload.get("title"),
        "snippet": snippet,
        "lang": lang,
        "domain": domain,
        "tier": tier,
        "source_type": source_type,
        "trust_tier": payload.get("trust_tier"),
        "site_name": payload.get("site_name"),
        "anchor": native_anchor if tier == "A" else None,
        "last_seen": payload.get("last_seen"),
        "country": country,
        "license": lic,
        "content_digest": digest,
        "point_id": point_id,
        "id_salt": id_salt or None,   # 325: persisted collision salt (rare)
        "rule_epochs": {**ID.EPOCHS, "tok": tok_epoch, "chunk": chunk_epoch},
        "build_id": "aw_b1",
    }
    if chunk_n_words is not None:
        out_payload["chunk_n_words"] = chunk_n_words
    if chunk_rule is not None:
        out_payload["chunk_rule"] = chunk_rule
    if prov_id is not None:
        out_payload["prov_id"] = prov_id
    if grace:
        # retained-under-grace (task 319): visible, filterable, expirable. The
        # weekly retry job re-resolves these hosts and evicts on decline or on
        # grace expiry without resolution (architecture.md §5 grace rule).
        out_payload["consent_state"] = "grace"
        out_payload["consent_reason"] = grace[len("grace:"):]
        out_payload["consent_grace_until"] = gate.grace_until
    # passthrough register/card fields
    # NOTE (task 358 R1): `canton` was omitted here for the entire aw_b1
    # rebuild — the serving-side cantonal-scope penalty (task 123/144,
    # `canton_scope` in fusion_weights.json, retrieval.py keys on
    # pl.get("canton")) went silently index-wide no-op (0/336,723 stamped)
    # while the code that reads it kept working, a rule-vs-data-drift class
    # bug. `canton` restored to the whitelist; see (l) payload_contract below
    # for the gate that now catches a repeat of exactly this class of bug.
    for k in ("register_id", "register_scheme", "lei", "names", "provenance",
              "entity_kind", "category", "retail_kind", "qid", "disc_class",
              "agent_policy", "canton", "search_ok", "train_ok",
              "consent_checked_at"):
        if payload.get(k) is not None:
            out_payload[k] = payload[k]

    # --- ADR-076 dump-basis evidence, emitted (task 468; task 433 WI-2) ---
    # `consent_state` / `consent_basis` / `consent_source` are the three of the
    # four fields `dump_basis_evidence()` reads that this projection used to
    # drop (the fourth, `license`, already survives via LIC.resolve_license,
    # which trusts an existing value). Consequence of dropping them, stated
    # plainly: the exemption lasted exactly one generation. The gate admitted
    # the record on its dump evidence, the projection emitted it stripped, and
    # the NEXT rebuild — reading this build's own output as its base
    # collection — found no evidence and evicted it on the cached host
    # decline. A fix that re-arms next rebuild is not a fix. Two cohorts
    # depend on the round-trip: legifrance/`fr_dila_deep` (499,302 points,
    # WI-1) and rii_courts (38,159, task 465).
    #
    # Emitted from `dump_basis_evidence` rather than by adding the three names
    # to the passthrough tuple above, deliberately, for two reasons:
    #   * a raw passthrough would also carry a STALE `consent_state="grace"`
    #     out of a previous build's payload onto a record this build admitted
    #     normally — a resurrected grace stamp with no `consent_grace_until`
    #     beside it, which is a lie about which rule holds the record;
    #   * the emitted evidence is then exactly what the gate recognized: the
    #     same closed-set predicate decides admission and emission, so the
    #     round trip is idempotent and no unrecognized `consent_state` string
    #     can accumulate serving semantics by riding through a rebuild. The
    #     default-exclude design of the whitelist is preserved — nothing new
    #     reaches a serving payload except on evidence this build validated.
    dump = dump_basis_evidence(payload)
    if dump is not None:
        if "consent_state" in out_payload:
            # Unreachable from ConsentGate (branch 0 admits on exactly this
            # predicate, so a dump record never reaches the grace branch).
            # Loud rather than silent if that ever stops being true: two
            # consent states on one payload cannot both govern eviction.
            raise ValueError(
                f"contradictory consent state for {url}: "
                f"{out_payload['consent_state']!r} beside dump evidence")
        out_payload["consent_state"] = DUMP_CONSENT_STATE
        out_payload["consent_basis"] = dump["consent_basis"]
        out_payload["consent_source"] = dump["consent_source"]

    if published is not None:
        out_payload["published"] = published
    if resolve_conf is not None:
        out_payload["resolve_confidence"] = resolve_conf
    # master-only fields ride under _master (ADR-039; stripped by compact dump)
    master = {}
    for k in ("attribution", "license_version"):
        if payload.get(k) is not None:
            master[k] = payload[k]
    corpus_bundle_digest = (corpus_row or {}).get("bundle_digest")
    if corpus_bundle_digest is not None and not is_card:
        raise ValueError("bundle_digest corpus channel is entity_card-only")
    if (is_card and corpus_row
            and (corpus_row.get("dense_text") is not None
                 or corpus_row.get("sparse_text") is not None)
            and corpus_bundle_digest is None):
        raise ValueError("entity_card corpus channels lack bundle_digest")
    bundle_value = (corpus_bundle_digest if corpus_bundle_digest is not None
                    else payload.get("bundle_digest"))
    if bundle_value is not None:
        value = bundle_value
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("invalid bundle_digest")
        out_payload["bundle_digest"] = value
    if master:
        out_payload["_master"] = master
    out_payload = {k: v for k, v in out_payload.items() if v is not None}

    vector = {"dense": list(dense)}
    if sidx:
        vector["bm25"] = qm.SparseVector(indices=sidx, values=sval)

    point = {"id": point_id, "vector": vector, "payload": out_payload,
             # Read off the SOURCE payload, so check (m) can compare what the
             # gate saw against what the payload carries — a contract rule
             # keyed on the emitted field alone would be a tautology.
             "dump_basis": dump is not None,
             "canon_key": canon + "#" + (anchor_component if tier == "A" else
                                         (digest if tier == "B" else "")),
             "digest": digest, "origin": origin, "tier": tier,
             "lic_exception": lic_exception, "tok_epoch": tok_epoch,
             "has_text": has_text, "src_dense": dense,
             "dense_reembed_text": (compose_index_text({
                 "text": dense_raw, "title": payload.get("title"),
                 "site_name": payload.get("site_name"),
                 "source_type": source_type}) if reembed_dense else None)}
    return point, {}

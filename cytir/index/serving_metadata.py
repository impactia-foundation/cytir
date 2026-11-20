#!/usr/bin/env python3
"""The one home for "what serving metadata does this source record carry?".

THE DEFECT THIS EXISTS TO FIX
-----------------------------
`build_aw_b1.project_record` derives identity, country, licence and snippet,
and **passes through** everything the ranker actually weighs: `title`,
`trust_tier`, `source_type`, `lang`, `site_name`, `last_seen`, `provenance`.
That is correct for its historical input -- a point scrolled out of a source
collection, i.e. an already-serving payload. It is silently wrong for the input
the 1024 assembly binder feeds it: a sealed *embedding-stage* record, whose
contract is identity + text.

Three sealed record shapes reach the binder and none of them is a payload:

  * `chunk@2 / epoch-4 materializer` (task471/472/477/482/492/494/495) --
    `point_id, url, canonical_url, text, canonical_text, sparse_text,
    content_digest, identity_tier, anchor_component, rule_epochs, prov_id,
    document_id`;
  * `task417 family pack` -- adds `source_profile`, `source_metadata`,
    `section_heading`, `parent_identity`, `source_links`, and carries `lang`
    only when the upstream L1 record had one;
  * `task467 entity-card re-embed` -- `domain, country, register_id,
    entity_kind, provenance, qid, card_recipe*`.

Task 514 measured the consequence on the built collection: `trust_tier` and
`source_type` empty on 11,273,954 rows (36.5%), `lang` on 4,169,939, `title`
missing wholesale -- and a fusion re-tune that could not improve anything
because the two heaviest stage-2 weights (trust 2.5, title 2.5) had nothing to
read. h@1 fell 276 -> 106 on 629 queries. No count check could see it: the
projection returns a point either way and conservation still balances.

THE RULE
--------
1. A value the RECORD carries always wins. Nothing here overwrites a stamped
   field; hydration only fills absences. So an aw_b1-native stage record
   (task427, task505, rii, news) projects byte-identically to before.
2. A record-native surface for the field comes next -- `source_metadata.title`
   for the pack shape, which is the L1 parent's own title.
3. Then the URL-keyed registry: the serving metadata this exact canonical URL
   carried in the collection this build REPLACES, read from the sealed
   membership snapshot (never from a live collection). This is the ADR-039
   amendment's posture -- "existing values ... are preserved rather than
   inferred" -- applied to the ranking signals, and it is what the binder's
   `twin` payload route already does for wikipedia_413a by point id.
4. Then the domain registry -- the snapshot's modal stamp for the host --
   accepted only above a DECLARED share threshold, never as a bare "most
   common". Measured on the snapshot: the modal domain stamp covers 99.66% of
   points for `source_type` and 99.36% for `trust_tier`, but only 96.66% for
   `lang`, which is why the URL key comes first and the domain is a fallback
   rather than the primary. The domain is tried as given and then with a
   leading `www.` stripped, because `aw_common.canonical_url` folds `www.` and
   production stamped only nine `www.`-prefixed domains out of 361,149 -- the
   staged records have no `domain` field at all, so the projection derives it
   from the URL netloc and gets `www.rfc-editor.org` where production says
   `rfc-editor.org`.
5. Whatever is still absent is COUNTED, per field, and the caller decides.
   Nothing is defaulted: a guessed `trust_tier` is a ranking weight nobody can
   see is wrong.

WHY A SNAPSHOT AND NOT A REGISTRY OF RULES
------------------------------------------
There is no host -> (source_type, trust_tier) registry anywhere in this repo:
every ingester stamps its own hosts at ingest time (`cc_ingest.TIER_TRUST`,
`discovery_ingest.CLASS_TRUST`, per-source constants in two dozen ingesters).
The stamping rule is therefore only recoverable from its output. That makes
this artifact a MIGRATION input, not the durable design: the durable fix is for
the chunk/embed stage record contract to carry the serving metadata forward
from its own L1 parent, so a cold-start build with no prior index needs no
snapshot. Until that lands, a rebuild that replaces an existing collection
carries the stamps across, and says so in its manifest.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3

# The fields the registry stores. `country` is included because the projection
# treats an existing value as the KEEP_EXISTING fallback for `aw_country`, and
# `license` because `aw_b1_license.resolve_license` trusts an existing value.
#
# `published` (task 536, closing task 535 sec.3): a property of the DOCUMENT,
# not the chunk, so the URL key here is the semantically correct join -- not a
# second-best one. Task 514's own class of defect, one field further: task 515
# built this exact URL-keyed snapshot mechanism from the exact artifact that
# still holds every one of the 5,208,018 stamps production carried, and
# `published` was never added to its field list -- so the field was
# structurally invisible to hydration, not merely unset in one run.
# `build_aw_b1.project_record` already reads `payload.get("published")` before
# falling back to its own `published_map` (news-sidecar dates); adding the
# field here means that read sees the hydrated value for every non-news row
# whose registry entry carries one, with the same record-wins-over-registry
# rule every other field in this tuple already gets from `hydrate()`.
REGISTRY_FIELDS = ("title", "lang", "source_type", "trust_tier", "country",
                   "license", "site_name", "last_seen", "provenance",
                   "published",
                   # `domain` is in the registry because production is not
                   # perfectly consistent about it: nine of its 361,149
                   # domains are stamped `www.`-prefixed (www.zh.ch,
                   # www.sz.ch, the two cantonal Gesetzessammlungen ...,
                   # 63,150 points). Folding those to bare would be a second
                   # drift in the other direction, so the exact value this
                   # URL carried wins and `serving_domain()` is only the
                   # fallback for a URL the snapshot does not know.
                   "domain")

# The fields whose absence made task 514's gate fail. `_stage_partition`
# reports these separately so a future regression is visible in the receipt.
SERVING_SIGNAL_FIELDS = ("title", "trust_tier", "source_type", "lang")


def serving_domain(url: str) -> str:
    """The host under the convention production's payloads actually use.

    Not `urlsplit(url).netloc`: that keeps `www.`, and every production
    ingester stamped the canonicalized host. `aw_common.canonical_url` already
    performs exactly this fold, so the host is taken FROM it rather than
    re-implemented beside it -- there is one definition of what `www.` folding
    means and both sides import it.
    """
    from urllib.parse import urlsplit
    import cytir.core.common as AC
    return urlsplit(AC.canonical_url(url or "")).netloc.lower()


def url_key(canonical_url: str) -> str:
    """Registry key. `hash()` is salted per process; blake2b is not.

    Must stay byte-identical to the task-515 registry builder (`build_serving_metadata_registry`)
    -- this is a mirror contract, so there is exactly one definition and the
    builder imports it.
    """
    return hashlib.blake2b(canonical_url.encode("utf-8"),
                           digest_size=8).hexdigest()


class ServingMetadataRegistry:
    """Read-only reader over the URL registry + the domain fallback.

    Opened per process (sqlite connections are not fork-safe); the binder's
    stage workers each construct their own.
    """

    def __init__(self, db_path: str, domain_registry_path: str | None = None,
                 manifest_path: str | None = None,
                 min_domain_share: float = 0.90):
        self.db_path = db_path
        self.min_domain_share = float(min_domain_share)
        self._con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self._con.execute("PRAGMA query_only=ON")
        self._cur = self._con.cursor()
        self._sql = ("SELECT " + ",".join(REGISTRY_FIELDS)
                     + " FROM url_meta WHERE k=?")
        self.domains: dict = {}
        if domain_registry_path:
            with open(domain_registry_path, encoding="utf-8") as fh:
                self.domains = json.load(fh)["domains_map"]
        self.manifest = None
        if manifest_path:
            with open(manifest_path, encoding="utf-8") as fh:
                self.manifest = json.load(fh)

    def by_url(self, canonical_url: str) -> dict | None:
        row = self._cur.execute(self._sql, (url_key(canonical_url),)).fetchone()
        if row is None:
            return None
        return {f: (json.loads(v) if v is not None else None)
                for f, v in zip(REGISTRY_FIELDS, row)}

    def by_domain(self, domain: str) -> dict:
        """The snapshot's modal stamp for the host, above the declared share
        threshold only. `www.` is folded exactly as `canonical_url` folds it."""
        dom = (domain or "").lower()
        d = self.domains.get(dom)
        if d is None and dom.startswith("www."):
            d = self.domains.get(dom[4:])
        if not d:
            return {}
        out = {}
        for f in REGISTRY_FIELDS:
            e = d.get(f)
            if (e and e.get("value") is not None
                    and float(e.get("share", 0.0)) >= self.min_domain_share):
                out[f] = e["value"]
        return out

    def close(self):
        self._con.close()


# `entity_card` is a RECORD KIND, not a property of a URL. In
# `build_aw_b1.project_record` it activates the ADR-060 card contract --
# recipe-owned sparse regeneration from paired dense/sparse corpus channels
# plus a `bundle_digest` -- and a re-chunked passage that happens to share a
# URL with a card satisfies none of it. Inheriting the label through a URL or
# id join therefore does not restore a card; it raises, or worse, it declares a
# passage to be a card. So it is never carried, and it is counted.
#
# The live consequence, named with its number: task467's 339,307 entity-card
# re-embeds keep `title`, `trust_tier`, `lang` and `country` from the snapshot
# but reach the collection WITHOUT `source_type="entity_card"`, because their
# sealed stage records carry one text surface and no bundle_digest while the
# card contract demands two channels and a digest. Production itself does not
# satisfy that contract: 361,468 of its 370,432 cards carry no `bundle_digest`
# at all. Resolving that is a ruling, not a code change -- see the task 515
# report.
NON_INHERITABLE_SOURCE_TYPES = frozenset({"entity_card"})

# ...unless the record says it is a card IN ITS OWN WORDS. Task 516's ruling.
#
# The refusal above is right about the JOIN and wrong about this population.
# What must never happen is a re-chunked passage inheriting `entity_card`
# because it shares a URL with a card. But task467's 336,254 sealed stage
# records carry `card_recipe`, `card_recipe_source`, `card_recipe_tier` and
# `card_recipe_join_basis` -- the recipe that MADE the card -- plus the single
# `dense_text` surface its own stage.json declares
# ("dense_text_contract: record['dense_text'] (ADR-048 entity_card exemption:
# compose_index_text() returns text unchanged for entity_card)"). That is not
# an inherited label; it is the producer's own statement of what the record is,
# and it is the discriminator ADR-060 was missing.
#
# The remaining gap is the bundle_digest, and it is a gap PRODUCTION ALSO HAS:
# 361,468 of its 370,432 serving cards carry no `bundle_digest` at all (8,964
# do, all tok@2). Stamping the label without one is therefore production
# parity, not a new liberty -- and shipping 336,254 cards with `source_type`
# blank is a ranking regression against a collection that has it on all but 10
# points. The bundle-digest half of ADR-060 stays open, recorded for Tim, and
# is not invented here.
CARD_RECIPE_FIELD = "card_recipe"


def is_recipe_owned_card(record) -> bool:
    """True when the record carries the ADR-060 card recipe that produced it.

    Deliberately a property of the DATA, never of a set-id list: a future card
    population is recognised by carrying its recipe, and a passage that merely
    shares a URL with a card carries none.
    """
    return bool(isinstance(record, dict) and record.get(CARD_RECIPE_FIELD))


def _pack_title(record: dict):
    """The task417 pack shape keeps the L1 parent's title under
    `source_metadata`, which the projection never looked at."""
    sm = record.get("source_metadata")
    if isinstance(sm, dict):
        for k in ("title", "headline"):
            v = sm.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def hydrate(payload: dict, canonical_url: str, registry, stats) -> dict:
    """Fill ONLY the serving fields this record does not carry. Mutates and
    returns `payload`; counts every decision into `stats`.

    `registry` may be None, in which case this is a no-op -- the full-rebuild
    path, whose input really is a serving payload, is unchanged.
    """
    if registry is None:
        return payload

    row_refused = False
    missing = [f for f in REGISTRY_FIELDS
               if payload.get(f) is None or payload.get(f) == ""]
    if not missing:
        stats["meta_record_complete"] += 1
        return payload

    # (2) record-native surfaces before any registry read
    if "title" in missing:
        t = _pack_title(payload)
        if t:
            payload["title"] = t
            missing.remove("title")
            stats["meta_title_from_source_metadata"] += 1

    if not missing:
        return payload

    # (3) the URL-keyed snapshot
    hit = registry.by_url(canonical_url)
    if hit is not None:
        stats["meta_url_hit"] += 1
        for f in list(missing):
            v = hit.get(f)
            if v is None or v == "":
                continue
            if f == "source_type" and v in NON_INHERITABLE_SOURCE_TYPES:
                if v == "entity_card" and is_recipe_owned_card(payload):
                    stats["meta_entity_card_stamped_from_recipe"] += 1
                else:
                    stats[f"meta_refused_source_type_{v}"] += 1
                    row_refused = True
                    continue
            payload[f] = v
            missing.remove(f)
            stats[f"meta_url_filled_{f}"] += 1
    else:
        stats["meta_url_miss"] += 1

    # (4) the domain fallback, single-valued facets only
    if missing:
        dom = (payload.get("domain") or "").lower()
        if not dom:
            from urllib.parse import urlsplit
            dom = urlsplit(payload.get("url") or "").netloc.lower()
        dhit = registry.by_domain(dom)
        if dhit:
            stats["meta_domain_hit"] += 1
        for f in list(missing):
            v = dhit.get(f)
            if v is None or v == "":
                continue
            if f == "source_type" and v in NON_INHERITABLE_SOURCE_TYPES:
                if v == "entity_card" and is_recipe_owned_card(payload):
                    stats["meta_entity_card_stamped_from_recipe"] += 1
                else:
                    stats[f"meta_refused_source_type_{v}"] += 1
                    row_refused = True
                    continue
            payload[f] = v
            missing.remove(f)
            stats[f"meta_domain_filled_{f}"] += 1

    # (5) whatever is left is counted, never defaulted
    for f in missing:
        stats[f"meta_unresolved_{f}"] += 1
    # A field left empty because the registry does not know it is a COVERAGE
    # problem, fixable by a better registry. A field left empty because the
    # snapshot's answer was `entity_card` and the label is not inheritable is a
    # RULING problem with a different remedy. Counting them together would let
    # 339,307 cards hide inside a coverage number, or a coverage regression
    # hide inside a known exemption.
    refused = ["source_type"] if (row_refused and "source_type" in missing) else []
    unresolved = [f for f in missing
                  if f in SERVING_SIGNAL_FIELDS and f not in refused]
    if unresolved:
        stats["meta_row_missing_serving_signal"] += 1
    elif refused:
        stats["meta_row_source_type_refused_entity_card"] += 1
    return payload

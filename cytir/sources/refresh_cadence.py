"""cytir.sources.refresh_cadence (`cyt source refresh-cadence`) — generate data/refresh-cadence.v1.json (task 632).

Reads every declared source pack (`sources/**/*.yaml`, `sources/_schema/pack.schema.json`
shape) and PORTFOLIO.md §3c's freshness-class table, and emits ONE deterministic
per-source periodic-re-fetch cadence, plus a daily-new-refs-discovery eligibility
flag. This file is the "per-source cadence table ... generated from PORTFOLIO (a
script, with a test that the table matches the policy)" required by task 632's
brief.

This script NEVER fetches anything and never touches a network. It only reads
local YAML/Markdown already in the repo. Re-running it against an unchanged
`sources/` tree must produce a byte-identical `data/refresh-cadence.v1.json`
(determinism is asserted by tests/operator/test_aw_source_refresh_cadence.py).

--- The cadence rule, and where each number is cited from -----------------

Tim's mission framing (task 632 brief, 2026-09-22) names four concrete cadences
by layer: "legal spines + registers monthly, statistics quarterly or on release
calendar, docs weekly, news never re-fetched (aged-out instead)". That is the
PRIMARY rule, keyed on the pack's `layer` field (POLICY.md@v1 §4, enumerated in
sources/_schema/pack.schema.json). Layers the mission did not name (scholarship,
reference, health, community) fall back to PORTFOLIO.md §3c's own freshness-class
table, which independently states: "static: re-verify ~quarterly" (the class
every non-law/non-docs/non-stats layer currently carries) and singles out
community/news as NOT a re-fetch target (news ages out; community is a hard-capped
discussion tier, POLICY §4.8, never a freshness target on its own).

A pack's own `freshness` field (its upstream SOURCE's *publish* cadence, e.g. an
"amendment-driven" law feed) can only SHARPEN the layer default to something
FASTER — never slower — because PORTFOLIO §3c's "amendment-driven" class is
explicitly <7 days for exactly this reason (source-native change feeds). It never
overrides the "never" rulings (news, or a source with declined consent).

ADR-103/ADR-024 am.5: a source whose pack `status` is `declined` (consent denied
or unreadable) is `non_refreshable` — this generator hard-excludes it from BOTH
periodic re-fetch and new-refs discovery, regardless of layer. Consent governs
fetch, not retention (ADR-103); a declined origin is never re-crawled, refreshed,
or newly discovered from again.

Only packs with `status: ingested` (the schema's "actually serving" state) are
`eligible_now`. Everything else (staged/planned/deferred) gets its cadence
assignment recorded for when it lands, but is excluded from `eligible_now` so the
scheduler and the dry-run report never plan to touch a family with nothing in the
index yet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

SCHEMA = "aw.refresh-cadence.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]

# Layer -> periodic re-fetch cadence in days. None = never re-fetched via this
# lane (news ages out instead; community is a discussion-tier cap, not a
# freshness target). Cites: task 632 brief (law/stats/docs/news) +
# sources/PORTFOLIO.md §3c ("static ... quarterly" fallback for the rest).
LAYER_CADENCE_DAYS = {
    "law-government": 30,     # "legal spines + registers monthly"
    "commercial": 30,         # "... registers monthly" (registry/register-derived dumps)
    "statistics-data": 90,    # "statistics quarterly or on release calendar"
    "tech-docs": 7,           # "docs weekly"
    "health": 30,             # not named by the mission; PORTFOLIO §4.6 floor is the
                               # strictest of any layer (Cochrane/WHO-class) — held to
                               # the same monthly cadence as law/registers rather than
                               # the generic quarterly static default. Flagged for Tim
                               # review in specs/refresh-cadence.md §7 (not yet ruled).
    "scholarship": 90,        # PORTFOLIO §3c "static: re-verify ~quarterly"
    "reference": 90,          # PORTFOLIO §3c "static: re-verify ~quarterly"
    "community": None,        # POLICY §4.8 hard-capped discussion tier, not a
                               # freshness target; individual threads are not re-fetched.
    "news": None,              # "news never re-fetched (aged-out instead)" — owned by
                               # the separate news lane (the news refresh cron).
}

# freshness field (the pack's own upstream publish cadence) -> a cadence CEILING
# in days. Only ever tightens (min()) the layer default; never loosens it.
# amendment-driven: PORTFOLIO §3c "<7 days of change" via source-native change feeds.
FRESHNESS_CEILING_DAYS = {
    "amendment-driven": 7,
}

# Adapters that are AW-GENERATED derivations (cards synthesized from a register or
# other already-refreshed source) rather than an independent external fetch target.
# They inherit the layer's cadence but never a tighter one from `freshness`, and
# they are never a new-refs discovery source (nothing external to discover "new"
# from — the register leg already is one).
GENERATED_ADAPTERS = {"cards"}

# Adapters usable as a genuine daily new-refs discovery ROUTE (source_owned feed/
# dump/API per data/acquisition-policy.v1.json route precedence). `cc` (Common
# Crawl) is rank 4 in that policy and is a legitimate discovery route only when
# nothing higher-priority exists for the family; it is included here because the
# eligibility question this table answers is "does an admitted-source-owned or
# policy-sanctioned discovery route exist", not "which route wins" (that
# precedence is applied at run time by the new-refs refresh lane).
DISCOVERY_ADAPTERS = {"dump", "sitemap", "api", "registry", "cc"}


def iter_source_packs(sources_dir: Path):
    for path in sorted(sources_dir.glob("**/*.yaml")):
        if path.name.startswith("_"):
            continue
        # Templates under sources/_examples/ are not declared sources (task 999).
        if any(part.startswith("_") for part in path.relative_to(sources_dir).parts[:-1]):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        if not isinstance(doc, dict) or "id" not in doc:
            continue
        yield path, doc


def cadence_for(layer: str, freshness: str, adapter: str, status: str) -> tuple[int | None, str]:
    """Return (periodic_refetch_days, rationale) for one pack."""
    if status == "declined":
        return None, "consent declined (ADR-103/ADR-024 am.5) — non_refreshable, never re-fetched"

    base = LAYER_CADENCE_DAYS.get(layer)
    if base is None:
        if layer == "news":
            return None, "news lane owns freshness for this layer (aw-news-freshness-policy) — aged out, never re-fetched here"
        return None, f"layer '{layer}' is not a periodic re-fetch target (POLICY §4.8 discussion-tier or equivalent)"

    days = base
    rationale = f"layer '{layer}' default ({base}d, task 632 brief / PORTFOLIO §3c)"

    if adapter not in GENERATED_ADAPTERS:
        ceiling = FRESHNESS_CEILING_DAYS.get(freshness)
        if ceiling is not None and ceiling < days:
            days = ceiling
            rationale = (
                f"layer '{layer}' default ({base}d) tightened to {ceiling}d by the pack's own "
                f"'{freshness}' change-feed class (PORTFOLIO §3c)"
            )
    else:
        rationale += " (adapter=cards: AW-generated derivation, layer cadence unmodified)"

    return days, rationale


def new_refs_eligible(adapter: str, status: str) -> tuple[bool, str]:
    if status == "declined":
        return False, "consent declined — no discovery, ADR-103"
    if adapter in GENERATED_ADAPTERS:
        return False, "adapter=cards: AW-generated, nothing external to discover from"
    if adapter in DISCOVERY_ADAPTERS:
        return True, f"adapter='{adapter}' is a source-owned/policy-sanctioned discovery route (data/acquisition-policy.v1.json)"
    return False, f"adapter='{adapter}' has no declared discovery route"


def build_table(sources_dir: Path) -> dict:
    rows = []
    for path, doc in iter_source_packs(sources_dir):
        layer = doc.get("layer", "")
        freshness = doc.get("freshness", "")
        adapter = doc.get("adapter", "")
        status = doc.get("status", "")
        pid = doc["id"]

        refetch_days, refetch_reason = cadence_for(layer, freshness, adapter, status)
        discover, discover_reason = new_refs_eligible(adapter, status)

        rows.append(
            {
                "id": pid,
                "pack_path": str(path.relative_to(REPO_ROOT)),
                "country": doc.get("country"),
                "layer": layer,
                "policy_section": doc.get("policy_section"),
                "adapter": adapter,
                "source_freshness_class": freshness,
                "status": status,
                "eligible_now": status == "ingested",
                "periodic_refetch_days": refetch_days,
                "periodic_refetch_rationale": refetch_reason,
                "new_refs_discovery": discover,
                "new_refs_discovery_rationale": discover_reason,
            }
        )

    rows.sort(key=lambda r: r["id"])

    rules = {
        "layer_cadence_days": LAYER_CADENCE_DAYS,
        "freshness_ceiling_days": FRESHNESS_CEILING_DAYS,
        "generated_adapters": sorted(GENERATED_ADAPTERS),
        "discovery_adapters": sorted(DISCOVERY_ADAPTERS),
        "cites": [
            "task 632 brief (Tim 2026-09-22): legal+registers monthly, stats quarterly, docs weekly, news never",
            "sources/PORTFOLIO.md §3c (freshness classes table)",
            "sources/POLICY.md §4 (per-layer admission/authority)",
            "decisions.md ADR-103 / ADR-024 am.5 (consent governs fetch, not retention)",
            "data/acquisition-policy.v1.json (route precedence for discovery)",
        ],
    }

    return {
        "schema": SCHEMA,
        "generated_by": "cyt source refresh-cadence",
        "source_packs_glob": "sources/**/*.yaml",
        "rules": rules,
        "sources": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cyt source refresh-cadence")
    ap.add_argument("--sources-dir", default=str(REPO_ROOT / "sources"))
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "refresh-cadence.v1.json"))
    ap.add_argument("--check", action="store_true", help="exit 1 if regenerating would change the committed file")
    args = ap.parse_args(argv)

    table = build_table(Path(args.sources_dir))
    out_text = json.dumps(table, indent=2, ensure_ascii=False, sort_keys=False) + "\n"

    out_path = Path(args.out)
    if args.check:
        if not out_path.exists():
            print(f"MISSING: {out_path}", file=sys.stderr)
            return 1
        current = out_path.read_text(encoding="utf-8")
        if current != out_text:
            print(f"DRIFT: {out_path} does not match a fresh generation — re-run without --check", file=sys.stderr)
            return 1
        print(f"ok: {out_path} matches a fresh generation ({len(table['sources'])} sources)")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_text, encoding="utf-8")
    digest = hashlib.sha256(out_text.encode("utf-8")).hexdigest()
    print(f"wrote {out_path} ({len(table['sources'])} sources) sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

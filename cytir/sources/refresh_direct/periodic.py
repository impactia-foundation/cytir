"""periodic — PERIODIC re-fetch lane for EXISTING targets
(task 632, mission item 2), cadence per source class (data/refresh-cadence.v1.json).

Two real, generic steps run for every DUE, `eligible_now` source when armed:

1. **Consent re-resolved live** (ADR-024/ADR-103): for each `routes[].host`
   declared `ingested` in the source's own pack (falling back to none for a
   pack with `consent.basis: official-no-copyright`, which by definition needs
   no per-host robots probe), re-resolve search_ok/train_ok via
   `cytir.consent.resolver.resolve_cached_host` — the SAME resolver every
   other consent-gated lane in this repo uses, so a policy change (a new
   signal, a new precedence rule) is picked up here automatically instead of
   forked. A host that now reads declined/unreadable where it previously
   served STOPS this source's re-fetch (ADR-103: consent governs fetch, not
   retention) and is queued + alerted, never silently skipped or silently
   re-fetched anyway.

2. **Digest compare + re-embed** is NOT yet wired to a live fetch: per-family
   fetch adapters (what actually re-GETs a page and re-extracts it) are
   follow-up work exactly like the new-refs lane (see new_refs.py
   ADAPTER_REGISTRY). A due source that clears its consent re-check but has no
   registered fetch adapter is reported `status: not_implemented` and queued
   for a human/agent follow-up rather than silently marked complete.

This script never writes a canonical (twin or prod) collection. "Only changed
pages re-embedded and applied as supersede rows" (mission item 2) is future
work behind a per-family adapter AND the twin-first content-pack transaction
(the twin-first apply tool) — this lane queues a
candidate pack; it is never the thing that applies one.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from cytir.core.paths import REPO_ROOT
from cytir.sources.refresh_direct import lane_lib as lib

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

# adapter -> callable(source_row, consent_ok_hosts) -> dict (fetch+digest result).
# Empty today by design (see module docstring).
FETCH_ADAPTER_REGISTRY: dict[str, object] = {}


def load_cadence_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["sources"]


def ingested_hosts_for_pack(pack_path: Path) -> list[str]:
    """Real hosts to re-resolve consent for, straight from the pack's own
    `routes[]` where `status: ingested` (the schema's "actually serving"
    marker per sub-host). A pack with no `routes` array, or a
    `consent.basis: official-no-copyright` posture (no robots signal needed,
    e.g. fedlex's own structured dump — POLICY §2 U1 / the pack's own consent
    notes), returns no hosts: there is nothing to re-probe.

    Prefers `canonical_domain` over `host` (task 661 F5, 2026-09-28): most
    packs fan `routes[]` out one-per-SUB-HOST (cantonal-law's `be.clex.ch`
    etc.), where `host` already IS the bare hostname to re-probe. The
    GitHub-releases/HF-orgs/vendor-blogs packs fan `routes[]` out one-per-
    TRACKED-ITEM instead (`host: "github.com/qdrant/qdrant"`, a label their
    own discovery adapters read via `pack_path` — see
    `github_releases_adapter.py`'s own doc comment) — passing that whole
    string as a "host" here built a syntactically-valid but semantically
    wrong URL (`https://github.com/qdrant/qdrant/robots.txt`, a real
    connection to github.com but the WRONG path, found live via this
    module's own test suite hitting the real network with a malformed
    host). `canonical_domain` is these packs' real, bare hostname for
    exactly this re-probe; falls back to `host` for packs that don't set
    it (the majority — `canonical_domain` is optional per the schema).
    Deduped: several tracked-item routes commonly share one host."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to read source packs")
    doc = yaml.safe_load(pack_path.read_text(encoding="utf-8"))
    if doc.get("consent", {}).get("basis") == "official-no-copyright":
        return []
    seen: list[str] = []
    for r in doc.get("routes", []):
        if r.get("status") != "ingested":
            continue
        host = r.get("canonical_domain") or r["host"]
        if host not in seen:
            seen.append(host)
    return seen


def check_consent(hosts: list[str], resolver_cache_dir: str) -> dict:
    """Re-resolve consent for each host via the shared resolver's persistent
    cache (`consent_resolver.resolve_cached_host`). Imports lazily so this
    module still loads (for the cadence/due logic, and for tests of THAT
    logic) in an environment without `httpx`.
    """
    if not hosts:
        return {"hosts": [], "all_ok": True}
    from cytir.consent import resolver as consent_resolver  # local import: only needed on the armed path

    client = consent_resolver._client()
    results = []
    all_ok = True
    try:
        for host in hosts:
            receipt = consent_resolver.resolve_cached_host(client, host, cache_dir=resolver_cache_dir)
            ok = bool(receipt.get("search_ok")) and receipt.get("status") != "declined"
            all_ok = all_ok and ok
            results.append({"host": host, "search_ok": receipt.get("search_ok"), "status": receipt.get("status")})
    finally:
        client.close()
    return {"hosts": results, "all_ok": all_ok}


def run(
    cadence_path: Path,
    state_path: Path,
    now: datetime,
    armed: bool,
    resolver_cache_dir: str,
) -> dict:
    rows = load_cadence_rows(cadence_path)
    state = lib.load_state(state_path)
    due = lib.due_report(rows, state, "periodic_refetch", now=now)
    rows_by_id = {r["id"]: r for r in rows}

    entries = []
    for d in due:
        row = rows_by_id[d.source_id]
        entry = {
            "id": d.source_id,
            "layer": row["layer"],
            "country": row["country"],
            "adapter": row["adapter"],
            "cadence_days": d.cadence_days,
            "due": d.due,
            "alarm": d.alarm,
            "last_run": d.to_json_dict()["last_run"],
            "days_since_last_run": d.to_json_dict()["days_since_last_run"],
        }
        if not d.due:
            entry["status"] = "not_due"
            entries.append(entry)
            continue
        if not armed:
            entry["status"] = "would_refetch (dry-run)"
            entries.append(entry)
            continue

        hosts = ingested_hosts_for_pack(REPO_ROOT / row["pack_path"])
        consent = check_consent(hosts, resolver_cache_dir)
        entry["consent_recheck"] = consent
        if not consent["all_ok"]:
            entry["status"] = "consent_declined_stop"
            entry["reason"] = "ADR-103: a declined/unreadable host stops re-fetch — queued for review, never re-fetched"
            entries.append(entry)
            continue

        impl = FETCH_ADAPTER_REGISTRY.get(row["adapter"])
        if impl is None:
            entry["status"] = "not_implemented"
            entry["reason"] = (
                f"consent re-check passed; no fetch/digest adapter registered for "
                f"adapter='{row['adapter']}' (id={d.source_id}) yet"
            )
        else:
            result = impl(row, consent)
            entry["status"] = "checked"
            entry.update(result)
            state = lib.mark_run_complete(state, d.source_id, "periodic_refetch", lib.utcnow_iso())
        entries.append(entry)

    if armed:
        lib.save_state(state_path, state)

    return {
        "schema": "aw.refresh-lane-periodic-report.v1",
        "lane": "periodic_refetch",
        "armed": armed,
        "generated_at": lib.utcnow_iso(),
        "as_of": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "due_count": sum(1 for e in entries if e["due"]),
        "alarm_count": sum(1 for e in entries if e["alarm"]),
        "consent_declined_count": sum(1 for e in entries if e.get("status") == "consent_declined_stop"),
        "not_implemented_count": sum(1 for e in entries if e.get("status") == "not_implemented"),
        "sources": entries,
    }


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("--cadence-table", default=str(REPO_ROOT / "data" / "refresh-cadence.v1.json"))
    ap.add_argument("--state", required=True)
    ap.add_argument("--out", default="-")
    ap.add_argument("--now", default=None, help="override 'now' as ISO UTC (tests only)")
    ap.add_argument("--armed", action="store_true", default=False)
    ap.add_argument("--resolver-cache-dir", required=True)
    args = ap.parse_args(argv)

    now = (
        datetime.strptime(args.now, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        if args.now
        else datetime.now(timezone.utc)
    )
    report = run(Path(args.cadence_table), Path(args.state), now, args.armed, args.resolver_cache_dir)
    text = json.dumps(report, indent=2)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:], "python -m cytir.sources.refresh_direct.periodic"))

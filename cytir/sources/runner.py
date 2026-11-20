#!/usr/bin/env python3
"""Task 434 — run a source pack end to end. The tool task 429 Blocker A asked for.

    python3 run_source.py sources/ch/fedlex.yaml
    python3 run_source.py ch-media --plan

WHAT THIS IS NOT: a new ingest engine. The survey (task 434) established that
the engines already existed and were simply never bound to the packs --
`pack_fetch.run_pack` (task 195/ADR-026) was called by 5 of 86 ingesters, and
`cytir/sources/cov/engine.py` (task 351) already ran five route modes off a routes YAML.
**The missing piece was a binding, not an engine.** This module dispatches; it
does not fetch, extract, chunk, tokenize, canonicalize or mint ids.

NAMING: this is the *source runner*. It is deliberately NOT called a dispatcher
-- that name already denotes the chunk-policy dispatcher
(`aw_common.chunk_markdown_units`, ADR-079/090), and reusing it would fork a
meaning.

ROUTE -> ENGINE, per data/acquisition-policy.v1.json route_precedence:

  source_owned_sitemap_fetch  -> pack_fetch.run_pack(mode="sitemap")
  direct_origin_crawl         -> pack_fetch.run_pack(mode="bfs")   [rank 6]
  source_owned_bulk_dump      -> the pack's declared `extractor:`
  documented_source_api       -> the pack's declared `extractor:`
  source_owned_incremental_feed -> the pack's declared `extractor:`
  common_crawl                -> the pack's declared `extractor:` (cc_columnar path)

The extractor-backed routes are NOT a cop-out: architecture.md §4.1 "adapter
honesty" (F5) ruled that real sources are bespoke (DE Genesis, FR DILA, StatCan
WDS -- each its own API dialect) and that a pack may ship a thin per-source
extractor plugging into a family's interface. The survey confirmed it: `api` is
28 of 86 scripts and the most diverse bucket. What the runner standardises is
everything AROUND the dialect -- consent, politeness, L0/L1/L2 staging,
receipts -- not the dialect itself.

SCOPE (orchestrator ruling 2026-08-20): the 10 bulk-dump ingesters are NOT
rewritten. They produced 90.9% of the corpus, they are bespoke for good reasons,
and automating them would buy nothing. The runner targets the long tail, where
the per-source agent-hours actually go. Success is measured in
**scripts-not-written**, not rows-ingested.

CONSENT is resolved live, per host, before any fetch, via the shared modules.
`consent_resolver.py` is imported as-is and never modified (three consent changes
are queued behind PR #67 and land in order).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

# cytir/sources/runner.py -> repo root (moved from the retired `source_runner` script, task
# 677: one level deeper). The sitemap/BFS engine is `cytir.acquire.crawl` (moved
# from the retired `pack_fetch` script, task 677 S5 acquire slice); it is imported
# lazily because `--plan` and the extractor routes do not need httpx.
REPO_ROOT = Path(__file__).resolve().parents[2]

from cytir.sources import pack_loader as PL  # noqa: E402


def _consent_modules():
    """Import the consent stack lazily.

    Deliberate: `consent_resolver` imports `user_agent`, which REFUSES at import
    time unless a real crawler identity is configured (task 432) -- correctly, since
    every host we fetch logs that string. But `--plan` touches no network and needs
    no identity, and an operator inspecting a pack on a laptop should not have to
    configure a crawler to do it. So the refusal fires when we are about to fetch,
    not when we are about to think.

    These modules are imported as-is and never modified: three consent changes are
    queued behind PR #67 and land in order.
    """
    from cytir.consent import block as consent_block
    from cytir.consent import policy as consent_policy
    from cytir.consent import resolver as consent_resolver
    return consent_block, consent_policy, consent_resolver


# --- consent -----------------------------------------------------------------

def resolve_consent(host: str, *, direct_crawl: bool, client=None) -> dict:
    """Live consent for one host. Fail-closed on a decided decline.

    Order matters and is not ours to invent:
      1. consent_block.assert_crawlable  -- recorded decline decisions
      2. consent_resolver.resolve_cached_host -- live AIPREF/robots/TDMRep
      3. consent_policy.gate_bucket(search_indexing=True) -- the index rubric

    `status == "unknown"` is transient and RETRYABLE; it is never a refusal and
    must never be recorded as one (verify-liveness-before-trusting-misses).
    """
    consent_block, consent_policy, consent_resolver = _consent_modules()
    consent_block.assert_crawlable(host)
    close = False
    if client is None:
        client = consent_resolver._client()
        close = True
    try:
        rec = consent_resolver.resolve_cached_host(
            client, host, direct_crawl=direct_crawl)
    finally:
        if close:
            try:
                client.close()
            except Exception:
                pass
    bucket = consent_policy.gate_bucket(rec, search_indexing=True)
    return {
        "host": host,
        "mode": rec.get("mode"),
        "bucket": bucket,
        "status": rec.get("status"),
        "search_ok": rec.get("search_ok"),
        "train_ok": rec.get("train_ok"),
        "checked_at": rec.get("checked_at"),
        "reasons": rec.get("reasons", []),
    }


# --- engines -----------------------------------------------------------------

PACK_FETCH_MODES = {
    "source_owned_sitemap_fetch": "sitemap",
    "direct_origin_crawl": "bfs",
}

# Routes that fetch by following or generating origin URLs rather than by
# consuming a publisher-declared artifact. A pack with no_fallback may not use
# these. Kept as a named set so the rule has one home.
CRAWL_ROUTES = {"direct_origin_crawl"}

EXTRACTOR_ROUTES = {
    "source_owned_bulk_dump",
    "documented_source_api",
    "source_owned_incremental_feed",
    "common_crawl",
}


class RunError(RuntimeError):
    pass


def run_via_pack_fetch(plans, *, pack_id: str, args) -> dict:
    """Sitemap / bounded-crawl routes -> the existing shared fetch layer.

    Everything below the HOSTS dict is pack_fetch's: robots verification,
    per-host politeness, L0 retention, outlink capture, resume, heartbeat.
    We supply configuration compiled from YAML; we add no fetch logic.
    """
    from cytir.acquire import crawl as pack_fetch  # lazy: httpx is not needed for --plan

    hosts = {}
    for p in plans:
        mode = PACK_FETCH_MODES[p.route]
        if not p.host:
            raise RunError(f"{p.pack_id}: route {p.route} needs a `host:`")
        hosts[p.route_id] = p.to_host_cfg(mode)

    ap = argparse.ArgumentParser()
    pack_fetch.add_standard_args(ap, out_dir=args.out_dir)
    fetch_args = ap.parse_args([
        "--out-dir", args.out_dir,
        "--limit", str(args.limit),
        "--budget", str(args.budget),
    ] + (["--no-l0"] if args.no_l0 else []))

    return pack_fetch.run_pack(
        pack=pack_id,
        hosts=hosts,
        args=fetch_args,
        canon=pack_fetch.make_canon(),
    )


def _package_module(extractor: str) -> str | None:
    """`cytir/sources/families/fedlex.py` -> `cytir.sources.families.fedlex`; None for any other path."""
    parts = PurePosixPath(extractor).parts
    if len(parts) < 2 or parts[0] != "cytir" or not parts[-1].endswith(".py"):
        return None
    return ".".join(parts[:-1] + (parts[-1][:-3],))


def run_via_extractor(plan, *, args) -> dict:
    """Bespoke-dialect routes -> the pack's declared `extractor:`.

    The extractor is invoked as a subprocess with a documented, stable
    contract, so an extractor stays an ordinary runnable script rather than
    becoming a plugin API nobody can test standalone.
    """
    if not plan.extractor:
        raise RunError(
            f"{plan.pack_id}/{plan.route_id}: route {plan.route} needs an "
            f"`extractor:` and the pack declares none. Per architecture.md "
            f"§4.1 (F5) a bespoke dialect ships a thin per-source extractor.")
    ext = REPO_ROOT / plan.extractor
    if not ext.exists():
        raise RunError(f"{plan.pack_id}: extractor {plan.extractor} does not exist")

    # An extractor inside the package (`aw/...py`) runs as a module (`python -m cytir.x.y`) with the
    # checkout on PYTHONPATH: the script directory is then NOT put first on `sys.path`, and the
    # package imports resolve from any working directory. Any other path is a script and runs as
    # it always did (task 730: the pack line names the package module, nothing else changes).
    module = _package_module(plan.extractor)
    head = [sys.executable, "-m", module] if module else [sys.executable, str(ext)]
    cmd = head + list(args.extractor_args or [])
    print(f"[run] extractor: {' '.join(cmd)}", flush=True)
    if args.dry_run:
        return {"skipped": "dry-run", "cmd": cmd}
    # aw.source-extractor-env.v1: route identity and bounded working settings
    # are supplied uniformly without inventing a bespoke CLI for every adapter.
    # Existing extractors ignore these variables; family adapters can opt in.
    env = os.environ.copy()
    if module:
        env["PYTHONPATH"] = os.pathsep.join(
            [str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.update({
        "AW_SOURCE_PACK_ID": plan.pack_id,
        "AW_SOURCE_ROUTE_ID": plan.route_id,
        "AW_SOURCE_OUT_DIR": args.out_dir,
        "AW_SOURCE_LIMIT": str(args.limit),
        "AW_SOURCE_BUDGET": str(args.budget),
        "AW_SOURCE_NO_L0": "1" if args.no_l0 else "0",
        "AW_SOURCE_CONFIG_REF": json.dumps(
            plan.config_ref or {}, sort_keys=True, separators=(",", ":")),
    })
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    if proc.returncode != 0:
        raise RunError(
            f"{plan.pack_id}: extractor exited {proc.returncode} — "
            f"not masking this as a partial success")
    return {"extractor": plan.extractor, "returncode": 0,
            "elapsed_s": round(time.time() - t0, 2)}


# --- orchestration -----------------------------------------------------------

def run_pack_plans(doc, plans, args) -> dict:
    pack_id = doc["id"]
    receipt = {
        "schema": "aw.source-run.v1",
        "pack": pack_id,
        "pack_path": doc.get("_path"),
        "adapter": doc.get("adapter"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "plans": [], "consent": [], "results": {},
    }

    # 1. consent preflight for every host BEFORE any fetch.
    # Consent mode is acquisition provenance, not a synonym for BFS. Every
    # origin we contact ourselves is direct-mode; only Common Crawl provenance
    # uses the CC-mode receipt. Keep (host, mode) distinct so a host appearing
    # in both lanes cannot borrow the wrong cached verdict.
    consent_checks = sorted({
        (h, p.route != "common_crawl")
        for p in plans for h in p.effective_hosts()
    })
    blocked = set()
    if consent_checks and not args.skip_consent:
        print(f"[consent] resolving {len(consent_checks)} host/mode check(s) live…",
              flush=True)
        for h, direct in consent_checks:
            try:
                verdict = resolve_consent(h, direct_crawl=direct)
            except Exception as e:
                # A recorded decline (consent_block) or a transport failure.
                # Do NOT silently continue -- an unchecked host must not be fetched.
                verdict = {"host": h,
                           "mode": "direct" if direct else "cc",
                           "bucket": "error", "error": str(e)}
            receipt["consent"].append(verdict)
            mark = verdict["bucket"]
            print(f"  {h:38s} {'direct' if direct else 'cc':6s} {mark}",
                  flush=True)
            if mark not in ("ok", "ok_search_only"):
                blocked.add((h, direct))
        if blocked:
            print(f"[consent] {len(blocked)} host(s) not clear to fetch: "
                  f"{sorted(blocked)}", flush=True)

    def consent_keyset(plan):
        direct = plan.route != "common_crawl"
        return {(host, direct) for host in plan.effective_hosts()}

    runnable = [p for p in plans if not (consent_keyset(p) & blocked)]
    receipt["plans"] = [
        {"route_id": p.route_id, "route": p.route,
         "hosts": p.effective_hosts(),
         "consent_mode": "cc" if p.route == "common_crawl" else "direct",
         "skipped": bool(consent_keyset(p) & blocked)} for p in plans]

    if not runnable:
        print("[run] nothing runnable after the consent gate.")
        receipt["results"] = {"ran": 0}
        return receipt

    # 2. enforce no_fallback BEFORE dispatch.
    # The schema already rejects the contradiction statically. This is the
    # second, independent check: a plan built by hand or by a future caller
    # must not be able to route a no-crawl pack into a crawl engine. A
    # commitment enforced in only one place is a commitment with a bypass.
    for p in runnable:
        if p.no_fallback and p.route in CRAWL_ROUTES:
            raise RunError(
                f"{p.pack_id}/{p.route_id}: pack declares no_fallback (never "
                f"crawl this source) but this plan's route is {p.route}. "
                f"Refusing to run.")

    forced_extractors = [p for p in runnable if p.engine == "extractor"]
    pf_plans = [p for p in runnable
                if p.engine != "extractor" and p.route in PACK_FETCH_MODES]
    ex_plans = forced_extractors + [
        p for p in runnable
        if p.engine != "extractor" and p.route in EXTRACTOR_ROUTES]

    if pf_plans:
        print(f"[run] pack_fetch: {len(pf_plans)} route(s)", flush=True)
        if args.dry_run:
            receipt["results"]["pack_fetch"] = {"skipped": "dry-run"}
        else:
            receipt["results"]["pack_fetch"] = run_via_pack_fetch(
                pf_plans, pack_id=pack_id, args=args)

    for p in ex_plans:
        print(f"[run] {p.route_id}: route={p.route}", flush=True)
        receipt["results"].setdefault("extractors", []).append(
            run_via_extractor(p, args=args))

    receipt["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return receipt


def build_parser():
    ap = argparse.ArgumentParser(
        prog="run_source.py",
        description="Run a source pack end to end (task 434). "
                    "Makes sources/*.yaml executable.")
    ap.add_argument("pack", help="path to a pack YAML, or its `id:` (e.g. ch-fedlex)")
    ap.add_argument("--plan", action="store_true",
                    help="print the compiled plans and exit (no network)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve consent, then show what would run without running it")
    ap.add_argument("--out-dir", default="build/run_source",
                    help="working dir for L0/L1/L2 + receipts")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N new fetches (smoke test)")
    ap.add_argument("--budget", type=int, default=45000, help="hard total-fetch stop")
    ap.add_argument("--no-l0", action="store_true",
                    help="disable raw retention (tests ONLY; production keeps L0, ADR-026)")
    ap.add_argument("--skip-consent", action="store_true",
                    help="DANGEROUS: skip the LIVE consent resolution (AIPREF, robots, TDMRep). "
                         "Only for offline replay of already-consented L0. The recorded-decline "
                         "guard in the HTTP client stays on: a declined host still raises "
                         "ConsentBlocked at the first request.")
    ap.add_argument("--receipt", default="",
                    help="write the run receipt JSON here")
    ap.add_argument("--extractor-args", nargs=argparse.REMAINDER,
                    help="everything after this is passed to the extractor verbatim")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)

    path = Path(args.pack)
    try:
        if not path.exists():
            path = PL.find_pack(args.pack)
        doc = PL.load_pack(path)
    except PL.PackError as exc:       # no sources/ in reach, no such pack, an invalid pack
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    plans = PL.plans(doc)

    print(f"pack   {doc['id']}  ({path})")
    print(f"adapter={doc['adapter']}  country={doc['country']}  "
          f"layer={doc['layer']}  freshness={doc['freshness']}")

    if not plans:
        if doc.get("adapter") == "cards":
            # Not unmigrated -- non-acquiring. A cards pack synthesises entity
            # cards from data already held; it fetches nothing, so it has no
            # acquisition route and never will. Saying "unmigrated" here would
            # be a false backlog item.
            print("\nCARDS PACK — no acquisition route by design.")
            print("Entity cards are built from data already held; nothing is "
                  "fetched. Build it with the recipe named in `card_recipe:`.")
            return 0
        print("\nNO RUNNABLE PLANS — this pack declares no `acquire:` block.")
        print("It is still documentation. Add an `acquire:` block to make it runnable.")
        return 2

    print(f"\n{len(plans)} plan(s):")
    for p in plans:
        print(f"  - {p.route_id:28s} route={p.route:30s} "
              f"params={sorted(p.params)} extractor={p.extractor or '-'}")
    if args.plan:
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    receipt = run_pack_plans(doc, plans, args)

    dest = args.receipt or os.path.join(args.out_dir, f"run_{doc['id']}.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\n[receipt] {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

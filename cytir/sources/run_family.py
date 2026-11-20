"""Run one registered source family end to end (task 683, S7): `cyt source run <family_id>`.

    discover -> consent -> acquire (sealed) -> extract -> chunk -> compose a pack 

The adapter (`cytir.sources.family.SourceFamily`) supplies the dialect; this module supplies
everything around it, once, for every family:

* the admission (`cytir.sources.registry.admit`): an undeclared family does not run;
* the consent preflight through the ONE resolver (`cytir.sources.runner.resolve_consent`), before
  any byte is acquired; a recorded decline EXCLUDES the unit, an unreadable policy QUARANTINES
  it (held for a re-probe; our failure is never recorded as the publisher's decision);
* the L0 seal: the acquired bytes and their transport metadata are written content-addressed
  BEFORE `extract` runs (corpus-doctrine #6);
* the terminals: every discovered unit ends `admitted`, `excluded(reason)` or
  `quarantined(reason)`, and the run refuses when `in != a + e + q` (corpus-doctrine #2);
* the pack: `records.jsonl` and a `manifest.json` that carries the family's declaration (its
  digest, consent basis, licence instrument, chunk spec), the consent verdicts and the
  terminals, so a pack can be read without the run that made it;
* the embed precondition (`assert_embeddable`): the consent refusal every embed door shares
  (recorded declines AND the declared consent basis; `unknown` is a refusal). `--embed` itself
  is RETIRED: it refused-by-message and names `cyt index build` (slice 8 step 2).

NO WALL CLOCK is written into the pack: two runs over the same sealed inputs and the same
`today` give byte-identical `records.jsonl`, which is what the exemplar goldens pin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Callable, Sequence
from urllib.parse import urlsplit

from cytir.sources import contract as contract_mod
from cytir.sources import registry, soft404, staged_text_guard
from cytir.sources.family import (ADMITTED, EXCLUDED, QUARANTINED, Captured, Ledger,
                               HttpSpec, RunContext, Unit, UnitExcluded)

PACK_SCHEMA = "aw.family-pack.v1"
CLEAR_BUCKETS = ("ok", "ok_search_only")


class RunError(RuntimeError):
    """The run cannot produce a trustworthy pack (an outage, a broken adapter)."""


class EmbedRefused(RuntimeError):
    """The pack may not be embedded: consent unchecked or the declared basis is `unknown`."""


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def contract_digest(row: dict) -> str:
    """sha256 of the family's declaration as canonical JSON: what the pack was made under."""
    return _sha256(json.dumps(row, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False).encode("utf-8"))


# --- stages ---------------------------------------------------------------------------

def default_consent(host: str, direct: bool) -> dict:
    """The ONE resolver (`cytir.consent`), through the wrapper the pack runner already uses."""
    from cytir.sources import runner
    return runner.resolve_consent(host, direct_crawl=direct)


def _consent_hosts(u: Unit) -> tuple[str, ...]:
    """The hosts a unit will contact: the ones it declares, else its URL's host. A `file:` unit
    (a local dump member) contacts no host, so it has none to resolve; any other URL without a
    host keeps the empty host, which the resolver never clears (held, never admitted)."""
    if u.consent_hosts:
        return tuple(u.consent_hosts)
    if urlsplit(u.url).scheme == "file":
        return ()
    return (_host(u.url),)


def undeclared_consent(unit: Unit, contract: dict) -> str | None:
    """The ONE refusal rule for an undeclared consent row (task 1406, ADR-115 rule 3).

    A unit that will contact a host is refused when its family's contract row has consent basis
    `unknown`: nobody has declared what the host allows, so no request goes out. Returns the
    reason `consent_undeclared:<family_id>:<host>`, else None. A declared row whose `search_ok`
    is None means "take the host's current verdict" and is NOT refused here: it runs through the
    live gate in `resolve_unit_consent`. A unit with no host (`file:`, a sealed archive) is never
    refused: consent governs fetch, not files already held."""
    hosts = _consent_hosts(unit)
    if not hosts or (contract.get("consent") or {}).get("basis") != "unknown":
        return None
    return f"consent_undeclared:{contract.get('family_id')}:{hosts[0]}"


def resolve_unit_consent(units: Sequence[Unit], ctx: RunContext, *, direct: bool
                         ) -> tuple[dict[tuple[str, bool], dict], dict[str, tuple[str, str]]]:
    """Verdict per (host, mode) and, per unit key, `(terminal, reason)` for blocked units."""
    resolver = ctx.consent or default_consent
    hosts = sorted({h for u in units for h in _consent_hosts(u) if h})
    verdicts: dict[tuple[str, bool], dict] = {}
    for h in hosts:
        try:
            verdicts[(h, direct)] = resolver(h, direct)
        except Exception as exc:  # noqa: BLE001 -- a recorded decline or an instrument failure
            name = type(exc).__name__
            verdicts[(h, direct)] = {
                "host": h, "bucket": "declined" if name == "ConsentBlocked" else "error",
                "error": f"{name}: {exc}"[:300]}
    blocked: dict[str, tuple[str, str]] = {}
    for u in units:
        for h in _consent_hosts(u):
            v = verdicts.get((h, direct))
            if v is None:
                blocked[u.key] = (QUARANTINED, f"consent_unresolved:{h}")
                break
            if v.get("bucket") in CLEAR_BUCKETS:
                continue
            if v.get("bucket") == "declined":
                blocked[u.key] = (EXCLUDED, f"consent_declined:{h}")
            else:
                # unknown / error: our instrument did not produce a verdict. HELD, re-probe.
                blocked[u.key] = (QUARANTINED, f"consent_unresolved:{h}")
            break
    return verdicts, blocked


def seal(out_dir: Path, cap: Captured) -> dict:
    """Write the captured bytes content-addressed and append the L0 index row."""
    digest = _sha256(cap.body)
    path = out_dir / "l0" / digest[:2] / digest
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(cap.body)
        tmp.replace(path)
    row = {"unit": cap.unit.key, "url": cap.unit.url, "sha256": digest,
           "bytes": len(cap.body), "meta": cap.meta}
    with (out_dir / "l0" / "index.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return row


def collapse_identical_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Within ONE unit: a row whose `point_id` AND `text` equal an earlier row's is a copy (same
    url + same chunk text mint the same id); keep the first, drop the rest, and return the
    detail `[{point_id, url, dropped}]` so the count stays visible (task 759). The same id with
    DIFFERENT text is a real id defect and is left in place for the duplicate_point_id
    quarantine. Rows without a point_id are left alone for the shape check."""
    first: dict[str, str] = {}
    out: list[dict] = []
    dropped: dict[str, dict] = {}
    for r in rows:
        pid = r.get("point_id")
        if pid and pid in first and first[pid] == r.get("text"):
            d = dropped.setdefault(pid, {"point_id": pid, "url": r.get("url"), "dropped": 0})
            d["dropped"] += 1
            continue
        if pid:
            first.setdefault(pid, r.get("text"))
        out.append(r)
    return out, list(dropped.values())


def _reason(prefix: str, exc: BaseException) -> str:
    return f"{prefix}:{type(exc).__name__}"


def build_http_client(spec: HttpSpec):
    """The client a family's `HttpSpec` describes (the ONE place a family client is built)."""
    if spec.kind == "curl":
        from cytir.sources.ctgov_psy import harvest
        return harvest.CurlClient(timeout=spec.timeout)
    from cytir.sources.http import make_client     # the factory: its transport is always guarded
    return make_client(spec)


def ensure_guarded(client):
    """Put the consent guard under an injected httpx client (spec section 2, factory rule).

    Untouched: a factory-built client (already guarded), a `MockTransport` tape (replay, no
    socket) and anything that is not an `httpx.Client` (the curl client guards itself)."""
    import httpx

    from cytir.consent.guard import guard_client, is_guarded
    if not isinstance(client, httpx.Client) or is_guarded(client):
        return client
    if isinstance(getattr(client, "_transport", None), httpx.MockTransport):
        return client
    return guard_client(client)


def run_family(family_id: str, *, out_dir: str | Path, today: str, limit: int = 0,
               http=None, consent: Callable[[str, bool], dict] | None = None,
               skip_consent: bool = False, strict: bool = False,
               options: dict | None = None, log=print, dry_run: bool = False,
               contracts_path: str | Path | None = None) -> dict:
    """Run `family_id` and compose its pack under `out_dir/pack/`. Returns the manifest.

    `dry_run` stops after discover + consent: it returns the plan (units, hosts, verdicts,
    what would be read) and writes nothing; no unit is acquired.

    The runner owns the HTTP client (task 1556): a family with an `http_spec` gets a client built
    from it (after `preflight`, so a replay mode may inject its own), closed when the run ends.
    An `http` passed in, or set by `preflight`, is never closed here; an httpx client of that kind
    gets the consent guard under its transport (`ensure_guarded`) unless it is a MockTransport tape."""
    with ExitStack() as stack:
        return _run_family(stack, family_id, out_dir=out_dir, today=today, limit=limit,
                           http=http, consent=consent, skip_consent=skip_consent,
                           strict=strict, options=options, log=log, dry_run=dry_run,
                           contracts_path=contracts_path)


def _run_family(stack: ExitStack, family_id: str, *, out_dir, today, limit, http, consent,
                skip_consent, strict, options, log, dry_run, contracts_path) -> dict:
    adapter, row, spec = registry.admit(family_id, strict=strict,
                                        contracts_path=contracts_path)
    out = Path(out_dir)
    if not dry_run:
        if (out / "pack" / "manifest.json").exists():
            raise RunError(f"{out} already holds a pack; a run never overwrites one "
                           "(pick a new --out-dir)")
        (out / "l0").mkdir(parents=True, exist_ok=True)
        (out / "l0" / "index.jsonl").write_text("")
    ctx = RunContext(out_dir=out, today=today, limit=limit, http=http, consent=consent,
                     skip_consent=skip_consent, options=dict(options or {}),
                     chunk_spec=spec, contract=row, log=log)
    ledger = Ledger()

    preflight = getattr(adapter, "preflight", None)
    if preflight is not None:
        preflight(ctx)          # cheap checks that must fail BEFORE a unit is discovered
    ctx.http = ensure_guarded(ctx.http)      # an injected or preflight-set client, guarded
    spec_http = getattr(adapter, "http_spec", None)
    if ctx.http is None and spec_http is not None:
        ctx.http = build_http_client(spec_http)
        stack.callback(ctx.http.close)
    l1_manifest = None
    l1 = getattr(adapter, "l1", None)    # OPTIONAL stage (ADR-105 am.1); same probe as preflight
    if l1 is not None and not dry_run:   # a dry run writes nothing
        l1_manifest = l1(ctx)            # persisted L1 inputs, BEFORE the stages that need them
        if not isinstance(l1_manifest, dict):
            raise RunError(f"{family_id}: l1() must return a manifest dict, got "
                           f"{type(l1_manifest).__name__}")
        log(f"[{family_id}] l1 persisted")
    units = list(adapter.discover(ctx))
    ledger.discover(len(units))
    seen_keys = Counter(u.key for u in units)
    dup = sorted(k for k, n in seen_keys.items() if n > 1)
    if dup:
        raise RunError(f"{family_id}: discover() returned duplicate unit keys {dup[:5]}; a "
                       "unit key is stable and unique (the ledger keys on it)")
    log(f"[{family_id}] discovered {len(units)} unit(s)")

    if limit and len(units) > limit:
        for u in units[limit:]:
            ledger.close(u.key, EXCLUDED, "run_limit")
        units = units[:limit]

    # consent, BEFORE acquire
    verdicts: dict[tuple[str, bool], dict] = {}
    # the declared-basis rule holds in BOTH paths: --skip-consent bypasses only the LIVE
    # per-origin gate, never this one. A refused unit contacts nothing.
    refused = {u.key: r for u in units if (r := undeclared_consent(u, row))}
    if skip_consent:
        log(f"[{family_id}] consent SKIPPED (offline replay); the pack cannot be embedded")
        blocked = {k: (QUARANTINED, r) for k, r in refused.items()}
    else:
        verdicts, blocked = resolve_unit_consent(
            [u for u in units if u.key not in refused], ctx, direct=adapter.direct_crawl)
        blocked.update({k: (QUARANTINED, r) for k, r in refused.items()})
    runnable = []
    for u in units:
        if u.key in blocked:
            ledger.close(u.key, *blocked[u.key])
        else:
            runnable.append(u)
    if refused and len(refused) == len(units) and not dry_run:
        raise RunError(f"{family_id}: every one of {len(units)} unit(s) refused: the "
                       "contract row declares no consent basis (basis 'unknown'); declare "
                       "it before any host is contacted (nothing composed)")

    if dry_run:
        plan = {"family_id": family_id, "dry_run": True, "units": [u.key for u in units],
                "runnable": [u.key for u in runnable], "discover": ctx.notes,
                "consent_checked": not skip_consent,
                "verdicts": [{"host": k[0], "direct": k[1], "bucket": v.get("bucket")}
                             for k, v in sorted(verdicts.items())],
                "blocked": {u.key: ledger._t[u.key].reason for u in units
                            if u.key in ledger._t}}
        log(f"[{family_id}] DRY RUN: {len(runnable)} of {len(units)} unit(s) would be "
            f"acquired; nothing was read or written")
        return plan

    # acquire (sealed) -> extract -> chunk
    records: list[dict] = []
    seen_ids: set[str] = set()
    collapsed_total = 0
    collapsed_detail: list[dict] = []
    failures = 0
    failure_reasons: list[tuple[str, str]] = []   # (unit key, class: message) for the loud error
    guard = staged_text_guard.StagedTextGuard(
        family_id, min_coverage=getattr(adapter, "staged_text_min_coverage", None))
    for u in runnable:
        try:
            caps = list(adapter.acquire(u, ctx))
        except UnitExcluded as exc:          # the origin says it is not there to take
            ledger.close(u.key, EXCLUDED, exc.reason)
            continue
        except Exception as exc:  # noqa: BLE001 -- quarantined with its class, never an empty
            failures += 1
            log(f"[{family_id}] acquire failed for {u.key}: {exc!r}")
            failure_reasons.append((u.key, f"acquire_failed:{type(exc).__name__}: {exc}"))
            ledger.close(u.key, QUARANTINED, _reason("acquire_failed", exc))
            continue
        if not caps:
            ledger.close(u.key, EXCLUDED, "acquire_empty")
            continue
        for cap in caps:
            seal(out, cap)
        unit_records: list[dict] = []
        try:
            docs = [d for cap in caps for d in adapter.extract(cap, ctx)]
            if not docs:
                ledger.close(u.key, EXCLUDED, "extract_empty")
                continue
            short = False
            for d in docs:
                doc_records = list(adapter.chunk(d, ctx))
                if getattr(d, "source_parts", None) is not None and d.part_of is not None:
                    cov = guard.observe_parts(u.key, d.source_parts, doc_records, d.part_of)
                else:
                    cov = guard.observe(u.key, d.source_text,
                                        [str(r.get("text") or "") for r in doc_records])
                short = short or guard.shortfall(cov)
                unit_records.extend(doc_records)
        except UnitExcluded as exc:
            ledger.close(u.key, EXCLUDED, exc.reason)
            continue
        except Exception as exc:  # noqa: BLE001
            failures += 1
            log(f"[{family_id}] extract/chunk failed for {u.key}: {exc!r}")
            failure_reasons.append((u.key, f"extract_failed:{type(exc).__name__}: {exc}"))
            ledger.close(u.key, QUARANTINED, _reason("extract_failed", exc))
            continue
        if short:       # a cap, not a declared transformation, removed text (task 706)
            ledger.close(u.key, QUARANTINED, staged_text_guard.SHORTFALL_REASON)
            continue
        if not unit_records:
            ledger.close(u.key, EXCLUDED, "chunk_empty")
            continue
        soft = soft404.detect_records(unit_records)
        if soft:                 # an error page served as 200: the bytes stay sealed, no rows
            ledger.close(u.key, EXCLUDED, f"soft_404:{soft}")
            continue
        bad = [r for r in unit_records
               if not (r.get("url") and r.get("text") and r.get("point_id"))]
        if bad:
            ledger.close(u.key, QUARANTINED, "record_shape:needs url,text,point_id")
            continue
        unit_records, dropped = collapse_identical_rows(unit_records)
        if dropped:             # same id AND same bytes: one row, the copies counted (task 759)
            collapsed_total += sum(d["dropped"] for d in dropped)
            collapsed_detail.extend({"unit": u.key, **d} for d in dropped)
            note = ", ".join(f"{d['url']} x{d['dropped'] + 1}" for d in dropped[:3])
            log(f"[{family_id}] {u.key}: collapsed {sum(d['dropped'] for d in dropped)} "
                f"identical row(s) ({note})")
        ids = [r["point_id"] for r in unit_records]
        if len(set(ids)) != len(ids) or seen_ids.intersection(ids):
            ledger.close(u.key, QUARANTINED, "duplicate_point_id")
            continue
        seen_ids.update(ids)
        for r in unit_records:      # ADR-124: the one stamp -- the door names the family on every row
            if r.setdefault("family_id", family_id) != family_id:
                raise RunError(f"{family_id}: a record carries family_id={r['family_id']!r}")
        records.extend(unit_records)
        ledger.close(u.key, ADMITTED, records=len(unit_records))
    if runnable and failures == len(runnable):
        # task 1824: CI showed only this sentence; the cause lives in the failures, so name it.
        classes = {r.split(":", 1)[0] + ":" + r.split(":", 2)[1].strip() for _, r in failure_reasons}
        shown = "; ".join(f"{k[:12]} -> {r[:300]}" for k, r in failure_reasons[:3])
        raise RunError(f"{family_id}: every one of {len(runnable)} unit(s) failed; that is an "
                       "outage or a broken adapter, not an empty source (nothing composed); "
                       f"{len(classes)} distinct reason class(es); first {min(3, len(failure_reasons))}: "
                       f"{shown}")

    ledger.assert_balanced()
    loud = guard.loud_line()
    if loud:
        log(loud)
    guard_summary = guard.summary([str(r.get("text") or "") for r in records])
    if runnable and not records and getattr(adapter, "require_records", False):
        # an adapter that sets `require_records` (a dump family: an input that yields nothing is
        # a wrong input or a broken parser, never an empty source; task 440/634: a month of
        # 0-passage runs with green tests). A sitemap family may legitimately end all-404.
        raise RunError(f"{family_id}: {len(runnable)} unit(s) read, 0 records admitted "
                       f"({ledger.summary()['excluded_reasons'] or ledger.summary()['quarantined_reasons']}); "
                       "nothing composed")
    return compose_pack(out, family_id=family_id, adapter=adapter, row=row, spec=spec,
                        ctx=ctx, ledger=ledger, verdicts=verdicts, records=records,
                        contracts_path=contracts_path, guard_summary=guard_summary,
                        collapsed_rows=collapsed_total, collapsed_detail=collapsed_detail,
                        l1_manifest=l1_manifest)


def compose_pack(out: Path, *, family_id, adapter, row, spec, ctx, ledger, verdicts,
                 records, contracts_path=None, guard_summary=None, collapsed_rows=0,
                 collapsed_detail=(), l1_manifest=None) -> dict:
    pack = out / "pack"
    pack.mkdir(parents=True, exist_ok=True)
    # records keep the ADAPTER'S key order (no sort): an adapter ported from a script then
    # writes the same bytes the script wrote, which is what a golden compares
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")
    (pack / "records.jsonl").write_bytes(body)
    consent = row.get("consent") or {}
    licence = row.get("licence") or {}
    manifest = {
        "schema": PACK_SCHEMA,
        "family_id": family_id,
        "route": adapter.route,
        "contract": {
            "path": str(Path(contracts_path or contract_mod.DEFAULT_PATH)),
            "row_sha256": contract_digest(row),
            "status": row.get("status"),
            "consent_basis": consent.get("basis"),
            "consent_resolution": consent.get("resolution"),
            "licence_instrument": licence.get("instrument"),
            "unknowns": sorted(row.get("unknowns") or []),
            "chunk": spec.as_manifest(),
        },
        "run": {"today": ctx.today, "limit": ctx.limit, "options": ctx.options},
        "discover": ctx.notes,
        **({"l1": l1_manifest} if l1_manifest is not None else {}),
        "consent": {
            "checked": not ctx.skip_consent,
            "direct_crawl": adapter.direct_crawl,
            "verdicts": [
                {k: v.get(k) for k in ("host", "bucket", "status", "search_ok", "train_ok",
                                       "checked_at", "error") if k in v}
                for _k, v in sorted(verdicts.items())],
        },
        "terminals": ledger.summary(),
        "staged_text_guard": guard_summary,
        "collapsed_identical_rows": collapsed_rows,
        **({"collapsed_detail": list(collapsed_detail)} if collapsed_detail else {}),
        "units": [{"unit": t.unit_key, "terminal": t.terminal, "reason": t.reason,
                   "records": t.records} for t in ledger.terminals()],
        "records": {
            "count": len(records),
            "sha256": _sha256(body),
            "hosts": dict(sorted(Counter(_host(r["url"]) for r in records).items())),
        },
    }
    (pack / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    t = manifest["terminals"]
    ctx.log(f"[{family_id}] pack: in={t['units_in']} admitted={t['units_admitted']} "
            f"excluded={t['units_excluded']}{t['excluded_reasons'] or ''} "
            f"quarantined={t['units_quarantined']}{t['quarantined_reasons'] or ''} "
            f"records={t['records']} collapsed_identical_rows={collapsed_rows} -> {pack}")
    return manifest


# --- embed bridge ---------------------------------------------------------------------

def load_pack(pack_dir: str | Path) -> tuple[dict, list[dict]]:
    pack = Path(pack_dir)
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != PACK_SCHEMA:
        raise EmbedRefused(f"{pack}: not an {PACK_SCHEMA} pack")
    raw = (pack / "records.jsonl").read_bytes()
    if _sha256(raw) != manifest["records"]["sha256"]:
        raise EmbedRefused(f"{pack}: records.jsonl does not match its manifest digest")
    recs = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return manifest, recs


def assert_embeddable(manifest: dict, recs: list[dict], *, registry_path: str | None = None,
                      log=lambda m: None) -> dict:
    """The consent refusal in front of the embed stage. Refuses when the pack's consent was
    not checked, when the family's declared consent basis is `unknown`, or when any record
    host has a recorded blocking decision (`cytir.consent.embed_gate`)."""
    from cytir.consent import embed_gate

    fid = manifest["family_id"]
    if not (manifest.get("consent") or {}).get("checked"):
        raise EmbedRefused(
            f"EMBED REFUSED: the {fid} pack was composed with the consent check skipped "
            "(offline replay). Re-run it with the live consent check; vectors are never "
            "made from an unchecked pack.")
    basis = (manifest.get("contract") or {}).get("consent_basis")
    if basis in (None, contract_mod.UNKNOWN):
        raise EmbedRefused(
            f"EMBED REFUSED: family {fid!r} declares its consent basis as {basis!r}. "
            "`unknown` is a refusal, not a value (corpus-doctrine #7): the declaration needs "
            "a real basis first (POLICY PR against data/source-family-contracts.v1.json).")
    return embed_gate.assert_records_consent(
        recs, registry=registry_path, label=f"the {fid} pack", log=log)


def parse_options(pairs: Sequence[str]) -> dict:
    out: dict = {}
    for p in pairs or ():
        if "=" not in p:
            raise SystemExit(f"--opt expects KEY=VALUE, got {p!r}")
        k, v = p.split("=", 1)
        out.setdefault(k, []).append(v)
    return {k: (v[0] if len(v) == 1 else v) for k, v in out.items()}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="cyt source run", description="Run a registered source family and compose a pack.")
    ap.add_argument("family", help="a registered family id (`cyt source families` lists them)")
    ap.add_argument("--out-dir", default=None,
                    help="working dir: L0 seal + pack/ (required unless --out is given; with "
                         "--out it keeps the seal and the pack, else they are discarded)")
    ap.add_argument("--from", dest="from_dir", default=None, metavar="DIR",
                    help="OFFLINE: read this local directory (the family's `root` option) instead "
                         "of the network. Only for a family whose route is a source-owned bulk "
                         "dump and that contacts no host; no consent is asked, no client is made. "
                         "For a family with the replay-l0 route (PDFs) it is a sealed L0 listing "
                         "(a JSONL file, or a directory with listing.jsonl); no page is fetched "
                         "and the host's consent is still resolved live")
    ap.add_argument("--out", default=None, metavar="STAGE",
                    help="also write the sealed chunk stage `cyt index build --pack` reads "
                         "(PACK.json + chunks/); STAGE must not exist")
    ap.add_argument("--today", default=None,
                    help="the run date stamped into rows (default: today, UTC)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N units (smoke test)")
    ap.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE",
                    help="family option (repeatable), e.g. --opt eli=cc/1948/195")
    ap.add_argument("--dry-run", action="store_true",
                    help="discover + consent only: print what would be acquired, read and "
                         "write nothing")
    ap.add_argument("--strict", action="store_true",
                    help="refuse a family whose declaration holds any `unknown` value")
    ap.add_argument("--embed", action="store_true",
                    help="RETIRED: refuses and names `cyt index build`, the one build verb")
    ap.add_argument("--skip-consent", action="store_true",
                    help="DANGEROUS: skips the LIVE consent resolution only (AIPREF, robots, TDMRep); "
                         "the pack is marked unchecked and can never be embedded. The "
                         "recorded-decline guard is built into the HTTP client and stays on: "
                         "a declined host still raises ConsentBlocked at the first request")
    return ap


EMBED_RETIRED_MESSAGE = (
    "REFUSED: `--embed` is retired. It built an index that differs from the one production "
    "serves (point ids, dense text, BM25 weights and payload all differ). The one build verb "
    "is `cyt index build`. Run without "
    "`--embed` to stop at the sealed pack; vectors for a pack come from `cyt embed shards`. "
    "Nothing was read or written.")


def main(argv=None) -> int:
    import datetime as dt

    args = build_parser().parse_args(argv)
    if args.embed:      # before any discovery or fetch: nothing is read or written
        print(EMBED_RETIRED_MESSAGE, file=sys.stderr)
        return 2
    today = args.today or dt.datetime.now(dt.timezone.utc).date().isoformat()
    if args.out or args.from_dir:
        return _stage_main(args, today)
    if not args.out_dir and not args.dry_run:     # a dry run writes nothing: no dir needed
        print("REFUSED: give --out-dir (the working dir) or --out (the stage)", file=sys.stderr)
        return 2
    try:
        manifest = run_family(
            args.family, out_dir=args.out_dir or ".", today=today, limit=args.limit,
            skip_consent=args.skip_consent, strict=args.strict, dry_run=args.dry_run,
            options=parse_options(args.opt), log=lambda m: print(m, flush=True))
    except (registry.FamilyRefused, RunError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    if args.dry_run:
        print(json.dumps(manifest, indent=1, sort_keys=True))
        return 0
    if not manifest["terminals"]["units_admitted"]:
        return 3
    return 0


def _stage_main(args, today: str) -> int:
    """`--from` and/or `--out`: run through the family door, then write the stage (stage_out)."""
    from cytir.sources import stage_out

    if args.dry_run or args.skip_consent:
        print("REFUSED: --dry-run and --skip-consent do not combine with --from/--out (a stage "
              "is made only from a pack whose consent was checked)", file=sys.stderr)
        return 2
    if not args.out:
        print("REFUSED: --from needs --out STAGE (or drop --from and use --out-dir for a pack)",
              file=sys.stderr)
        return 2
    if not args.from_dir and not args.out_dir:
        print("REFUSED: --out without --from needs --out-dir (a network run keeps its sealed source "
              "data in a directory you name)", file=sys.stderr)
        return 2
    try:
        meta = stage_out.run_to_stage(
            args.family, out=args.out, today=today, from_dir=args.from_dir,
            work_dir=args.out_dir, limit=args.limit, strict=args.strict,
            options=parse_options(args.opt), log=lambda m: print(m, flush=True))
    except (registry.FamilyRefused, RunError, stage_out.StageOutRefused, ValueError,
            RuntimeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    return 0 if meta["rows"] else 3


def list_main(argv=None) -> int:
    registry_rows = []
    doc = contract_mod.load()
    by = contract_mod.by_id(doc)
    for fid in registry.registered():
        row = by.get(fid)
        c = (row or {}).get("consent", {}).get("basis") if row else "UNDECLARED"
        registry_rows.append((fid, (row or {}).get("status", "UNDECLARED"), c))
    for fid, status, basis in registry_rows:
        print(f"{fid:32s} {status:11s} consent={str(basis)[:40]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

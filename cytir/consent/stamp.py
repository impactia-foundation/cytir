"""Stamp consent flags on a sealed pack's rows at COMPOSE time (task 684, ruling 20).

A growth writer used to seal rows with no `search_ok` / `train_ok` / `consent_basis` /
`consent_checked_at`, and the backfill then had to infer them from the payload. 2.7M rows
landed that way (1.2M ERIC official PDF). This module is the writer's side of the same
derivation: it calls `cytir.consent.derive.derive_consent_flags` -- the ONE rule engine
(ADR-115) -- with the family the pack's manifest DECLARES, and writes the four fields into
the chunk rows before the manifest is sealed. There is no second copy of any rule.

Contract:

* A row that already carries both flags is class E: kept as it is.
* A row the engine stamps (class A or B) gets exactly `Derivation.writes()`.
* Everything else REFUSES the whole pack, with counts by reason and sample urls, and
  nothing is written: a row with no derivable basis is never defaulted to true (and never
  written as an `unknown` value -- ADR-115 defines none). A row that would be stamped
  `search_ok=false` refuses too: a declined row does not belong in a new pack.
* A family that is not declared in the contracts file refuses every non-E row.
* arXiv rows refuse when no per-paper licence map was supplied: a missing instrument is
  not an empty one.

The chunk rows are re-written to a sibling directory (the pattern `compose` already uses
for the language backfill); the original materialisation is never edited in place. The
payload profile is applied later, at the drain's write: the twin (`aw_e8`, profile `prod`)
takes the raw strings, the served collection (`aw_e9`, `normalized`) takes the coded
`consent_basis` -- every basis written here has a code in `data/payload-enum-tables.v1.json`.

Pure stdlib + the engine; no network, no Qdrant.
"""
from __future__ import annotations

import collections
import glob
import gzip
import hashlib
import json
import os
import re
import time
from pathlib import Path

from .derive import (DUMP_CONSENT_BASES, RULES_VERSION, WRITTEN_BASES, DeriveContext, derive_consent_flags,
                     host_of, load_cache_index, load_sealed_probes)

SCHEMA = "aw.consent-stamp.v1"
REPO = Path(__file__).resolve().parents[2]
CONTRACTS_PATH = REPO / "data" / "source-family-contracts.v1.json"
PER_URL_RULES = frozenset({"r1-arxiv", "r1-pubmed"})
FLAG_FIELDS = ("search_ok", "train_ok", "consent_checked_at", "consent_basis")


class ConsentStampRefused(RuntimeError):
    """The pack carries rows the engine cannot stamp. Nothing was written."""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_cache_dir() -> str:
    # Same location and variable as `cytir.consent.resolver.CACHE_DIR`. Not imported from
    # there: importing the resolver demands a configured crawler identity, and composing a
    # pack fetches nothing. the consent-stamp compose test pins the two together.
    return os.environ.get("AW_CONSENT_CACHE_DIR") or os.path.expanduser("~/aw-data/consent")


def load_context(*, as_of: str | None = None, contracts_path: str | os.PathLike | None = None,
                 cache_dir: str | None = None,
                 arxiv_licences: dict | None = None) -> DeriveContext:
    """The engine's context for a compose run. A missing cache directory REFUSES: a cache
    that failed to load would otherwise read as 'no host has a verdict'."""
    cdir = os.path.expanduser(cache_dir or default_cache_dir())
    if not os.path.isdir(cdir):
        raise ConsentStampRefused(
            f"consent cache directory {cdir!r} does not exist: refusing to stamp from an "
            "empty cache (set CYTIR_CONSENT_CACHE_DIR; verification-doctrine: an error is "
            "not an empty)")
    contracts = json.loads(Path(contracts_path or CONTRACTS_PATH).read_text())
    return DeriveContext(families=contracts["families"], cache_index=load_cache_index(cdir),
                         as_of=as_of or utc_now(), arxiv_licences=arxiv_licences,
                         sealed_probes=load_sealed_probes(contracts["families"], str(REPO)))


def _dotted(row: dict, path: str):
    cur = row
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _norm_licence(v) -> str:
    """'CC BY-SA 2.5' (the Stack Exchange dump's spelling) -> 'CC-BY-SA-2.5', the form
    the ADR-115 licence prefix test reads. A view for the derivation only: the row's own
    licence value is never rewritten."""
    s = str(v or "").strip()
    return re.sub(r"\s+", "-", s) if s.startswith("CC ") else s


def derivation_view(row: dict, *, license_from: str | None, family_license: str | None) -> dict:
    """The row as the engine reads it. A chunk-stage row carries no top-level `license`
    or `domain`; the manifest's licence door (`license_from`, `family_license`) supplies
    the licence the projection will resolve, and the host comes from the url."""
    lic = row.get("license")
    if not lic and license_from:
        lic = _dotted(row, license_from)
    if not lic:
        lic = family_license
    view = dict(row)
    view["license"] = _norm_licence(lic)
    return view


def stamp_row(row: dict, ctx: DeriveContext, *, family: str, license_from: str | None = None,
              family_license: str | None = None):
    """(row', derivation, refusal_reason|None). `row'` is `row` itself for class E."""
    view = derivation_view(row, license_from=license_from, family_license=family_license)
    d = derive_consent_flags(view, ctx, family_id=family)
    if d.klass == "E":
        return row, d, None
    if d.klass == "D":
        return row, d, d.rule
    if family not in ctx.by_id:
        return row, d, f"family-undeclared ({family})"
    if d.search_ok is not True:
        return row, d, f"would-stamp-search_ok-false [{d.rule}]"
    if d.rule == "r1-arxiv" and ctx.arxiv_licences is None:
        return row, d, "arxiv-licence-map-not-supplied"
    if d.consent_basis not in WRITTEN_BASES | DUMP_CONSENT_BASES:
        return row, d, f"basis-not-closed ({d.consent_basis!r})"
    out = dict(row)
    out.update(d.writes())
    return out, d, None


def stamp_payload_rows(rows: list, ctx: DeriveContext, *, family: str,
                       replace_existing: bool = False, sample_urls: int = 5):
    """(stamped rows, quarantined, report): stamp the PAYLOAD of content-pack rows
    (`{"id","payload","vector"}`, the shape a daily news pack carries) through the same
    engine `stamp_chunks_dir` uses.

    `replace_existing=True` is for a stage that already wrote its own four consent fields
    (the news stage writes `consent_basis: robots-live-resolve` with the cache record's
    date): the engine then reads the row WITHOUT them, so rule E does not freeze the stage's
    older stamp, and the host's CURRENT verdict (r3/r4, its own `checked_at`) is written in
    their place. Only those four fields change; ids, vectors and every other payload field
    are kept as they are, in the same key order.

    A row that cannot be stamped is NOT returned in `stamped rows`: it goes to
    `quarantined` (`{"id","url","host","reason","kind"}`). Nothing is written for it and
    nothing is defaulted. Three kinds:
      `no_record`     the engine cannot stamp it (no usable cache record: rule D, a held
                      record, ...);
      `declines_search` the host HAS a usable record and it says search_ok=false (the host
                      declines search; not a missing record);
      `would_loosen`  the stage flag was FALSE but the host's current verdict reads TRUE.
    Ruled (Tim 2026-10-05, plan.md ruling 26 Q1): when the current record would loosen a flag
    the stage set to false, only those rows are left out and the rest of the pack lands.
    Leaving a row out never loosens anything. The caller decides what a quarantine
    means for the pack (see `cytir.news.compose.stamp_consent`)."""
    out = []
    quarantined = []
    samples: dict = collections.defaultdict(list)
    rules = collections.Counter()
    by_host: dict = collections.defaultdict(collections.Counter)
    changed = collections.Counter()
    for row in rows:
        payload = row.get("payload") or {}
        view = ({k: v for k, v in payload.items() if k not in FLAG_FIELDS}
                if replace_existing else payload)
        _new, d, why = stamp_row(view, ctx, family=family)
        kind = "declines_search" if ("search-false" in (why or "") or "search_ok-false" in (why or "")) else "no_record"
        if why is None and replace_existing:
            for f in ("search_ok", "train_ok"):
                if payload.get(f) is False and getattr(d, f) is True:
                    why = f"would-loosen-{f}-stage-false-host-true"
                    kind = "would_loosen"
        if why is not None:
            url = str(payload.get("url"))
            quarantined.append({"id": row.get("id"), "url": url,
                                "host": host_of(payload.get("url")), "reason": why,
                                "kind": kind})
            if len(samples[why]) < sample_urls:
                samples[why].append(url)
            continue
        np = dict(payload)
        np.update(d.writes())          # class E: writes() is empty -> the row is kept as it is
        if replace_existing:
            for f in FLAG_FIELDS:
                if np.get(f) != payload.get(f):
                    changed[f] += 1
        rules[(d.klass, d.rule)] += 1
        by_host[host_of(payload.get("url"))][
            f"search_ok={np.get('search_ok')},train_ok={np.get('train_ok')}"] += 1
        out.append({**row, "payload": np})
    return out, quarantined, {
        "quarantine_samples": dict(samples),
        "schema": SCHEMA, "rules_version": RULES_VERSION, "family": family, "as_of": ctx.as_of,
        "rows": len(rows), "replace_existing": replace_existing,
        "rules": {f"{k}|{r}": n for (k, r), n in sorted(rules.items())},
        "fields_changed": dict(changed),
        "flags_by_host": {h: dict(c) for h, c in sorted(by_host.items())},
        "contracts_sha256": hashlib.sha256(CONTRACTS_PATH.read_bytes()).hexdigest()}


def stamp_chunks_dir(chunks_dir: str, out_dir: str, ctx: DeriveContext, *, family: str,
                     license_from: str | None = None, family_license: str | None = None,
                     sample_urls: int = 5, dry: bool = False) -> dict:
    """Stamp every row of `chunks_dir/shard-*.texts.jsonl.gz` into `out_dir` (same shard
    names, same row order). Refuses the whole pack on any unstampable row. `dry=True` runs the
    same derivation and refusal and writes nothing (the pre-embed check)."""
    shards = sorted(glob.glob(os.path.join(chunks_dir, "shard-*.texts.jsonl.gz")))
    if not shards:
        raise ConsentStampRefused(f"no shard-*.texts.jsonl.gz under {chunks_dir}")
    out = Path(out_dir or "dry-run")
    tmp = out.with_name(out.name + ".partial")
    if not dry:
        if tmp.exists():
            for f in tmp.glob("*"):
                f.unlink()
        tmp.mkdir(parents=True, exist_ok=True)
    rules = collections.Counter()
    refused = collections.Counter()
    samples: dict = collections.defaultdict(list)
    memo: dict = {}
    rows = e_rows = e_no_basis = 0
    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8") as fin, \
                (open(os.devnull, "w") if dry else
                 gzip.open(tmp / os.path.basename(shard), "wt", encoding="utf-8")) as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows += 1
                key = None
                if not (isinstance(row.get("search_ok"), bool) and isinstance(row.get("train_ok"), bool)):
                    view = derivation_view(row, license_from=license_from, family_license=family_license)
                    fam_rule_per_url = family in ("named_slice:arxiv", "named_slice:pubmed")
                    # exactly what the engine reads: the licence view, the provenance (r5),
                    # the consent state/basis (r5/r0/carried), the partial flags (D) and the
                    # host it looks up (`host_of(url)` or the domain)
                    key = None if fam_rule_per_url else (
                        view["license"], row.get("provenance"), row.get("consent_state"),
                        row.get("consent_basis"), row.get("search_ok"), row.get("train_ok"),
                        host_of(row.get("url")) or str(row.get("domain") or "").lower())
                hit = memo.get(key) if key is not None else None
                if hit is None:
                    new, d, why = stamp_row(row, ctx, family=family, license_from=license_from,
                                            family_license=family_license)
                    if key is not None:
                        memo[key] = (d, why)
                else:
                    d, why = hit
                    new = row
                    if why is None and d.klass in ("A", "B"):
                        new = dict(row)
                        new.update(d.writes())
                if why is not None:
                    refused[why] += 1
                    if len(samples[why]) < sample_urls:
                        samples[why].append(str(row.get("url")))
                    continue
                if d.klass == "E":
                    e_rows += 1
                    if not row.get("consent_basis"):
                        e_no_basis += 1
                rules[(d.klass, d.rule)] += 1
                fout.write(json.dumps(new, ensure_ascii=False) + "\n")
    if refused:
        if not dry:
            for f in tmp.glob("*"):
                f.unlink()
            tmp.rmdir()
        lines = [f"  {n:,} x {why}  e.g. {samples[why]}" for why, n in refused.most_common()]
        raise ConsentStampRefused(
            f"{family}: {sum(refused.values()):,} of {rows:,} row(s) have no derivable consent "
            "basis under the ADR-115 rules (no flags are ever defaulted, and 'unknown' is not a "
            "value). Nothing was written.\n" + "\n".join(lines))
    if not dry:
        if out.exists():
            for f in out.glob("*"):
                f.unlink()
            out.rmdir()
        tmp.rename(out)
    report = {
        "schema": SCHEMA, "rules_version": RULES_VERSION, "family": family, "as_of": ctx.as_of,
        "rows_in": rows, "rows_out": rows, "class_e_kept": e_rows,
        "class_e_without_consent_basis": e_no_basis,
        "rules": {f"{k}|{r}": n for (k, r), n in sorted(rules.items())},
        "contracts_sha256": hashlib.sha256(CONTRACTS_PATH.read_bytes()).hexdigest(),
        "stamped_from": chunks_dir,
    }
    if dry:
        report["dry"] = True
        return report
    report.update(verify_stamped(str(out), expect_rows=rows))
    return report


def verify_stamped(chunks_dir: str, *, expect_rows: int | None = None) -> dict:
    """Re-read the stamped directory (the stage AFTER the write) and count the rows that
    carry both flags as booleans. Refuses unless every row does."""
    n = ok = with_checked = 0
    for shard in sorted(glob.glob(os.path.join(chunks_dir, "shard-*.texts.jsonl.gz"))):
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                n += 1
                if isinstance(r.get("search_ok"), bool) and isinstance(r.get("train_ok"), bool):
                    ok += 1
                    if r.get("consent_checked_at"):
                        with_checked += 1
    if expect_rows is not None and n != expect_rows:
        raise ConsentStampRefused(f"{chunks_dir}: {n} rows after stamping, {expect_rows} before")
    if ok != n:
        raise ConsentStampRefused(f"{chunks_dir}: only {ok}/{n} rows carry both flags after stamping")
    return {"verified_rows": n, "verified_with_flags": ok,
            "verified_with_checked_at": with_checked}

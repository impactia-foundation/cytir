#!/usr/bin/env python3
"""Authored prose must never be indexed as a page's content (task 586).

WHY
---
``data/refresh_manifest.json`` carries 199 curated "freshness" sources. Each
entry's ``text`` is not the page: it is prose a model wrote on 2026-07-08/09
stating what the page says ("The current latest stable React version is
19.2…"). ``freshness_ingest.build_points`` embedded that string, tokenised it
into the BM25 surface and shipped it as the row for the URL. Production still
serves those rows; on 2026-09-16, 7 of 9 spot-checked claims were false
(the task 586 stale-facts artifact), and 97 of production's 429
held-out h@1 stand on a row at one of those URLs.

Task 581 then made both lanes file their own authored text into the ADR-026 L1
store so the ADR-105 persisted-inputs guard would pass. That guard asks whether
a document exists; it never asks whether the document is the page. So the
authored body is now a first-class stored input that a rebuild can replay --
and one already did: 178 of production's un-saturated rows carry ``build_id
aw_b1`` and the manifest's own authored snippet.

WHAT THIS MODULE REFUSES
------------------------
1. embedding an authored body as a page's content (``assert_embeddable``);
2. filing an authored body into L1 under a fetched-looking origin, or letting
   a downstream reader treat an authored L1 record as page text
   (``assert_l1_records_fetched``, ``is_authored_origin``);
3. refreshing a row whose own newest dated claim has expired
   (``assert_claims_fresh``) -- a refresh that re-embeds a stale assertion is
   worse than no refresh, because it restamps ``last_seen``.

It also offers the replacement shape: ``manifest_roster()`` turns the manifest
into what it should always have been -- a URL and refresh roster with no
indexable body.

The list of authored origins is DATA (``data/authored-text-origins.v1.json``),
not strings repeated in code: adding a lane is a data edit, and one reader.

A GUARD THAT CANNOT FAIL IS DECORATION. Every ``assert_*`` here raises
``AuthoredTextError``; the CLI returns a non-zero exit code; and a missing or
unreadable policy file is a refusal, never a default.

  python3 authored_text_guard.py scan-manifest --manifest data/refresh_manifest.json
  python3 authored_text_guard.py scan-l1 --path l1/release=freshness/…jsonl.gz
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import re
import sys
from typing import Iterable, Optional

from cytir.core.paths import REPO_ROOT, data_path

APP = str(REPO_ROOT)
DEFAULT_POLICY = str(data_path("authored-text-origins.v1.json"))

POLICY_SCHEMA = "aw.authored-text-origins.v1"


class AuthoredTextError(RuntimeError):
    """A write this module refuses: authored prose standing in for a page."""


# ------------------------------------------------------------------ policy
def load_policy(path: Optional[str] = None) -> dict:
    """Read the authored-origin policy. A missing file is a refusal.

    Failing closed matters more here than anywhere: a guard that silently
    defaults to an empty origin list would let every authored body through
    while still printing a verdict.
    """
    path = path or os.environ.get("AW_AUTHORED_TEXT_POLICY") or DEFAULT_POLICY
    try:
        with open(path, encoding="utf-8") as fh:
            pol = json.load(fh)
    except OSError as exc:
        raise AuthoredTextError(
            f"authored-text policy unreadable at {path}: {exc}. Refusing to "
            "run: an empty origin list would pass every authored body.") from None
    if pol.get("schema") != POLICY_SCHEMA:
        raise AuthoredTextError(
            f"{path}: schema {pol.get('schema')!r} != {POLICY_SCHEMA!r}")
    if not pol.get("authored_l1_origins") and not pol.get("authored_l1_origin_prefixes"):
        raise AuthoredTextError(
            f"{path}: names no authored origins. A policy that forbids nothing "
            "is decoration; remove the guard or fill the policy.")
    pol["_path"] = path
    return pol


def is_authored_origin(origin: Optional[str], policy: dict) -> bool:
    """True when `origin` marks text that was WRITTEN ABOUT a page."""
    if not origin:
        # An L1 record with no origin cannot be shown to be the page. Under
        # this guard "unknown" is not "fetched" -- the records that caused
        # this all predate the field.
        return True
    if origin in set(policy.get("authored_l1_origins") or ()):
        return True
    return any(origin.startswith(p)
               for p in (policy.get("authored_l1_origin_prefixes") or ()))


def authored_manifest_spec(manifest_path: str, policy: dict) -> Optional[dict]:
    """The policy row for a manifest path, matched on basename."""
    base = os.path.basename(manifest_path)
    for spec in policy.get("authored_manifests") or ():
        if os.path.basename(spec.get("path", "")) == base:
            return spec
    return None


# ------------------------------------------------------- dated-claim ageing
#: Month names in the four indexed languages. A current-state body dates itself
#: in the page's own language, so an English-only matcher would read a French
#: statistic as undated and pass it forever.
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "januar": 1, "februar": 2, "maerz": 3, "märz": 3, "mai": 5, "juni": 6,
    "juli": 7, "oktober": 10, "dezember": 12,
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4,
    "juin": 6, "juillet": 7, "août": 8, "aout": 8, "septembre": 9,
    "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12,
    "gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4, "maggio": 5,
    "giugno": 6, "luglio": 7, "agosto": 8, "settembre": 9, "ottobre": 10,
    "novembre_it": 11, "dicembre": 12,
}
_MONTH_RE = "|".join(sorted((m for m in _MONTHS if not m.endswith("_it")),
                            key=len, reverse=True))

_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DMY = re.compile(rf"\b(\d{{1,2}})\.?\s+({_MONTH_RE})\.?\s+(\d{{4}})\b", re.I)
_MDY = re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.I)
_MY = re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{4}})\b", re.I)


def _month(name: str) -> int:
    return _MONTHS[name.lower()]


def dated_claims(text: str) -> list[dict]:
    """Every date the body asserts, most recent last.

    Precision matters: a month-only claim is dated to the LAST day of that
    month, so a body saying "June 2026" is not called stale one day into July.
    Erring towards fresh keeps the guard from refusing work it should not.
    """
    if not text:
        return []
    found: list[dict] = []
    for m in _ISO.finditer(text):
        y, mo, d = (int(x) for x in m.groups())
        try:
            found.append({"claim": m.group(0), "date": dt.date(y, mo, d),
                          "precision": "day"})
        except ValueError:
            continue
    for m in _DMY.finditer(text):
        d, mo, y = int(m.group(1)), _month(m.group(2)), int(m.group(3))
        try:
            found.append({"claim": m.group(0), "date": dt.date(y, mo, d),
                          "precision": "day"})
        except ValueError:
            continue
    for m in _MDY.finditer(text):
        mo, d, y = _month(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            found.append({"claim": m.group(0), "date": dt.date(y, mo, d),
                          "precision": "day"})
        except ValueError:
            continue
    spans = {(m.start(), m.end()) for m in _DMY.finditer(text)}
    spans |= {(m.start(), m.end()) for m in _MDY.finditer(text)}
    for m in _MY.finditer(text):
        if any(s <= m.start() and m.end() <= e for s, e in spans):
            continue
        mo, y = _month(m.group(1)), int(m.group(2))
        nxt = dt.date(y + (mo == 12), (mo % 12) + 1, 1)
        found.append({"claim": m.group(0), "date": nxt - dt.timedelta(days=1),
                      "precision": "month"})
    found.sort(key=lambda c: c["date"])
    return found


def claim_age_days(text: str, *, now: dt.date) -> Optional[int]:
    """Age in days of the body's NEWEST dated claim, or None if it has none.

    The newest claim is the right one: it is the body's own statement about how
    current it is. An older date further up the text is history, not staleness.
    """
    claims = dated_claims(text)
    if not claims:
        return None
    return (now - claims[-1]["date"]).days


def assert_claims_fresh(text: str, *, now: dt.date, label: str,
                        policy: Optional[dict] = None,
                        max_age_days: Optional[int] = None) -> dict:
    """Refuse a refresh whose body's newest dated claim has expired."""
    policy = policy or load_policy()
    limit = (max_age_days if max_age_days is not None
             else int(policy.get("stale_claim_max_age_days", 30)))
    age = claim_age_days(text, now=now)
    if age is not None and age > limit:
        newest = dated_claims(text)[-1]
        raise AuthoredTextError(
            f"{label}: body's newest dated claim {newest['claim']!r} is {age} "
            f"days old (limit {limit}). A refresh may not re-embed an expired "
            f"current-state assertion and restamp last_seen; re-derive the "
            f"text from a fetch or do not write the row.")
    return {"verdict": "FRESH", "age_days": age, "limit_days": limit,
            "label": label}


# ------------------------------------------------------------ the refusals
def assert_embeddable(entry: dict, *, label: str, policy: Optional[dict] = None,
                      text_field: str = "text") -> None:
    """Refuse to turn an authored body into a page's embedding input.

    `entry` is a would-be row: it must say where its text came from, in
    ``text_origin`` (or ``l1_origin``). A row that cannot name a fetched origin
    is refused -- that is the whole lesson, and defaulting it to "probably
    fetched" is how 966 rows of prose reached production.
    """
    policy = policy or load_policy()
    origin = entry.get("text_origin") or entry.get("l1_origin")
    if is_authored_origin(origin, policy):
        raise AuthoredTextError(
            f"{label}: refusing to embed {entry.get('url')!r} -- its "
            f"{text_field!r} has origin {origin!r}, which is authored prose "
            f"about the page, not the page. Fetch it, persist the fetch as L1, "
            f"and take the text from there; or do not write the row. "
            f"(policy {policy['_path']})")


def assert_chunk_embeddable(chunk: Iterable[dict], *, label: str,
                            policy: Optional[dict] = None) -> dict:
    """Refuse a whole embed batch if any member carries an authored body.

    Batch-level rather than row-level on purpose: the writer embeds a chunk in
    one call, so a per-row skip would write a partial batch and report success.
    The accounting is returned so the caller can log input beside output.
    """
    policy = policy or load_policy()
    rows = list(chunk)
    for entry in rows:
        assert_embeddable(entry, label=label, policy=policy)
    return {"verdict": "FETCHED", "checked": len(rows), "admitted": len(rows),
            "refused": 0, "label": label}


def assert_l1_records_fetched(records: Iterable[dict], *, label: str,
                              policy: Optional[dict] = None) -> dict:
    """Refuse an L1 batch that carries authored bodies.

    Returns the accounting -- checked, admitted, refused -- because a stage
    that reports success without its input count beside it is how rows stop
    being mentioned (corpus-doctrine 2).
    """
    policy = policy or load_policy()
    checked = 0
    refused: list[dict] = []
    for r in records:
        checked += 1
        origin = r.get("l1_origin") or r.get("text_origin")
        if is_authored_origin(origin, policy):
            refused.append({"url": r.get("url") or r.get("canonical_url"),
                            "l1_origin": origin})
    if refused:
        head = ", ".join(str(x["url"]) for x in refused[:3])
        raise AuthoredTextError(
            f"{label}: {len(refused)} of {checked} L1 record(s) carry an "
            f"authored origin ({head}{'…' if len(refused) > 3 else ''}). "
            f"An authored body may be stored as a note, never as the page's "
            f"L1 document -- a rebuild replays L1 as content. "
            f"(policy {policy['_path']})")
    return {"verdict": "FETCHED", "checked": checked, "admitted": checked,
            "refused": 0, "label": label}


def fetched_text_for(url: str, l1_records: Iterable[dict],
                     policy: Optional[dict] = None) -> Optional[str]:
    """The page's own text from a persisted L1 record, or None.

    None means "do not write the row". It never means "fall back to the
    manifest body": that fallback is the defect.
    """
    policy = policy or load_policy()
    for r in l1_records:
        if r.get("url") != url and r.get("canonical_url") != url:
            continue
        if is_authored_origin(r.get("l1_origin"), policy):
            continue
        text = r.get("markdown") or r.get("text")
        if text and str(text).strip():
            return str(text)
    return None


# --------------------------------------------------------------- the roster
def manifest_roster(manifest: dict, *, manifest_path: str = "data/refresh_manifest.json",
                    policy: Optional[dict] = None) -> list[dict]:
    """The manifest as it should be used: a URL and refresh roster, no body.

    Every indexable text field named by the policy is DROPPED, not blanked:
    a present-but-empty ``text`` is an invitation to fall back to it.
    """
    policy = policy or load_policy()
    spec = authored_manifest_spec(manifest_path, policy)
    drop = set((spec or {}).get("text_fields") or ("text", "snippet"))
    keep = (spec or {}).get("roster_fields")
    out = []
    for e in manifest.get("sources") or ():
        row = {k: v for k, v in e.items() if k not in drop}
        if keep:
            row = {k: v for k, v in row.items() if k in keep}
        out.append(row)
    return out


# ------------------------------------------------------------------- CLI
def _scan_manifest(args, policy) -> int:
    try:
        with open(args.manifest, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except FileNotFoundError:
        # the freshness manifest is operator-supplied and the public cut does not ship it (task 2277)
        print(f"{args.manifest}: manifest not found (operator-supplied input, not shipped in the public "
              "cut); nothing scanned", file=sys.stderr)
        return 2
    spec = authored_manifest_spec(args.manifest, policy)
    if spec is None:
        print(f"{args.manifest}: not listed as an authored manifest — nothing "
              f"to scan under {policy['_path']}")
        return 0
    now = (dt.date.fromisoformat(args.now) if args.now else dt.date.today())
    fields = spec.get("text_fields") or ["text"]
    rows, stale, undated = [], 0, 0
    for e in manifest.get("sources") or ():
        body = " ".join(str(e.get(f) or "") for f in fields)
        age = claim_age_days(body, now=now)
        if age is None:
            undated += 1
        elif age > int(args.max_age_days or policy["stale_claim_max_age_days"]):
            stale += 1
            rows.append({"url": e.get("url"), "age_days": age,
                         "newest_claim": dated_claims(body)[-1]["claim"]})
    report = {"schema": "aw.authored-text-scan.v1", "manifest": args.manifest,
              "policy": policy["_path"], "as_of": now.isoformat(),
              "entries": len(manifest.get("sources") or ()),
              "indexable_text_fields": fields,
              "with_expired_claim": stale, "undated": undated,
              "expired": rows}
    if args.out:
        json.dump(report, open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "expired"},
                     indent=1))
    print(f"{stale} of {report['entries']} entries carry an expired dated "
          f"claim as of {now}.")
    # The manifest exists and carries indexable bodies: that alone is the
    # finding this command reports, so it exits non-zero even at age 0.
    return 1 if (stale or fields) else 0


def _scan_l1(args, policy) -> int:
    checked = refused = 0
    bad = []
    for path in args.path:
        op = gzip.open if path.endswith(".gz") else open
        with op(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                checked += 1
                if is_authored_origin(r.get("l1_origin"), policy):
                    refused += 1
                    if len(bad) < 20:
                        bad.append({"url": r.get("url"),
                                    "l1_origin": r.get("l1_origin")})
    print(json.dumps({"checked": checked, "authored": refused,
                      "sample": bad}, ensure_ascii=False, indent=1))
    return 1 if refused else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--policy", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("scan-manifest")
    m.add_argument("--manifest", required=True)
    m.add_argument("--now", default=None, help="YYYY-MM-DD (default: today)")
    m.add_argument("--max-age-days", type=int, default=None)
    m.add_argument("--out", default=None)
    s = sub.add_parser("scan-l1")
    s.add_argument("--path", action="append", required=True)
    args = ap.parse_args(argv)
    policy = load_policy(args.policy)
    if args.cmd == "scan-manifest":
        return _scan_manifest(args, policy)
    return _scan_l1(args, policy)


if __name__ == "__main__":
    sys.exit(main())

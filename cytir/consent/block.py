#!/usr/bin/env python3
"""Consent-block guard — refuse to crawl a host whose consent decision is on record.

WHY (task 433): the retired `ingest_vd` driver (deleted, task 1552) carried, in its own docstring, a 2026-07-09
CONSENT-BLOCKED determination for prestations.vd.ch (robots.txt 403 +
X-Robots-Tag noindex) together with a "DO NOT RUN this ingester" instruction --
and then nothing in the code enforced it. The block lived in prose only; a
single `python ingest_vd.py ingest` (the retired driver, deleted in task 1552) would have crawled a source that declined.
`cytir.sources.families.de_laender_juris` had the same shape of decision recorded for five
Laender, enforced only implicitly by their absence from a dict -- and its arg
parser SILENTLY DROPPED unknown Laender codes rather than refusing, so the
failure mode of a re-added host was quiet, not loud.

This module makes those recorded decisions executable. It is enforcement, not
policy: every entry in data/consent_declined_hosts.json mirrors a determination
already documented in a script docstring or task doc.

Design:
  * NO NETWORK. This guard never resolves consent; it only reads what was
    already decided. Re-resolving a host that previously declined is a human
    decision, and a guard that phoned home would make "am I allowed?" a
    question we ask by knocking on the door we were told to stay away from.
  * FAILS CLOSED. Unreadable/corrupt registry raises rather than allowing.
    A consent guard that degrades to "allow" is worse than no guard.
  * Registry is DATA (data/consent_declined_hosts.json), not literals in
    ingesters -- the list is referenced by more than one script.

Usage in an ingester, before the first request to a host:

    from cytir.consent.block import assert_crawlable, ConsentBlocked
    assert_crawlable("prestations.vd.ch")     # raises ConsentBlocked

Override: there is deliberately no env-var bypass. Un-blocking a host means
editing the registry via PR, which is the ratification path for policy.
"""
from __future__ import annotations

import json
import os

__all__ = ["ConsentBlocked", "ConsentRegistryError", "assert_crawlable",
           "lookup", "is_blocked", "registry_path", "load_registry"]

# Config is data: overridable, defaulted in exactly one place.
# cytir/consent/block.py -> <repo>/data (moved from pipeline/, task 677: one level deeper).
_DEFAULT_REGISTRY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "consent_declined_hosts.json")

# States that forbid crawling. "unknown" is included on purpose: consent that
# could not be established is not consent.
BLOCKING_STATES = ("declined", "unknown")


class ConsentBlocked(RuntimeError):
    """Raised when an ingester targets a host with a recorded block."""


class ConsentRegistryError(RuntimeError):
    """Raised when the registry cannot be read or is malformed (fail closed)."""


def registry_path() -> str:
    """Path to the consent registry (env-overridable, one default)."""
    return os.path.normpath(
        os.environ.get("AW_CONSENT_REGISTRY", _DEFAULT_REGISTRY))


def load_registry(path: str | None = None) -> dict:
    """Load the registry. Raises ConsentRegistryError rather than defaulting."""
    p = path or registry_path()
    try:
        with open(p, encoding="utf-8") as fh:
            doc = json.load(fh)
    except FileNotFoundError as e:
        raise ConsentRegistryError(
            f"consent registry not found at {p!r}; refusing to crawl without "
            f"it (fail closed). Set AW_CONSENT_REGISTRY if it moved.") from e
    except (OSError, ValueError) as e:
        raise ConsentRegistryError(
            f"consent registry at {p!r} is unreadable/malformed: {e}") from e
    hosts = doc.get("hosts")
    if not isinstance(hosts, dict):
        raise ConsentRegistryError(
            f"consent registry at {p!r} has no 'hosts' object")
    return hosts


def _normalise(host: str) -> str:
    """Bare lowercase hostname: strip scheme, path, port, userinfo, trailing dot."""
    h = (host or "").strip().lower()
    if "://" in h:
        h = h.split("://", 1)[1]
    h = h.split("/", 1)[0].split("?", 1)[0]
    if "@" in h:
        h = h.rsplit("@", 1)[1]
    if h.startswith("["):                      # IPv6 literal
        h = h.split("]", 1)[0] + "]"
    elif ":" in h:
        h = h.split(":", 1)[0]
    return h.rstrip(".")


def _candidates(host: str):
    """The host and its registrable parents, so a block on a domain covers
    its subdomains (a decline at landesrecht.rlp.de covers www.landesrecht.rlp.de)."""
    h = _normalise(host)
    if not h:
        return []
    out = [h]
    if h.startswith("www."):
        out.append(h[4:])
    parts = h.split(".")
    for i in range(1, len(parts) - 1):
        out.append(".".join(parts[i:]))
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def lookup(host: str, path: str | None = None) -> tuple[str, dict] | None:
    """Return (matched_host, entry) if `host` or a parent is registered, else None."""
    hosts = load_registry(path)
    for cand in _candidates(host):
        entry = hosts.get(cand)
        if isinstance(entry, dict):
            return cand, entry
    return None


def is_blocked(host: str, path: str | None = None) -> bool:
    """True if `host` has a recorded blocking consent state."""
    hit = lookup(host, path)
    return bool(hit and hit[1].get("state") in BLOCKING_STATES)


def assert_crawlable(host: str, path: str | None = None) -> None:
    """Raise ConsentBlocked if `host` has a recorded blocking consent state.

    Call this BEFORE the first network request to `host`.
    """
    hit = lookup(host, path)
    if not hit:
        return
    matched, entry = hit
    state = entry.get("state")
    if state not in BLOCKING_STATES:
        return
    raise ConsentBlocked(
        f"CONSENT-BLOCKED: refusing to crawl {_normalise(host)!r} "
        f"(matched registry entry {matched!r}, state={state!r}, "
        f"decided {entry.get('decided_on')}).\n"
        f"  basis: {entry.get('basis')}\n"
        f"  recorded in: {entry.get('recorded_in')}\n"
        f"  un-block condition: {entry.get('unblock_condition')}\n"
        f"This is an enforcement of a decision already on record, not a bug. "
        f"Do not work around it, and do not re-resolve consent for a host that "
        f"declined without a ruling -- edit {registry_path()} via PR.")


def main(argv: list[str] | None = None) -> int:      # tiny CLI for operators
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 1:
        print("usage: consent_block.py <host> [<host>...]", file=sys.stderr)
        return 2
    rc = 0
    for h in argv:
        try:
            assert_crawlable(h)
            print(f"OK        {h}")
        except ConsentBlocked as e:
            rc = 1
            print(f"BLOCKED   {h}\n{e}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

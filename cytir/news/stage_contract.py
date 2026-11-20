#!/usr/bin/env python3
"""Fail-closed boundary checks for the recurring news staging worker.

``sitemap_ingest`` correctly records an unreachable publisher as ``error`` so
that an outage cannot masquerade as an empty feed.  The scheduler must turn
that state into a failed *run*, rather than embedding the other publishers and
leaving an ambiguous partial handoff.  This dependency-free helper is invoked
before projection/embedding and before the stage manifest is written.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from cytir.news import publishers as REG


#: The daily run's REQUIRED publishers (registry role `daily_required`); `ALL_PUBLISHERS` adds the
#: backfill-only hosts, which only an explicit AW_NEWS_EXPECTED_HOSTS may name.
PUBLISHERS = REG.hosts(daily=True)
ALL_PUBLISHERS = REG.hosts()
VALID_PUBLISHER_STATES = {"ok", "empty"}

#: host -> staging arm, read from the registry (`cytir/news/publishers.v1.json`), the ONE home of this
#: grouping. `news_refresh_stage.sh` asks this module (`routes`, `status-hosts` subcommands), as the
#: apply driver does (`arms`); `daily._ARM_HOST_MAP` is generated from the same rows.
HOST_ARM = REG.host_arm()
#: Arm order the pack step has always used (`ARMS=(chmedia swissinfo)`).
ARM_ORDER = ("chmedia", "swissinfo")

# The r7 route's default width. Every collection this lane has EVER staged at
# was named without a width suffix -- `aw_dnews_<day>_<arm>` -- so that shape
# stays the width-128 name forever, byte-identical to what already exists.
# A width other than 128 (the flip-day repoint to 1024, task 508 / task 500
# §7.1) targets a DIFFERENT, explicitly named staging collection instead of
# writing 1024-dim vectors into a 128-dim collection under the old name --
# which Qdrant would simply reject on the first upsert (fail loud), but the
# whole point is to never get there.
DEFAULT_NEWS_MRL_DIM = 128

# Tim's ruling, 2026-09-23 ~11:15Z: "in general we probably don't want to be
# continuously re-collecting previously ingested news articles. Maybe a 1
# week overhang is useful but more sounds like it will just slow us down."
# The ONE home for this number -- fetch-time selection (news_recapture_skip.py's
# `apply_recapture_overhang`, called by sitemap_ingest.process_host) and compose-time supersede
# (news_backlog_composer.py's `filter_unchanged_recaptures`) both read it
# rather than each carrying their own literal `7`. An article already served
# in prod is re-fetched/re-packed only if its `published` date is within
# this many days of "now" (catches corrections/updates); older than this and
# already served, it is skipped -- at fetch time, not fetched-then-dropped.
# A url NOT yet in prod is never subject to this window (a backfill catching
# up on a genuinely missing old article is not a "re-collection").
DEFAULT_NEWS_RECAPTURE_OVERHANG_DAYS = 7


def news_stage_collection_name(day: str, label: str, mrl_dim: int) -> str:
    """The one place the news lane's per-day, per-arm collection name is
    computed -- called from both `news_refresh_stage.sh` (which stages) and
    task 452's `news_tier2_apply.sh` (which reads the same staged
    collection back for sealing/apply). Two independent shell re-derivations
    of this string is exactly the fork this project's doctrine forbids for a
    mirror contract (engineering-doctrine #6); this is the shared source.
    """
    if not isinstance(mrl_dim, int) or mrl_dim <= 0:
        raise ValueError(f"mrl_dim must be a positive int, got {mrl_dim!r}")
    compact_day = day.replace("-", "")
    suffix = "" if mrl_dim == DEFAULT_NEWS_MRL_DIM else str(mrl_dim)
    return f"aw_dnews{suffix}_{compact_day}_{label}"


class NewsStageContractError(RuntimeError):
    pass


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise NewsStageContractError(f"unreadable JSON evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise NewsStageContractError(f"JSON evidence is not an object: {path}")
    return value


def parse_expected_hosts(value: str | None) -> tuple[str, ...]:
    """`None`/empty -> the daily run's required publishers (its strict default).
    Otherwise a comma list, every name a registered publisher (daily or backfill-only):
    an unknown name refuses, so a typo can never shrink the boundary to nothing."""
    if value is None or not value.strip():
        return PUBLISHERS
    hosts = tuple(dict.fromkeys(h.strip() for h in value.split(",") if h.strip()))
    unknown = [h for h in hosts if h not in ALL_PUBLISHERS]
    if not hosts or unknown:
        raise NewsStageContractError(
            f"expected hosts must be a non-empty subset of {list(ALL_PUBLISHERS)}; "
            f"got {value!r} (unknown: {unknown})")
    return hosts


def arms_for_hosts(expected_hosts: tuple[str, ...]) -> tuple[str, ...]:
    """The staging arms the given hosts route to, in the pack step's order."""
    wanted = {HOST_ARM[h] for h in expected_hosts}
    return tuple(a for a in ARM_ORDER if a in wanted)


def validate_sources(state: dict, hn_receipt: dict,
                     expected_hosts: tuple[str, ...] = PUBLISHERS) -> dict:
    """Every EXPECTED publisher must be `ok`/`empty`. `expected_hosts` defaults to
    all four; a run that deliberately covers one publisher (the coverage backfill)
    names it, and only it is checked -- the other hosts are not fetched, so their
    state is not evidence of anything."""
    hosts = state.get("hosts")
    if not isinstance(hosts, dict):
        raise NewsStageContractError("sitemap state has no hosts object")
    invalid = {
        host: (hosts.get(host) or {}).get("status", "missing")
        for host in expected_hosts
        if (hosts.get(host) or {}).get("status") not in VALID_PUBLISHER_STATES
    }
    if invalid:
        raise NewsStageContractError(
            "publisher source boundary incomplete (retry; do not stage partial run): "
            + ", ".join(f"{host}={status}" for host, status in sorted(invalid.items()))
        )
    if hn_receipt.get("completeness") != "time-window-complete":
        raise NewsStageContractError(
            "HN source boundary incomplete (no partial/fallback handoff): "
            f"{hn_receipt.get('completeness', 'missing-receipt')}"
        )
    return {
        "publishers": {host: hosts[host]["status"] for host in expected_hosts},
        "hn": hn_receipt["completeness"],
    }


def assert_staging_ready(url: str) -> dict:
    endpoint = url.rstrip("/") + "/collections"
    try:
        with urlopen(endpoint, timeout=10) as response:  # nosec B310: configured local endpoint
            value = json.loads(response.read())
    except (OSError, URLError, json.JSONDecodeError) as exc:
        raise NewsStageContractError(
            f"staging Qdrant unavailable at {endpoint}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(value, dict) or value.get("status") != "ok":
        raise NewsStageContractError(f"staging Qdrant unhealthy response at {endpoint}")
    return {"endpoint": endpoint, "collections": len(value.get("result", {}).get("collections", []))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sources = sub.add_parser("source-boundary")
    sources.add_argument("--state", type=Path, required=True)
    sources.add_argument("--hn-receipt", type=Path, required=True)
    sources.add_argument("--expected-hosts", default=None,
                         help="comma list of publisher hosts this run covers "
                              "(default: all four)")
    arms = sub.add_parser("arms", help="staging arms the expected hosts route to")
    arms.add_argument("--expected-hosts", default=None)
    sub.add_parser("routes", help="JSON {bare host: arm} of every registered publisher (stage script)")
    status = sub.add_parser("status-hosts", help="hosts the stage manifest's source_status reports")
    status.add_argument("--expected-hosts", default=None)
    ready = sub.add_parser("staging-ready")
    ready.add_argument("--url", required=True)
    name = sub.add_parser("collection-name",
                          help="the one place the news lane's per-day, "
                               "per-arm staging collection name is computed")
    name.add_argument("--day", required=True, help="YYYY-MM-DD")
    name.add_argument("--label", required=True, help="arm, e.g. chmedia/swissinfo/hn")
    name.add_argument("--mrl-dim", type=int, required=True)
    args = parser.parse_args()
    try:
        if args.command == "source-boundary":
            result = validate_sources(load_json(args.state), load_json(args.hn_receipt),
                                      parse_expected_hosts(args.expected_hosts))
        elif args.command == "arms":
            print(" ".join(arms_for_hosts(parse_expected_hosts(args.expected_hosts))))
            return
        elif args.command == "routes":
            print(json.dumps(REG.routes()))
            return
        elif args.command == "status-hosts":
            expected = parse_expected_hosts(args.expected_hosts) if args.expected_hosts else ()
            print(" ".join(REG.status_hosts(expected)))
            return
        elif args.command == "collection-name":
            print(news_stage_collection_name(args.day, args.label, args.mrl_dim))
            return
        else:
            result = assert_staging_ready(args.url)
        print(json.dumps({"schema": "aw.news-stage-contract.v1", "status": "PASS", **result}, sort_keys=True))
    except NewsStageContractError as exc:
        raise SystemExit(f"FATAL news stage contract: {exc}") from exc


if __name__ == "__main__":
    main()

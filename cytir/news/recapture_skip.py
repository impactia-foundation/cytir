#!/usr/bin/env python3
"""Fetch-time half of the news re-collection ruling (plan.md ruling 8, Tim
2026-09-23 11:15Z): an article already ingested is not fetched again once it is
older than `news_recapture_overhang_days` (`news_stage_contract`).

The other half lives at compose time (`news_backlog_composer.
filter_unchanged_recaptures`): it drops the same urls AFTER they were fetched,
embedded and staged. Until this module, the fetch step re-fetched the newest
1,500 urls of every publisher every day (SRF: the same 1,500 undated urls, 1,880
of them dropped as `undated_dropped` on 2026-09-28) and a same-day run would have
re-fetched everything again.

What "already ingested" and "older" mean here
---------------------------------------------
* Ingested = the url is stored in the serving twin collection (the composer's own
  test), joined by the canonical url, never by string equality: the selected url
  and the stored one differ by scheme and `www.` (`upsert_supersede.url_variants`,
  `group_key`).
* Older = the newest `published` DAY carried by any stored point of that url is
  more than `overhang_days` before the run's reference day. The STORED date is used,
  not the sitemap's: SRF's sitemaps carry no `lastmod`, and the composer will see
  the page date of a recapture, which for an unchanged article is the stored one.
  A stored url with no readable day (none, a bare year or year-month) has an
  unknown age and is FETCHED: the composer decides it later. Unknown is never
  treated as old.
* A url that is not stored is never skipped (a backfill catching up on a missing
  article is not a re-collection), and neither is one dated in the future.

Every skip is accounted: the caller writes each one (url, stored day, age) to a
`recapture_skipped.jsonl` beside the stage state, and the host's counts carry it.
A lookup that fails is an error for the host (the stage boundary then fails and
retries), never an empty answer: "could not look" must not resolve to "fetch
everything" or to "skip everything".
"""
from __future__ import annotations

import datetime as _dt
import os
import re

from cytir.core.published import normalize_published
from cytir.index import supersede as US
from cytir.news import stage_contract as NSC

_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class RecaptureLookupError(RuntimeError):
    """The served-url lookup failed; the caller must not fall back silently."""


def stored_day(payloads: list[dict]) -> _dt.date | None:
    """The newest full day among the stored points' `published`, in any spelling
    `cytir.core.published` reads. None when no point names a day."""
    days = []
    for pl in payloads:
        value = normalize_published(pl.get("published"))
        if value and _DAY.match(value):
            days.append(_dt.date.fromisoformat(value))
    return max(days) if days else None


def served_days_by_group(client, collection: str, urls: list[str]) -> dict:
    """{canonical url group -> newest stored day or None} for every url that has at
    least one stored point in `collection`."""
    try:
        by_stored = US.served_payloads_by_url(
            client, collection, urls, ("published",))
    except Exception as exc:  # noqa: BLE001 -- re-raised as the one typed refusal
        raise RecaptureLookupError(
            f"served-url lookup on {collection} failed: {type(exc).__name__}: {exc}"
        ) from exc
    groups: dict[str, list[dict]] = {}
    for stored_url, payloads in by_stored.items():
        groups.setdefault(US.group_key(stored_url), []).extend(payloads)
    return {group: stored_day(payloads) for group, payloads in groups.items()}


def apply_recapture_overhang(
        selected: list[dict], served_days: dict, *, now_ref: _dt.date,
        overhang_days: int = NSC.DEFAULT_NEWS_RECAPTURE_OVERHANG_DAYS,
) -> tuple[list[dict], list[dict]]:
    """Split `selected` (sitemap rows with a `url`) into (to_fetch, skipped).

    `served_days` is `served_days_by_group`'s answer. A skipped entry is
    `{"url", "stored_published", "age_days", "overhang_days", "reason"}`. Order of
    `to_fetch` is the order of `selected`; nothing is lost: len(to_fetch) +
    len(skipped) == len(selected)."""
    if overhang_days < 0:
        raise ValueError("overhang_days must not be negative")
    keep: list[dict] = []
    skipped: list[dict] = []
    for row in selected:
        group = US.group_key(row["url"])
        if group not in served_days:            # not ingested: never skipped
            keep.append(row)
            continue
        day = served_days[group]
        if day is None:                         # ingested, age unknown: fetch
            keep.append(row)
            continue
        age = (now_ref - day).days
        if age > overhang_days:
            skipped.append({"url": row["url"], "stored_published": day.isoformat(),
                            "age_days": age, "overhang_days": overhang_days,
                            "reason": "served_older_than_overhang"})
        else:                                   # inside the overhang (or future)
            keep.append(row)
    return keep, skipped


def open_client(url: str):
    from cytir.core.qdrant import qdrant_client  # noqa: PLC0415 -- the one client factory
    return qdrant_client(url, timeout=120, check_compatibility=False)


def skip_served(selected: list[dict], *, twin_url: str, collection: str,
                now_ref: _dt.date, overhang_days: int, client=None):
    """The one call `sitemap_ingest.process_host` makes. Returns
    (to_fetch, skipped); raises RecaptureLookupError on a failed lookup."""
    if not selected:
        return [], []
    client = client or open_client(twin_url)
    served = served_days_by_group(client, collection, [r["url"] for r in selected])
    return apply_recapture_overhang(
        selected, served, now_ref=now_ref, overhang_days=overhang_days)


def settings_from_env(environ=None) -> dict | None:
    """The twin the stage already knows (`AW_NEWS_TWIN_URL`/`AW_NEWS_TWIN_COLLECTION`,
    the variables the pack step reads too). None when either is unset: a stage run
    with no twin configured does no fetch-time skip and says so."""
    e = os.environ if environ is None else environ
    url, coll = e.get("AW_NEWS_TWIN_URL", ""), e.get("AW_NEWS_TWIN_COLLECTION", "")
    return {"twin_url": url, "collection": coll} if url and coll else None

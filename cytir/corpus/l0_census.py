"""L0 census: how far is a served population from "rebuildable without refetch"? (task 1036)

ADR-026: a rebuild is replay L0 -> L1 -> L2, no refetch. ADR-093 counted 492,211 served pages
with no text on disk. This module answers the same question per source family for a SAMPLE of
served urls, by asking the persisted-input index (``cytir.corpus.persisted``, ADR-105) which layer
holds each page.

Every sampled page ends in exactly ONE terminal, the strongest rebuild basis it has:

  ``l0``    raw bytes as fetched sit in a registered L0 store, in a framework-sealed L0 tree, or
            inside a sealed upstream dump the caller can test per page (``dump_l0``);
  ``warc``  no local L0, but a Common Crawl WARC locator is known: CC is the durable L0
            (``warc_keys``, page keys of captures with a locator);
  ``l1``    only extracted text/markdown (a weaker but real basis; re-chunk is local);
  ``l2``    only passages (rebuildable by the ADR-093 overlap-chaining route, never a replay);
  ``none``  nothing on disk: only a refetch brings the page back (a fresh consent question).

Pure functions plus a tiny accumulator; no IO beyond ``sample_urls`` and ``roster_from_index``.
The roster of stores comes from the index's own ``scanned_files`` table (a manifest of what was
globbed), never from a hand list.
"""
from __future__ import annotations

import gzip
import json
import math
import os
from collections import defaultdict
from typing import Callable, Iterator, Optional

from cytir.corpus.persisted import page_keys

TERMINALS = ("l0", "warc", "l1", "l2", "none")


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for k successes in n draws (95 % by default)."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def _any_key(url: str, keys) -> bool:
    return bool(keys) and any(k in keys for k in page_keys(url))


def classify(url: str, *, index, extra_l0=frozenset(), warc_keys=frozenset(),
             dump_l0: Optional[Callable[[str], bool]] = None,
             pack_text: bool = False) -> str:
    """The strongest rebuild basis of one page (see the module docstring).

    ``pack_text`` is True for a row sampled from a sealed pack's own chunk files: its passage
    text is on disk by construction, so a page the index does not know is ``l2``, not ``none``."""
    if index.lookup(url, order=("l0",)) is not None:
        return "l0"
    if _any_key(url, extra_l0):
        return "l0"
    if dump_l0 is not None and dump_l0(url):
        return "l0"
    if _any_key(url, warc_keys):
        return "warc"
    if index.lookup(url, order=("l1",)) is not None:
        return "l1"
    if pack_text or index.lookup(url, order=("l2",)) is not None:
        return "l2"
    return "none"


class Census:
    """Per-family terminal counts with 95 % Wilson intervals."""

    def __init__(self) -> None:
        self.counts: dict = defaultdict(lambda: dict.fromkeys(TERMINALS, 0))

    def add(self, family: str, terminal: str) -> None:
        if terminal not in TERMINALS:
            raise ValueError(f"unknown terminal {terminal!r}")
        self.counts[family][terminal] += 1

    def merge(self, other: "Census") -> None:
        for fam, c in other.counts.items():
            for t, v in c.items():
                self.counts[fam][t] += v

    def rows(self, served: Optional[dict] = None) -> list:
        """One record per family: sample size, terminal shares + CIs, and, when ``served`` (the
        family's served-row count) is given, the share scaled to rows."""
        out = []
        for fam in sorted(self.counts):
            c = self.counts[fam]
            n = sum(c.values())
            rec = {"family": fam, "sample": n}
            for t in TERMINALS:
                lo, hi = wilson(c[t], n)
                rec[t] = c[t]
                rec[t + "_pct"] = round(100.0 * c[t] / n, 2) if n else None
                rec[t + "_ci"] = [round(100 * lo, 2), round(100 * hi, 2)]
            if served and fam in served:
                rec["served"] = served[fam]
                rec["none_rows_est"] = round(served[fam] * c["none"] / n) if n else None
            out.append(rec)
        return out


def sample_urls(path: str, limit: int, *, max_files: int = 12) -> Iterator[str]:
    """Up to ``limit`` urls from a pack's rows: a jsonl / jsonl.gz / parquet file, or a directory
    of them (the first ``max_files``, in name order). A row's url is ``url`` or ``payload.url``."""
    files = []
    if os.path.isdir(path):
        for root, _d, names in sorted(os.walk(path)):
            for nm in sorted(names):
                if nm.endswith((".jsonl", ".jsonl.gz", ".parquet")):
                    files.append(os.path.join(root, nm))
            if len(files) >= max_files:
                break
    else:
        files = [path]
    got = 0
    per_file = max(1, limit // max(1, min(len(files), max_files)))
    for f in files[:max_files]:
        n = 0
        if f.endswith(".parquet"):
            import pyarrow.parquet as pq
            for batch in pq.ParquetFile(f).iter_batches(batch_size=2000, columns=["url"]):
                for u in batch.column("url").to_pylist():
                    if u:
                        yield u
                        got += 1
                        n += 1
                        if n >= per_file or got >= limit:
                            break
                if n >= per_file or got >= limit:
                    break
        else:
            op = gzip.open if f.endswith(".gz") else open
            with op(f, "rt", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    u = r.get("url") or (r.get("payload") or {}).get("url")
                    if u:
                        yield u
                        got += 1
                        n += 1
                        if n >= per_file or got >= limit:
                            break
        if got >= limit:
            return


def roster_from_index(conn) -> list:
    """The store roster as the index recorded it: layer -> files, bytes, rows, unreadable files.
    ``conn`` is a read-only sqlite connection to the persisted index (table ``scanned_files``)."""
    cur = conn.execute(
        "SELECT layer, count(*), coalesce(sum(size),0), coalesce(sum(rows),0), "
        "coalesce(sum(err IS NOT NULL),0) FROM scanned_files GROUP BY layer ORDER BY layer")
    return [{"layer": layer, "files": f, "bytes": b, "rows": r, "unreadable_files": e}
            for layer, f, b, r, e in cur.fetchall()]

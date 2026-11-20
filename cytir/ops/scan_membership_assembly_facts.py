"""One pass over the membership snapshot for every fact the assembly gate needs.

READ-ONLY over the corpus; writes only under ``--out``.

Two guard obligations both require walking all 345 sealed membership shards,
and the corpus spindle is shared with live production lanes.  Walking it twice
is not a neutral choice: at the ~20 MB/s the disk sustains under contention, a
second pass is another hour of IO taken from a lane that needs it.  So both
jobs ride the same read.

**Job A -- named-slice claims (guards G1, G2).**  Seven sealed 1024-dim sets are
subtracted from the ADR-092 remainder by a *payload predicate*
(``census_r0_remainder.py::named_slice``: domain, source_type, URL prefix)
rather than by an enumerated point witness.  Two of the seven also publish a
real witness; five do not.  A predicate is weaker than the "counts only" case
guard G1 already refuses -- it is a rule, not a claim, and two rules can select
the same point with nothing raising.  Task 456 found the 176 arxiv points task
427 also claims only by *elimination*, because there was no arxiv claim set to
intersect against.  This enumerates each predicate once and seals the result.

**Job B -- tombstoned-URL cohabitation (guard G4, ADR-101a Correction 2).**  A
URL bearing an ADR-091 natural-person tombstone accepts no replacement rows.
The rule is at URL granularity and no claim artifact in the assembly carries a
URL -- they are all bare point-ID lists -- so the join has to come from
membership.  ``nunige.ch`` and ``ivan-herman.net`` are personal homepages that
both an entity card and a crawl covered, so "the crawl lanes are probably
disjoint from the card tombstones" is a guess, not a measurement.

The ``named_slice`` rule below is reproduced verbatim from
the retired task-417 census script (census_r0_remainder).  It is deliberately duplicated
rather than imported: if the census rule ever drifts, the counts this emits
stop matching the census's own ``named_counts`` and the run refuses.
"""
from __future__ import annotations

import argparse
import gzip
from cytir.acquire.gzip_pinned import deterministic_gzip_open
import hashlib
import io
import json
import multiprocessing as mp
import sys
import uuid
from pathlib import Path

WIKIPEDIA_DOMAINS = {"en.wikipedia.org", "de.wikipedia.org",
                     "fr.wikipedia.org", "it.wikipedia.org"}

#: The five slices that publish no point witness.  wikipedia and gesetze do,
#: and the census cross-checks those against the same rule.
UNWITNESSED = ("arxiv", "bfs-pxweb", "legifrance-codes", "pubmed",
               "swisstopo-v2")

TOMBSTONE_BIT = 0

_SWIS_RESIDUAL: set[str] = set()
_TRACKED: dict[bytes, int] = {}
_MEMBERSHIP: Path | None = None
_SLICE_INDEX: dict[str, int] = {name: index
                                for index, name in enumerate(UNWITNESSED)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def named_slice(point_id: str, payload: dict) -> str | None:
    """Byte-for-byte the rule in the retired task-417 census script (census_r0_remainder)."""
    domain = str(payload.get("domain") or "").lower()
    source_type = str(payload.get("source_type") or "")
    url = str(payload.get("url") or payload.get("canonical_url") or "")
    if domain in WIKIPEDIA_DOMAINS:
        return "wikipedia"
    if domain == "swisstopo.admin.ch" and source_type == "gov-geo":
        return None if point_id in _SWIS_RESIDUAL else "swisstopo-v2"
    if (domain == "bfs.admin.ch" and source_type == "gov-stat"
            and url.startswith("https://www.pxweb.bfs.admin.ch/api/")):
        return "bfs-pxweb"
    if domain == "arxiv.org":
        return "arxiv"
    if domain == "pubmed.ncbi.nlm.nih.gov":
        return "pubmed"
    if domain in {"gesetze-im-internet.de", "www.gesetze-im-internet.de"}:
        return "gesetze"
    if (domain == "legifrance.gouv.fr" and source_type == "official"
            and url.startswith("https://www.legifrance.gouv.fr/codes/article_lc/")):
        return "legifrance-codes"
    return None


def read_point_ids(path: Path) -> set[bytes]:
    out: set[bytes] = set()
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if isinstance(row, dict):
                # producers publish bare strings, {"point_id": ...}, or the
                # task-427 stage's {"old_point_id": ..., "url": ...}
                value = row.get("point_id") or row.get("old_point_id")
                if value is None:
                    raise SystemExit(f"no point ID in a row of {path}")
            else:
                value = row
            out.add(uuid.UUID(str(value)).bytes)
    return out


def write_point_ids(path: Path, point_ids: list[str]) -> None:
    """Deterministic gzip: fixed mtime, so the digest is reproducible."""
    with path.open("wb") as handle:
        with deterministic_gzip_open(handle, compresslevel=6) as raw_out:
            with io.TextIOWrapper(raw_out, encoding="utf-8") as target:
                for point_id in point_ids:
                    target.write(
                        json.dumps({"point_id": point_id}, sort_keys=True) + "\n")


def _init(residual: set[str], tracked: dict[bytes, int],
          membership: Path) -> None:
    global _SWIS_RESIDUAL, _TRACKED, _MEMBERSHIP
    _SWIS_RESIDUAL = residual
    _TRACKED = tracked
    _MEMBERSHIP = membership


def scan_shard(job: tuple[str, int]):
    """Return (rows, per-slice point IDs, per-URL label bitmask)."""
    shard_file, expect_rows = job
    slices: list[list[bytes]] = [[] for _ in UNWITNESSED]
    per_url: dict[str, int] = {}
    rows = 0
    with gzip.open(_MEMBERSHIP / shard_file, "rt", encoding="utf-8") as source:
        for line in source:
            rows += 1
            if line[:7] != '{"id":"':
                raise SystemExit(f"unexpected membership row shape in {shard_file}")
            text_id = line[7:43]
            key = uuid.UUID(text_id).bytes
            bits = _TRACKED.get(key)
            # Parsing the payload is the expensive half; do it only for a row
            # one of the two jobs actually needs.
            if bits is None and not _cheap_prefilter(line):
                continue
            payload = json.loads(line).get("payload") or {}
            if bits is not None:
                url = str(payload.get("url") or payload.get("canonical_url") or "")
                per_url[url] = per_url.get(url, 0) | bits
            name = named_slice(text_id, payload)
            index = _SLICE_INDEX.get(name)
            if index is not None:
                slices[index].append(key)
    if rows != expect_rows:
        raise SystemExit(f"membership shard row drift: {shard_file}")
    return rows, slices, per_url


#: Every named-slice domain, as a substring test on the raw line.  A false
#: positive costs one JSON parse; a false negative would lose a claim, so the
#: test is deliberately loose and the authoritative decision stays with
#: ``named_slice`` on the parsed payload.
_PREFILTER = ('"arxiv.org"', '"pubmed.ncbi.nlm.nih.gov"',
              '"swisstopo.admin.ch"', '"bfs.admin.ch"',
              '"legifrance.gouv.fr"')


def _cheap_prefilter(line: str) -> bool:
    return any(needle in line for needle in _PREFILTER)


def main(argv=None, prog=None) -> int:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--tombstones", required=True,
                        help="corpus-relative ADR-091 tombstone point IDs")
    parser.add_argument("--track", action="append", default=[],
                        metavar="LABEL=corpus-relative-path")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifact-prefix", default="_work/task462/named-slices",
                        help="corpus-relative directory the emitted claim "
                             "files are published under; the URIs resolve "
                             "against it, so downstream stages can read them "
                             "without knowing where the scan ran")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)

    root: Path = args.corpus_root
    args.out.mkdir(parents=True, exist_ok=True)
    membership = root / "_preservation/aw_e3-membership-20260814"
    census_run = root / "_work/task417/r0-remainder-census-v31/RUN.json"

    residual: set[str] = set()
    residual_path = (root / "l1/release=aw-e3-rebuild-20260815"
                     / "source=swisstopo-structured-v3-exact-witness"
                     / "selector-residual.jsonl.gz")
    if residual_path.exists():
        with gzip.open(residual_path, "rt", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                point = row.get("point_id") or row.get("id")
                if point:
                    residual.add(str(point))
    print(f"[scan] swisstopo selector residual {len(residual):,}", flush=True)

    tombstone_path = root / args.tombstones.removeprefix("artifact://")
    tracked: dict[bytes, int] = {}
    labels = ["__tombstone__"]
    tracked_inputs = [{"label": "__tombstone__", "uri": args.tombstones,
                       "sha256": sha256_file(tombstone_path)}]
    for key in read_point_ids(tombstone_path):
        tracked[key] = 1 << TOMBSTONE_BIT
    print(f"[scan] tombstones {len(tracked):,}", flush=True)
    for spec in args.track:
        label, separator, rel = spec.partition("=")
        if not separator:
            raise SystemExit("--track needs LABEL=corpus-relative-path")
        if label in labels:
            raise SystemExit(f"duplicate track label: {label}")
        path = root / rel.removeprefix("artifact://")
        bit = 1 << len(labels)
        labels.append(label)
        tracked_inputs.append({"label": label, "uri": rel,
                               "sha256": sha256_file(path)})
        ids = read_point_ids(path)
        for key in ids:
            tracked[key] = tracked.get(key, 0) | bit
        print(f"[scan] {label:<30} {len(ids):>9,} ids "
              f"(tracked {len(tracked):,})", flush=True)

    receipt = json.loads((membership / "RECEIPT.json").read_text())
    jobs = [(shard["file"], shard["rows"]) for shard in receipt["shards"]]
    collected: list[list[bytes]] = [[] for _ in UNWITNESSED]
    merged: dict[str, int] = {}
    scanned = 0
    with mp.Pool(args.workers, initializer=_init,
                 initargs=(residual, tracked, membership)) as pool:
        for index, (rows, slices, per_url) in enumerate(
                pool.imap_unordered(scan_shard, jobs), 1):
            scanned += rows
            for bucket, ids in zip(collected, slices):
                bucket.extend(ids)
            for url, bits in per_url.items():
                merged[url] = merged.get(url, 0) | bits
            if index % 25 == 0:
                print(f"[scan] shards {index}/{len(jobs)} scanned={scanned:,} "
                      f"urls={len(merged):,}", flush=True)
    print(f"[scan] scanned={scanned:,} urls touched={len(merged):,}", flush=True)

    # ---- Job A: seal the named-slice claims ------------------------------
    census = json.loads(census_run.read_text())
    named_counts = census["named_counts"]
    manifests = []
    for name, bucket in zip(UNWITNESSED, collected):
        unique = sorted({key for key in bucket})
        if len(unique) != len(bucket):
            raise SystemExit(f"duplicate point ID in slice {name}")
        expected = int(named_counts[name])
        if len(unique) != expected:
            raise SystemExit(
                f"slice {name}: enumerated {len(unique):,} != census "
                f"named_counts {expected:,}; the payload rule has drifted")
        path = args.out / f"claims-{name}.jsonl.gz"
        write_point_ids(path, [str(uuid.UUID(bytes=key)) for key in unique])
        manifests.append({
            "set_id": name, "granularity": "point_id",
            "uri": f"artifact://{args.artifact_prefix.strip('/')}/{path.name}",
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size, "count": len(unique),
            "verified": True,
            "derived_from": "payload predicate census_r0_remainder.py::"
                            "named_slice, applied once to the sealed "
                            "membership snapshot and enumerated",
        })
        print(f"[scan] slice {name:<20} {len(unique):>9,} ids", flush=True)

    (args.out / "named-slice-claims.v1.json").write_text(json.dumps({
        "schema": "aw.assembly-named-slice-claims.v1",
        "producer": "aw ops scan-membership-assembly-facts",
        "membership": {
            "uri": "artifact://_preservation/aw_e3-membership-20260814/RECEIPT.json",
            "sha256": sha256_file(membership / "RECEIPT.json"),
            "points": scanned},
        "census": {"uri": "artifact://_work/task417/r0-remainder-census-v31/RUN.json",
                   "sha256": sha256_file(census_run)},
        "claims": manifests,
        "total_points": sum(row["count"] for row in manifests),
    }, indent=2, sort_keys=True) + "\n")

    # ---- Job B: the tombstoned-URL cohabitation --------------------------
    tombstone_bit = 1 << TOMBSTONE_BIT
    tombstoned_urls = {url for url, bits in merged.items()
                       if bits & tombstone_bit}
    findings: dict[str, list[str]] = {}
    for url in sorted(tombstoned_urls):
        bits = merged[url]
        for index, label in enumerate(labels):
            if index != TOMBSTONE_BIT and bits & (1 << index):
                findings.setdefault(label, []).append(url)
    (args.out / "tombstone-url-cohabitation.v1.json").write_text(json.dumps({
        "schema": "aw.assembly-tombstone-url-cohabitation.v1",
        "producer": "aw ops scan-membership-assembly-facts",
        "rule": "ADR-101a Correction 2 -- a URL bearing an ADR-091 "
                "natural-person tombstone accepts no replacement rows",
        "membership": {
            "uri": "artifact://_preservation/aw_e3-membership-20260814/RECEIPT.json",
            "sha256": sha256_file(membership / "RECEIPT.json"),
            "points": scanned},
        "inputs": tracked_inputs,
        "tombstoned_urls": len(tombstoned_urls),
        "cohabitation": {label: {"urls": len(urls), "url_list": urls}
                         for label, urls in sorted(findings.items())},
        "clean": sorted(label for label in labels[1:] if label not in findings),
        "totals": {"labels_tested": len(labels) - 1,
                   "labels_with_cohabitation": len(findings)},
    }, indent=2, sort_keys=True) + "\n")
    print(f"[scan] tombstoned urls={len(tombstoned_urls):,} "
          f"cohabiting={ {k: len(v) for k, v in findings.items()} }", flush=True)
    return 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main() or 0)

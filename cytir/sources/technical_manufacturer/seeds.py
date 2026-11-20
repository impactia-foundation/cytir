"""Derive technical-manufacturer discovery domains from query-gap evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import urllib.parse

from cytir.acquire.artifact_io import atomic_publish, canonical_json, sha256_file
from cytir.sources.technical_manufacturer.urls import manufacturer_url  # noqa: F401 (moved by task 1401; re-exported)


VERSION = "technical-manufacturer-query-gap-seeds.v1"
TECHNICAL_QUERY = re.compile(
    r"\b(robot(?:ics)?|raspberry\s*pi|arduino|microcontroller|electronics?|"
    r"single[- ]board|maker\s+(?:kit|hardware)|3d\s+(?:printer|printing)|"
    r"embedded\s+systems?)\b", re.I)


def _locator(path: Path, artifact_root: Path | None) -> str:
    if artifact_root is None:
        return str(path)
    return "artifact://" + path.resolve().relative_to(artifact_root.resolve()).as_posix()


def derive(*, source: Path, frontier: Path, receipt: Path,
           artifact_root: Path | None = None) -> dict:
    domains: dict[str, dict] = {}
    source_rows = matched_rows = 0
    for line in source.open(encoding="utf-8"):
        if not line.strip():
            continue
        source_rows += 1
        row = json.loads(line)
        query = str(row.get("query") or "")
        if not TECHNICAL_QUERY.search(query):
            continue
        matched_rows += 1
        for raw_url in row.get("preferred_urls") or []:
            url = manufacturer_url(str(raw_url))
            if not url:
                continue
            domain = urllib.parse.urlsplit(url).hostname or ""
            value = domains.setdefault(domain, {
                "domain": domain, "seed_url": url, "evidence": []})
            value["evidence"].append({
                "query_id": row.get("id"), "query_sha256": hashlib.sha256(
                    query.encode()).hexdigest(), "preferred_url": url,
                "source": "agent-query-gap-preferred-url",
            })
    rows = []
    for domain, value in sorted(domains.items()):
        evidence = sorted(value["evidence"], key=lambda x: (
            str(x["query_id"]), x["preferred_url"]))
        rows.append({**value, "evidence": evidence,
                     "evidence_count": len(evidence),
                     "discovery_surfaces": [
                         f"https://{domain}/agents.md",
                         f"https://{domain}/sitemap.xml"]})
    encoded = b"".join((canonical_json(row) + "\n").encode() for row in rows)
    if frontier.exists() and frontier.read_bytes() != encoded:
        raise ValueError("immutable technical-manufacturer frontier collision")
    atomic_publish(frontier, encoded)
    value = {
        "schema": "aw.technical-manufacturer-seed-receipt.v1",
        "version": VERSION, "status": "complete",
        "selection": "all preferred-URL domains from query rows matching the tracked technical-hardware intent regex; no domain allowlist",
        "source": {"uri": _locator(source, artifact_root), "bytes": source.stat().st_size,
                   "sha256": sha256_file(source), "rows": source_rows},
        "matched_query_rows": matched_rows,
        "frontier": {"uri": _locator(frontier, artifact_root), "bytes": len(encoded),
                     "sha256": hashlib.sha256(encoded).hexdigest(),
                     "domains": len(rows)},
        "producer": {"id": "pipeline/prepare_technical_manufacturer_seeds.py",
                     "version": VERSION},
    }
    receipt_bytes = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if receipt.exists() and receipt.read_bytes() != receipt_bytes:
        raise ValueError("immutable technical-manufacturer seed receipt collision")
    atomic_publish(receipt, receipt_bytes)
    return value

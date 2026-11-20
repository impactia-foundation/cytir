#!/usr/bin/env python3
"""BFS (Swiss Federal Statistical Office) PX-Web metadata ingester.

Source (policy-a, documented API): the BFS PX-Web / STAT-TAB API at
https://www.pxweb.bfs.admin.ch/api/v1/{de,fr,it,en}/ . We walk the database
tree (a flat list of table dbids at the px-x root) and, for each table, fetch
its METADATA in every available language:

    GET /api/v1/{lang}/{dbid}/{dbid}.px   -> {title, variables:[{text,valueTexts}]}

One Passage per (table, language): the localized title + each dimension's label
+ its value labels (capped) + the time period covered. This is statistical-table
*metadata only* — never the data matrix (we never POST a data query).

Not every table exists in every language (it/en are partial); a 4xx for a
language is treated as "absent in that language" and skipped, NOT as an error.
Transient 429/5xx are retried with backoff — the BFS endpoint rate-limits bursts
and will intermittently 4xx/5xx under load, so retries are essential.

  python3 -m cytir.extract.bfs_pxweb --out bfs.jsonl [--limit N] [--langs de,fr,it,en]

Tier A / gov-stat. opendata.swiss / open by default.

Moved here from the flat pipeline driver (task 1412, S7); the old path forwards to this module.
`cytir.sources.families.bfs_pxweb` wraps the parse functions on the `SourceFamily` Protocol.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict

import httpx
from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client

from cytir.core.common import Passage

API = "https://www.pxweb.bfs.admin.ch/api/v1"
DOMAIN = "bfs.admin.ch"
LANGS = ["de", "fr", "it", "en"]

# How many value labels of a dimension to fold into the passage text. Some
# dimensions have hundreds of members (e.g. every Swiss commune); we keep the
# metadata passage compact and representative rather than dumping the matrix.
MAX_VALUES_PER_VAR = 12


def get_json(client: httpx.Client, url: str, *, tries: int = 6):
    """Polite GET returning (status, json|None).

    - 200            -> (200, parsed json)
    - 4xx (not 429)  -> (status, None)   # legitimately absent (e.g. lang missing)
    - 429 / 5xx      -> retry with exponential backoff
    After exhausting retries on a transient failure, RAISE (never silently empty).
    """
    delay = 2.0
    last_transient = None
    for attempt in range(tries):
        try:
            r = client.get(url)
        except httpx.HTTPError as e:  # network-level — retry
            last_transient = e
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue
        sc = r.status_code
        if sc == 200:
            try:
                return 200, r.json()
            except ValueError as e:
                # 200 but unparseable -> treat as transient (BFS sometimes
                # returns an HTML error body with a 200 under load)
                last_transient = e
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
        if sc == 429 or 500 <= sc < 600:
            ra = r.headers.get("retry-after")
            wait = float(ra) if (ra and ra.isdigit()) else delay
            print(f"  {sc} on {url} — retry in {wait:.0f}s "
                  f"({attempt + 1}/{tries})", file=sys.stderr)
            time.sleep(wait)
            delay = min(delay * 2, 60)
            last_transient = RuntimeError(f"{sc} on {url}")
            continue
        # other 4xx — table/lang genuinely not available
        return sc, None
    raise RuntimeError(f"BFS API kept failing for {url}: {last_transient}")


def list_tables(client: httpx.Client) -> list[str]:
    """Walk the PX-Web db tree from the root, collecting every table dbid.

    Entries with type 'l' are sub-folders to recurse into; everything carrying a
    'dbid' is a table. The px-x root currently returns a flat list of dbids, but
    we recurse defensively so a future nested tree is handled too."""
    seen: list[str] = []
    seen_set: set[str] = set()

    def walk(path: str):
        sc, data = get_json(client, f"{API}/de/{path}".rstrip("/") + "/")
        if sc != 200 or not isinstance(data, list):
            return
        for e in data:
            if not isinstance(e, dict):
                continue
            if e.get("type") == "l" and e.get("id"):
                walk(f"{path}/{e['id']}" if path else e["id"])
            dbid = e.get("dbid")
            if dbid and dbid not in seen_set:
                seen_set.add(dbid)
                seen.append(dbid)

    walk("")
    return seen


def table_meta_url(lang: str, dbid: str) -> str:
    return f"{API}/{lang}/{dbid}/{dbid}.px"


def public_url(dbid: str, lang: str) -> str:
    """Stable, human/agent-dereferenceable URL for the table. The PX-Web API
    metadata endpoint itself is the canonical, stable locator for the cube
    metadata across crawls."""
    return table_meta_url(lang, dbid)


def build_passage(dbid: str, lang: str, meta: dict) -> Passage | None:
    title = (meta.get("title") or "").strip()
    variables = meta.get("variables") or []
    if not title or not variables:
        return None

    # Period: the time dimension is usually the variable flagged "time" or named
    # Jahr/Année/Anno/Year; summarize its span.
    period = ""
    var_lines = []
    for v in variables:
        vtext = (v.get("text") or v.get("code") or "").strip()
        vals = v.get("valueTexts") or v.get("values") or []
        is_time = bool(v.get("time")) or vtext.lower() in (
            "jahr", "année", "annee", "anno", "year", "periode", "période", "period")
        if is_time and vals:
            period = f"{vals[0]}–{vals[-1]}" if len(vals) > 1 else f"{vals[0]}"
        shown = vals[:MAX_VALUES_PER_VAR]
        more = f" (+{len(vals) - len(shown)} more)" if len(vals) > len(shown) else ""
        if shown:
            var_lines.append(f"{vtext}: {', '.join(map(str, shown))}{more}")
        else:
            var_lines.append(vtext)

    label = {"de": "BFS-Tabelle", "fr": "Tableau OFS",
             "it": "Tabella UST", "en": "FSO table"}.get(lang, "BFS table")
    agency = {
        "de": "Bundesamt für Statistik (BFS)",
        "fr": "Office fédéral de la statistique (OFS)",
        "it": "Ufficio federale di statistica (UST)",
        "en": "Swiss Federal Statistical Office (FSO)",
    }.get(lang, "Swiss Federal Statistical Office (FSO)")
    parts = [f"{title}.", f"{label} {dbid} — {agency}."]
    if period:
        parts.append(f"Period covered: {period}.")
    parts.append("Dimensions: " + "; ".join(var_lines) + ".")
    text = " ".join(parts)

    snippet = f"{title} — official Swiss statistical table {dbid} (BFS/FSO)"
    return Passage(
        url=public_url(dbid, lang),
        title=title,
        snippet=snippet[:200],
        text=text,
        lang=lang,
        domain=DOMAIN,
        tier="A",
        source_type="gov-stat",
        trust_tier=0.95,
        last_seen=period or None,
    )


def structured_record(dbid: str, lang: str, meta: dict, *, response_uri: str,
                      response_sha256: str, retrieved_at: str) -> dict | None:
    """Project an exact PXWeb metadata response to structured record Markdown."""
    passage = build_passage(dbid, lang, meta)
    if passage is None:
        return None
    variables = meta.get("variables") or []
    lines = [f"# {passage.title}", "", "## Record identity", "",
             f"- PXWeb database ID: `{dbid}`", f"- Language: `{lang}`",
             f"- API endpoint: `{passage.url}`",
             f"- Parent response: `{response_uri}`",
             f"- Parent response SHA-256: `{response_sha256}`",
             f"- Retrieved at: `{retrieved_at}`", "", "## Dimensions", ""]
    for index, variable in enumerate(variables, 1):
        code = " ".join(str(variable.get("code") or "").split())
        label = " ".join(str(variable.get("text") or code).split())
        values = variable.get("values") or []
        value_texts = variable.get("valueTexts") or []
        lines.extend((f"### {index}. {label}", "", f"- Code: `{code}`",
                      f"- Time dimension: {str(bool(variable.get('time'))).lower()}",
                      f"- Values: {len(values)}", ""))
        for value_index, value in enumerate(values):
            value_label = value_texts[value_index] if value_index < len(value_texts) else value
            lines.append(f"- `{value}` — {value_label}")
        lines.append("")
    agency_url = "https://www.bfs.admin.ch/"
    links = [
        {"url": passage.url, "relation": "source-record", "context": dbid},
        {"url": agency_url, "relation": "publisher", "context": "BFS/FSO"},
    ]
    lines.extend(("## Source links", "", f"- [PXWeb metadata]({passage.url})",
                  f"- [Swiss Federal Statistical Office]({agency_url})"))
    record = asdict(passage)
    record["text"] = "\n".join(lines).rstrip()
    record["source_record_id"] = f"{lang}:{dbid}"
    record["parent_response"] = {"uri": response_uri, "sha256": response_sha256,
                                 "retrieved_at": retrieved_at}
    record["source_metadata"] = meta
    record["source_links"] = links
    record["outlinks"] = [{"url": link["url"], "nofollow": False} for link in links]
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N passages (0 = all)")
    ap.add_argument("--langs", default=",".join(LANGS),
                    help="comma-separated language codes to harvest")
    ap.add_argument("--sleep", type=float, default=0.05,
                    help="polite delay between requests (s)")
    args = ap.parse_args()
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]

    client = make_client(HttpSpec(timeout=120, follow_redirects=True,
                                  headers={"user-agent": "authoritative-web/0.1 "
                                           "(+https://websear.ch) bfs-pxweb-ingest"}))

    tables = list_tables(client)
    print(f"discovered {len(tables)} BFS tables", flush=True)

    total = 0
    per_lang = {l: 0 for l in langs}
    missing = {l: 0 for l in langs}
    with open(args.out, "w", encoding="utf-8") as out:
        for i, dbid in enumerate(tables):
            for lang in langs:
                sc, meta = get_json(client, table_meta_url(lang, dbid))
                if sc != 200 or not meta:
                    missing[lang] += 1
                    continue
                p = build_passage(dbid, lang, meta)
                if not p:
                    continue
                out.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
                total += 1
                per_lang[lang] += 1
                if args.sleep:
                    time.sleep(args.sleep)
                if args.limit and total >= args.limit:
                    print(f"hit --limit {args.limit}", flush=True)
                    print(f"wrote {total} passages -> {args.out}  "
                          f"per_lang={per_lang} missing={missing}")
                    return
            if (i + 1) % 50 == 0:
                print(f"  ...{i + 1}/{len(tables)} tables, {total} passages",
                      flush=True)
    print(f"wrote {total} passages -> {args.out}")
    print(f"per_lang={per_lang}  missing(lang/table absent)={missing}")


if __name__ == "__main__":
    main()

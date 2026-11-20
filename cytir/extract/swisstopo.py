#!/usr/bin/env python3
"""SwissNAMES3D ingester: swisstopo's authoritative Swiss gazetteer.

Source (policy-a, official dump): swisstopo STAC API exposes a bulk CSV asset
for ch.swisstopo.swissnames3d. We resolve the *latest* swissnames3d_<year>
item from the STAC collection, download its `*_2056.csv.zip` (LV95 / EPSG:2056),
and emit one Passage per named place/feature.

The zip holds three CSVs — points (PKT), lines (LIN), polygons (PLY) — all sharing
NAME / SPRACHCODE (language) / OBJEKTART (feature type) / E,N,Z (LV95 coords).
Each named feature becomes one passage: name + feature type + language/locale +
coordinate context, with a stable, dereferenceable map.geo.admin.ch URL pinned to
its LV95 easting/northing. Metadata only — never the full geometry.

  python -m cytir.extract.swisstopo --out swisstopo.jsonl [--limit N]

Task 1413 (S7): MOVED here from the flat driver (which now forwards). The source-family adapter is
`cytir.sources.families.swisstopo`; it reads an already-downloaded zip and calls `rows` and
`structured_record` below. The download (`stac_latest_csv`, `get_with_retry`, `main`) stays here.

Tier A / gov-geo. Open swisstopo OGD (free, commercial use OK).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import zipfile
from dataclasses import asdict

import httpx
from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client

from cytir.core.common import Passage

STAC_ITEMS = ("https://data.geo.admin.ch/api/stac/v1/collections/"
              "ch.swisstopo.swissnames3d/items")
DOMAIN = "swisstopo.admin.ch"

# SPRACHCODE (free-text in the dump) -> AW lang code. The dump labels the
# language family of the name; we keep rm (Romansh) as given per the record even
# though it's out of P0 scope, because the task asks for de/fr/it/rm as present.
_LANG = {
    "Hochdeutsch": "de",
    "Deutsch": "de",
    "Franzoesisch": "fr",
    "Italienisch": "it",
    "Rumantsch": "rm",
    "Rätoromanisch": "rm",
}


def lang_of(sprachcode: str) -> str:
    s = (sprachcode or "").strip()
    for prefix, code in _LANG.items():
        if s.startswith(prefix):
            return code
    return "de"  # 'mehrsprachig' / blanks default to de (German is the majority)


def stac_latest_csv(client: httpx.Client) -> tuple[str, str]:
    """Return (item_id, csv_zip_href) for the most recent swissnames3d item."""
    r = get_with_retry(client, STAC_ITEMS)
    feats = r.json().get("features", [])
    if not feats:
        raise RuntimeError("swisstopo STAC returned no swissnames3d items")
    feats.sort(key=lambda f: f.get("id", ""))
    latest = feats[-1]
    for name, asset in (latest.get("assets") or {}).items():
        if name.endswith(".csv.zip"):
            return latest["id"], asset["href"]
    raise RuntimeError(f"no CSV asset on STAC item {latest.get('id')}")


def get_with_retry(client: httpx.Client, url: str, *, stream: bool = False,
                   tries: int = 6) -> httpx.Response:
    """Polite GET: exponential backoff on 429/5xx; other errors raise. Never
    swallows a failure into an empty result."""
    delay = 2.0
    last = None
    for attempt in range(tries):
        try:
            if stream:
                # caller is responsible for closing; we only retry the open
                r = client.send(client.build_request("GET", url), stream=True)
            else:
                r = client.get(url)
        except httpx.HTTPError as e:  # network-level — retry
            last = e
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue
        if r.status_code in (429,) or 500 <= r.status_code < 600:
            r.close()
            ra = r.headers.get("retry-after")
            wait = float(ra) if (ra and ra.isdigit()) else delay
            print(f"  {r.status_code} on {url} — retry in {wait:.0f}s "
                  f"({attempt + 1}/{tries})", file=sys.stderr)
            time.sleep(wait)
            delay = min(delay * 2, 60)
            continue
        r.raise_for_status()
        return r
    if last:
        raise last
    raise RuntimeError(f"giving up on {url} after {tries} tries")


def map_url(e: str, n: str) -> str | None:
    """Stable dereferenceable URL: pin the LV95 (EPSG:2056) easting/northing on
    map.geo.admin.ch with a crosshair marker. None if coords are unusable."""
    try:
        ei, ni = int(float(e)), int(float(n))
    except (TypeError, ValueError):
        return None
    return (f"https://map.geo.admin.ch/?E={ei}&N={ni}"
            f"&zoom=8&crosshair=marker&swisssearch=1")


def rows(zf: zipfile.ZipFile):
    """Yield (member, dict) for every data row across all three CSVs."""
    for member in zf.namelist():
        if not member.lower().endswith(".csv"):
            continue
        with zf.open(member) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text, delimiter=";")
            for row in reader:
                yield member, row


def build_passage(row: dict, item_id: str) -> Passage | None:
    name = (row.get("NAME") or "").strip()
    if not name:
        return None
    e, n, z = row.get("E"), row.get("N"), row.get("Z")
    url = map_url(e, n)
    if not url:
        return None
    objektart = (row.get("OBJEKTART") or "").strip()
    objklasse = (row.get("OBJEKTKLASSE_TLM") or "").strip()
    sprachcode = (row.get("SPRACHCODE") or "").strip()
    lang = lang_of(sprachcode)
    feature_type = objektart or objklasse or "named feature"

    # Build descriptive text: name + what it is + where (LV95) + language.
    parts = [f"{name} — {feature_type}." if feature_type else f"{name}."]
    if objklasse and objklasse != feature_type:
        parts.append(f"Feature class: {objklasse}.")
    if sprachcode:
        parts.append(f"Name language: {sprachcode}.")
    coord = ""
    try:
        coord = f"LV95 (EPSG:2056) E={int(float(e))} N={int(float(n))}"
        if z and z not in ("", "k_W"):
            coord += f" Z={int(float(z))}m"
    except (TypeError, ValueError):
        pass
    if coord:
        parts.append(f"Location: {coord}, Switzerland.")
    parts.append("Authoritative geographic name from swisstopo SwissNAMES3D "
                 "(the official Swiss gazetteer).")
    text = " ".join(parts)

    snippet = f"{name} — {feature_type}, Switzerland" if feature_type else name
    return Passage(
        url=url,
        title=name,
        snippet=snippet[:200],
        text=text,
        lang=lang,
        domain=DOMAIN,
        tier="A",
        source_type="gov-geo",
        trust_tier=0.95,
        last_seen=item_id,  # which SwissNAMES3D vintage this name came from
    )


def structured_record(row: dict, *, item_id: str, item_datetime: str,
                      parent_member: str, parent_archive_sha256: str,
                      archive_url: str) -> dict | None:
    """Project one CSV row to source-profile-complete record Markdown."""
    passage = build_passage(row, item_id)
    if passage is None:
        return None
    source_uuid = (row.get("UUID") or "").strip()
    name_uuid = (row.get("NAME_UUID") or "").strip()
    group_uuid = (row.get("NAMENGRUPPE_UUID") or "").strip()
    fields = [
        ("Status", row.get("STATUS")),
        ("Language code", row.get("SPRACHCODE")),
        ("Name type", row.get("NAMEN_TYP")),
        ("Feature type", row.get("OBJEKTART")),
        ("TLM feature class", row.get("OBJEKTKLASSE_TLM")),
        ("Engineering structure", row.get("KUNSTBAUTE")),
    ]
    lines = [f"# {passage.title}", "", "## Record identity", "",
             f"- SwissNAMES3D UUID: `{source_uuid}`",
             f"- Name UUID: `{name_uuid}`",
             f"- Release: `{item_id}`",
             f"- Release datetime: `{item_datetime}`",
             f"- Parent archive member: `{parent_member}`",
             f"- Parent archive SHA-256: `{parent_archive_sha256}`"]
    if group_uuid:
        lines.append(f"- Name group UUID: `{group_uuid}`")
    lines.extend(("", "## Classification", ""))
    lines.extend(f"- {name}: {' '.join(str(value).split())}"
                 for name, value in fields if value and str(value).strip())
    lines.extend(("", "## Coordinates", "",
                  "- Coordinate reference system: LV95 (EPSG:2056)",
                  f"- Easting: {row.get('E', '')}", f"- Northing: {row.get('N', '')}"))
    if row.get("Z"):
        lines.append(f"- Elevation: {row['Z']} m")
    stac_url = ("https://data.geo.admin.ch/api/stac/v1/collections/"
                f"ch.swisstopo.swissnames3d/items/{item_id}")
    links = [
        {"url": passage.url, "relation": "map-location", "context": source_uuid},
        {"url": stac_url, "relation": "source-release", "context": item_id},
        {"url": archive_url, "relation": "parent-feed", "context": parent_member},
    ]
    lines.extend(("", "## Source links", "",
                  f"- [Map location]({passage.url})",
                  f"- [STAC release]({stac_url})",
                  f"- [Bulk CSV archive]({archive_url})"))
    record = asdict(passage)
    record["text"] = "\n".join(lines).rstrip()
    record["source_record_id"] = source_uuid
    record["source_release"] = {"id": item_id, "datetime": item_datetime}
    record["parent_archive"] = {"member": parent_member,
                                "sha256": parent_archive_sha256}
    record["source_metadata"] = {key: value for key, value in row.items()}
    record["source_links"] = links
    record["outlinks"] = [{"url": link["url"], "nofollow": False} for link in links]
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N passages (0 = all)")
    ap.add_argument("--cache", default="",
                    help="optional local path to a pre-downloaded csv.zip")
    args = ap.parse_args()

    client = make_client(HttpSpec(timeout=300, follow_redirects=True,
                                  headers={"user-agent": "authoritative-web/0.1 "
                                           "(+https://websear.ch) swisstopo-ingest"}))
    item_id, csv_href = stac_latest_csv(client)
    print(f"latest SwissNAMES3D item: {item_id}", flush=True)
    print(f"CSV asset: {csv_href}", flush=True)

    if args.cache:
        print(f"using cached zip {args.cache}", flush=True)
        blob = open(args.cache, "rb").read()
    else:
        r = get_with_retry(client, csv_href)
        blob = r.content
        print(f"downloaded {len(blob):,} bytes", flush=True)

    zf = zipfile.ZipFile(io.BytesIO(blob))
    print(f"members: {zf.namelist()}", flush=True)

    total = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for member, row in rows(zf):
            p = build_passage(row, item_id)
            if not p:
                continue
            out.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
            total += 1
            if total % 50000 == 0:
                print(f"  ...{total:,} passages", flush=True)
            if args.limit and total >= args.limit:
                break
        else:
            pass
    print(f"wrote {total:,} passages -> {args.out}")


if __name__ == "__main__":
    main()

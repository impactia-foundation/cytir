"""Generator of data/fedlex-versions.v1.json, the side table behind `read_url` (task 2709).

    cyt source fedlex-versions [--out FILE] [--elis-from FILE] [--langs de,fr,it,en]

For every ELI (the curated law roster, plus one ELI per line of --elis-from) it resolves the
in-force consolidation date (`fedlex.resolve_date`) and GETs the filestore HTML twin per language.
A wrong date answers HTTP 200 with a ~9 KB Angular shell, so a twin is vouched for only when it
holds an `id="art_` anchor and not the shell marker (a short real law passes, the shell does not): logged and
recorded under `_dropped` as "<eli>/<lang>"; an ELI with no date is left out. The reader
(`cytir.serve.read_url`) emits no `read_url` for a dropped pair. At most 2 requests per second.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
import time

from cytir.core.paths import data_path
from cytir.sources.families.fedlex_contract import fedlex_html_url

SCHEMA = "aw.fedlex-versions.v1"
ANCHOR = b'id="art_'
SHELL_MARKER = b"<app-root"   # the Angular SPA shell a wrong date serves with HTTP 200 (~9 KB)
LANGS = ("de", "fr", "it", "en")
MIN_INTERVAL = 0.5   # seconds between requests: <= 2 req/s


def is_real_twin(status: int, body: bytes) -> bool:
    return status == 200 and ANCHOR in body and SHELL_MARKER not in body


def build_table(elis, *, resolve, get, langs=LANGS, built_at=None, sleep=time.sleep,
                log=lambda m: print(m, file=sys.stderr)) -> dict:
    """`resolve(eli) -> 'YYYYMMDD' | None`; `get(url) -> (status, body_bytes)`. Injected for tests."""
    laws, dropped, verified = {}, [], 0
    for eli in elis:
        date = resolve(eli)
        sleep(MIN_INTERVAL)
        if not date:
            log(f"drop {eli}: no in-force consolidation date")
            continue
        ok = 0
        for lang in langs:
            url = fedlex_html_url(eli, date, lang)
            status, body = get(url)
            sleep(MIN_INTERVAL)
            if is_real_twin(status, body):
                ok += 1
                verified += 1
            else:
                log(f"drop {eli}/{lang}: HTTP {status}, {len(body)} bytes, no article anchor or shell page ({url})")
                dropped.append(f"{eli}/{lang}")
        if ok:
            laws[eli] = date
    out = {"_comment": "ELI -> in-force consolidation date (YYYYMMDD) for the filestore HTML twin "
                       "behind `read_url`. _langs: languages verified; _dropped: <eli>/<lang> pairs "
                       "whose twin failed the content check (HTTP 200, an id=art_ anchor, no SPA shell). Regenerate with "
                       "`aw source fedlex-versions`.",  # rename-codemod: keep (persisted: _comment written into data/fedlex-versions.v1.json)
           "schema": SCHEMA,
           "built_at": built_at or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "_langs": list(langs), "_dropped": sorted(dropped),
           "_verified": verified}
    out.update(dict(sorted(laws.items())))
    return out


def main(argv, prog="cyt source fedlex-versions") -> int:
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(data_path("fedlex-versions.v1.json")))
    ap.add_argument("--elis-from", help="file with one ELI per line (e.g. cc/27/317_321_377)")
    ap.add_argument("--langs", default=",".join(LANGS))
    a = ap.parse_args(argv)
    from cytir.sources.families import fedlex
    from cytir.sources.families.fedlex_law_meta import load_curated_names
    elis = list(load_curated_names())
    if a.elis_from:
        extra = [ln.strip() for ln in pathlib.Path(a.elis_from).read_text().splitlines()
                 if ln.strip() and not ln.startswith("#")]
        elis += [e for e in extra if e not in elis]
    client = fedlex._h()

    def get(url):
        r = client.get(url)
        return r.status_code, r.content

    table = build_table(elis, resolve=lambda e: fedlex.resolve_date(e), get=get,
                        langs=tuple(a.langs.split(",")))
    pathlib.Path(a.out).write_text(json.dumps(table, indent=1, ensure_ascii=False) + "\n")
    laws = [k for k in table if "/" in k]
    print(f"laws {len(laws)}/{len(elis)}  twins verified {table['_verified']}  "
          f"dropped {len(table['_dropped'])}  -> {a.out}")
    return 0

"""Shared rows and declarations for the coded-twin reader tests (task 855).

* `real_rows()`: the 20 REAL twin rows of `fixtures/row_decode/twin_rows.v1.json` (raw profile).
* `representative_rows()`: those, the six mirror-guard rows (twin and snapshot copy), the news
  daily sample rows and the consent rows below -- one list of `(label, raw payload)`.
* `declare(monkeypatch, tmp_path, ...)`: the declaration file with ONE more collection, a twin
  declared like the post-flip twin (`aw_e10_twin`, profile `rebuild_e10`, not serving); the real
  declarations stay as they are. Every reader resolves its profile from the file at call time.
* `coded(payload, profile)`: the payload as that profile STORES it (the writer's projection).
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys

# the repo root first: a direct run must import THIS tree's `aw`, never another worktree's
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cytir.index import payload_profile as PP  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIX = HERE / "fixtures"
CODED_TWIN = "aw_e10_twin"
RAW_TWIN = "aw_e8"

#: Real-shaped consent rows (the families the ADR-115 backfill tells apart). `weko_docs` is
#: provenance code 0, which a raw reader read as "no provenance" (`0 or ""`).
UNSEEN_CONSENT_SOURCE = "synthetic robots.txt sentence that no enum table holds (task 923)"
CONSENT_ROWS = [
    {"url": "https://www.weko.admin.ch/dam/fr/sd-web/X1c86wvpMLVZ/dpc_2015-4.pdf",
     "domain": "weko.admin.ch", "provenance": "weko_docs", "license": "ch-gov-open-cite",
     "consent_state": "carried", "consent_basis": "carried-from-superseded-row",
     "consent_checked_at": "2026-09-09"},
    {"url": "https://www.legifrance.gouv.fr/loda/article_lc/LEGIARTI000024415582",
     "domain": "legifrance.gouv.fr", "provenance": "loda", "license": "Etalab-2.0",
     "consent_state": "dump", "consent_basis": "official-dump-license"},
    {"url": "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000006902584",
     "domain": "legifrance.gouv.fr", "provenance": "legi_code", "license": "Etalab-2.0"},
    {"url": "https://www.legifrance.gouv.fr/conv_coll/article/KALIARTI000054454368",
     "domain": "legifrance.gouv.fr", "provenance": "kali_article", "license": "Etalab-2.0"},
    {"url": "https://www.rechtsprechung-im-internet.de/jportal/?quelle=jlink&docid=KORE625042017",
     "domain": "rechtsprechung-im-internet.de", "provenance": "rii_courts",
     "license": "public-domain-official-work-UrhG-5"},
    {"url": "https://en.wikipedia.org/wiki/Craigavon", "domain": "en.wikipedia.org",
     "source_type": "reference", "license": "CC-BY-SA-4.0", "consent_checked_at": "2026-08-02T02:08:30Z"},
]


def real_rows() -> list[dict]:
    return json.loads((FIX / "row_decode" / "twin_rows.v1.json").read_text(encoding="utf-8"))["rows"]


def representative_rows() -> list[tuple[str, dict]]:
    out = [(f"twin:{r['id']}", r["payload"]) for r in real_rows()]
    mg = json.loads((FIX / "mirror_guards" / "rows.v1.json").read_text(encoding="utf-8"))
    for cls in ("prod_only_old", "equal"):
        for e in mg[cls]:
            for side in ("twin", "copy"):
                out.append((f"mirror_guards:{cls}:{side}:{e['id']}", e[side]["payload"]))
    for line in (FIX / "news-daily-sample-rows.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out.append((f"news-daily:{r['id']}", r["payload"]))
    out += [(f"consent:{i}", dict(r)) for i, r in enumerate(CONSENT_ROWS)]
    # the aw_e10 T0 table codes every consent_source the twin holds, so the open-key "unseen value"
    # case is a synthetic row (task 923)
    out.append(("synthetic:unseen-open-value", {**CONSENT_ROWS[1], "consent_source": UNSEEN_CONSENT_SOURCE}))
    return out


def tables() -> dict:
    return PP.load_tables()


def coded(payload: dict, profile: str = "rebuild_e10") -> dict:
    """The payload as `profile` stores it (the one projection; open key `consent_source`)."""
    return PP.project_payload(payload, profile, tables(), open_keys=frozenset({"consent_source"}))


def declare(monkeypatch, tmp_path, name: str = CODED_TWIN, profile: str = "rebuild_e10") -> pathlib.Path:
    """A copy of the real declaration file with `name` declared like the post-flip twin, pointed
    at by `payload_profile.PROFILES_PATH` for the test."""
    doc = json.loads(PP.PROFILES_PATH.read_text(encoding="utf-8"))
    base = doc["collections"]["aw_e9"]
    doc["collections"][name] = {
        "profile": profile, "role": "test: the twin after the aw_e10 flip (coded copy of prod)",
        "tables": base["tables"], "tables_sha256": base["tables_sha256"],
        "open_keys": ["consent_source"], "sparse_rule": copy.deepcopy(base["sparse_rule"]),
        "decided": "test: the twin takes prod's BM25 rule after the flip (task 847: both legs)"}
    path = tmp_path / "collection-payload-profiles.v1.json"
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    monkeypatch.setattr(PP, "PROFILES_PATH", path)
    return path

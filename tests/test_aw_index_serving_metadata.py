"""`cytir.index.serving_metadata` on its own (task 1535): no builder, no pipeline path.

The projection-level contract tests stay with the builder's characterization tests; these pin
the module's own surface: the registry key, the www fold, the URL tier.
"""
from __future__ import annotations

import json
import sqlite3

from cytir.index import serving_metadata as SM


def _registry(tmp_path, rows):
    db = tmp_path / "reg.sqlite"
    con = sqlite3.connect(str(db))
    cols = ", ".join(f"{f} TEXT" for f in SM.REGISTRY_FIELDS)
    con.execute(f"CREATE TABLE url_meta (k TEXT PRIMARY KEY, canon TEXT, "
                f"points INTEGER, {cols}) WITHOUT ROWID")
    for canon, vals in rows.items():
        con.execute(
            f"INSERT INTO url_meta VALUES (?,?,?,{','.join('?' * len(SM.REGISTRY_FIELDS))})",
            (SM.url_key(canon), canon, 1,
             *[json.dumps(vals[f]) if vals.get(f) is not None else None
               for f in SM.REGISTRY_FIELDS]))
    con.commit()
    con.close()
    return SM.ServingMetadataRegistry(str(db))


def test_url_key_is_stable_and_distinguishes_urls():
    assert SM.url_key("https://example.org/a") == SM.url_key("https://example.org/a")
    assert SM.url_key("https://example.org/a") != SM.url_key("https://example.org/b")
    assert len(SM.url_key("https://example.org/a")) == 16


def test_serving_domain_folds_www():
    assert SM.serving_domain("https://www.zh.ch/de/x") == "zh.ch"
    assert SM.serving_domain("https://zh.ch/de/x") == "zh.ch"


def test_registry_answers_by_url_and_misses_unknown(tmp_path):
    url = "https://example.org/a"
    reg = _registry(tmp_path, {url: {"title": "A", "trust_tier": 1.0,
                                     "source_type": "official", "lang": "de"}})
    try:
        hit = reg.by_url(url)
        assert hit and hit["title"] == "A" and hit["trust_tier"] == 1.0
        assert reg.by_url("https://example.org/zzz") is None
    finally:
        reg.close()

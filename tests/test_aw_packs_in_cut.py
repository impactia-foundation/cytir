"""Packs in the public cut (task 999): the cut ships only packs that name nothing private, and
the example pack in ADDING-A-SOURCE validates and compiles to a sitemap plan with no network."""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from cytir.sources import pack_loader
from cytir.sources import schema as pack_schema

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = re.compile(r"\b(?:" + "|".join(("pipe" + "line", "tas" + "ks")) + r")/")
EXAMPLE = ROOT / "sources" / "_examples" / "example-authority.yaml"


def _shipped_packs() -> list[str]:
    cut = json.loads((ROOT / "data" / "public-cut.v1.json").read_text())
    rule = next(r for r in cut["rules"] if r["id"] == "packs")
    return [p for p in rule["paths"] if p.endswith(".yaml") and "*" not in p]


def test_shipped_packs_exist_and_name_nothing_private():
    packs = _shipped_packs()
    assert EXAMPLE.relative_to(ROOT).as_posix() in packs
    for rel in packs:
        text = (ROOT / rel).read_text()
        assert not PRIVATE.search(text), rel
        doc = yaml.safe_load(text)
        ext = doc.get("extractor") or ""
        assert not ext or ext.startswith("cytir/"), rel


def test_packs_rule_is_explicit_not_a_glob():
    cut = json.loads((ROOT / "data" / "public-cut.v1.json").read_text())
    rule = next(r for r in cut["rules"] if r["id"] == "packs")
    assert "sources/**/*.yaml" not in rule["paths"]


def test_example_pack_validates_and_plans_offline():
    doc = yaml.safe_load(EXAMPLE.read_text())
    assert doc["adapter"] == "sitemap" and doc["id"] == "example-authority"
    assert doc["license"] and doc["consent"]["basis"]
    plans = pack_loader.plans(doc)
    assert [p.route for p in plans] == ["source_owned_sitemap_fetch"]
    assert plans[0].extractor is None
    errors = pack_schema.validate_pack(doc, pack_schema.load_schema(), "example-authority")
    assert not errors


def test_authority_packs_and_schema_ship_so_the_adapter_runs_from_the_cut():
    """The authority-packs adapter loads its packs by id at runtime (task 1471): the cut must
    select each of those pack files and the schema that validates them."""
    from cytir.ops import public_cut
    from cytir.sources.families import authority_packs as AP

    doc = public_cut.load_allowlist(ROOT)
    files = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "sources").rglob("*") if p.is_file())
    shipped, problems = public_cut.select(doc, files)
    assert not [p for p in problems if "sources/" in p], problems
    ids = {cls.pack_id for cls in vars(AP).values()
           if isinstance(cls, type) and cls.__module__ == AP.__name__ and getattr(cls, "pack_id", None)}
    assert len(ids) == 4, ids
    for pack_id in sorted(ids):
        rel = pack_loader.find_pack(pack_id).relative_to(ROOT).as_posix()
        assert rel in shipped, f"{pack_id}: {rel} is not in the cut"
    assert any(p.startswith("sources/_schema/") and p.endswith("pack.schema.json") for p in shipped)

"""`gesetze` on the family Protocol (task 1400, S7): gesetze-im-internet.de, the BMJ's bulk law XML.

No network. The law is a SYNTHETIC ZIP built here from `tests/fixtures/gesetze/testg.xml` (the gii
layout: `<norm>` with `enbez`, `titel`, `Content`), cut to hit every filter of the parser: a
structural norm without `enbez`, a table of contents, a repealed placeholder, a 31-char one-liner, an
`(XXXX)` marker, a too-short body, and a duplicate label. `golden-pre-move.json` is the output of the
PRE-move driver (the old flat script at origin/main 02ce21304, the base this task
started from; the file was unchanged on main when the golden was cut) on that ZIP: the moved parser must equal it.

Pins: registered and conformant; the parser is unchanged by the move; a dry run plans one unit per
law; a full run through a mocked transport ends with the four expected records and a balanced
ledger; a 404 is the origin's word (excluded); consent, licence and route are read from the
contract row, never written by the adapter (the row stays `unknown`, the rows carry no licence).
"""
from __future__ import annotations

import io
import json
import pathlib
import sys
import zipfile
from dataclasses import asdict

import httpx
import pytest

from cytir.sources import contract as contract_mod
from cytir.sources import registry
from cytir.sources.family import RunContext, SourceFamily, UnitExcluded
from cytir.sources.families.gesetze import GesetzeFamily
from cytir.sources.gesetze import ingest as G
from cytir.sources.run_family import run_family

FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "gesetze"
ZIP_URL = "https://www.gesetze-im-internet.de/testg/xml.zip"
TOC = ('<?xml version="1.0"?><items><item><title>TestG</title><link>' + ZIP_URL
       + '</link></item><item><title>x</title><link>https://www.gesetze-im-internet.de/x/index.html'
       '</link></item></items>').encode()


def _zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("testg.xml", (FIX / "testg.xml").read_bytes())
    return buf.getvalue()


def _http(zip_status: int = 200) -> httpx.Client:
    def handler(req: httpx.Request) -> httpx.Response:
        if str(req.url) == G.TOC:
            return httpx.Response(200, content=TOC)
        if str(req.url) == ZIP_URL:
            return httpx.Response(zip_status, content=_zip() if zip_status == 200 else b"")
        return httpx.Response(404)
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(G, "POLITE_DELAY", 0)
    monkeypatch.setattr("time.sleep", lambda s: None)


def _ok(host, direct):
    return {"bucket": "ok", "host": host}


def _declared_contracts(tmp_path):
    """The shipped row declares no consent basis, so the framework refuses a host-contacting run
    (task 1406). These tests exercise the adapter, not the declaration: they run a copy whose row
    declares the live per-origin basis. The shipped file stays untouched."""
    doc = json.loads((FIX.parents[2] / contract_mod.DEFAULT_PATH).read_text(encoding="utf-8"))
    row = next(f for f in doc["families"] if f["family_id"] == "gesetze")
    row["consent"].update(basis="declared: take the host's verdict", resolution="live-per-origin")
    row["unknowns"] = sorted(contract_mod.actual_unknowns(row))
    path = tmp_path / "contracts.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _row() -> dict:
    return contract_mod.by_id(contract_mod.load())["gesetze"]


def test_registered_and_conformant():
    assert "gesetze" in registry.registered()
    adapter, row, spec = registry.admit("gesetze")
    assert isinstance(adapter, SourceFamily)
    assert row["family_id"] == "gesetze" and spec.declared == "native"
    assert adapter.PARTIAL == ("l1",)


def test_the_moved_parser_equals_the_pre_move_driver():
    rows = G.parse_law_zip(ZIP_URL, _zip())
    assert [asdict(r) for r in rows] == json.loads((FIX / "golden-pre-move.json").read_text())
    assert len(rows) == 4


def test_a_dry_run_plans_one_unit_per_law_and_fetches_no_law(tmp_path):
    plan = run_family("gesetze", out_dir="unused", today="2026-10-30", http=_http(),
                      consent=_ok, dry_run=True, log=lambda m: None,
                      contracts_path=_declared_contracts(tmp_path))
    assert plan["units"] == ["testg"] and plan["runnable"] == ["testg"]
    assert plan["discover"]["laws"] == 1            # the non-ZIP link is not a law
    assert plan["verdicts"][0]["host"] == "www.gesetze-im-internet.de"


def test_a_full_run_ends_in_four_records_and_a_balanced_ledger(tmp_path):
    m = run_family("gesetze", out_dir=tmp_path / "run", today="2026-10-30", http=_http(),
                   consent=_ok, log=lambda m: None,
                   contracts_path=_declared_contracts(tmp_path))
    recs = [json.loads(line) for line in
            (tmp_path / "run" / "pack" / "records.jsonl").read_text().splitlines()]
    assert [r["url"].rsplit("/", 1)[1] for r in recs] == [
        "abschnitt_1_allgemeines.html", "__1.html", "art_31.html", "__1_7.html"]
    assert all(r["domain"] == "gesetze-im-internet.de" and r["source_type"] == "official"
               and r["trust_tier"] == 1.0 and r["last_seen"] == "2026-10-30" for r in recs)
    # the driver stamped no licence; the adapter does not invent one
    assert all(not r.get("license") for r in recs)
    t = m["terminals"]
    assert (t["units_in"], t["units_admitted"], t["records"]) == (1, 1, 4)


def test_a_404_is_the_origins_word_and_other_failures_are_ours(tmp_path):
    fam = GesetzeFamily()
    ctx = RunContext(out_dir=tmp_path, today="2026-10-30", http=_http(404), skip_consent=True)
    (unit,) = fam.discover(ctx)
    with pytest.raises(UnitExcluded, match="http_404"):
        fam.acquire(unit, ctx)
    ctx.http = _http(500)
    with pytest.raises(RuntimeError, match="500"):
        fam.acquire(unit, ctx)


def test_an_empty_table_of_contents_is_a_wrong_source(tmp_path):
    toc = tmp_path / "toc.xml"
    toc.write_bytes(b"<items/>")
    ctx = RunContext(out_dir=tmp_path, today="2026-10-30", options={"toc": str(toc)})
    with pytest.raises(RuntimeError, match="lists no law ZIP"):
        GesetzeFamily().discover(ctx)


def test_consent_is_read_from_the_row_ruling_77_and_licence_stays_as_declared():
    row = _row()
    assert row["route"] == "bulk-dump"      # task 2262 declared the route from the adapter; consent and licence stay unknown
    c = row["consent"]   # plan.md RULING 77 declared the consent block (task 2854); the adapter reads it, writes none
    assert c["resolution"] == "explicit-grant" and c["search_ok"] is True and c["train_ok"] is True
    assert c["basis"] != "unknown" and "UrhG §5" in c["basis"]
    assert row["licence"]["instrument"] == "public_domain_official_work (UrhG §5)" and row["licence"]["spdx"] is None
    # the adapter reads nothing of the three from itself: a strict admission refuses the row
    with pytest.raises(registry.FamilyRefused):
        registry.admit("gesetze", strict=True)


def test_importing_the_adapter_needs_no_crawler_identity(monkeypatch, tmp_path):
    """Cut rehearsal (task 1400): a subprocess with every AW_* identity var stripped must import the family."""
    import importlib

    for var in ("AW_OPERATOR_URL", "AW_OPERATOR_CONTACT", "AW_ALLOW_UNIDENTIFIED_CRAWL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # no operator.json in scope
    importlib.reload(G)
    importlib.reload(sys.modules["cytir.sources.families.gesetze"])

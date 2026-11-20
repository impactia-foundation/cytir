"""`us_federal_register_ecfr` eCFR leg (task 698) on a REAL govinfo title XML (title 1, trimmed to
8 sections: 5 kept, 2 reserved, the longest of the title), through `cyt source run`'s own door under
the contract's declared chunk policy. The network is a `httpx.MockTransport` over the same bytes.

Pins: one record per kept section; reserved sections are counted, not dropped silently; the
section ledger balances; the record carries the licence, the canonical eCFR section URL and the
"unofficial consolidation" statement; a reserved title is excluded with a reason; the Federal
Register leg refuses; a run that admits 0 sections refuses.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import httpx
import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

FIX = HERE / "fixtures" / "us_ecfr" / "ECFR-title1-sample.xml"
FAMILY = "us_federal_register_ecfr"
TODAY = "2026-10-03"
POLICY_FILE = REPO / "data" / ("chunk-policy-struct-md-768w-o0-adjacent-sections-"
                               "tail-1152-tok2560.v1.json")
OK = {"bucket": "ok", "status": "ok", "search_ok": True, "train_ok": True,
      "checked_at": "2026-10-03T00:00:00Z"}
TITLES = {"titles": [
    {"number": 1, "name": "General Provisions", "up_to_date_as_of": "2026-10-01",
     "reserved": False},
    {"number": 35, "name": "[Reserved]", "up_to_date_as_of": "2026-10-01", "reserved": True}]}
EMPTY_TITLE = (b"<DLPSTEXTCLASS><TEXT><BODY><ECFRBRWS><DIV1 N='1' TYPE='TITLE'></DIV1>"
               b"</ECFRBRWS></BODY></TEXT></DLPSTEXTCLASS>")


@pytest.fixture(autouse=True)
def _declared_policy(monkeypatch):
    from cytir.core import model_pin
    monkeypatch.setenv("AW_LANGGATE_OPTIONAL", "1")
    monkeypatch.setenv("AW_LID_MODEL", "/nonexistent/lid.176.bin")
    monkeypatch.setenv("AW_CHUNKER", "structured")
    monkeypatch.setenv("AW_CHUNK_POLICY", str(POLICY_FILE))
    monkeypatch.setenv("AW_CHUNK_POLICY_SHA256",
                       hashlib.sha256(POLICY_FILE.read_bytes()).hexdigest())
    monkeypatch.setenv("AW_CHUNK_TOKENIZER_REVISION", model_pin.EMBED_MODEL_REVISION)
    yield
    from cytir.chunk import chunker
    chunker._apply_policy(chunker.load_chunk_policy(None), allow_env_overrides=True)
    chunker.select_tokenizer_revision(None)


def _client(xml=None):
    xml = FIX.read_bytes() if xml is None else xml

    def handler(req: httpx.Request):
        if req.url.path.endswith("/titles.json"):
            return httpx.Response(200, json=TITLES)
        assert "/full/" not in req.url.path and "/renderer/" not in req.url.path
        if req.url.host == "www.govinfo.gov" and req.url.path.endswith("ECFR-title1.xml"):
            return httpx.Response(200, content=xml, headers={"content-type": "text/xml"})
        return httpx.Response(404)
    return httpx.Client(transport=httpx.MockTransport(handler))


def _run(tmp_path, http=None, **kw):
    from cytir.sources import run_family
    return run_family.run_family(
        FAMILY, out_dir=tmp_path / "o", today=TODAY, consent=lambda h, d: {**OK, "host": h},
        http=http or _client(), log=lambda m: None, **kw)


def _rows(tmp_path):
    return [json.loads(x) for x in
            (tmp_path / "o" / "pack" / "records.jsonl").read_text(encoding="utf-8").splitlines()]


def test_one_record_per_kept_section_and_the_ledger_balances(tmp_path, word_only_chunking_allowed):
    m = _run(tmp_path)
    rows = _rows(tmp_path)
    t = m["terminals"]
    assert t["units_in"] == 2 and t["units_admitted"] == 1
    assert t["excluded_reasons"] == {"title_reserved": 1}
    s = m["discover"]["sections"]
    assert s["seen"] == 8 and s["admitted"] == 6 and s["section_reserved"] == 2
    urls = {r["url"] for r in rows}
    assert len(urls) == 6
    assert all(u.startswith("https://www.ecfr.gov/current/title-1/section-") for u in urls)
    for r in rows:
        assert r["license"] == "public-domain" and r["domain"] == "ecfr.gov"
        assert r["lang"] == "en" and r["tier"] == "A" and r["source_type"] == "official"
        assert "unofficial" in r["text"].lower()
        assert r["last_seen"] == "2026-10-01" and r["point_id"]
    assert len(rows) >= len(urls)          # a long section may split; none is lost
    assert all("annual CFR edition" in " ".join(r["text"] for r in rows if r["url"] == u)
               for u in urls)


def test_the_read_rate_is_capped(tmp_path, monkeypatch, word_only_chunking_allowed):
    slept = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    _run(tmp_path, options={"max_mb_s": "0.01"})       # 43 KB at 10 KB/s -> it must wait
    assert slept and sum(slept) > 1.0


def _sec(n, head, body="Text of the section that is long enough to keep."):
    return f'<DIV8 N="{n}" TYPE="SECTION"><HEAD>{head}</HEAD><P>{body}</P></DIV8>'


def test_section_identifiers_keep_parentheses_bare_numbers_and_repeats_are_counted():
    """Real shapes seen on the 2026-10-03 full run: title 26 "§ 1.1(a)-1" (the first regex cut it
    to "1.1" and quarantined the whole title on a duplicate id), title 48 "1.000" with no "§"
    (11,152 sections excluded), title 14 "Section 03" (not a section number)."""
    from cytir.sources.families import us_federal_register_ecfr as E
    xml = ("<R>" + _sec("§ 1.1(a)-1", "§ 1.1(a)-1 First") + _sec("§ 1.1(h)-1", "§ 1.1(h)-1 Second")
           + _sec("1.000", "1.000 Scope of part.") + _sec("Section 03", "Section 03 Definitions")
           + _sec("§ 2.1", "§ 2.1 One") + _sec("§ 2.1", "§ 2.1 Again") + "</R>").encode()
    out = list(E.parse_title(xml, number=26, name="x", asof="2026-10-01"))
    kept = [p["sec"] for k, p in out if k == "section"]
    assert kept == ["1.1(a)-1", "1.1(h)-1", "1.000", "2.1"]
    assert sorted(p for k, p in out if k == "excluded") == ["duplicate_section_number",
                                                           "no_section_number"]


def test_the_federal_register_leg_is_refused(tmp_path):
    with pytest.raises(ValueError, match="leg='federal_register'"):
        _run(tmp_path, options={"leg": "federal_register"})


def test_a_run_that_admits_zero_sections_refuses(tmp_path):
    with pytest.raises(Exception) as exc:
        _run(tmp_path, http=_client(xml=EMPTY_TITLE))
    assert "title_without_sections" in str(exc.value) or "record" in str(exc.value).lower()

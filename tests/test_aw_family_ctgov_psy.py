"""`scratch_ctgov_psy_exact` on the family Protocol (task 1407).

No network. The fixture under `data/samples/ctgov_psy/` is SYNTHETIC: it follows the field paths the
code reads from the ClinicalTrials.gov v2 and EU CTIS responses, plus a small invented regulator page
(no live capture, `data/samples/ctgov_psy/README.md`). The golden rows are what the PRE-MOVE driver
produced over the same responses; the adapter must reproduce them byte for byte.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from cytir.sources import contract as contract_mod
from cytir.sources import registry
from cytir.sources.ctgov_psy import harvest as H
from cytir.sources.family import RunContext, SourceFamily, UnitExcluded
from cytir.sources.families import ctgov_psy as fam_mod
from cytir.sources.families.ctgov_psy import CtgovPsyFamily

REPO = pathlib.Path(__file__).resolve().parent.parent
FIX = REPO / "data" / "samples" / "ctgov_psy"
FAMILY = "scratch_ctgov_psy_exact"


class _Resp:
    def __init__(self, status, text, ctype="application/json"):
        self.status_code, self.text, self.headers = status, text, {"content-type": ctype}

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(self.status_code)


class _Fake:
    """The CurlClient shape, answering from the sealed fixture."""

    def __init__(self, html_status=200, robots="", html_error=None):
        self.html_status, self.robots, self.html_error = html_status, robots, html_error

    def get(self, url, params=None):
        if url.endswith("/robots.txt"):
            return _Resp(200 if self.robots else 404, self.robots)
        if url == H.CTG_API:
            name = "ctg-page-2.json" if (params or {}).get("pageToken") else "ctg-page-1.json"
            return _Resp(200, (FIX / name).read_text())
        if self.html_error:
            raise self.html_error
        return _Resp(self.html_status, (FIX / "regulator-page.html").read_text(), "text/html; charset=utf-8")

    def post_json(self, url, body):
        return _Resp(200, (FIX / "ctis-page-1.json").read_text())


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("AW_LANGGATE_OPTIONAL", "1")
    monkeypatch.setattr(H.fp, "detect_lang", lambda text, url, default: (default, False))
    monkeypatch.setattr(fam_mod, "PAUSE", lambda s: None)
    monkeypatch.setattr(H, "CTG_QUERIES", H.CTG_QUERIES[:2])
    monkeypatch.setattr(H, "CTIS_TERMS", ["psilocybin", "ketamine"])
    monkeypatch.setattr(H, "HTML_PAGES", H.HTML_PAGES[3:4])


def _row() -> dict:
    return contract_mod.by_id(contract_mod.load())[FAMILY]


def _ctx(tmp_path, http=None, contract=None, **options) -> RunContext:
    return RunContext(out_dir=tmp_path / "out", today="2026-10-30", skip_consent=True, http=http or _Fake(),
                      options=options, contract=contract or _row(), log=lambda m: None)


def _rows(fam, ctx):
    rows = []
    for unit in fam.discover(ctx):
        for cap in fam.acquire(unit, ctx):
            for doc in fam.extract(cap, ctx):
                rows.extend(fam.chunk(doc, ctx))
    return rows


def test_registered_and_conformant():
    assert FAMILY in registry.registered()
    adapter, row, spec = registry.admit(FAMILY)
    assert isinstance(adapter, SourceFamily)
    assert row["family_id"] == FAMILY and spec.declared == "dispatcher"
    assert adapter.PARTIAL == ("l1",)


def test_plan_lists_one_unit_per_slice_term_and_page(tmp_path):
    ctx = _ctx(tmp_path)
    units = list(CtgovPsyFamily().discover(ctx))
    assert [u.key.split("-")[0] for u in units] == ["ctg", "ctg", "ctis", "ctis", "html"]
    assert ctx.notes["units"] == {"ctg": 2, "ctis": 2, "html": 1}
    assert units[0].consent_hosts == ("clinicaltrials.gov",) and units[2].consent_hosts == ("euclinicaltrials.eu",)
    skip = list(CtgovPsyFamily().discover(_ctx(tmp_path, skip_ctis="1")))
    assert not any(u.key.startswith("ctis-") for u in skip)


def test_rows_equal_the_pre_move_golden(tmp_path):
    rows = _rows(CtgovPsyFamily(), _ctx(tmp_path))
    got = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    assert got == (FIX / "golden" / "records.jsonl").read_text()


def test_one_url_is_one_record_across_pages_and_units(tmp_path):
    rows = _rows(CtgovPsyFamily(), _ctx(tmp_path))
    urls = [r["url"] for r in rows if r["domain"] in ("clinicaltrials.gov", "euclinicaltrials.eu")]
    assert len(urls) == len(set(urls)) == 5          # NCT...01 sits on both pages and in the second slice


def test_consent_and_licence_are_read_from_the_row_and_stay_unknown(tmp_path):
    ctx = _ctx(tmp_path)
    CtgovPsyFamily().discover(ctx)
    t, row = ctx.notes["contract_terms"], _row()
    assert t["search_ok"] is row["consent"]["search_ok"] and t["train_ok"] is row["consent"]["train_ok"]
    assert t["consent_basis"] == row["consent"]["basis"] and t["licence_basis"] == row["licence"]["basis"]
    with pytest.raises(ValueError, match="contract row"):
        CtgovPsyFamily().discover(_ctx(tmp_path, contract={"family_id": FAMILY}))


def test_regulator_page_robots_and_waf_end_the_unit_excluded(tmp_path):
    fam = CtgovPsyFamily()
    (html,) = [u for u in fam.discover(_ctx(tmp_path)) if u.key.startswith("html")]
    with pytest.raises(UnitExcluded, match="robots"):
        fam.acquire(html, _ctx(tmp_path, http=_Fake(robots="User-agent: *\nDisallow: /")))
    with pytest.raises(UnitExcluded, match="http_403"):
        CtgovPsyFamily().acquire(html, _ctx(tmp_path, http=_Fake(html_status=403)))


def test_a_transport_error_or_5xx_is_an_instrument_failure_not_an_exclusion(tmp_path):
    fam = CtgovPsyFamily()
    (html,) = [u for u in fam.discover(_ctx(tmp_path)) if u.key.startswith("html")]
    with pytest.raises(RuntimeError, match="instrument failure"):
        CtgovPsyFamily().acquire(html, _ctx(tmp_path, http=_Fake(html_status=503)))
    with pytest.raises(OSError):
        CtgovPsyFamily().acquire(html, _ctx(tmp_path, http=_Fake(html_error=OSError("reset"))))


def test_ketamine_hit_is_kept_only_when_psychiatric(tmp_path):
    hits = json.loads((FIX / "ctis-page-1.json").read_text())["data"]
    rec, why = H.ctis_record(hits[1], "ketamine", "2026-10-30")
    assert rec is None and why == "ketamine_not_psy"
    rec, why = H.ctis_record(hits[1], "psilocybin", "2026-10-30")
    assert rec["trust_tier"] == 0.9 and rec["source_type"] == "registry_record"


def test_the_point_id_is_the_s2_card_id():
    from cytir.core.common import mint_card_point_id
    rec, _ = H.ctg_record(json.loads((FIX / "ctg-page-1.json").read_text())["studies"][0], "2026-10-30")
    assert rec["url"] == "https://clinicaltrials.gov/study/NCT00000001"
    assert rec["point_id"] == mint_card_point_id(rec["url"])
    assert rec["point_id"] == "c13c93ad-79c2-598e-875e-9dbdd7f32acb"      # the id the pre-move driver minted


def test_declared_chunk_policy_is_what_the_emitted_passages_carry(tmp_path):
    """The label cannot drift from the code: the regulator passages come out of the dispatcher's
    selection (`process_html`), and the declaration names exactly that; registry records carry no
    chunker stamps, so nothing else claims a policy."""
    from cytir.core import common as C
    from cytir.sources.family import CHUNK_DISPATCHER

    assert CtgovPsyFamily.chunk_policy_id == CHUNK_DISPATCHER == "dispatcher"
    rows = _rows(CtgovPsyFamily(), _ctx(tmp_path))
    passages = [r for r in rows if r["domain"] not in ("clinicaltrials.gov", "euclinicaltrials.eu")]
    records = [r for r in rows if r["domain"] in ("clinicaltrials.gov", "euclinicaltrials.eu")]
    assert passages and records
    assert all(r["stamps"]["chunk"] == C.chunker_selection().stamps()["chunk"] for r in passages)
    assert not any("stamps" in r for r in records)


def test_regulator_page_outlinks_stream_to_the_out_dir_and_the_sink_is_disarmed(tmp_path):
    ctx = _ctx(tmp_path)
    _rows(CtgovPsyFamily(), ctx)
    sink = ctx.out_dir / "outlinks.jsonl"
    lines = [json.loads(x) for x in sink.read_text().splitlines()]
    assert lines and all(x.get("page") or x.get("url") for x in lines), lines[:1]
    from cytir.core import common as C
    assert C._OUTLINK_SINK["fh"] is None

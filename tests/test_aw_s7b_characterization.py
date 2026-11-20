"""Characterization of `consent_gate`, the source-card builders, `cc_columnar`, `cc_digest` and
`corpus_store` BEFORE they move into `cytir/` (task 683, S7 part 2).

Written and pinned against the OLD flat modules first; the same file, unchanged, must pass
against the moved code (the old names are shims that alias the package modules). Each snapshot
is a canonical-JSON digest of real outputs over a fixed input grid, plus a few readable asserts
so a failure says what moved. Nothing here touches the network, a disk outside tmp_path, or a
model. Pins: `AW_PRINT_PINS=1 pytest -s tests/test_aw_s7b_characterization.py` prints them.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import types

import pytest

HERE = pathlib.Path(__file__).resolve().parent
for _p in (str(HERE.parent), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cytir.corpus import source_cards as BSC  # noqa: E402
from cytir.acquire import cc_columnar as CCC  # noqa: E402
from cytir.acquire import cc_digest as CCD  # noqa: E402
import cytir.consent.gate as CG  # noqa: E402

PINS = {
    "source_cards": "6a30f76d4a7e28c062ac244aa8b2fb2cb726c47fa998f344399d9a5635c524ef",
    "gate_bucket": "648a7c2e751ff78e4fc9e2b1de1555aa2d6bc148b906a3cd0286ba99c3f145d1",
    "gate_helpers": "e3459b0e1810cc74a1c3a4813668b684e4549d8973cefdc2e3ca4947d49ce329",
    "gate_run": "f045ecc52db2ad3ad64a5da4a332c577c56a7c1d1fb0ed9e056eb164b3d62efc",
    "gate_run_unclear": "4633385e9422d198a21318fd1a03fa96f3e672b17c0755ef090eae04eee7cd99",
    "cc_columnar_pure": "80437bae4583fc161ca93acd735bd7f24036480d21c530f95803711d93ab415f",
    "cc_digest": "1355975c3bb86723b1da40b9f9f488e36f9fd706c712e747f09f819b4442bc7b",
    "corpus_store": "5a60c5af6c583dbb65e03f602997d8d87213c254a920a1f38a3d6c3400b101a8",
}


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()).hexdigest()


def _check(name: str, obj) -> None:
    got = _digest(obj)
    if os.environ.get("AW_PRINT_PINS"):
        print(f"PIN {name} {got}")
        return
    assert got == PINS[name], f"{name}: behaviour moved (was {PINS[name][:12]}, now {got[:12]})"


# -- build_source_cards ---------------------------------------------------------------------

DOMAINS = ["admin.ch", "vs.ch", "zh.ch", "epfl.ch", "unibe.ch", "swisstopo.ch", "hug.ch",
           "lib4ri.ch", "landesmuseum.ch", "srf.ch", "nzz.ch", "nytimes.com",
           "theguardian.com", "doi.org", "example.org", "ahv-iv.ch", "meteoschweiz.ch"]
CLASSES = ["gov", "edu", "health", "ref", "news", "tech", "other", "unknown-class"]


def _cards_snapshot(tmp_path):
    grid = []
    for d in DOMAINS:
        for c in CLASSES:
            card = {"domain": d, "disc_class": c,
                    "authoritative_for": BSC.derive_topics(d, c, ["federal-gov", "news", "x"]),
                    "jurisdiction": "CH", "description": "desc"}
            grid.append({"d": d, "c": c, "topics": card["authoritative_for"],
                         "nav": BSC.navigation_for(d, c), "trust": BSC.trust_tier(card),
                         "disp": BSC.disposition_for(card), "text": BSC.card_text(card)})
    disc = tmp_path / "disc.json"
    disc.write_text(json.dumps([
        {"domain": "vs.ch", "swiss": True, "class": "gov", "score": 0.91234,
         "endorse_score": 0.5, "by_classes": ["federal-gov"], "why": "w"},
        {"domain": "nytimes.com", "swiss": False, "class": "news"},
        {"domain": "hug.ch", "swiss": True, "class": "health"}]))
    return {"grid": grid, "swiss": BSC.build_swiss_cards(str(disc)),
            "hn": BSC.build_hn_head_cards(),
            "consts": [BSC.CLASS_TRUST, BSC.CLASS_FRESHNESS, sorted(BSC.KNOWN_NAV),
                       sorted(BSC.PAYWALLED), sorted(BSC.PUBLIC_BROADCASTER),
                       sorted(BSC.KNOWN_OPEN_CORE)]}


def test_source_cards_are_unchanged(tmp_path):
    snap = _cards_snapshot(tmp_path)
    assert BSC.derive_topics("vs.ch", "gov")[:1] == ["government"]
    assert "canton VS" in BSC.derive_topics("vs.ch", "gov")
    assert BSC.navigation_for("admin.ch", "gov")["derivation"] == "pattern-default"
    _check("source_cards", snap)


# -- consent_gate ---------------------------------------------------------------------------

def _receipt(status="ok", search_ok=True, train_ok=True, reasons=(), ccbot=False):
    return {"status": status, "search_ok": search_ok, "train_ok": train_ok,
            "signals": {"robots": {"fetched": True, "http_status": 200, "wildcard_root": True}},
            "reasons": list(reasons), "checked_at": "2026-09-30T00:00:00Z"}


RECEIPTS = {
    "ok": _receipt(), "search_only": _receipt(train_ok=False),
    "declined": _receipt(search_ok=False, train_ok=False), "unknown": _receipt(status="unknown"),
    "error": _receipt(status="error"),
}


def test_gate_bucket_matrix_is_unchanged():
    rows = []
    for name, rec in RECEIPTS.items():
        for si in (False, True):
            for ag in (False, True):
                for uo in (False, True):
                    rows.append([name, si, ag, uo, CG.gate_bucket(
                        rec, search_indexing=si, accept_search_grant=ag, unclear_ok=uo)])
    assert CG.gate_bucket(RECEIPTS["ok"]) == "ok"
    _check("gate_bucket", rows)


def test_gate_helpers_are_unchanged(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    from cytir.consent.resolver import _cache_path  # the resolver's own cache layout
    p = _cache_path(str(cache), "robots.example")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"_robots_body": "User-agent: *\nDisallow: /private\n",
                             "signals": {"robots": {"fetched": True, "http_status": 200}}}))
    q = _cache_path(str(cache), "norobots.example")
    q.write_text(json.dumps({"signals": {"robots": {"fetched": True, "http_status": 404}}}))
    z = _cache_path(str(cache), "blocked.example")
    z.write_text(json.dumps({"signals": {"robots": {"fetched": True, "http_status": 403}}}))
    snap = {
        "robots": [CG._robots_allows(h, pth, str(cache)) for h, pth in [
            ("robots.example", "/"), ("robots.example", "/private/x"),
            ("norobots.example", "/"), ("blocked.example", "/"), ("absent.example", "/")]],
        "frag": [CG._clean_frag(s) for s in
                 ["<b>Bern &amp; Z&Auml;rich</b>  page", "", None, "x" * 400]],
        "guess": CG._guess_nav("a.example"),
        "group": CG._group_declined([
            {"host": "a.ch", "rows_excluded": 3}, {"host": "www.a.ch", "rows_excluded": 9},
            {"host": "x.b.ch", "rows_excluded": 1}, {"host": "www.c.ch", "rows_excluded": 2},
            {"host": "y.c.ch", "rows_excluded": 4}]),
        "host_of": [CG._host_of(u) for u in ["https://WWW.A.ch/x", "", "notaurl"]],
    }
    _check("gate_helpers", snap)


def _passages(path):
    rows = [{"url": "https://ok.example/a", "text": "a"}, {"url": "https://ok.example/b", "text": "b"},
            {"url": "https://search.example/a", "text": "c"},
            {"url": "https://no.example/a", "text": "d"}, {"url": "https://unk.example/a", "text": "e"},
            {"url": "", "text": "no host"}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + "\nnot json\n")


def _fake_resolve(hosts, **kw):
    table = {"ok.example": RECEIPTS["ok"], "search.example": RECEIPTS["search_only"],
             "no.example": RECEIPTS["declined"], "unk.example": RECEIPTS["unknown"]}
    return {h: table[h] for h in hosts}


def _run_gate(tmp_path, monkeypatch, **extra):
    _passages(tmp_path / "p.jsonl")
    monkeypatch.setattr(CG, "resolve_hosts", _fake_resolve)
    monkeypatch.setattr(CG.cr, "_now_iso", lambda: "2026-09-30T00:00:00Z")
    out = tmp_path / "out"
    args = types.SimpleNamespace(passages=str(tmp_path / "p.jsonl"), state="", cache_dir=str(tmp_path / "c"),
                                 out_dir=str(out), concurrency=1, direct_crawl=False,
                                 corpus_wave="", **extra)
    CG.cmd_gate(args)
    snap = {f.name: f.read_text().replace(str(tmp_path), "<TMP>") for f in sorted(out.iterdir())}
    return snap


def test_gate_run_is_unchanged(tmp_path, monkeypatch, capsys):
    snap = _run_gate(tmp_path, monkeypatch, accept_search_grant=True, search_indexing=False,
                     unclear_register="")
    s = json.loads(snap["gate_summary.json"])
    assert s["rows"]["total"] == 6 and s["hosts"]["total"] == 4
    assert json.loads(snap["passages_ok.jsonl"].splitlines()[0])["search_ok"] is True
    assert "passages_declined.jsonl" in snap and "passages_unknown.jsonl" in snap
    _check("gate_run", {"files": snap, "stdout": capsys.readouterr().out.replace(str(tmp_path), "<TMP>")})


def test_gate_run_with_the_unclear_register_is_unchanged(tmp_path, monkeypatch, capsys):
    reg = tmp_path / "register.yaml"
    snap = _run_gate(tmp_path, monkeypatch, accept_search_grant=False, search_indexing=True,
                     unclear_register=str(reg))
    _check("gate_run_unclear", {"files": snap,
                                "stdout": capsys.readouterr().out.replace(str(tmp_path), "<TMP>"),
                                "register_exists": reg.exists()})


# -- cc_columnar / cc_digest ----------------------------------------------------------------

def test_cc_columnar_pure_functions_are_unchanged():
    from datetime import datetime, timezone
    snap = {
        "surt": [CCC.url_surtkey(u) for u in [
            "https://www.admin.ch/de/nsb?id=1", "http://estv.admin.ch/", "https://BBC.co.uk/news/x",
            "not a url", ""]],
        "fetch_time": [CCC._fetch_time_iso(v) for v in [
            None, "", datetime(2026, 6, 15, 5, 45, 8), datetime(2026, 6, 15, 5, 45, 8, tzinfo=timezone.utc),
            "2026-06-15 05:45:08", "2026-06-15T05:45:08+00:00", "2026-06-15T05:45:08Z",
            "2026-06-15T05:45:08+02:00"]],
        "trunc": [CCC._truncation_metadata(v) for v in [None, "", "length", " disconnect ", "false"]],
        "record": CCC._record("https://x/", "f.warc.gz", "10", 5, "D", "2026-06-15 05:45:08", "length"),
        "mime": CCC._mime_predicate(("text/html", "application/pdf"), column="c"),
        "digest_pred": CCC._nonempty_digest_predicate(column="d"),
        "manifest": CCC._manifest_key("CC-MAIN-2026-25"),
        "prefix": [CCC.domain_surtkey_prefix(d) for d in ["lib4ri.ch", "estv.admin.ch", "", ".x.ch", "WWW.a.ch"]],
        "match": [CCC.match_prefix(k, ["ch,admin)", "ch,admin,estv)", "ch,lib4ri)"]) for k in [
            "ch,admin)/de", "ch,admin,estv)/x", "ch,lib4ri)/", "ch,zzz)/", "ch,admin)x"]],
    }
    with pytest.raises(ValueError):
        CCC._mime_predicate(("text/html'; DROP",), column="c")
    with pytest.raises(ValueError):
        CCC._mime_predicate((), column="c")
    _check("cc_columnar_pure", snap)


def test_cc_digest_is_unchanged():
    names = sorted(n for n in dir(CCD) if not n.startswith("_"))
    snap = {"names": names,
            "consts": {n: getattr(CCD, n) for n in names if n.isupper()}}
    _check("cc_digest", snap)


# -- corpus_store ---------------------------------------------------------------------------

def test_corpus_store_is_unchanged(tmp_path):
    from cytir.corpus import store as CS
    rows = [{"url": "https://a.example/x", "text": "hello", "title": "T", "point_id": None},
            {"url": "https://a.example/y", "text": "world", "title": "U", "lang": "de"}]
    p = tmp_path / "rows.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows) + "\n")
    snap = {
        "iter": list(CS._iter_jsonl(str(p))),
        "lines": CS._count_lines(str(p)),
        "sha": CS._sha256(str(p)),
        "pid": [CS.stable_point_id(r) for r in rows],
        "norm": [CS.normalize_row(r, "wave-x", "2026-09-30T00:00:00Z") for r in rows],
        "wave_dir": str(CS.wave_dir("/d", "w")),
    }
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        snap["parquet"] = "skipped: no pyarrow"
    else:
        dest = tmp_path / "store"
        meta = CS.write_wave(iter(rows), "wave-x", dest=str(dest), extraction_version="t",
                             force=True, meta={"k": "v"})
        snap["parquet"] = {"rows": meta["rows"], "count": CS.wave_row_count("wave-x", str(dest)),
                           "urls": sorted(r["url"] for r in CS.read_wave(
                               "wave-x", str(dest), columns=["url"]))}
    _check("corpus_store", snap)

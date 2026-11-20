#!/usr/bin/env python3
"""Exemplar 2 of the source-family interface (task 683, S7): a sitemap pack, ported.

`scratch_elcom_docs_exact` is the ElCom (federal electricity regulator) route of
`sources/ch/regulator-docs.yaml`. A live sitemap crawl cannot be byte-compared (the site moves
between runs), so two proofs, as task 434 established (`verify_authority_pack_equivalence.py`):

1. SELECTION EQUIVALENCE. The thing the port changed is the selection decision: before, the
   route's YAML `include`/`exclude` regexes in `data/cov351_routes.yaml` applied by
   `cytir.sources.cov.engine.enum_sitemap`; after, the pack's compiled `select`/`lang` applied by the
   family's `discover`. Both run over the SAME inventory (ElCom's three real sitemaps, sealed,
   plus a synthetic tail of adversarial URLs: other hosts, `http://`, query strings, duplicates,
   a `/kontakt` page, a PDF) and must agree on every URL and its language.
2. A GOLDEN over the sealed pages (four pages of the real site): the records, terminals and
   discover counts are pinned, so a change in extraction, chunking, identity or stamping shows.

The recording is `fixtures/sitemap_elcom/tape.json` (task 683's elcom tape recorder).
No network; a request the tape does not hold fails the test.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys

import pytest

httpx = pytest.importorskip("httpx")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
FIX = HERE / "fixtures" / "sitemap_elcom" / "tape.json"
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
from tests.helpers.repo_files import repo_file_or_skip
from cytir.core import model_pin  # noqa: E402

FAMILY = "scratch_elcom_docs_exact"
POLICY_FILE = REPO / "data" / ("chunk-policy-struct-md-768w-o0-adjacent-sections-"
                               "tail-1152-tok2560.v1.json")
POLICY_SHA = hashlib.sha256(POLICY_FILE.read_bytes()).hexdigest()
OK = {"bucket": "ok", "status": "ok", "search_ok": True, "train_ok": True,
      "checked_at": "2026-09-30T00:00:00Z"}
TODAY = "2026-09-30"

#: the golden, pinned from this tape on 2026-10-10 (task 852) under the DECLARED generic epoch-4
#: policy (word-bounded only: no tokenizer in CI), lang gate off. Was 3 records / d9cb20bd... under
#: the legacy 180w chunker (2026-09-30), and 5 records / 89943428... before the framework soft-404
#: exclusion dropped /de/404 and /de/500.
GOLDEN = {"records": 2, "sha256": "344c35d8ad99bc3e09888b58acd78532249953384a41492437a7c0a2cd1ffb97",
          "sha256_stamped": "8047bab835702e8d97788ec63f99dfa913d6b5a34f3b980036d81d76e25456e3"}


@pytest.fixture
def regdocs_pack():
    """The pack is withheld from the public cut: the tests that read it skip there."""
    return repo_file_or_skip("sources/ch/regulator-docs.yaml")


@pytest.fixture(autouse=True)
def _offline_deterministic(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)        # the pack's politeness floor
    monkeypatch.setenv("AW_LANGGATE_OPTIONAL", "1")
    monkeypatch.setenv("AW_LID_MODEL", "/nonexistent/lid.176.bin")   # the gate is off everywhere
    select_declared_policy(monkeypatch)
    yield
    restore_chunker_defaults()


def select_declared_policy(monkeypatch):
    """Select the family's declared chunk policy the way a candidate run does (task 852: the
    contract row now declares the generic epoch-4 policy, so the dispatcher must select it)."""
    monkeypatch.setenv("AW_CHUNKER", "structured")
    monkeypatch.setenv("AW_CHUNK_POLICY", str(POLICY_FILE))
    monkeypatch.setenv("AW_CHUNK_POLICY_SHA256", POLICY_SHA)
    monkeypatch.setenv("AW_CHUNK_TOKENIZER_REVISION", model_pin.EMBED_MODEL_REVISION)


def restore_chunker_defaults():
    """`chunk_markdown_channels` ACTIVATES the policy in `cytir.chunk.chunker` globals; put the
    import-time default back so later tests are not affected (same as test_aw_family_cc)."""
    from cytir.chunk import chunker
    chunker._apply_policy(chunker.load_chunk_policy(None), allow_env_overrides=True)
    chunker.select_tokenizer_revision(None)


def _tape(extra=None, replace=None):
    from cytir.acquire.tape import Tape
    doc = json.loads(FIX.read_text(encoding="utf-8"))
    entries = [e for e in doc["tape"] if not (replace and e["url"] in replace)]
    entries += (extra or []) + list((replace or {}).values())
    return Tape(entries)


def _pages(tape) -> list[str]:
    return [e["url"] for e in doc_entries() if not e["url"].endswith((".xml", "robots.txt"))]


def doc_entries():
    return json.loads(FIX.read_text(encoding="utf-8"))["tape"]


def _run(tmp_path, tape=None, **kw):
    from cytir.sources import run_family
    tape = tape or _tape()
    opts = kw.pop("options", {"url": _pages(tape)})
    manifest = run_family.run_family(
        FAMILY, out_dir=tmp_path, today=TODAY, http=tape.client(),
        consent=kw.pop("consent", lambda h, d: {**OK, "host": h}), options=opts,
        log=lambda m: None, **kw)
    return manifest, tape


# -- proof 1: selection equivalence ----------------------------------------------------

def _xml(urls):
    return ('<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            + "".join(f"<url><loc>{u}</loc></url>" for u in urls) + "</urlset>")


ADVERSARIAL = [
    "https://www.elcom.admin.ch/de/aufgaben",                    # duplicate of a real URL
    "http://www.elcom.admin.ch/de/aufgaben",                     # scheme variant: same page
    "https://WWW.ELCOM.ADMIN.CH/de/aufgaben?utm_source=x",       # host case + query: same page
    "https://www.elcom.admin.ch/de/kontakt",                     # excluded by path_deny_regex
    "https://www.elcom.admin.ch/fr/newsletter-abonnieren",       # excluded
    "https://www.elcom.admin.ch/en/home",                        # /en/ is not in (de|fr|it)
    "https://www.elcom.admin.ch/dam/de/sd-web/abc/weisung.pdf",  # PDF under /dam/: not in scope
    "https://www.elcom.admin.ch/it/documenti/guida.pdf",         # PDF inside scope
    "https://elcom.example.org/de/fremde-seite",                 # another host
    "https://www.elcom.admin.ch/",                               # root
]


def _inventory_tape():
    real = {e["url"]: e for e in doc_entries() if e["url"].endswith(".xml")}
    de = re.findall(r"<loc>(.*?)</loc>", real["https://www.elcom.admin.ch/sitemap/de.xml"]["body"])
    repl = {"https://www.elcom.admin.ch/sitemap/de.xml": {
        **real["https://www.elcom.admin.ch/sitemap/de.xml"], "body": _xml(de + ADVERSARIAL)}}
    return _tape(replace=repl)


def test_selection_equals_the_old_enumeration_on_the_same_inventory(tmp_path, regdocs_pack):
    from cytir.sources.cov import engine as COV
    from cytir.sources import registry

    cfg = COV.load_routes(str(REPO / "data" / "cov351_routes.yaml"),
                          "regdocs2")["sources"]["elcom_docs"]
    old = {row["url"]: row["lang"]
           for row in COV.enum_sitemap(_inventory_tape().client(), cfg, None, lambda m: None)}

    adapter, _row, spec = registry.admit(FAMILY)
    from cytir.sources.family import RunContext
    ctx = RunContext(out_dir=tmp_path, today=TODAY, http=_inventory_tape().client(),
                     chunk_spec=spec)
    new = {u.url: u.meta["lang"] for u in adapter.discover(ctx)}

    assert old == new, (sorted(set(old) ^ set(new))[:10])
    assert len(new) > 50
    # the tail really exercised the rules: each adversarial class is decided, and decided alike
    assert "https://www.elcom.admin.ch/de/kontakt" not in new
    assert "https://www.elcom.admin.ch/en/home" not in new
    assert "https://www.elcom.admin.ch/dam/de/sd-web/abc/weisung.pdf" not in new
    assert "https://elcom.example.org/de/fremde-seite" not in new
    assert new["https://www.elcom.admin.ch/de/aufgaben"] == "de"
    assert new["https://www.elcom.admin.ch/it/documenti/guida.pdf"] == "it"   # in scope (PDF)
    # canonicalisation collapsed the three spellings of one page into one unit
    assert list(new).count("https://www.elcom.admin.ch/de/aufgaben") == 1
    n = ctx.notes
    assert n["other_host_locs"] == 1 and n["inventory_locs"] > n["inventory_canonical"]


def test_the_pdf_is_a_counted_exclusion_not_a_silent_drop(tmp_path, word_only_chunking_allowed, regdocs_pack):
    pdf = "https://www.elcom.admin.ch/it/documenti/guida.pdf"
    manifest, _ = _run(tmp_path, _inventory_tape(),
                       options={"url": [pdf, "https://www.elcom.admin.ch/de/aufgaben"]})
    t = manifest["terminals"]
    assert t["units_in"] == 2 and t["units_admitted"] == 1
    assert t["excluded_reasons"] == {"unsupported_kind:pdf": 1}


# -- proof 2: the golden ---------------------------------------------------------------

def test_golden_over_the_sealed_pages(tmp_path, word_only_chunking_allowed, regdocs_pack):
    manifest, tape = _run(tmp_path)
    assert tape.unexpected == []
    t = manifest["terminals"]
    assert (t["units_in"], t["units_admitted"], t["units_excluded"],
            t["units_quarantined"]) == (4, 2, 2, 0)
    # /de/404 and /de/500 answer HTTP 200 with an error page: framework soft-404 exclusion
    assert t["excluded_reasons"] == {"soft_404:title_0": 1, "soft_404:title_6": 1}
    assert manifest["records"]["count"] == GOLDEN["records"]
    # ADR-124: the door stamps `family_id`; the manifest digest is of the stamped bytes (re-pinned
    # 2026-10-30) and the unstamped bytes still hash to the original golden.
    assert manifest["records"]["sha256"] == GOLDEN["sha256_stamped"]
    raw = (tmp_path / "pack" / "records.jsonl").read_text(encoding="utf-8").splitlines()
    unstamped = "".join(json.dumps({k: v for k, v in json.loads(x).items() if k != "family_id"},
                                   ensure_ascii=False) + "\n" for x in raw)
    assert hashlib.sha256(unstamped.encode("utf-8")).hexdigest() == GOLDEN["sha256"]
    rows = [json.loads(line) for line in
            (tmp_path / "pack" / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["domain"] for r in rows} == {"elcom.admin.ch"}
    assert {r["license"] for r in rows} == {"ch-gov-open-cite"}          # from the pack's route
    assert {r["trust_tier"] for r in rows} == {0.95} and {r["tier"] for r in rows} == {"A"}
    assert {r["last_seen"] for r in rows} == {TODAY}
    assert all(r["point_id"] and r["text"] for r in rows)
    d = manifest["discover"]
    assert d["pack"] == "ch-regulator-docs" and d["cap"] == 900 and d["units"] == 4
    assert d["chunker"]["mode"] == "structured"


def test_the_run_is_reproducible_byte_for_byte(tmp_path, word_only_chunking_allowed, regdocs_pack):
    a, _ = _run(tmp_path / "a")
    b, _ = _run(tmp_path / "b")
    assert (tmp_path / "a/pack/records.jsonl").read_bytes() == \
        (tmp_path / "b/pack/records.jsonl").read_bytes()
    assert a["records"]["sha256"] == b["records"]["sha256"]


def test_the_seal_holds_each_pages_exact_bytes(tmp_path, word_only_chunking_allowed, regdocs_pack):
    import hashlib
    _run(tmp_path)
    index = [json.loads(line) for line in
             (tmp_path / "l0" / "index.jsonl").read_text().splitlines()]
    assert len(index) == 4
    by_url = {e["url"]: e["body"].encode("utf-8") for e in doc_entries()}
    for row in index:
        data = (tmp_path / "l0" / row["sha256"][:2] / row["sha256"]).read_bytes()
        assert data == by_url[row["url"]] and hashlib.sha256(data).hexdigest() == row["sha256"]
        assert row["meta"]["status"] == 200


# -- terminals: whose failure is it? ---------------------------------------------------

def test_a_404_is_the_origins_answer_a_503_is_ours(tmp_path, word_only_chunking_allowed, regdocs_pack):
    gone = "https://www.elcom.admin.ch/de/aufgaben"
    flaky = "https://www.elcom.admin.ch/de/anmeldung-forum"
    tape = _tape(replace={
        gone: {"method": "GET", "url": gone, "status": 404, "content_type": "text/html",
               "body": "gone"},
        flaky: {"method": "GET", "url": flaky, "status": 503, "content_type": "text/html",
                "body": "busy"}})
    manifest, _ = _run(tmp_path, tape)
    t = manifest["terminals"]
    assert t["excluded_reasons"] == {"origin_gone:404": 1, "soft_404:title_0": 1,
                                     "soft_404:title_6": 1}
    assert t["quarantined_reasons"] == {"acquire_failed:RuntimeError": 1}
    assert t["units_in"] == t["units_admitted"] + t["units_excluded"] + t["units_quarantined"]


def test_when_every_unit_fails_it_is_an_outage_not_an_empty_source(tmp_path, regdocs_pack):
    from cytir.sources import run_family
    tape = _tape(replace={u: {"method": "GET", "url": u, "status": 503,
                              "content_type": "text/html", "body": "x"}
                          for u in _pages(_tape())})
    with pytest.raises(run_family.RunError, match="outage"):
        _run(tmp_path, tape)


def test_an_unreadable_sitemap_is_a_loud_discovery_error(tmp_path, regdocs_pack):
    from cytir.sources.families.sitemap import DiscoveryError
    bad = "https://www.elcom.admin.ch/sitemap/fr.xml"
    tape = _tape(replace={bad: {"method": "GET", "url": bad, "status": 500,
                                "content_type": "text/plain", "body": ""}})
    with pytest.raises(DiscoveryError, match="fr.xml"):
        _run(tmp_path, tape)


def test_dry_run_reads_no_page_and_writes_nothing(tmp_path, regdocs_pack):
    tape = _tape()
    from cytir.sources import run_family
    plan = run_family.run_family(
        FAMILY, out_dir=tmp_path / "out", today=TODAY, http=tape.client(),
        consent=lambda h, d: {**OK, "host": h}, options={"url": _pages(tape)},
        dry_run=True, log=lambda m: None)
    assert plan["dry_run"] and len(plan["runnable"]) == 4
    assert not (tmp_path / "out").exists()
    page_fetches = [k for k in tape.served if not k[1].endswith(".xml")]
    assert page_fetches == []                       # only the sitemaps were read


def test_the_family_is_declared_and_its_declared_name_is_a_task_era_one():
    from cytir.sources import contract
    row = contract.by_id(contract.load())[FAMILY]
    assert row["status"] == "backfilled"
    assert FAMILY.startswith("scratch_")            # proposed rename: task 683's table

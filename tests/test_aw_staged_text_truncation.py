#!/usr/bin/env python3
"""Task 706: staged text is never cut by a cap, and a stage-time guard says so when it is.

Real records only. `tests/fixtures/legi_loda_long/` holds four REAL in-force DILA LEGI articles
whose embed text is over 8,000 characters (source, sha and the personal-data check in its
PROVENANCE.json). `data/samples/legi-loda` holds real short articles.

What this pins:
  1. a long article is split on paragraph boundaries, every chunk carries the same title and
     article header, no paragraph is lost (the old `[:8000]` lost 6 to 94 % of these four);
  2. a short article stages byte-identically to the task-633 row (the legacy formula is
     written out below, not imported: it no longer exists in the code);
  3. the guard FAILS on the task-633 behaviour (commit f2199fac: `[:8000]`) for every one of the
     four, and passes the real chunking;
  4. through `run_family`, a family that cuts is quarantined `staged_text_shortfall` and the
     pack manifest counts it; the declared caps of dila_juri / dila_kali / cards are REPORTED.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys

import pytest

pytest.importorskip("lxml")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

LONG = HERE / "fixtures" / "legi_loda_long"
SHORT = REPO / "data" / "samples" / "legi-loda"
PROV = json.loads((LONG / "PROVENANCE.json").read_text(encoding="utf-8"))
LONG_MEMBERS = sorted(PROV["files"])
RELEASE = "aw-task633-dila-loda-full-v1"


STAND_IN_CHARS_PER_TOKEN = 3


@pytest.fixture(autouse=True)
def _stand_in_tokenizer(monkeypatch):
    """CI has no `transformers`: a deterministic stand-in counter (3 chars per token) takes the
    place of the arctic tokenizer. The REAL tokenizer is exercised by
    `test_real_tokenizer_rows_fit_the_embed_window` wherever the model is cached."""
    from cytir.sources.families import loda
    monkeypatch.setattr(loda, "embed_token_counter",
                        lambda: (lambda text: len(text) // STAND_IN_CHARS_PER_TOKEN))


def _read(member: str) -> bytes:
    return (LONG / member).read_bytes()


def _build(member: str, body: bytes):
    from cytir.sources.families import loda
    accept, reason = loda.evaluate_article(body)
    assert accept is not None, reason
    sha = hashlib.sha256(body).hexdigest()
    return loda.build_records(accept, body=body, archive_member=member, archive_sha256=sha,
                              source_release=RELEASE)


def _legacy_row(l1: dict, accept: dict) -> dict:
    """The task-633 row (commit f2199fac), written out: the embed text cut at 8000."""
    return {
        "point_id": l1["point_id"], "text": f"{l1['title']}\n\n{l1['markdown']}"[:8000],
        "url": l1["url"], "anchor": l1["anchor"], "title": l1["title"],
        "nature": accept["nature"], "article_id": accept["aid"],
        "source_record_sha256": l1["source_record_sha256"],
        "country": "FR", "provenance": "loda", "domain": "legifrance.gouv.fr",
        "lang": "fr", "tier": "A", "source_type": "official", "trust_tier": 1.0,
        "license": "Etalab-2.0", "consent_state": "dump",
        "consent_basis": "official-dump-license",
    }


def test_the_fixture_bytes_are_the_ones_the_provenance_names():
    for member, meta in PROV["files"].items():
        assert hashlib.sha256(_read(member)).hexdigest() == meta["sha256"], member
    assert PROV["natural_person_data"].startswith("none found")


@pytest.mark.parametrize("member", LONG_MEMBERS)
def test_a_long_article_is_split_not_cut(member):
    from cytir.extract import legi
    l1, rows, source = _build(member, _read(member))
    assert len(source) == PROV["files"][member]["embed_chars"]
    assert len(rows) >= 2
    # anchors, ids, urls
    ids = {r["point_id"] for r in rows}
    assert len(ids) == len(rows)
    aid = rows[0]["article_id"]
    assert [r["anchor"] for r in rows] == [f"article:{aid}:chunk-{i}" for i in range(1, len(rows) + 1)]
    assert {r["url"] for r in rows} == {l1["url"]}
    # no row over the bound (the header leaves room for body text in all four)
    assert max(len(r["text"]) for r in rows) <= legi.ROW_CHARS
    # the representation contract is the same on every chunk: title line + the whole header
    header = l1["markdown"].split("\n\n### Paragraph ")[0]
    for r in rows:
        assert r["text"].startswith(f"{l1['title']}\n\n{header}\n\n### Paragraph ")
    # nothing is lost: every paragraph's words are in the rows, in order
    para_re = re.compile(r"### Paragraph (\d+)(?: \(part \d+/\d+\))? \{#[^}]+\}\n\n([^\n]*)")
    want = [(m.group(1), m.group(2)) for m in para_re.finditer(l1["markdown"])]
    got: dict[str, list[str]] = {}
    for r in rows:
        body = r["text"][len(l1["title"]) + 2 + len(header):]
        for m in para_re.finditer(body):
            got.setdefault(m.group(1), []).append(m.group(2))
    assert [n for n, _ in want] == list(got)
    for n, text in want:
        assert " ".join(got[n]) == text, f"paragraph {n} changed"
    # the old cut would have lost text; the chunks lose none
    from cytir.sources import staged_text_guard as G
    assert G.coverage(source, [r["text"] for r in rows]) >= G.MIN_COVERAGE
    assert G.coverage(source, [source[:8000]]) < G.MIN_COVERAGE


def test_a_paragraph_larger_than_one_chunk_is_split_on_spaces_into_numbered_parts():
    member = next(m for m in LONG_MEMBERS if "LEGIARTI000022732629" in m)
    _l1, rows, _src = _build(member, _read(member))
    assert len(rows) == 2                    # one 11,017-char paragraph
    assert "(part 1/2)" in rows[0]["text"] and "(part 2/2)" in rows[1]["text"]


def _kept_short_samples():
    from cytir.sources.families import loda
    return [p for p in sorted(SHORT.rglob("LEGIARTI*.xml"))
            if loda.evaluate_article(p.read_bytes())[0] is not None]


@pytest.mark.parametrize("path", _kept_short_samples(), ids=lambda p: p.name)
def test_a_short_article_stages_byte_identically_to_the_task_633_row(path):
    from cytir.sources.families import loda
    body = path.read_bytes()
    accept, _reason = loda.evaluate_article(body)
    l1, rows, _src = loda.build_records(accept, body=body, archive_member="m",
                                        archive_sha256="0" * 64, source_release=RELEASE)
    assert len(l1["markdown"]) + len(l1["title"]) + 2 <= 8000
    assert len(rows) == 1
    assert json.dumps(rows[0], sort_keys=False) == json.dumps(_legacy_row(l1, accept))


def test_the_borderline_article_is_two_balanced_rows():
    member = next(m for m in LONG_MEMBERS if "LEGIARTI000039233648" in m)
    _l1, rows, src = _build(member, _read(member))
    assert len(src) == 8502 and len(rows) == 2
    a, b = (len(r["text"]) for r in rows)
    assert abs(a - b) < 1500                 # balanced, not 8000 + 700


# --- the guard ------------------------------------------------------------------------------

def test_the_guard_catches_the_task_633_cut_on_all_four_real_articles():
    """The test that would have caught f2199fac: staged = source[:8000]."""
    from cytir.sources import staged_text_guard as G
    for member in LONG_MEMBERS:
        _l1, _rows, source = _build(member, _read(member))
        guard = G.StagedTextGuard("loda_exact")
        cov = guard.observe(member, source, [source[:8000]])
        assert guard.shortfall(cov) and guard.docs_shortfall == 1, member
        assert guard.summary()["shortfall_examples"][0]["staged_chars"] == 8000


def test_the_guard_does_not_flag_headers_whitespace_or_a_short_row():
    from cytir.sources import staged_text_guard as G
    guard = G.StagedTextGuard("x")
    text = "Article 1 du code. " * 40
    assert guard.observe("u", text, ["titre\n\n" + text.replace(" ", "  ")]) == 1.0
    assert guard.observe("u2", "", []) == 1.0
    assert guard.observe("u3", None, ["whatever"]) is None
    s = guard.summary()
    assert (s["docs_checked"], s["docs_unchecked"], s["docs_shortfall"]) == (2, 1, 0)


def _stage_root(tmp_path):
    """The four long + the real short samples under one `TNC_en_vigueur` root."""
    import shutil
    root = tmp_path / "root"
    shutil.copytree(LONG / "TNC_en_vigueur", root / "TNC_en_vigueur")
    for p in SHORT.rglob("LEGIARTI*.xml"):
        dest = root / "TNC_en_vigueur" / "JORF" / p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, dest)
    return root


def _run(tmp_path, name, **options):
    pytest.importorskip("httpx")
    pytest.importorskip("qdrant_client")
    from cytir.sources import run_family
    return run_family.run_family(
        "loda_exact", out_dir=tmp_path / name, today="2026-10-05",
        consent=lambda h, d: pytest.fail(f"consent for {h}"),
        options={"root": str(_stage_root(tmp_path)), "release": RELEASE, **options},
        log=lambda m: None)


def test_only_split_restages_just_the_articles_that_split(tmp_path):
    """The restage of the task-633 cut: single-row articles are byte-identical to the served rows,
    so they end `excluded(single_row_unchanged)` and only the split articles are staged."""
    man = _run(tmp_path, "split", only_split="true")
    t = man["terminals"]
    assert t["units_admitted"] == 4 and t["excluded_reasons"]["single_row_unchanged"] == 3
    assert {r["article_id"] for r in map(json.loads, (tmp_path / "split" / "pack" /
            "records.jsonl").read_text().splitlines())} == {
        "LEGIARTI000049864993", "LEGIARTI000039233648", "LEGIARTI000006680074",
        "LEGIARTI000022732629"}


def test_run_family_admits_every_long_article_with_zero_shortfall(tmp_path):
    man = _run(tmp_path, "ok")
    g = man["staged_text_guard"]
    assert g["docs_checked"] == man["terminals"]["units_admitted"] == 7
    assert g["docs_shortfall"] == 0 and g["lowest_coverage"] >= 0.98
    assert man["terminals"]["quarantined_reasons"] == {}
    # 4 long articles -> >= 8 rows, 3 short -> 3 rows
    assert man["records"]["count"] >= 11
    assert g["declared_caps"] == []           # loda declares no cap any more


def test_run_family_quarantines_a_family_that_cuts_and_counts_it(tmp_path, monkeypatch):
    """f2199fac re-created: the family's chunk step cuts at 8000. The unit is quarantined with
    the reason, the bytes stay sealed, the manifest counts it."""
    from cytir.sources.families import loda
    real = loda.LodaFamily.chunk

    def cutting(self, doc, ctx):
        rows = real(self, doc, ctx)
        whole = {**rows[0], "text": doc.source_text[:8000]}
        return [whole]

    monkeypatch.setattr(loda.LodaFamily, "chunk", cutting)
    man = _run(tmp_path, "cut")
    g = man["staged_text_guard"]
    assert g["docs_shortfall"] == 4 and len(g["shortfall_examples"]) == 4
    assert man["terminals"]["quarantined_reasons"] == {"staged_text_shortfall": 4}
    assert man["terminals"]["units_admitted"] == 3
    assert man["terminals"]["units_in"] == 12      # 7 kept + 5 excluded (below floor, not in force)
    assert all(e["staged_chars"] == 8000 for e in g["shortfall_examples"])


# --- declared caps (reported, not changed) ----------------------------------------------------

def test_declared_caps_are_reported_with_row_counts_for_juri_only():
    """`dila_kali` and the entity cards lost their caps (ADR-090 am., ADR-060 am., 2026-10-07):
    only `dila_juri` still declares one."""
    from cytir.sources import staged_text_guard as G
    cut_juri = "Titre\nSolution : x\n" + "a" * 2000
    assert [c["rows_at_cap"] for c in G.count_declared_caps("dila_juri", [cut_juri, "court"])] == [1]
    (c,) = G.count_declared_caps("dila_juri", ["x"])
    assert c["founder_question"] and c["where"] and c["status"].startswith("declared")
    for fam in ("dila_kali", "task467_entity_card_1024_reembed", "loda_exact"):
        assert G.count_declared_caps(fam, ["x" * 4000]) == []


def test_the_declared_caps_name_the_lines_they_live_on():
    """If someone moves or removes a cap, this fails and the declaration is updated with it."""
    from cytir.extract import dila_juri, dila_kali
    assert dila_juri.MAX_BODY_CHARS == 2000
    assert not hasattr(dila_kali, "MAX_ARTICLE_CHARS")      # the 4,000-char cap is gone


# --- the embed window ----------------------------------------------------------------------

def test_a_chunk_never_exceeds_the_embed_window_the_budget_shrinks_until_it_fits():
    """Characters alone cannot promise the window (on the real population an 8,000-char chunk is
    1,500 to 3,700 tokens). With a denser tokenizer the same article splits into more, smaller
    chunks and still loses nothing; with no counter a split is refused, never guessed."""
    from cytir.extract import legi
    from cytir.sources import staged_text_guard as G
    member = next(m for m in LONG_MEMBERS if "LEGIARTI000049864993" in m)
    l1, _rows, source = _build(member, _read(member))
    title, md = l1["title"], l1["markdown"]
    sparse = legi.split_article_markdown(md, title=title, count_tokens=lambda t: len(t) // 3)
    dense = legi.split_article_markdown(md, title=title, count_tokens=lambda t: len(t) // 2)
    assert len(dense) > len(sparse)
    for chunks, per in ((sparse, 3), (dense, 2)):
        assert max(len(f"{title}\n\n{c}") // per for c in chunks) <= legi.MAX_ROW_TOKENS
        assert G.coverage(source, [f"{title}\n\n{c}" for c in chunks]) >= G.MIN_COVERAGE
    with pytest.raises(ValueError, match="count_tokens"):
        legi.split_article_markdown(md, title=title)
    # a short article never asks for the counter
    assert legi.split_article_markdown("# t\n\nshort", title="t") == ["# t\n\nshort"]


@pytest.mark.parametrize("member", LONG_MEMBERS)
def test_real_tokenizer_rows_fit_the_embed_window(member, monkeypatch):
    pytest.importorskip("transformers")
    from cytir.sources.families import loda
    monkeypatch.undo()                          # the real counter, not the stand-in
    try:
        counter = loda.embed_token_counter()
    except Exception as exc:                    # model not cached on this machine
        pytest.skip(f"embed tokenizer not available: {exc}")
    _l1, rows, _src = _build(member, _read(member))
    assert len(rows) >= 2
    assert max(counter(r["text"]) for r in rows) <= 2560


# --- review fixes (task 706, Opus review: MERGE AFTER FIXES) -----------------------------------

CUT8000 = HERE / "fixtures" / "legi_loda_cut8000"
CUT_PROV = json.loads((CUT8000 / "PROVENANCE.json").read_text(encoding="utf-8"))
CUT_MEMBER = next(iter(CUT_PROV["files"]))
PINNED = "ac6544c8a46e00af67e330e85a9028c66b8cfd9a"


def _tokenizer_missing(monkeypatch):
    from cytir.chunk.token_counter import TokenCounterError
    from cytir.sources.families import loda

    def boom():
        raise TokenCounterError("no such model in the cache")
    monkeypatch.setattr(loda, "embed_token_counter", boom)


def test_a_missing_tokenizer_refuses_the_run_at_start_and_stages_no_row(tmp_path, monkeypatch):
    """Fix 1. Before: the short articles were admitted and every long one was quarantined
    `extract_failed:TokenCounterError`: a stageable pack that had silently lost all long articles."""
    pytest.importorskip("httpx")
    pytest.importorskip("qdrant_client")
    from cytir.sources import run_family
    _tokenizer_missing(monkeypatch)
    with pytest.raises(run_family.RunError) as err:
        _run(tmp_path, "norun")
    msg = str(err.value)
    assert PINNED in msg and "Snowflake/snowflake-arctic-embed-l-v2.0" in msg
    assert "snapshot_download" in msg and "\n" not in msg           # one clear line, with the fix
    assert not (tmp_path / "norun" / "pack").exists()


def test_an_empty_hf_cache_refuses_the_run_with_the_real_counter(tmp_path):
    """Fix 1, end to end: a fresh process, an empty HF cache, offline. Nothing is stubbed."""
    pytest.importorskip("httpx")
    pytest.importorskip("qdrant_client")
    import os
    import subprocess
    root = _stage_root(tmp_path)
    code = (
        "from cytir.sources import run_family\n"
        f"run_family.run_family('loda_exact', out_dir={str(tmp_path / 'cold')!r}, "
        "today='2026-10-05', consent=lambda h, d: None, log=lambda m: None, "
        f"options={{'root': {str(root)!r}, 'release': {RELEASE!r}}})\n")
    env = {**os.environ, "HF_HOME": str(tmp_path / "empty-hf"),
           "HUGGINGFACE_HUB_CACHE": str(tmp_path / "empty-hf" / "hub"),
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONPATH": str(REPO)}
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode != 0
    assert "RunError" in proc.stderr and PINNED in proc.stderr
    assert not (tmp_path / "cold" / "pack" / "records.jsonl").exists()


def test_the_loda_threshold_is_1_0_in_one_place_and_the_manifest_records_it(tmp_path):
    from cytir.sources import staged_text_guard as G
    from cytir.sources.families import loda
    assert loda.LodaFamily.staged_text_min_coverage == 1.0 and G.MIN_COVERAGE == 0.98
    g = _run(tmp_path, "thr")["staged_text_guard"]
    assert g["min_coverage"] == 1.0 and g["threshold_declared"] is True
    assert g["docs_shortfall"] == 0 and g["lowest_coverage"] == 1.0


def test_a_real_article_of_8015_chars_cut_at_8000_is_flagged_at_1_0_and_split_loses_nothing():
    """Fix 2. A REAL article of 8,001-8,100 chars (the 325 cuts that 0.98 let through were every
    cut article up to 8,100 chars). Provenance, sha and the personal-data check:
    tests/fixtures/legi_loda_cut8000/PROVENANCE.json."""
    from cytir.sources import staged_text_guard as G
    from cytir.sources.families import loda
    body = (CUT8000 / CUT_MEMBER).read_bytes()
    assert hashlib.sha256(body).hexdigest() == CUT_PROV["files"][CUT_MEMBER]["sha256"]
    assert CUT_PROV["natural_person_data"].startswith("none found")
    accept, reason = loda.evaluate_article(body)
    assert accept is not None, reason
    l1, rows, source = loda.build_records(accept, body=body, archive_member=CUT_MEMBER,
                                          archive_sha256="0" * 64, source_release=RELEASE)
    assert 8001 <= len(source) <= 8100
    assert len(source) == CUT_PROV["files"][CUT_MEMBER]["embed_chars"]
    cut = [source[:8000]]                                    # the task-633 row
    threshold = loda.LodaFamily.staged_text_min_coverage
    guard = G.StagedTextGuard("loda_exact", min_coverage=threshold)
    assert guard.shortfall(guard.observe(CUT_MEMBER, source, cut)), "the guard let the cut pass"
    # the old default let exactly this class through: that is why the family declares 1.0
    assert G.coverage(source, cut) >= G.MIN_COVERAGE
    # and the real split holds all of it
    assert len(rows) == 2
    assert G.coverage(source, [r["text"] for r in rows]) == 1.0


# --- the stage door (family_stage.check_pack) -------------------------------------------------

def _door_pack(tmp_path, name, guard):
    """A minimal valid family pack; `guard` is the `staged_text_guard` block (or the string
    "absent" for a pack staged before the guard existed)."""
    from cytir.sources import family_stage as FS
    pack = tmp_path / name
    pack.mkdir()
    rec = json.dumps({"point_id": "p1", "text": "t", "url": "https://x.example/a"}) + "\n"
    (pack / "records.jsonl").write_text(rec, encoding="utf-8")
    man = {"schema": FS.PACK_SCHEMA, "family_id": "loda_exact",
           "terminals": {"units_in": 2, "units_admitted": 1, "units_excluded": 0,
                         "units_quarantined": 1},
           "records": {"count": 1, "sha256": hashlib.sha256(rec.encode()).hexdigest()},
           "contract": {"consent_basis": "official-dump-license", "unknowns": []},
           "consent": {"checked": True}}
    if guard != "absent":
        man["staged_text_guard"] = guard
    (pack / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return pack


def _block(**kw):
    return {"min_coverage": 1.0, "threshold_declared": True, "docs_checked": 5,
            "docs_unchecked": 0, "docs_shortfall": 0, "lowest_coverage": 1.0,
            "shortfall_examples": [], **kw}


def test_the_door_refuses_a_pack_with_a_shortfall_for_a_family_that_declares_a_threshold(tmp_path):
    from cytir.sources import family_stage as FS
    bad = _door_pack(tmp_path, "bad", _block(docs_shortfall=2, lowest_coverage=0.91))
    with pytest.raises(FS.StageError, match="staged-text guard: 2 doc"):
        FS.check_pack(bad, "loda_exact")
    FS.check_pack(_door_pack(tmp_path, "clean", _block()), "loda_exact")


def test_the_door_still_passes_packs_staged_before_the_guard_and_default_threshold_packs(tmp_path):
    from cytir.sources import family_stage as FS
    FS.check_pack(_door_pack(tmp_path, "pre-guard", "absent"), "loda_exact")      # no key
    FS.check_pack(_door_pack(tmp_path, "null-guard", None), "loda_exact")         # key, no block
    # a family on the module default is not refused here: its quarantine stays visible, as before
    FS.check_pack(_door_pack(tmp_path, "default", _block(
        threshold_declared=False, min_coverage=0.98, docs_shortfall=1)), "loda_exact")


def test_a_run_that_quarantines_a_cut_produces_a_pack_the_door_refuses(tmp_path, monkeypatch):
    """The whole path: f2199fac re-created in the chunk step, then the real door."""
    from cytir.sources import family_stage as FS
    from cytir.sources.families import loda
    real = loda.LodaFamily.chunk
    monkeypatch.setattr(loda.LodaFamily, "chunk", lambda self, doc, ctx: [
        {**real(self, doc, ctx)[0], "text": doc.source_text[:8000]}])
    _run(tmp_path, "cutdoor")
    pack = tmp_path / "cutdoor" / "pack"
    # the loda contract row still has its four rulings open (POLICY PR); clear them so the door
    # reaches the guard check, which is what this pins
    man = json.loads((pack / "manifest.json").read_text())
    man["contract"]["unknowns"] = []
    man["consent"]["checked"] = True
    (pack / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(FS.StageError, match="staged-text guard"):
        FS.check_pack(pack, "loda_exact")


# --- `native` vs `native-chunked` at the stage door (POLICY v3.3 §4.1, ruling 27 T6 a) --------

def _door_ready(tmp_path, name, root, declared):
    """A real loda pack from `root`, with the door's other preconditions met and the chunk policy
    the manifest names set to `declared` (`native` = one row per article; `native-chunked` may
    split an over-window article)."""
    pytest.importorskip("httpx")
    pytest.importorskip("qdrant_client")
    from cytir.sources import run_family
    run_family.run_family(
        "loda_exact", out_dir=tmp_path / name, today="2026-10-05",
        consent=lambda h, d: pytest.fail(f"consent for {h}"),
        options={"root": str(root), "release": RELEASE}, log=lambda m: None)
    pack = tmp_path / name / "pack"
    man = json.loads((pack / "manifest.json").read_text())
    man["consent"]["checked"] = True
    man["contract"]["chunk"]["declared"] = declared
    (pack / "manifest.json").write_text(json.dumps(man))
    return pack


def _short_root(tmp_path):
    import shutil
    root = tmp_path / "short-root"
    for p in SHORT.rglob("LEGIARTI*.xml"):
        dest = root / "TNC_en_vigueur" / "JORF" / p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, dest)
    return root


def test_the_door_lets_native_chunked_split_a_long_loda_article(tmp_path):
    from cytir.sources import family_stage as FS
    man = FS.check_pack(_door_ready(tmp_path, "nc", _stage_root(tmp_path), "native-chunked"),
                        "loda_exact")
    d = man["door_article_rows"]
    assert d["policy"] == "native-chunked" and d["articles"] == 7
    assert d["split_articles"] == 4 and d["rows_in_split"] >= 8        # the four real long ones


def test_the_door_refuses_native_when_one_article_became_several_rows(tmp_path):
    from cytir.sources import family_stage as FS
    pack = _door_ready(tmp_path, "n", _stage_root(tmp_path), "native")
    with pytest.raises(FS.StageError, match=r"'native' is one row per article, but 4 article"):
        FS.check_pack(pack, "loda_exact")


def test_a_short_article_is_one_row_under_native_and_native_chunked(tmp_path):
    from cytir.sources import family_stage as FS
    for declared in ("native", "native-chunked"):
        pack = _door_ready(tmp_path, f"s-{declared}", _short_root(tmp_path), declared)
        d = FS.check_pack(pack, "loda_exact")["door_article_rows"]
        assert d["split_articles"] == 0 and d["articles"] > 0
        n = sum(1 for _ in open(pack / "records.jsonl"))
        assert n == d["articles"]                                   # one row per article


def test_the_door_reads_fedlex_windows_as_a_split_and_counts_them():
    """fedlex rows: url `...#<eId>-cN`, `base_anchor` = `<eId>` (its 240-word windows are legacy
    but are a split, never a cut): a `native` fedlex pack of them is refused, `native-chunked` not."""
    from cytir.sources import family_stage as FS
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rec = pathlib.Path(td) / "records.jsonl"
        rows = [{"url": "https://www.fedlex.admin.ch/eli/cc/1999/404/de#art_1-c1", "anchor": "art_1-c1",
                 "base_anchor": "art_1", "point_id": "a"},
                {"url": "https://www.fedlex.admin.ch/eli/cc/1999/404/de#art_1-c2", "anchor": "art_1-c2",
                 "base_anchor": "art_1", "point_id": "b"},
                {"url": "https://www.fedlex.admin.ch/eli/cc/1999/404/de#art_2", "anchor": "art_2",
                 "base_anchor": "art_2", "point_id": "c"}]
        rec.write_text("".join(json.dumps(r) + "\n" for r in rows))
        counts = FS.article_row_counts(rec)
    assert (counts["articles"], counts["split_articles"], counts["rows_in_split"]) == (2, 1, 2)
    assert FS.chunk_policy_refusal("native", counts) is not None
    assert FS.chunk_policy_refusal("native-chunked", counts) is None
    assert FS.chunk_policy_refusal("chunk-policy-atomic-record.v1", counts) is None   # not judged here


def test_rows_with_no_url_and_no_point_id_are_each_their_own_article_and_so_chunk_anchors_stay():
    from cytir.sources import family_stage as FS
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rec = pathlib.Path(td) / "records.jsonl"
        rows = [{"title": "news a"}, {"title": "news b"}, {"title": "news c"},
                {"url": "https://so.example/q/1", "anchor": "q:1:chunk-1", "point_id": "p1"},
                {"url": "https://so.example/q/1", "anchor": "q:1:chunk-2", "point_id": "p2"}]
        rec.write_text("".join(json.dumps(r) + "\n" for r in rows))
        counts = FS.article_row_counts(rec)
    # three news rows = three articles; the two StackOverflow-shaped anchors are not merged
    assert (counts["articles"], counts["split_articles"]) == (5, 0)


# --- nits ------------------------------------------------------------------------------------

@pytest.mark.parametrize("value,want", [("1", True), ("true", True), ("YES", True),
                                        ("0", False), ("false", False), ("no", False),
                                        ("", False), (None, False), (True, True)])
def test_only_split_accepts_only_the_documented_values(value, want):
    from cytir.sources.families import loda
    assert loda.only_split({} if value is None else {"only_split": value}) is want


def test_a_misspelt_only_split_refuses_the_run_instead_of_meaning_a_full_run(tmp_path):
    with pytest.raises(ValueError, match="only_split='treu'"):
        _run(tmp_path, "typo", only_split="treu")
    assert not (tmp_path / "typo" / "pack").exists()


def test_lowest_coverage_is_null_when_no_document_was_checked():
    from cytir.sources import staged_text_guard as G
    guard = G.StagedTextGuard("x")
    assert guard.summary()["lowest_coverage"] is None
    guard.observe("u", None, ["t"])                              # unchecked: still nothing measured
    assert guard.summary()["lowest_coverage"] is None
    guard.observe("v", "a b c d e f g", ["a b c d e f g"])
    assert guard.summary()["lowest_coverage"] == 1.0


def test_the_loud_line_says_quarantined_only_when_something_was():
    from cytir.sources import staged_text_guard as G
    guard = G.StagedTextGuard("x")
    guard.observe("u", None, ["t"])
    line = guard.loud_line()
    assert line and "UNCHECKED" in line and "quarantined" not in line.lower()
    guard.observe("v", "a b c d e f g h", ["a"])
    assert "1 SHORT of their source (quarantined)" in guard.loud_line()
    assert G.StagedTextGuard("y").loud_line() is None


def test_short_part_inside_longer_row_is_held_task_2198():
    from cytir.sources.staged_text_guard import SourcePart, part_coverage
    part = SourcePart(key="k", text="Schedule 2 has effect.")
    row = "Title breadcrumb > Section 4\n\nSchedule 2 has effect.\n\nmore"
    assert part_coverage(part, [row]) == 1.0


def test_short_part_absent_is_not_held_task_2198():
    from cytir.sources.staged_text_guard import SourcePart, part_coverage
    part = SourcePart(key="k", text="Schedule 2 has effect.")
    assert part_coverage(part, ["Title Schedule 3 has effect, and more words follow"]) == 0.0
    assert part_coverage(part, []) == 0.0


def test_short_part_across_word_boundary_is_not_held_task_2198():
    from cytir.sources.staged_text_guard import SourcePart, part_coverage
    assert part_coverage(SourcePart(key="k", text="2 has"), ["see 12 hash of the row text"]) == 0.0

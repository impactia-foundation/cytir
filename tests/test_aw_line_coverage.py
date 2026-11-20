#!/usr/bin/env python3
"""Ruling 27 T6: per-id line coverage lets a url-group supersede skip the snippet guard per URL.

REAL ROWS. The new text is the real article LEGIARTI000049864993 (`tests/fixtures/legi_loda_long`,
55k chars, 531 distinct body lines); the ids and URLs are the real twin rows of
`fixtures/loda-old-generation-rows.json` (as in `test_aw_url_group_supersede.py`). The OLD text of a
URL is built from the real article's own body lines: `found` of them kept, the rest replaced by
invented lines, so the coverage is exactly the number a test names (0.79, 0.80, ...). Two things are
placed by the tests, not twin facts: the real article's text stands at the real URL of the fixture
rows, and the snippets (the fixture has none) are set so that the SNIPPET guard would refuse.

The qdrant is the in-memory fake of the url-group tests. No network.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import shutil
import sys

import pytest

pytest.importorskip("lxml")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_aw_url_group_supersede as T  # noqa: E402
from test_aw_url_group_supersede import _stage_root, settings  # noqa: E402,F401  (fixtures)

from cytir.extract import legi  # noqa: E402
from cytir.index import compose  # noqa: E402
from cytir.index import line_coverage as LC  # noqa: E402
from cytir.index import promote_queue as Q  # noqa: E402
from cytir.index import url_group_supersede as UGS  # noqa: E402

LONG = HERE / "fixtures" / "legi_loda_long"
ARTICLE = next(LONG.rglob("LEGIARTI000049864993.xml"))
URL = T.URL_B                       # the real fixture URL that carries 1 old row + 1 new (cut) row
OLD_ID, CUT_ID = T.OLD_B["id"], T.NEW_B["id"]


def real_text() -> str:
    body = ARTICLE.read_bytes()
    l1 = legi.parse_article_xml_legal_markdown(
        body, archive_member="m", archive_sha256=hashlib.sha256(body).hexdigest(),
        source_release="aw-task633-dila-loda-full-v1", article_url_family="loda",
        source_id="dila-legi-loda-full-v1")
    return f"{l1['title']}\n\n{l1['markdown']}"


NEW_TEXT = real_text()
REAL_LINES = sorted(LC.body_lines(NEW_TEXT))


def old_text(total: int, found: int) -> str:
    """An old-generation-shaped text (the identity block first, which the measure skips) with `total` distinct body lines of
    which exactly `found` are lines of the real new text."""
    assert found <= total <= len(REAL_LINES)
    kept = REAL_LINES[:found]
    invented = [f"Ancienne redaction, alinea {i} abroge." for i in range(total - found)]
    return ("Article identity\n- Article ID: `X`\n- Archive member: `old`\n\n"
            + "\n".join(kept + invented) + "\n")


# ------------------------------------------------------------------ the definition
def test_the_measure_skips_identity_hierarchy_headings_and_one_char_lines():
    text = ("Titre\n\n# Heading\n\nArticle identity\n- Article ID: `A`\n- Status: VIGUEUR\n\n"
            "Legal hierarchy\n- Code X\n\n### Paragraph 1 {#A-p1}\n\nPremiere ligne.\n.\n"
            "Premiere   ligne.\nDeuxieme ligne.\n")
    assert LC.body_lines(text) == {"Titre", "Premiere ligne.", "Deuxieme ligne."}


def test_coverage_is_found_over_old_lines_and_one_when_there_is_no_old_line():
    new = LC.normalise("\n".join(["a b", "c d"]))
    assert LC.line_coverage("Titre\n\nab\nc d\nx y\n", LC.normalise("Titre\nc d\n")) \
        == (0.5, 4, 2)
    assert LC.line_coverage("Article identity\n- Article ID: `A`\n", new) == (1.0, 0, 0)


def test_the_real_article_has_the_lines_the_tests_rely_on():
    assert len(REAL_LINES) >= 531 - 5 and LC.line_coverage(NEW_TEXT, LC.normalise(NEW_TEXT))[0] == 1.0
    for total, found, want in ((100, 79, 0.79), (100, 80, 0.8), (100, 100, 1.0)):
        cov, n, f = LC.line_coverage(old_text(total, found), LC.normalise(NEW_TEXT))
        assert (cov, n, f) == (want, total, found)


# ------------------------------------------------------------------ the block
def write_shard(path: pathlib.Path, rows) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


def pack_dir(tmp_path, man):
    """The pack's own dir in a queue under `tmp_path` (the drain reads the block file there)."""
    d = tmp_path / "qd" / man["pack_id"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def plan(man, joined, c, tmp_path):
    return T.plan(man, joined, c, tmp_path, queue_dir=tmp_path / "qd")


def seal_rows(man, rows):
    """Put real stage rows into the manifest's sealed shard and bind them in PACK.json (the drain
    reads the stage rows to see which URLs it does not write in full: task 888)."""
    chunks = pathlib.Path(man["pack_chunks"])
    shard = write_shard(chunks / "shard-0000.texts.jsonl.gz", rows)
    pj = chunks.parent / "PACK.json"
    meta = json.loads(pj.read_text())
    meta["chunks_sha256"] = LC.sha256_file(shard)
    pj.write_text(json.dumps(meta))


def block_for(tmp_path, man, texts: dict[str, str], *, wanted=None, stage_rows=None):
    """The block file for `man` (a sealed url-group manifest), the old texts given by id."""
    seal_rows(man, stage_rows or [{"point_id": T.NEW_ID, "url": URL, "text": NEW_TEXT}])
    src = write_shard(tmp_path / "old" / "shard-0000.texts.jsonl.gz",
                      [{"point_id": i, "url": URL, "text": t} for i, t in texts.items()])
    block = LC.build_block(
        new_rows=stage_rows or [{"point_id": T.NEW_ID, "url": URL, "text": NEW_TEXT}],
        new_chunks_sha256=UGS.sealed_chunks_sha256(man), old_sources=[src], wanted=wanted)
    return block, compose.write_line_coverage_block(block, pack_dir(tmp_path, man))


def cluster_with_snippets_that_fail_the_snippet_guard(monkeypatch):
    """Real rows of URL B; 25 words per old row against 3 in the new row: the snippet guard
    would refuse this URL (3 < 50 % of 50)."""
    return T.make_cluster(monkeypatch, T.served(T.OLD_B, T.NEW_B, snippets=T.WORDS25))


def run_plan(tmp_path, man, c):
    return plan(man, T.joined_of(T.pack_row(T.NEW_ID, URL, snippet="trois mots seulement")),
                c, tmp_path)


def sealed_manifest(tmp_path, **fields):
    man = T.manifest(docs=1)
    man.update(fields)
    return man


def test_measure_gives_each_old_row_its_own_number_and_no_entry_without_a_text(tmp_path):
    new = [{"point_id": T.NEW_ID, "url": URL, "text": NEW_TEXT}]
    old = [{"point_id": OLD_ID, "url": URL, "text": old_text(100, 79)},
           {"point_id": CUT_ID, "url": URL, "text": NEW_TEXT[:8000]},
           {"point_id": "other", "url": "https://elsewhere.example/x", "text": "x"},
           {"point_id": T.NEW_ID, "url": URL, "text": "the pack's own id is never superseded"}]
    wanted = {OLD_ID: URL, CUT_ID: URL, "gone": URL}
    entries, unmeasured = LC.measure(old, new, pack_ids=[T.NEW_ID], wanted=wanted)
    assert unmeasured == ["gone"]                     # no text in the sources: no number guessed
    assert entries[OLD_ID] == {"url": URL, "coverage": 0.79, "old_lines": 100, "found_lines": 79}
    assert entries[CUT_ID]["coverage"] == 1.0 and set(entries) == {OLD_ID, CUT_ID}


def test_a_source_row_that_names_another_page_than_the_served_row_is_unmeasured():
    new = [{"point_id": T.NEW_ID, "url": URL, "text": NEW_TEXT}]
    old = [{"point_id": OLD_ID, "url": URL, "text": NEW_TEXT}]
    entries, unmeasured = LC.measure(old, new, wanted={OLD_ID: T.URL_A})
    assert entries == {} and unmeasured == []         # out of scope: the wanted url is not a pack url
    entries, unmeasured = LC.measure(old, new + [{"point_id": "z", "url": T.URL_A, "text": "t"}],
                                     wanted={OLD_ID: T.URL_A})
    assert entries == {} and unmeasured == [OLD_ID]


def test_the_block_names_its_sources_and_the_sealed_chunks_it_was_measured_against(tmp_path):
    man = sealed_manifest(tmp_path)
    block, fields = block_for(tmp_path, man, {OLD_ID: old_text(100, 80)}, wanted={OLD_ID: URL})
    assert block["new_chunks_sha256"] == UGS.sealed_chunks_sha256(man)
    assert block["old_text_sources"][0]["sha256"] and block["unmeasured_ids"] == 0
    assert fields["line_coverage"]["at_or_above_min"] == 1
    assert fields["line_coverage_file"] == LC.FILE_NAME             # a plain name, not a path
    assert fields["line_coverage_sha256"] == hashlib.sha256(
        (pack_dir(tmp_path, man) / fields["line_coverage_file"]).read_bytes()).hexdigest()


def test_a_block_with_no_entry_at_all_is_not_written(tmp_path):
    with pytest.raises(LC.LineCoverageError, match="nothing to write"):
        LC.build_block(new_rows=[{"point_id": "n", "url": URL, "text": "t"}],
                       new_chunks_sha256="a" * 64, wanted={OLD_ID: URL},
                       old_sources=[write_shard(tmp_path / "s" / "shard-0000.texts.jsonl.gz", [])])


# ------------------------------------------------------------------ the drain
def test_coverage_at_the_floor_skips_the_snippet_guard_for_its_url(tmp_path, monkeypatch, settings):
    c = cluster_with_snippets_that_fail_the_snippet_guard(monkeypatch)
    man = sealed_manifest(tmp_path)
    _, fields = block_for(tmp_path, man, {OLD_ID: old_text(100, 80), CUT_ID: NEW_TEXT[:8000]})
    man.update(fields)
    got = run_plan(tmp_path, man, c)
    s = got["summary"]
    assert got["ids"] == sorted([OLD_ID, CUT_ID])
    assert s["line_coverage"]["urls_skipped"] == 1 and s["line_coverage"]["ids_skipped"] == 2
    assert s["line_coverage"]["urls_snippet_guard"] == 0
    assert s["word_coverage"] == {"urls_checked": 0, "violations": 0}   # the snippet guard did not run


def test_coverage_just_below_the_floor_keeps_the_snippet_guard_which_refuses(
        tmp_path, monkeypatch, settings):
    c = cluster_with_snippets_that_fail_the_snippet_guard(monkeypatch)
    man = sealed_manifest(tmp_path)
    _, fields = block_for(tmp_path, man, {OLD_ID: old_text(100, 79), CUT_ID: NEW_TEXT[:8000]})
    man.update(fields)
    with pytest.raises(Q.QueueError, match="would lose text") as exc:
        run_plan(tmp_path, man, c)
    assert URL in str(exc.value)
    assert not list(settings.iterdir())                    # nothing exported


def test_one_unmeasured_row_sends_the_whole_url_to_the_snippet_guard(tmp_path, monkeypatch,
                                                                     settings):
    c = cluster_with_snippets_that_fail_the_snippet_guard(monkeypatch)
    man = sealed_manifest(tmp_path)
    _, fields = block_for(tmp_path, man, {OLD_ID: old_text(100, 100)})    # CUT_ID has no text
    man.update(fields)
    with pytest.raises(Q.QueueError, match="would lose text"):
        run_plan(tmp_path, man, c)


RULING = LC.RULED_SKIP_RULING


def ruled_file(tmp_path, man, entries):
    """Write the ruled-skip file beside the manifest and bind it, as compose does."""
    data = json.dumps(entries, indent=1).encode()
    path = pack_dir(tmp_path, man) / LC.RULED_SKIP_FILE_NAME
    path.write_bytes(data)
    man.update(LC.ruled_skip_fields(data, path, LC.parse_ruled_skip(data)))


def ruled_entry(pid=OLD_ID, url=URL, cov=7 / 16):
    return {"url": url, "point_id": pid, "measured_coverage": cov, "ruling": RULING}


def test_a_ruled_skip_id_lets_its_url_skip_the_snippet_guard_and_the_manifest_carries_it(
        tmp_path, monkeypatch, settings):
    c = cluster_with_snippets_that_fail_the_snippet_guard(monkeypatch)
    man = sealed_manifest(tmp_path)
    block, fields = block_for(tmp_path, man, {OLD_ID: old_text(16, 7), CUT_ID: NEW_TEXT[:8000]})
    assert block["entries"][OLD_ID]["coverage"] == 0.4375
    man.update(fields)
    with pytest.raises(Q.QueueError, match="would lose text"):      # unlisted: as today
        run_plan(tmp_path, man, c)
    ruled_file(tmp_path, man, [ruled_entry()])
    assert man["ruled_skip"] == {"count": 1, "entries": [ruled_entry()]}
    got = run_plan(tmp_path, man, c)
    s = got["summary"]["line_coverage"]
    assert s["urls_skipped"] == 1 and s["ids_ruled_skip"] == 1 and s["ids_below_min"] == 0
    assert s["ruled_skip"]["entries"] == [ruled_entry()]
    assert s["ruled_skip"]["sha256"] == man["ruled_skip_sha256"]
    assert got["summary"]["word_coverage"] == {"urls_checked": 0, "violations": 0}


def test_a_ruled_skip_whose_measure_drifted_refuses_the_pack(tmp_path, monkeypatch, settings):
    c = cluster_with_snippets_that_fail_the_snippet_guard(monkeypatch)
    man = sealed_manifest(tmp_path)
    _, fields = block_for(tmp_path, man, {OLD_ID: old_text(16, 8), CUT_ID: NEW_TEXT[:8000]})
    man.update(fields)                                              # measures 0.5, not 0.4375
    ruled_file(tmp_path, man, [ruled_entry()])
    with pytest.raises(Q.QueueError, match="measure drifted"):
        run_plan(tmp_path, man, c)
    assert not list(settings.iterdir())


@pytest.mark.parametrize("bad_cov", [float("nan"), float("inf"), 2.0, -0.1, True])
def test_a_ruled_skip_with_a_non_finite_or_out_of_range_measure_is_refused(bad_cov):
    with pytest.raises(LC.LineCoverageError, match="measured_coverage"):
        LC.parse_ruled_skip(json.dumps([ruled_entry(cov=bad_cov)]).encode())


def test_a_ruled_skip_id_that_is_not_at_that_url_refuses_the_pack(tmp_path, monkeypatch, settings):
    c = cluster_with_snippets_that_fail_the_snippet_guard(monkeypatch)
    man = sealed_manifest(tmp_path)
    _, fields = block_for(tmp_path, man, {OLD_ID: old_text(16, 7), CUT_ID: NEW_TEXT[:8000]})
    man.update(fields)
    ruled_file(tmp_path, man, [ruled_entry(url=T.URL_A)])
    with pytest.raises(Q.QueueError, match="measured at"):
        run_plan(tmp_path, man, c)
    ruled_file(tmp_path, man, [ruled_entry(pid="00000000-0000-0000-0000-000000000000")])
    with pytest.raises(Q.QueueError, match="no entry"):
        run_plan(tmp_path, man, c)


def test_a_ruled_skip_cannot_be_typed_forged_or_widened(tmp_path, monkeypatch, settings):
    man = sealed_manifest(tmp_path)
    _, fields = block_for(tmp_path, man, {OLD_ID: old_text(16, 7), CUT_ID: NEW_TEXT[:8000]})
    man.update(fields)
    ruled_file(tmp_path, man, [ruled_entry()])
    d = pack_dir(tmp_path, man)
    (d / LC.RULED_SKIP_FILE_NAME).write_text(json.dumps([ruled_entry(cov=0.5)]))
    with pytest.raises(LC.LineCoverageError, match="sha256"):
        LC.load_ruled_skip(man, pack_dir=d)
    typed = {k: v for k, v in man.items() if not k.startswith("ruled_skip_")}
    with pytest.raises(LC.LineCoverageError, match="typed in the manifest"):
        LC.load_ruled_skip(typed, pack_dir=d)
    for bad in ([{**ruled_entry(), "ruling": "ok"}], [{**ruled_entry(), "point_id": "*"}],
                [{**ruled_entry(), "extra": 1}], [ruled_entry(), ruled_entry()], []):
        with pytest.raises(LC.LineCoverageError):
            LC.parse_ruled_skip(json.dumps(bad).encode())


def test_a_ruled_skip_does_not_cover_a_second_low_id_of_the_same_url(tmp_path, monkeypatch, settings):
    c = cluster_with_snippets_that_fail_the_snippet_guard(monkeypatch)
    man = sealed_manifest(tmp_path)
    _, fields = block_for(tmp_path, man, {OLD_ID: old_text(16, 7), CUT_ID: old_text(16, 7)})
    man.update(fields)
    ruled_file(tmp_path, man, [ruled_entry()])
    with pytest.raises(Q.QueueError, match="would lose text"):
        run_plan(tmp_path, man, c)


def test_no_block_in_the_manifest_is_the_behaviour_of_main(tmp_path, monkeypatch, settings):
    c = cluster_with_snippets_that_fail_the_snippet_guard(monkeypatch)
    with pytest.raises(Q.QueueError, match="would lose text"):
        run_plan(tmp_path, sealed_manifest(tmp_path), c)
    c2 = T.make_cluster(monkeypatch, T.served(T.OLD_B, T.NEW_B, snippets=T.WORDS25))
    ok = plan(sealed_manifest(tmp_path), T.joined_of(T.pack_row(T.NEW_ID, URL)), c2, tmp_path)
    assert ok["summary"]["line_coverage"]["urls_snippet_guard"] == 1
    assert ok["summary"]["line_coverage"]["ids_unmeasured"] == 2


def test_a_hand_typed_block_is_refused(tmp_path, monkeypatch, settings):
    c = cluster_with_snippets_that_fail_the_snippet_guard(monkeypatch)
    man = sealed_manifest(tmp_path, line_coverage={"entries": {
        OLD_ID: {"url": URL, "coverage": 1.0, "old_lines": 10, "found_lines": 10},
        CUT_ID: {"url": URL, "coverage": 1.0, "old_lines": 10, "found_lines": 10}}})
    with pytest.raises(Q.QueueError, match="typed `line_coverage` block"):
        run_plan(tmp_path, man, c)


def test_a_block_file_that_is_not_the_one_the_manifest_hashed_is_refused(tmp_path, monkeypatch,
                                                                          settings):
    c = cluster_with_snippets_that_fail_the_snippet_guard(monkeypatch)
    man = sealed_manifest(tmp_path)
    _, fields = block_for(tmp_path, man, {OLD_ID: old_text(100, 100), CUT_ID: NEW_TEXT[:8000]})
    man.update(fields)
    f = pack_dir(tmp_path, man) / fields["line_coverage_file"]
    forged = json.loads(f.read_text())
    forged["entries"][OLD_ID]["coverage"] = 1.0
    forged["entries"][OLD_ID]["found_lines"] = forged["entries"][OLD_ID]["old_lines"] = 1
    f.write_text(json.dumps(forged))
    with pytest.raises(Q.QueueError, match="sha256"):
        run_plan(tmp_path, man, c)


def test_a_forged_entry_with_a_matching_sha_is_refused_when_it_is_not_a_measurement(
        tmp_path, monkeypatch, settings):
    c = cluster_with_snippets_that_fail_the_snippet_guard(monkeypatch)
    man = sealed_manifest(tmp_path)
    block, _ = block_for(tmp_path, man, {OLD_ID: old_text(100, 100), CUT_ID: NEW_TEXT[:8000]})
    block["entries"][OLD_ID]["coverage"] = 0.99        # not found / old (1.0)
    fields = compose.write_line_coverage_block(block, pack_dir(tmp_path, man))
    man.update(fields)
    with pytest.raises(Q.QueueError, match="not a measurement"):
        run_plan(tmp_path, man, c)


def test_a_block_measured_against_other_new_rows_is_refused(tmp_path, monkeypatch, settings):
    c = cluster_with_snippets_that_fail_the_snippet_guard(monkeypatch)
    man = sealed_manifest(tmp_path)
    block, _ = block_for(tmp_path, man, {OLD_ID: old_text(100, 100), CUT_ID: NEW_TEXT[:8000]})
    block["new_chunks_sha256"] = "0" * 64
    man.update(compose.write_line_coverage_block(block, pack_dir(tmp_path, man)))
    with pytest.raises(Q.QueueError, match="other new rows"):
        run_plan(tmp_path, man, c)


def test_an_entry_measured_at_another_url_does_not_count(tmp_path, monkeypatch, settings):
    c = cluster_with_snippets_that_fail_the_snippet_guard(monkeypatch)
    man = sealed_manifest(tmp_path)
    block, _ = block_for(tmp_path, man, {OLD_ID: old_text(100, 100), CUT_ID: NEW_TEXT[:8000]})
    block["entries"][OLD_ID]["url"] = T.URL_A
    man.update(compose.write_line_coverage_block(block, pack_dir(tmp_path, man)))
    with pytest.raises(Q.QueueError, match="would lose text"):
        run_plan(tmp_path, man, c)


def test_the_floor_is_a_constant_the_manifest_cannot_lower(tmp_path, monkeypatch, settings):
    c = cluster_with_snippets_that_fail_the_snippet_guard(monkeypatch)
    man = sealed_manifest(tmp_path, line_coverage_min=0.1, min_coverage=0.1)
    _, fields = block_for(tmp_path, man, {OLD_ID: old_text(100, 50), CUT_ID: NEW_TEXT[:8000]})
    man.update(fields)
    assert LC.MIN_COVERAGE == 0.8
    with pytest.raises(Q.QueueError, match="would lose text"):
        run_plan(tmp_path, man, c)


def test_a_url_over_the_floor_skips_while_another_url_of_the_same_pack_keeps_the_guard(
        tmp_path, monkeypatch, settings):
    """Two real URLs in one pack: B measured at 1.0 (skips), A has no measure (snippet guard,
    which passes here because A's replacement keeps the words)."""
    rows = T.served(T.OLD_B, T.NEW_B, *T.OLD_A[:2], snippets=T.WORDS25)
    c = T.make_cluster(monkeypatch, rows)
    man = T.manifest(docs=2)
    _, fields = block_for(tmp_path, man, {OLD_ID: old_text(100, 100), CUT_ID: NEW_TEXT[:8000]})
    man.update(fields)
    got = plan(man, T.joined_of(T.pack_row(T.NEW_ID, URL, snippet="trois mots seulement"),
                                  T.pack_row(T.NEW_ID_2, T.URL_A),
                                  T.pack_row("11111111-2222-5333-8444-555555555553", T.URL_A)),
                 c, tmp_path)
    lc = got["summary"]["line_coverage"]
    assert (lc["urls_skipped"], lc["urls_snippet_guard"]) == (1, 1)
    assert got["summary"]["word_coverage"]["urls_checked"] == 1


def test_a_stage_guard_below_1_0_skips_nothing_even_with_a_perfect_block(
        tmp_path, monkeypatch, settings):
    """Ruling 27 T6 c: the new rows are the COMPLETE article only when the stage guard ran at 1.0.
    A family that declared 0.98 (the drain accepted any declared threshold) keeps the snippet guard
    for every URL: here that refuses the pack, and the summary of a passing one says why."""
    c = cluster_with_snippets_that_fail_the_snippet_guard(monkeypatch)
    man = T.manifest(guard={**T.GUARD_OK, "docs_checked": 1, "min_coverage": 0.98})
    _, fields = block_for(tmp_path, man, {OLD_ID: old_text(100, 100), CUT_ID: NEW_TEXT[:8000]})
    man.update(fields)
    with pytest.raises(Q.QueueError, match="would lose text"):
        run_plan(tmp_path, man, c)
    for stage_min in (0.98, None, "1.0", True):
        m = T.manifest(guard={**T.GUARD_OK, "docs_checked": 1, "min_coverage": stage_min})
        _, f = block_for(tmp_path, m, {OLD_ID: old_text(100, 100), CUT_ID: NEW_TEXT[:8000]})
        m.update(f)
        got = UGS.line_coverage_plan(m, {OLD_ID: URL, CUT_ID: URL},
                                     pack_dir=pack_dir(tmp_path, m))
        assert got["skip_ids"] == frozenset()
        assert got["summary"]["urls_snippet_guard"] == 1 and "skip_refused" in got["summary"]
    ok = T.manifest(guard={**T.GUARD_OK, "docs_checked": 1, "min_coverage": 1.0})
    _, f = block_for(tmp_path, ok, {OLD_ID: old_text(100, 100), CUT_ID: NEW_TEXT[:8000]})
    ok.update(f)
    assert UGS.line_coverage_plan(ok, {OLD_ID: URL, CUT_ID: URL},
                                  pack_dir=pack_dir(tmp_path, ok))["skip_ids"] == {OLD_ID, CUT_ID}


def test_a_manifest_that_copied_the_guard_against_other_chunks_is_refused(tmp_path):
    man = sealed_manifest(tmp_path, staged_text_guard_chunks_sha256="0" * 64)
    with pytest.raises(Q.QueueError, match="composed against other rows"):
        UGS.sealed_guard_block(man)
    good = sealed_manifest(tmp_path)
    good["staged_text_guard_chunks_sha256"] = UGS.sealed_chunks_sha256(good)
    assert UGS.sealed_guard_block(good)["docs_checked"] == 1


def test_a_blind_family_never_reaches_the_line_coverage_even_with_a_perfect_block(
        tmp_path, monkeypatch, settings):
    c = cluster_with_snippets_that_fail_the_snippet_guard(monkeypatch)
    blind = {**T.GUARD_OK, "docs_checked": 1, "threshold_declared": False}
    man = T.manifest(guard=blind)
    _, fields = block_for(tmp_path, man, {OLD_ID: old_text(100, 100), CUT_ID: NEW_TEXT[:8000]})
    man.update(fields)
    with pytest.raises(Q.QueueError, match="BLIND"):
        run_plan(tmp_path, man, c)


def test_a_family_outside_the_allowlist_is_refused_before_any_read(tmp_path, monkeypatch, settings):
    c = cluster_with_snippets_that_fail_the_snippet_guard(monkeypatch)
    man = sealed_manifest(tmp_path)
    _, fields = block_for(tmp_path, man, {OLD_ID: old_text(100, 100), CUT_ID: NEW_TEXT[:8000]})
    man.update(fields, family="eric_official_pdf")
    n = len(c.calls)
    with pytest.raises(Q.QueueError, match="not permitted for family"):
        run_plan(tmp_path, man, c)
    assert len(c.calls) == n


# ------------------------------------------------------------------ the compose step
def test_a_url_with_a_new_row_the_drain_does_not_write_keeps_the_snippet_guard(
        tmp_path, monkeypatch, settings):
    """Task 888 D2: coverage >= 0.8 was measured against BOTH stage rows of the URL, but one is
    in drop_ids: the drain writes only the other. The URL must not skip the snippet guard."""
    c = cluster_with_snippets_that_fail_the_snippet_guard(monkeypatch)
    man = sealed_manifest(tmp_path)
    second = {"point_id": T.NEW_ID_2, "url": URL, "text": NEW_TEXT}
    stage = [{"point_id": T.NEW_ID, "url": URL, "text": NEW_TEXT}, second]
    _, fields = block_for(tmp_path, man, {OLD_ID: old_text(100, 100), CUT_ID: NEW_TEXT[:8000]},
                          stage_rows=stage)
    man.update(fields)
    joined = T.joined_of(T.pack_row(T.NEW_ID, URL, snippet="trois mots seulement"),
                         T.pack_row(T.NEW_ID_2, URL, snippet="trois mots seulement"))
    # all stage rows written: the URL skips (the control)
    got = T.plan(man, joined, c, tmp_path, queue_dir=tmp_path / "qd", write=False)
    assert got["summary"]["line_coverage"]["urls_skipped"] == 1
    # one of its new rows dropped: the URL keeps the guard, which refuses these snippets
    with pytest.raises(Q.QueueError):
        T.plan(man, joined, c, tmp_path / "again", queue_dir=tmp_path / "qd", write=False,
               drop=frozenset({T.NEW_ID_2}))


def test_unwritten_url_keys_names_only_the_urls_with_a_stage_row_not_written(tmp_path):
    man = sealed_manifest(tmp_path)
    seal_rows(man, [{"point_id": "a", "url": URL, "text": "x"},
                    {"point_id": "b", "url": URL, "text": "y"},
                    {"point_id": "c", "url": T.URL_A, "text": "z"}])
    assert UGS.unwritten_url_keys(man, {"a", "c"}) == {LC.url_key(URL)}
    assert UGS.unwritten_url_keys(man, {"a", "b", "c"}) == set()


def sealed_loda_stage(tmp_path, *, groups=1):
    """A sealed stage dir as `family_stage` writes it, holding one REAL new row of the real
    article, and a PACK.json whose guard block says `groups` URL groups were checked."""
    from cytir.core.common import mint_passage_point_id
    stage = tmp_path / "stage"
    anchor = "article:LEGIARTI000049864993"
    row = {"point_id": mint_passage_point_id(URL, anchor, NEW_TEXT.split("\n\n", 1)[1]),
           "text": NEW_TEXT, "url": URL, "anchor": anchor, "title": NEW_TEXT.split("\n\n")[0],
           "nature": "ARRETE", "article_id": "LEGIARTI000049864993", "country": "FR",
           "provenance": "loda", "domain": "legifrance.gouv.fr", "lang": "fr", "tier": "A",
           "source_type": "official", "trust_tier": 1.0, "license": "Etalab-2.0",
           "consent_state": "dump", "consent_basis": "official-dump-license"}
    shard = write_shard(stage / "chunks" / "shard-0000.texts.jsonl.gz", [row])
    guard = {**T.GUARD_OK, "docs_checked": groups}
    (stage / "PACK.json").write_text(json.dumps({
        "schema": "aw.family-stage.v1", "chunks_sha256": LC.sha256_file(shard),
        "pack_staged_text_guard": guard}))
    return stage, row, guard


def vectors_for(tmp_path, ids):
    import numpy as np
    v = tmp_path / "vectors"
    v.mkdir(exist_ok=True)
    np.zeros((len(ids), 1024), dtype="<f2").tofile(v / "shard-0000.vectors.f16")
    (v / "shard-0000.row_ids.jsonl").write_text("".join(json.dumps({"point_id": i}) + "\n"
                                                         for i in ids))
    (v / "shard-0000.vectors.manifest.json").write_text(json.dumps(
        {"schema": "aw.413a.vectors.v1", "shard": "shard-0000", "n": len(ids), "dim": 1024,
         "dtype": "float16"}))
    return v


def compose_loda(tmp_path, stage, row, **kw):
    return compose.compose_pack(
        pack_id="loda-restage", family="loda_exact", chunks_dir=str(stage / "chunks"),
        vectors_dir=str(vectors_for(tmp_path, [row["point_id"]])), collection="aw_e8", mrl=256,
        queue_dir=str(tmp_path / "queue"), **kw)


def test_compose_writes_the_url_group_fields_the_sealed_guard_and_the_block(tmp_path):
    stage, row, guard = sealed_loda_stage(tmp_path)
    src = write_shard(tmp_path / "old" / "shard-0000.texts.jsonl.gz",
                      [{"point_id": OLD_ID, "url": URL, "text": old_text(100, 90)}])
    wanted = write_shard(tmp_path / "wanted.jsonl.gz", [{"id": OLD_ID, "payload": {"url": URL}}])
    res = compose_loda(tmp_path, stage, row, supersede="url-group", allow_supersede="ruling 27 T6",
                       copy_sealed_guard=True, line_coverage_old_texts=[str(src)],
                       line_coverage_wanted_ids=[str(wanted)])
    m = res["manifest"]
    assert (m["supersede"], m["allow_supersede"]) == ("url-group", "ruling 27 T6")
    assert m["staged_text_guard"] == guard
    assert m["staged_text_guard_chunks_sha256"] == LC.sha256_file(
        stage / "chunks" / "shard-0000.texts.jsonl.gz")
    assert m["line_coverage"]["ids"] == 1 and m["line_coverage"]["lowest"] == 0.9
    # the manifest the drain reads is accepted by the drain's own readers
    assert UGS.declared(m) is True
    assert UGS.sealed_guard_block(m) == guard
    assert m["line_coverage_file"] == LC.FILE_NAME                  # relative to the pack dir
    entries = LC.load(m, sealed_chunks_sha256=UGS.sealed_chunks_sha256(m), pack_dir=res["pack_dir"])
    assert entries[OLD_ID]["coverage"] == 0.9
    assert json.loads((pathlib.Path(res["pack_dir"]) / "manifest.json").read_text()) == m


def test_compose_binds_a_ruled_skip_file_and_refuses_a_drifted_one(tmp_path):
    stage, row, guard = sealed_loda_stage(tmp_path)
    src = write_shard(tmp_path / "old" / "shard-0000.texts.jsonl.gz",
                      [{"point_id": OLD_ID, "url": URL, "text": old_text(16, 7)}])
    wanted = write_shard(tmp_path / "wanted.jsonl.gz", [{"id": OLD_ID, "payload": {"url": URL}}])
    f = tmp_path / "ruled.json"
    f.write_text(json.dumps([ruled_entry()]))
    kw = dict(supersede="url-group", allow_supersede="ruling 27 T6", copy_sealed_guard=True,
              line_coverage_old_texts=[str(src)], line_coverage_wanted_ids=[str(wanted)])
    res = compose_loda(tmp_path, stage, row, ruled_skip=str(f), **kw)
    m = res["manifest"]
    assert m["ruled_skip_file"] == LC.RULED_SKIP_FILE_NAME
    assert m["ruled_skip_sha256"] == LC.sha256_file(f)
    assert m["ruled_skip"]["entries"] == [ruled_entry()]
    assert LC.load_ruled_skip(m, pack_dir=res["pack_dir"]) == [ruled_entry()]
    f.write_text(json.dumps([ruled_entry(cov=0.5)]))
    with pytest.raises(compose.ComposeError, match="measure drifted"):
        compose_loda(tmp_path, stage, row, ruled_skip=str(f), **kw)
    with pytest.raises(compose.ComposeError, match="--ruled-skip"):
        compose_loda(tmp_path, stage, row, ruled_skip=str(f))


def test_compose_reads_a_generator_of_old_texts_once_and_still_writes_the_block(tmp_path):
    """A generator is spent after the first read: the option check must not eat it (it did: the
    block was silently not written and the drain kept the snippet guard for everything)."""
    stage, row, _ = sealed_loda_stage(tmp_path)
    src = write_shard(tmp_path / "old" / "shard-0000.texts.jsonl.gz",
                      [{"point_id": OLD_ID, "url": URL, "text": old_text(100, 90)}])
    res = compose_loda(tmp_path, stage, row, supersede="url-group", allow_supersede="r",
                       copy_sealed_guard=True, line_coverage_old_texts=(str(x) for x in [src]))
    assert res["manifest"]["line_coverage"]["ids"] == 1
    assert (pathlib.Path(res["pack_dir"]) / LC.FILE_NAME).is_file()


def test_compose_refuses_old_texts_that_measure_nothing(tmp_path):
    stage, row, _ = sealed_loda_stage(tmp_path)
    src = write_shard(tmp_path / "old" / "shard-0000.texts.jsonl.gz",
                      [{"point_id": OLD_ID, "url": "https://elsewhere.example/x", "text": "t"}])
    with pytest.raises(compose.ComposeError, match="line coverage block not written"):
        compose_loda(tmp_path, stage, row, supersede="url-group", allow_supersede="r",
                     copy_sealed_guard=True, line_coverage_old_texts=[str(src)])
    assert not (tmp_path / "queue" / "loda-x" / "manifest.json").exists()


def test_the_manifest_cannot_point_the_block_outside_the_pack_dir(tmp_path):
    man = sealed_manifest(tmp_path)
    _, fields = block_for(tmp_path, man, {OLD_ID: old_text(100, 100)})
    sha = UGS.sealed_chunks_sha256(man)
    for bad in (str(tmp_path / "qd" / man["pack_id"] / LC.FILE_NAME), "../x.json", "a/b.json"):
        with pytest.raises(LC.LineCoverageError, match="plain file name"):
            LC.load({**man, **fields, "line_coverage_file": bad}, sealed_chunks_sha256=sha,
                    pack_dir=pack_dir(tmp_path, man))
    with pytest.raises(LC.LineCoverageError, match="pack dir is unknown"):
        LC.load({**man, **fields}, sealed_chunks_sha256=sha)


def test_compose_without_the_options_writes_none_of_the_fields(tmp_path):
    stage, row, _ = sealed_loda_stage(tmp_path)
    m = compose_loda(tmp_path, stage, row)["manifest"]
    for k in ("supersede", "allow_supersede", "staged_text_guard", "line_coverage_file",
              "line_coverage_sha256", "line_coverage"):
        assert k not in m


def test_compose_refuses_the_options_for_a_family_outside_the_allowlist(tmp_path):
    stage, row, _ = sealed_loda_stage(tmp_path)
    with pytest.raises(compose.ComposeError, match="refused for this family"):
        compose.compose_pack(
            pack_id="x", family="eric_official_pdf", chunks_dir=str(stage / "chunks"),
            vectors_dir=str(vectors_for(tmp_path, [row["point_id"]])), collection="aw_e8",
            mrl=256, queue_dir=str(tmp_path / "queue"), supersede="url-group",
            allow_supersede="r", copy_sealed_guard=True)
    assert not (tmp_path / "queue").exists()


@pytest.mark.parametrize("kw,match", [
    ({"supersede": "url-group", "copy_sealed_guard": True}, "--allow-supersede"),
    ({"supersede": "url-group", "allow_supersede": "r"}, "--copy-sealed-guard"),
    ({"supersede": "all", "allow_supersede": "r", "copy_sealed_guard": True}, "only value"),
    ({"allow_supersede": "r"}, "only value"),
    ({"copy_sealed_guard": True}, "only value"),
    ({"supersede": "url-group", "allow_supersede": "r", "copy_sealed_guard": True,
      "line_coverage_wanted_ids": ["w"]}, "--line-coverage-old-texts"),
    ({"supersede": "url-group", "allow_supersede": "r", "copy_sealed_guard": True,
      "served_targets": {"twin": ("u", "c"), "prod": ("u", "c")}}, "already-served"),
    # task 888 D1: main's door (--supersede-url-group) refuses the exclusion as well
    ({"supersede_url_group": "task 876 reason",
      "served_targets": {"twin": ("u", "c"), "prod": ("u", "c")}}, "either door"),
])
def test_compose_refuses_an_incomplete_option_set_before_writing_anything(tmp_path, kw, match):
    stage, row, _ = sealed_loda_stage(tmp_path)
    with pytest.raises(compose.ComposeError, match=match):
        compose_loda(tmp_path, stage, row, **kw)
    assert not (tmp_path / "queue").exists()


@pytest.mark.parametrize("door", [
    {"supersede": "url-group", "allow_supersede": "r", "copy_sealed_guard": True},
    {"supersede_url_group": "task 876 reason"},
])
def test_compose_refuses_exclude_ids_file_with_either_supersede_door(tmp_path, door):
    """Task 892: an excluded unchanged row loses its served copy and is never rewritten."""
    stage, row, _ = sealed_loda_stage(tmp_path)
    with pytest.raises(compose.ComposeError, match="--exclude-ids-file"):
        compose_loda(tmp_path, stage, row, exclude_ids_file="x.jsonl",
                     exclude_ids_reason="r", **door)
    assert not (tmp_path / "queue").exists()


def test_compose_refuses_to_seal_a_pack_the_drain_would_refuse(tmp_path):
    """The sealed block says 5 URL groups were checked and the chunks hold 1: refused here."""
    stage, row, _ = sealed_loda_stage(tmp_path, groups=5)
    with pytest.raises(compose.ComposeError, match="refusing to seal a url-group pack"):
        compose_loda(tmp_path, stage, row, supersede="url-group", allow_supersede="r",
                     copy_sealed_guard=True)
    assert not (tmp_path / "queue").exists()
    stage2 = tmp_path / "nopack"
    shutil.copytree(stage, stage2)
    (stage2 / "PACK.json").unlink()
    with pytest.raises(compose.ComposeError, match="SEALED pack's own stage manifest"):
        compose_loda(tmp_path, stage2, row, supersede="url-group", allow_supersede="r",
                     copy_sealed_guard=True)


def test_compose_writes_no_block_when_no_old_text_is_found(tmp_path):
    stage, row, _ = sealed_loda_stage(tmp_path)
    src = write_shard(tmp_path / "old" / "shard-0000.texts.jsonl.gz",
                      [{"point_id": "zz", "url": "https://elsewhere.example/", "text": "t"}])
    with pytest.raises(compose.ComposeError, match="line coverage block not written"):
        compose_loda(tmp_path, stage, row, supersede="url-group", allow_supersede="r",
                     copy_sealed_guard=True, line_coverage_old_texts=[str(src)])
    assert not (tmp_path / "queue" / "loda-restage" / "READY").exists()


def test_the_cli_exposes_the_options():
    ns = compose.build_parser().parse_args([
        "--pack-id", "p", "--family", "loda_exact", "--chunks-dir", "c", "--vectors-dir", "v",
        "--collection", "aw_e8", "--mrl", "256", "--supersede", "url-group",
        "--allow-supersede", "r", "--copy-sealed-guard", "--line-coverage-old-texts", "a",
        "--line-coverage-old-texts", "b", "--line-coverage-wanted-ids", "w"])
    assert (ns.supersede, ns.copy_sealed_guard, ns.line_coverage_old_texts,
            ns.line_coverage_wanted_ids) == ("url-group", True, ["a", "b"], ["w"])
    with pytest.raises(SystemExit):
        compose.build_parser().parse_args([
            "--pack-id", "p", "--family", "f", "--chunks-dir", "c", "--vectors-dir", "v",
            "--collection", "x", "--mrl", "1", "--supersede", "everything"])


def test_wanted_ids_read_exports_stage_dirs_and_id_lines_united(tmp_path):
    exp = write_shard(tmp_path / "export.jsonl.gz", [{"id": "a", "payload": {"url": URL}}])
    stage = tmp_path / "stage"
    write_shard(stage / "shard-0000.texts.jsonl.gz", [{"point_id": "b", "url": T.URL_A, "text": "t"}])
    ids = tmp_path / "ids.txt"
    ids.write_text("c\n\n")
    assert LC.read_wanted([exp, stage, ids]) == {"a": URL, "b": T.URL_A, "c": None}
    with pytest.raises(LC.LineCoverageError, match="names no id"):
        LC.read_wanted(write_shard(tmp_path / "empty.jsonl.gz", []))

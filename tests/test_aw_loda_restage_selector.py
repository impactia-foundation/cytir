#!/usr/bin/env python3
"""Ruling 27 T6: the loda restage set is explicit. `only_split` (the cut articles) + the short rows
over the 2,560-token window + the URLs of a `urls_from` file; the stage guard sees all of them; the
restage is ONE slice.

REAL ARTICLES: the four long ones of `tests/fixtures/legi_loda_long` and the real short samples of
`data/samples/legi-loda` (2,990 / 2,176 / 2,235 chars), under one `TNC_en_vigueur` root as in
the staged-text-truncation test (its root builder is inlined here: that module is not in the
public cut). The stand-in tokenizer counts
3 chars per token here, and the 2,990-char article is put over the 2,560-token window; the
other two short ones are not (the real tokenizer's 574 over-window rows are measured on the
box, task 706).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import sys

import pytest

pytest.importorskip("lxml")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

LONG_DIR = HERE / "fixtures" / "legi_loda_long"
SHORT_DIR = REPO / "data" / "samples" / "legi-loda"
RELEASE = "aw-task633-dila-loda-full-v1"


def _stage_root(tmp_path):
    """The four long + the real short samples under one `TNC_en_vigueur` root."""
    import shutil
    root = tmp_path / "root"
    shutil.copytree(LONG_DIR / "TNC_en_vigueur", root / "TNC_en_vigueur")
    for p in SHORT_DIR.rglob("LEGIARTI*.xml"):
        dest = root / "TNC_en_vigueur" / "JORF" / p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, dest)
    return root

WINDOW = 2560


def counter(text: str) -> int:
    """3 chars per token, except that the one real short article, whole, is 3,000 tokens: it is
    over the window and its split chunks (under 2,900 chars) are not."""
    return 3000 if (OVER in text and len(text) >= 2900) else len(text) // 3


OVER = "LEGIARTI000020693146"        # real short article, 2,990 chars: over the stand-in window
FITS = "LEGIARTI000018943660"        # real short article, 2,176 chars: fits
URL_FITS_CANON = f"https://legifrance.gouv.fr/loda/article_lc/{FITS}"      # no www: as in #630's file
LONG_IDS = {"LEGIARTI000049864993", "LEGIARTI000039233648", "LEGIARTI000006680074",
            "LEGIARTI000022732629"}


@pytest.fixture(autouse=True)
def one_token_per_char(monkeypatch):
    from cytir.sources.families import loda
    monkeypatch.setattr(loda, "embed_token_counter", lambda: counter)


def run(tmp_path, name, **options):
    """`run_family` over the stage root, with the article root built once per test."""
    pytest.importorskip("httpx")
    pytest.importorskip("qdrant_client")
    from cytir.sources import run_family
    root = tmp_path / "root"
    if not root.exists():
        _stage_root(tmp_path)
    return run_family.run_family(
        "loda_exact", out_dir=tmp_path / name, today="2026-10-05",
        consent=lambda h, d: pytest.fail(f"consent for {h}"),
        options={"root": str(root), "release": RELEASE, **options}, log=lambda m: None)


def staged_articles(tmp_path, name):
    rows = [json.loads(x) for x in (tmp_path / name / "pack" / "records.jsonl").read_text().splitlines()]
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["article_id"], []).append(r)
    return out


def selector_file(tmp_path, urls, name="urls.jsonl", gz=False):
    path = tmp_path / (name + (".gz" if gz else ""))
    data = "".join(json.dumps({"url": u, "reasons": ["x"]}) + "\n" for u in urls)
    if gz:
        with gzip.open(path, "wt") as fh:
            fh.write(data)
    else:
        path.write_text(data)
    return path


def test_a_short_row_over_the_token_window_is_split_in_a_restage_and_kept_in_a_full_run(tmp_path):
    full = run(tmp_path, "full")
    got = staged_articles(tmp_path, "full")
    assert len(got[OVER]) == 1 and len(got[FITS]) == 1          # a full run never tokenises a short row
    assert got[OVER][0]["anchor"] == f"article:{OVER}"
    assert "restage_selector" not in full["discover"]

    man = run(tmp_path, "restage", only_split="true")
    got = staged_articles(tmp_path, "restage")
    assert set(got) == LONG_IDS | {OVER}                        # the two that fit stay unchanged
    assert len(got[OVER]) >= 2
    assert [r["anchor"] for r in got[OVER]] == [f"article:{OVER}:chunk-{i}"
                                                for i in range(1, len(got[OVER]) + 1)]
    assert max(len(r["text"]) // 3 for r in got[OVER]) <= WINDOW
    assert man["terminals"]["excluded_reasons"]["single_row_unchanged"] == 2
    sel = man["discover"]["restage_selector"]
    assert (sel["split"], sel["over_window"], sel["urls_from_only"]) == (4, 1, 0)


def test_a_url_of_the_selector_file_is_staged_unchanged_and_the_selector_is_recorded(tmp_path):
    f = selector_file(tmp_path, [URL_FITS_CANON, f"https://www.legifrance.gouv.fr/loda/article_lc/{OVER}"])
    man = run(tmp_path, "sel", only_split="true", urls_from=str(f))
    got = staged_articles(tmp_path, "sel")
    assert set(got) == LONG_IDS | {OVER, FITS}
    run(tmp_path, "full-ref")
    ref = staged_articles(tmp_path, "full-ref")[FITS]
    assert len(got[FITS]) == 1 and got[FITS][0] == ref[0]          # byte for byte: same id, same row
    assert got[FITS][0]["anchor"] == f"article:{FITS}"
    sel = man["discover"]["restage_selector"]
    assert (sel["split"], sel["over_window"], sel["urls_from_only"],
            sel["urls_from_also_split"]) == (4, 1, 1, 1)
    assert man["discover"]["urls_from"] == {
        "path": str(f), "urls": 2, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    assert man["run"]["options"]["urls_from"] == str(f)          # the selector is in the manifest


def test_the_stage_guard_sees_every_selected_document_including_the_extras(tmp_path):
    f = selector_file(tmp_path, [URL_FITS_CANON])
    man = run(tmp_path, "guard", only_split="true", urls_from=str(f))
    g = man["staged_text_guard"]
    assert g["docs_checked"] == man["terminals"]["units_admitted"] == 6   # 4 long + 1 over + 1 listed
    assert g["docs_unchecked"] == 0 and g["docs_shortfall"] == 0
    assert g["threshold_declared"] is True and g["min_coverage"] == 1.0


LONG_ONE = "LEGIARTI000049864993"
URL_LONG = f"https://www.legifrance.gouv.fr/loda/article_lc/{LONG_ONE}"


def test_a_listed_article_is_excluded_from_the_restage_and_the_list_is_in_the_manifest(tmp_path):
    ex = selector_file(tmp_path, [URL_LONG, URL_FITS_CANON], name="exclude.jsonl")
    sel = selector_file(tmp_path, [URL_FITS_CANON])
    man = run(tmp_path, "ex", only_split="true", urls_from=str(sel), exclude_urls=str(ex))
    got = staged_articles(tmp_path, "ex")
    assert set(got) == (LONG_IDS - {LONG_ONE}) | {OVER}            # the exclusion beats urls_from too
    assert man["terminals"]["excluded_reasons"]["restage_excluded_by_list"] == 2
    assert man["discover"]["exclude_urls"] == {
        "path": str(ex), "urls": 2, "sha256": hashlib.sha256(ex.read_bytes()).hexdigest()}
    assert man["run"]["options"]["exclude_urls"] == str(ex)
    s = man["discover"]["restage_selector"]
    assert (s["split"], s["over_window"], s["urls_from_only"]) == (3, 1, 0)   # an excluded one is not counted
    g = man["staged_text_guard"]
    assert g["docs_checked"] == man["terminals"]["units_admitted"] == 4      # = the URL groups
    assert g["docs_unchecked"] == 0 and g["docs_shortfall"] == 0


def test_the_exclusion_list_needs_only_split_and_a_real_loda_url_file(tmp_path):
    ex = selector_file(tmp_path, [URL_LONG], name="exclude.jsonl")
    with pytest.raises(ValueError, match="exclude_urls needs only_split=true"):
        run(tmp_path, "nosplit", exclude_urls=str(ex))
    with pytest.raises(ValueError, match="exclude_urls=.* is not a file"):
        run(tmp_path, "nofile", only_split="true", exclude_urls=str(tmp_path / "none.jsonl"))
    bad = tmp_path / "bad.txt"
    bad.write_text("https://example.org/x\n")
    with pytest.raises(ValueError, match="is not a loda article URL"):
        run(tmp_path, "bad", only_split="true", exclude_urls=str(bad))
    assert not (tmp_path / "bad" / "pack").exists()


def test_the_selector_reads_gz_and_plain_url_lines(tmp_path):
    from cytir.sources.families import loda
    gz = selector_file(tmp_path, [URL_FITS_CANON], gz=True)
    assert loda.urls_from({"only_split": "true", "urls_from": str(gz)}) == frozenset(
        {"https://legifrance.gouv.fr/loda/article_lc/" + FITS})
    txt = tmp_path / "u.txt"
    txt.write_text(f"https://www.legifrance.gouv.fr/loda/article_lc/{FITS}\n\n")
    assert loda.urls_from({"only_split": "1", "urls_from": str(txt)}) == frozenset(
        {"https://legifrance.gouv.fr/loda/article_lc/" + FITS})
    assert loda.urls_from({"only_split": "true"}) is None


@pytest.mark.parametrize("content,match", [
    ("", "names no URL"),
    ("not a url\n", "is not a loda article URL"),
    ('{"url": "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI1"}\n', "not a loda article URL"),
    ("{broken\n", "not JSON"),
])
def test_a_bad_selector_file_refuses_the_run_at_start(tmp_path, content, match):
    f = tmp_path / "bad.jsonl"
    f.write_text(content)
    with pytest.raises(ValueError, match=match):
        run(tmp_path, "bad", only_split="true", urls_from=str(f))
    assert not (tmp_path / "bad" / "pack").exists()


def test_the_selector_without_only_split_or_with_a_missing_file_refuses(tmp_path):
    f = selector_file(tmp_path, [URL_FITS_CANON])
    with pytest.raises(ValueError, match="needs only_split=true"):
        run(tmp_path, "nosplit", urls_from=str(f))
    with pytest.raises(ValueError, match="is not a file"):
        run(tmp_path, "nofile", only_split="true", urls_from=str(tmp_path / "missing.jsonl"))


def _opened_door(tmp_path, name, **options):
    """A restage run whose pack the door's contract check would pass (the POLICY PR is open)."""
    run(tmp_path, name, only_split="true", **options)
    pack = tmp_path / name / "pack"
    man = json.loads((pack / "manifest.json").read_text())
    man["contract"]["unknowns"] = []
    man["consent"]["checked"] = True
    (pack / "manifest.json").write_text(json.dumps(man))
    return pack


def test_a_restage_pack_that_would_be_cut_into_slices_is_refused_and_nothing_is_staged(tmp_path):
    from cytir.sources import family_stage as FS
    pack = _opened_door(tmp_path, "slices")
    corpus = tmp_path / "corpus"
    with pytest.raises(FS.StageError, match="ONE slice"):
        FS.stage_family_pack(pack, corpus_root=corpus, family_id="loda_exact", shard_rows=3)
    assert not corpus.exists() or not list(corpus.rglob("PACK.json"))


def test_a_restage_pack_refuses_drop_served(tmp_path):
    from cytir.sources import family_stage as FS
    pack = _opened_door(tmp_path, "dropserved")
    with pytest.raises(FS.StageError, match="--drop-served refuses a restage pack"):
        FS.stage_family_pack(pack, corpus_root=tmp_path / "corpus", family_id="loda_exact",
                             drop_served={"twin": ("http://t", "c")})


def test_the_single_slice_flag_refuses_any_pack_over_the_slice_size(tmp_path):
    from cytir.sources import family_stage as FS
    run(tmp_path, "full")
    pack = tmp_path / "full" / "pack"
    man = json.loads((pack / "manifest.json").read_text())
    man["contract"]["unknowns"] = []
    man["consent"]["checked"] = True
    (pack / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(FS.StageError, match="ONE slice"):
        FS.stage_family_pack(pack, corpus_root=tmp_path / "c", family_id="loda_exact",
                             shard_rows=3, single_slice=True)
    assert FS._parser().parse_args(["stage", "loda_exact", "--pack", "p", "--corpus-root", "c",
                                    "--single-slice"]).single_slice is True


def test_the_short_row_test_checks_chars_and_tokens_in_the_splitter():
    from cytir.extract import legi
    paras = [f"### Paragraph {i} {{#A-p{i}}}\n\nCorps de l'alinea {i}. " + "mot " * 90
             for i in range(1, 9)]
    md = "\n\n".join(["## Article identity\n- Article ID: `A`", *paras])
    assert 2560 < len("T\n\n" + md) < 8000          # short in characters, over the window in tokens
    count = len                                                        # noqa: E731  one token per char
    assert legi.split_article_markdown(md, title="T", count_tokens=count) == [md]       # default
    parts = legi.split_article_markdown(md, title="T", count_tokens=count,
                                        split_over_window=True, max_tokens=2560)
    assert len(parts) >= 2 and all(count("T\n\n" + c) <= 2560 for c in parts)
    ok = legi.split_article_markdown(md, title="T", count_tokens=lambda t: 100,
                                     split_over_window=True)
    assert ok == [md]                                                  # fits both: byte for byte
    with pytest.raises(ValueError, match="needs `count_tokens`"):
        legi.split_article_markdown(md, title="T", split_over_window=True)

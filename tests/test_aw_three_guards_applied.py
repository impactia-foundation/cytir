"""Task 1861: three computed-but-unused guards are now applied."""
from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from cytir.acquire.github_docs_tree import render_markdown
from cytir.embed import vector_rollup as vr
from cytir.extract.pubmed import iter_pubmed_articles


def _stage(tmp_path, ids_sha_override=None):
    root = tmp_path / "root"
    vec = root / "vec"
    stage = root / "stage"
    vec.mkdir(parents=True)
    stage.mkdir()
    dim = 2
    (vec / "shard-0.vectors.f16").write_bytes(b"\x00" * (3 * dim * 2))
    (vec / "shard-0.row_ids.jsonl").write_text('"a"\n"b"\n"c"\n')
    man = {"n": 3, "sha256": vr.sha256_file(vec / "shard-0.vectors.f16"),
           "row_ids_sha256": ids_sha_override or vr.sha256_file(vec / "shard-0.row_ids.jsonl")}
    (vec / "shard-0.vectors.manifest.json").write_text(json.dumps(man))
    (stage / "STAGE-RECEIPT.json").write_text(json.dumps({"status": "complete", "rows": 3}))
    ver = root / "verification.json"
    ver.write_text(json.dumps({"failed": 0, "rows": 3, "shards": 1}))
    return SimpleNamespace(vectors=str(vec), artifact_root=str(root), stage=str(stage),
                           verification=str(ver), dim=dim, output_name="RECEIPT.json",
                           supersedes="", evidence_prefix="ev")


def test_rollup_accepts_matching_row_ids_digest(tmp_path):
    assert vr.seal(_stage(tmp_path))["rows"] == 3


def test_rollup_refuses_wrong_row_ids_digest(tmp_path):
    with pytest.raises(ValueError, match="row-ids SHA mismatch"):
        vr.seal(_stage(tmp_path, ids_sha_override="0" * 64))


def test_longer_fence_is_not_closed_by_shorter_one():
    out = render_markdown("before\n````\none\n```\ntwo\n````\nafter")
    assert "one" in out and "two" in out and "after" in out
    assert re.search(r"one +two", out)


def test_tilde_inside_backtick_block_is_content():
    out = render_markdown("```\nalpha\n~~~\nbeta\n```\nafter")
    assert "alpha ~~~ beta" in out


def _xml(lang_xml):
    abstract = "word " * 30
    return (f"<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>1</PMID><Article>"
            f"<ArticleTitle>A title about things</ArticleTitle><Abstract><AbstractText>"
            f"{abstract}</AbstractText></Abstract>{lang_xml}</Article></MedlineCitation>"
            f"</PubmedArticle></PubmedArticleSet>").encode()


@pytest.mark.parametrize("code,want", [("eng", "en"), ("fre", "fr"), ("ger", "de"), ("ita", "it")])
def test_pubmed_language_follows_article(code, want):
    rows = list(iter_pubmed_articles(_xml(f"<Language>{code}</Language>")))
    assert len(rows) == 1 and rows[0]["lang"] == want


def test_pubmed_first_listed_language_wins():
    xml = _xml("<Language>ger</Language><Language>eng</Language>")
    assert [r["lang"] for r in iter_pubmed_articles(xml)] == ["de"]


@pytest.mark.parametrize("lang_xml", ["<Language>rus</Language>", ""])
def test_pubmed_out_of_target_or_missing_language_is_dropped(lang_xml):
    assert list(iter_pubmed_articles(_xml(lang_xml))) == []

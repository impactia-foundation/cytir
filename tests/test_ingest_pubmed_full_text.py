import io
import json

from cytir.extract.pubmed import iter_pubmed_articles, parse_gz


def test_pubmed_parser_preserves_abstract_beyond_6000_characters():
    abstract = "Complete biomedical evidence sentence. " * 250
    xml = f"""<PubmedArticleSet><PubmedArticle><MedlineCitation>
      <PMID>123</PMID><Article><ArticleTitle>Long abstract</ArticleTitle>
      <Abstract><AbstractText>{abstract}</AbstractText></Abstract>
      <Language>eng</Language></Article></MedlineCitation></PubmedArticle>
      </PubmedArticleSet>""".encode()
    output = io.StringIO()
    assert parse_gz(xml, output) == 1
    row = json.loads(output.getvalue())
    assert abstract.strip() in row["text"]
    assert len(row["text"]) > 6000


def test_pubmed_parser_preserves_structured_fields_inline_markup_and_links():
    long_methods = "Measured complete evidence. " * 260
    xml = f"""<PubmedArticleSet><PubmedArticle><MedlineCitation>
      <PMID>456</PMID><Article><ArticleTitle>Effect of <i>BRCA1</i> variants</ArticleTitle>
      <ELocationID EIdType="doi">10.1234/primary</ELocationID>
      <Abstract>
        <AbstractText Label="BACKGROUND" NlmCategory="BACKGROUND">Prior <b>inline</b> evidence establishes context.</AbstractText>
        <AbstractText Label="METHODS" NlmCategory="METHODS">{long_methods}</AbstractText>
      </Abstract><Language>eng</Language></Article>
      <OtherAbstract Type="publisher" Language="fre">
        <AbstractText Label="RÉSUMÉ">Résumé secondaire conservé intégralement.</AbstractText>
      </OtherAbstract></MedlineCitation>
      <PubmedData><ArticleIdList>
        <ArticleId IdType="doi">10.1234/primary</ArticleId>
        <ArticleId IdType="pmc">PMC987</ArticleId>
      </ArticleIdList><ReferenceList><Reference><Citation>Cited study</Citation>
        <ArticleIdList><ArticleId IdType="pubmed">789</ArticleId></ArticleIdList>
      </Reference></ReferenceList></PubmedData>
      </PubmedArticle></PubmedArticleSet>""".encode()
    archive = {"member": "_dumps/pubmed/pubmed26n0001.xml.gz", "sha256": "a" * 64}
    row = next(iter_pubmed_articles(xml, source_archive=archive))
    assert row["title"] == "Effect of BRCA1 variants"
    assert row["text"].startswith("# Effect of BRCA1 variants\n\n## Record identity")
    assert "- PMID: `456`" in row["text"]
    assert "- Parent archive: `_dumps/pubmed/pubmed26n0001.xml.gz`" in row["text"]
    assert "## Abstract\n\n### BACKGROUND" in row["text"]
    assert "Prior inline evidence" in row["text"]
    assert "### METHODS" in row["text"]
    assert long_methods.strip() in row["text"]
    assert "## Other abstract\n\n- Type: publisher\n- Language: fre" in row["text"]
    assert "Résumé secondaire conservé intégralement." in row["text"]
    assert row["source_record_id"] == "456"
    assert row["parent_archive"] == archive
    assert any(link["url"] == "https://doi.org/10.1234/primary" for link in row["source_links"])
    assert any(link["url"] == "https://pmc.ncbi.nlm.nih.gov/articles/PMC987/" for link in row["source_links"])
    assert any(link["url"] == "https://pubmed.ncbi.nlm.nih.gov/789/" and
               link["relation"] == "cited-source" for link in row["source_links"])


def test_pubmed_other_abstract_does_not_admit_record_without_primary_abstract():
    xml = b"""<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>999</PMID>
      <Article><ArticleTitle>Other only</ArticleTitle></Article>
      <OtherAbstract><AbstractText>This other abstract is deliberately longer than eighty characters but must not alter the exact primary-abstract population filter.</AbstractText></OtherAbstract>
      </MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    assert list(iter_pubmed_articles(xml)) == []


def test_the_old_driver_path_forwards_to_the_package_module():
    from cytir.extract import pubmed
    assert iter_pubmed_articles is pubmed.iter_pubmed_articles

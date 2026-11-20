import hashlib
import xml.etree.ElementTree as ET


import cytir.sources.justice_canada.render as justice


def test_act_slug_is_exact_and_rejects_other_routes():
    assert justice.act_slug("https://laws-lois.justice.gc.ca/eng/acts/A-1/") == "A-1"
    for url in (
        "https://laws-lois.justice.gc.ca/fra/lois/A-1/",
        "https://laws-lois.justice.gc.ca/eng/regulations/SOR-1/",
        "https://example.com/eng/acts/A-1/",
    ):
        try:
            justice.act_slug(url)
        except ValueError:
            pass
        else:
            raise AssertionError(url)


def test_renderer_preserves_inline_text_hierarchy_and_links():
    root = ET.fromstring(b'''<Statute xmlns:lims="http://justice.gc.ca/lims"
      lims:current-date="2026-08-14" lims:lastAmendedDate="2026-08-01" in-force="yes">
      <Identification><ShortTitle>Example Act</ShortTitle></Identification>
      <Body><Heading level="1"><TitleText>Purpose</TitleText></Heading>
      <Section lims:id="10"><Label>1</Label><Text>This Act cites the
      <XRefExternal reference-type="act" link="A-1">Access Act</XRefExternal>.</Text></Section></Body>
      </Statute>''')
    title, markdown, version = justice.render_official_markdown(
        root, url="https://laws-lois.justice.gc.ca/eng/acts/E-1/",
        member="root/eng/acts/E-1.xml", member_sha256="a" * 64)
    assert title == "Example Act"
    assert "### Section `10`" in markdown
    assert "This Act cites the Access Act." in markdown
    assert version["current_date"] == "2026-08-14"
    links = justice.source_links(root, source_slug="E-1")
    assert links == [{
        "relation": "act", "target": "https://laws-lois.justice.gc.ca/eng/acts/A-1/",
        "target_id": "A-1", "context": "Access Act", "direction": "outbound",
        "article_anchor": "E-1",
    }]


def test_generalized_descriptor_and_french_anchored_links():
    assert justice.law_descriptor(
        "https://laws-lois.justice.gc.ca/fra/reglements/DORS-93-348/"
    ) == {"source_class": "fra/reglements", "slug": "DORS-93-348",
          "lang": "fr", "kind": "regulation"}
    root = ET.fromstring(b'''<Regulation xmlns:lims="http://justice.gc.ca/lims">
      <Identification><ShortTitle>Exemple</ShortTitle></Identification>
      <Body><Section lims:id="42"><Label>7</Label><Text>Voir
      <XRefExternal reference-type="act" link="F-27">la loi</XRefExternal>.
      </Text></Section></Body></Regulation>''')
    title, markdown, _ = justice.render_official_markdown(
        root, url="https://laws-lois.justice.gc.ca/fra/reglements/DORS-93-348/",
        member="root/fra/reglements/DORS-93-348.xml", member_sha256="b" * 64)
    assert title == "Exemple"
    assert "### Section `42`" in markdown
    assert justice.legal_unit_anchors(root) == [{
        "kind": "section", "source_id": "42", "label": "7",
        "anchor": "section:42",
    }]
    assert justice.source_links_with_anchors(
        root, source_slug="fra/reglements:DORS-93-348", lang="fr") == [{
            "relation": "act",
            "target": "https://laws-lois.justice.gc.ca/fra/lois/F-27/",
            "target_id": "F-27", "context": "la loi", "direction": "outbound",
            "article_anchor": "section:42",
        }]


def test_plain_archive_binding_does_not_apply_decompressed_body_contract(tmp_path):
    archive = tmp_path / "snapshot.tar.gz"
    archive.write_bytes(b"plain-stage-artifact-bytes")
    ref = {
        "uri": "artifact://snapshot.tar.gz", "bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    assert justice._validate_plain_binding(tmp_path, ref) == archive

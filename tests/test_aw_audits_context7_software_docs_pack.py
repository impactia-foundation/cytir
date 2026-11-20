

import cytir.ops.audits.context7_software_docs_pack as subject


def test_the_lineage_is_read_back_out_of_the_url():
    """Task 517: the audit used to parse the lineage out of the front matter the
    same producer had written two lines earlier. The URL is composed
    independently of the identity block, so reading it back out is a real
    cross-check -- and a non-immutable ref fails it."""
    match = subject.URL_LINEAGE.fullmatch(
        "https://github.com/o/r/blob/" + "a" * 40 + "/docs/a.md")
    assert match is not None
    assert match.group("repo") == "o/r"
    assert match.group("revision") == "a" * 40
    assert match.group("path") == "docs/a.md"
    assert subject.URL_LINEAGE.fullmatch(
        "https://github.com/o/r/blob/main/docs/a.md") is None


def test_a_body_that_opens_with_the_provenance_is_REFUSED():
    """The defect this audit could not see: 72,330 of 126,427 staged context7
    rows shipped with the YAML block as their snippet and as the head of their
    dense embedding, and every check passed."""
    v1 = ('---\naw_repository: "o/r"\n'
          f'aw_revision: "{"a" * 40}"\n'
          'aw_source_path: "docs/a.md"\n'
          f'aw_parent_sha256: {"b" * 64}\n---\n\n# Heading\n')
    assert subject._front_matter_absent(v1) is False
    assert subject._front_matter_absent("# Heading\n\nbody\n") is True
    # a Markdown horizontal rule is not a provenance block
    assert subject._front_matter_absent("---\n\ntext\n") is True


def test_context7_immutable_blob_url_contract():
    assert subject.URL.fullmatch(
        "https://github.com/o/r/blob/" + "a" * 40 + "/docs/a.md")
    assert not subject.URL.fullmatch(
        "https://github.com/o/r/blob/main/docs/a.md")


def test_a_percent_encoded_path_still_joins_to_its_identity_block():
    """The 1024-lab tranche carries CJK paths, which the URL percent-encodes;
    the identity block does not. The audit unquotes before comparing."""
    match = subject.URL_LINEAGE.fullmatch(
        "https://github.com/1024-lab/smart-admin/blob/" + "a" * 40
        + "/%E6%95%B0%E6%8D%AE%E5%BA%93SQL%E8%84%9A%E6%9C%AC/README.md")
    assert match is not None
    from urllib.parse import unquote
    assert unquote(match.group("path")).endswith("SQL脚本/README.md")

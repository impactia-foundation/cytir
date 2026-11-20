import base64
import hashlib
import inspect

import cytir.sources.finma.materialize_pilot as subject


def test_decode_requires_exact_bytes_and_digest():
    body = b"%PDF-1.7\ncomplete"
    row = {"body_b64": base64.b64encode(body).decode(), "bytes": len(body),
           "sha256": hashlib.sha256(body).hexdigest()}
    assert subject._decode(row, "fixture") == body
    row["bytes"] += 1
    try:
        subject._decode(row, "fixture")
    except ValueError as exc:
        assert "body binding mismatch" in str(exc)
    else:
        raise AssertionError("body byte mismatch accepted")


def test_markdown_links_are_uncapped_and_deterministic():
    markdown = "See https://example.org/a and https://example.org/b."
    assert subject._markdown_links(markdown) == [
        {"target_url": "https://example.org/a", "page": None,
         "method": "markdown-uri", "relation": "citation-or-external-uri",
         "context": "Docling Markdown line 1: See https://example.org/a and https://example.org/b."},
        {"target_url": "https://example.org/b", "page": None,
         "method": "markdown-uri", "relation": "citation-or-external-uri",
         "context": "Docling Markdown line 1: See https://example.org/a and https://example.org/b."},
    ]


def test_link_dedupe_sorts_nullable_and_numeric_pages_deterministically():
    links = [
        {"target_url": "https://example.org/a", "page": None,
         "method": "markdown-uri", "context": "line 1"},
        {"target_url": "https://example.org/a", "page": 2,
         "method": "pdf-annotation-uri", "context": "page 2"},
        {"target_url": "https://example.org/a", "page": 1,
         "method": "pdf-annotation-uri", "context": "page 1"},
        {"target_url": "https://example.org/a", "page": None,
         "method": "markdown-uri", "context": "line 1"},
    ]
    assert subject._dedupe_links(links) == [links[2], links[1], links[0]]


def test_runtime_is_explicitly_cpu_pinned():
    source = inspect.getsource(subject.run)
    assert "AcceleratorDevice.CPU" in source
    assert '"accelerator_device": "cpu"' in source


def test_scale_tranches_are_disjoint_source_order_slices():
    source = inspect.getsource(subject.run)
    assert "start <= ordinal < start + limit" in source
    assert "ordinal = start + local_ordinal" in source


def test_source_identity_keeps_distinct_urls_for_identical_pdf_bytes():
    common = {"sha256": "a" * 64}
    de = {**common, "url": "https://www.finma.ch/doc.pdf?sc_lang=de&hash=ABC"}
    fr = {**common, "url": "https://www.finma.ch/doc.pdf?sc_lang=fr&hash=DEF"}
    assert subject._source_record_id(de) != subject._source_record_id(fr)
    assert subject._source_record_id(de).startswith("https://www.finma.ch/")


def test_original_stage_deduplicates_bodies_without_deduplicating_records():
    source = inspect.getsource(subject.run)
    assert "originals_by_sha" in source
    assert 'if source["sha256"] not in originals_by_sha' in source
    assert '"source_record_id": canonical_url' in source
    assert "records=limit" in source

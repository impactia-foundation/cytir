"""Task 450: a day's news spool seals as count-conserving stage evidence."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from cytir.acquire import preserved_page_fetch as PF
from cytir.news import seal_lane_stages as subject
from cytir.index.preservation_contract import _validate_edge, _validate_stage
from test_preserved_page_fetch import UA, Response


DAY = "2026-08-28"


def document_id(url):
    return hashlib.sha256(f"news-document\0{url}".encode()).hexdigest()


def envelope(headline, url, fetched_at, publisher, body):
    return (f"# {headline}\n\nByline: not declared by publisher\n\n"
            f"Publisher: {publisher}\n\nFetched: {fetched_at}\n\n"
            f"Source: {url}\n\n{body}\n")


def spool_with(tmp_path: Path, pages) -> Path:
    """`pages` = (url, body, status, content_type, make_document, outlinks)."""
    spool = tmp_path / "stage-2026-08-28"
    (spool / "pages").mkdir(parents=True)
    documents = []
    for index, (url, body, status, content_type, make_document,
                outlinks) in enumerate(pages):
        fetched_at = f"2026-08-28T0{index}:00:00Z"
        receipt = PF.seal_page_fetch(
            spool / "pages", url,
            Response(content=body.encode(), url=url, requested_url=url,
                     status_code=status, content_type=content_type),
            fetched_at, UA)
        PF.publish_receipt(spool / "pages", url, receipt)
        if not make_document:
            continue
        publisher = url.split("//", 1)[1].split("/", 1)[0].removeprefix("www.")
        text = envelope(f"Headline {index}", url, fetched_at, publisher,
                        f"Body {index}")
        documents.append({
            "schema": subject.DOCUMENT_SCHEMA,
            "document_id": document_id(url), "url": url,
            "headline": f"Headline {index}", "byline": None,
            "byline_status": "not_declared_by_publisher",
            "publication_time": "2026-08-27",
            "publication_time_status": "source_declared",
            "publisher": publisher, "fetched_at": fetched_at,
            "markdown": text,
            "markdown_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "body_markdown_sha256": hashlib.sha256(
                f"Body {index}".encode()).hexdigest(),
            "lineage": {"fetched_at": fetched_at,
                        "original_sha256": receipt["body_sha256"],
                        "original_bytes": receipt["body_bytes"],
                        "http_status": receipt["http_status"],
                        "final_url": receipt["final_url"],
                        "receipt_object_id": receipt["object_id"],
                        "transport_sha256":
                            receipt["artifacts"]["transport"]["body_sha256"]},
            "chunk_point_ids": [f"{index}1111111-1111-1111-1111-11111111111"
                                f"{index}"],
            "outlinks": outlinks,
        })
    (spool / "news_documents.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in documents))
    write_gated(spool, documents)
    return spool


def write_gated(spool: Path, documents) -> None:
    """One gated row per emitted passage, in the shape the projector reads."""
    rows = []
    for document in documents:
        for ordinal, point in enumerate(document["chunk_point_ids"]):
            rows.append({
                "url": f"{document['url']}#p{ordinal}",
                "text": f"{document['headline']} passage {ordinal} " * 8,
                "title": document["headline"],
                "source_type": "news",
                "domain": document["publisher"],
                "anchor": None,
                "point_id": point,
                "news_document_id": document["document_id"],
            })
    (spool / "gated_rows.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


LINK = [{"ordinal": 0, "target_url": "https://ref.test/law",
         "anchor_or_context": "Bundesgesetz", "nofollow": False}]


def default_pages():
    return [
        ("https://www.srf.ch/a", "<html>a</html>", 200,
         "text/html; charset=utf-8", True, LINK),
        ("https://www.rts.ch/b", "<html>b</html>", 200,
         "text/html; charset=utf-8", True, []),
        ("https://www.swissinfo.ch/z", "<html>z</html>", 200,
         "text/html; charset=utf-8", True, LINK),
    ]


POLICY = (Path(__file__).resolve().parents[1] / "data" /
          "chunk-policy-legacy-180w-o30.v1.json")


def emitted_points(spool, arm):
    """The SERVED ids this arm's surviving parents would stage.

    Stands in for the read-back of the arm's staged collection: the default
    case is "everything the chunker emitted was staged", so a test that wants
    a consent/projection drop passes its own smaller set. Derived through the
    same projector the sealer uses, from the same gated rows.
    """
    documents = [row for row in subject.load_documents(spool)
                 if subject.arm_of(row["url"]) == arm]
    resolved, _ = subject.resolve_parents(documents)
    served, _documents, _skipped = subject.served_identities(
        spool / "gated_rows.jsonl", arm)
    return {served[point] for row in resolved
            for point in row["chunk_point_ids"] if point in served}


def run(tmp_path, spool, arm="chmedia", staged=None):
    return subject.seal_arm(
        spool=spool, artifact_root=tmp_path, arm=arm,
        out_root=tmp_path / "preservation" / arm, day=DAY,
        producer_version="task450.test", chunk_policy_path=POLICY,
        staged_point_ids=(emitted_points(spool, arm) if staged is None
                          else staged),
        gated_rows=spool / "gated_rows.jsonl")


def rows_of(tmp_path, stage):
    checked = _validate_stage(stage, "s", artifact_root=tmp_path, deep=True)
    out = []
    for artifact in checked["artifacts"]:
        path = tmp_path / artifact["uri"].removeprefix("artifact://")
        if path.name.endswith(".jsonl.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                out.extend(json.loads(line) for line in handle if line.strip())
    return out


def test_arm_routing_follows_the_lane(tmp_path):
    assert subject.arm_of("https://www.srf.ch/x") == "chmedia"
    assert subject.arm_of("https://www.rts.ch/x") == "chmedia"
    assert subject.arm_of("https://rsi.ch/x") == "chmedia"
    assert subject.arm_of("https://www.swissinfo.ch/x") == "swissinfo"
    assert subject.arm_of("https://news.ycombinator.com/x") is None
    # A lookalike host must not be routed into a Swiss arm.
    assert subject.arm_of("https://evil-srf.ch.attacker.test/x") is None


def test_a_clean_spool_seals_conserving_stage_evidence(tmp_path):
    spool = spool_with(tmp_path, default_pages())
    result = run(tmp_path, spool)

    assert result["schema"] == subject.RUN_SCHEMA
    assert result["arm"] == "chmedia" and result["day"] == DAY
    assert result["counts"]["article_parents"] == 2
    assert result["counts"]["markdown_documents"] == 2
    assert result["counts"]["admitted_originals"] == 2
    assert result["counts"]["quarantine_records"] == 0
    assert result["counts"]["empty_outlink_parents"] == 1
    assert result["counts"]["emitted_points"] == 2

    for name in ("original", "markdown", "outlinks", "chunk_join"):
        _validate_stage(result["stages"][name], name,
                        artifact_root=tmp_path, deep=True)
    _validate_edge(result["lineage"]["original_to_markdown"],
                   "original_to_markdown",
                   result["stages"]["original"]["records"],
                   result["stages"]["markdown"]["records"],
                   artifact_root=tmp_path, deep=True)

    originals = rows_of(tmp_path, result["stages"]["original"])
    markdown = rows_of(tmp_path, result["stages"]["markdown"])
    outlinks = rows_of(tmp_path, result["stages"]["outlinks"])
    assert {row["url"] for row in originals} == {
        "https://www.srf.ch/a", "https://www.rts.ch/b"}
    # Every markdown parent is bound to a sealed original by digest.
    by_id = {row["document_id"]: row for row in originals}
    for row in markdown:
        assert row["parent_response"]["original_sha256"] == \
            by_id[row["document_id"]]["original_sha256"]
        assert "outlinks" not in row and "chunk_point_ids" not in row
        assert row["markdown"].startswith(f"# {row['headline']}\n\nByline: ")
    # The parent with no cited targets gets an explicit receipted empty row.
    empty = [row for row in outlinks if row["is_empty"]]
    assert len(empty) == 1 and empty[0]["target_url"] is None
    assert {row["document_id"] for row in outlinks} == set(by_id)


# --- task 452 / ADR-100: the chunks and vectors stages ----------------------

def test_the_chunks_stage_is_sealed_against_the_ratified_legacy_descriptor(
        tmp_path):
    spool = spool_with(tmp_path, default_pages())
    result = run(tmp_path, spool)
    assert result["status"] == subject.RUN_STATUS
    assert "blocking_gaps" not in result
    _validate_stage(result["stages"]["chunks"], "chunks",
                    artifact_root=tmp_path, deep=True)

    policy = json.loads(POLICY.read_text())
    declared = result["chunk_policy"]
    assert declared["policy_id"] == policy["policy_id"]
    assert declared["rule_legend"] == policy["rule_legend"]
    assert declared["algorithm"] == "legacy-fixed-window-pack-v1"
    # The SHA is the file's, re-derived here rather than taken from the run.
    assert declared["sha256"] == hashlib.sha256(POLICY.read_bytes()).hexdigest()

    chunks = rows_of(tmp_path, result["stages"]["chunks"])
    markdown = {row["document_id"]: row
                for row in rows_of(tmp_path, result["stages"]["markdown"])}
    assert len(chunks) == result["counts"]["emitted_points"] == 2
    for row in chunks:
        parent = markdown[row["document_id"]]
        assert row["parent_markdown_sha256"] == hashlib.sha256(
            parent["markdown"].encode()).hexdigest()
        assert row["chunk_ordinal"] == 0
        assert row["chunk_policy"] == declared
        assert row["source_metadata"] == {
            key: parent[key] for key in subject.NEWS_KEYS}


def test_the_pack_declares_the_descriptor_the_news_auditor_accepts(tmp_path):
    spool = spool_with(tmp_path, default_pages())
    run(tmp_path, spool)
    pack = json.loads(
        (tmp_path / "preservation" / "chmedia" / "PACK.json").read_text())
    assert pack["source_profile"] == "news_article"
    assert pack["chunk_policy"] == {
        "path": "data/chunk-policy-legacy-180w-o30.v1.json",
        "policy_id": "legacy-180w-o30",
        "sha256": hashlib.sha256(POLICY.read_bytes()).hexdigest()}
    # The exact shape audit_news_article_profile._validate_news_policy reads.
    import cytir.ops.audits.news_article_profile as audit
    audit._validate_news_policy(pack, POLICY.parent.parent / "data" /
                                "source-preservation-profiles.v1.json")


def test_ordinals_are_gapless_per_parent_over_the_distinct_points(tmp_path):
    spool = spool_with(tmp_path, default_pages())
    path = spool / "news_documents.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    first = rows[0]["chunk_point_ids"][0]
    second = "abcdef01-2222-3333-4444-555555555555"
    # emitted: [p0, p1, p0] — the repeat is one point, and the ordinals over
    # the distinct set stay 0..n-1 with no hole where the repeat was.
    rows[0]["chunk_point_ids"] = [first, second, first]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = run(tmp_path, spool)
    chunks = rows_of(tmp_path, result["stages"]["chunks"])
    ordinals = sorted(row["chunk_ordinal"] for row in chunks
                      if row["point_id"] in (first, second))
    assert ordinals == [0, 1]
    assert result["counts"]["repeated_points_within_documents"] == 1


def test_a_parent_that_emitted_no_chunk_is_quarantined_not_carried(tmp_path):
    spool = spool_with(tmp_path, default_pages())
    path = spool / "news_documents.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    rows[0]["chunk_point_ids"] = []
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = run(tmp_path, spool)

    assert result["counts"]["chunkless_parents"] == 1
    assert result["counts"]["article_parents"] == 1
    edge = result["lineage"]["original_to_markdown"]
    assert edge["input_records"] == 2 and edge["quarantined_records"] == 1
    _validate_edge(edge, "original_to_markdown", 2, 1,
                   artifact_root=tmp_path, deep=True)
    # Every parent that survives into `markdown` resolves to at least one chunk.
    markdown = rows_of(tmp_path, result["stages"]["markdown"])
    chunks = rows_of(tmp_path, result["stages"]["chunks"])
    assert {row["document_id"] for row in markdown} == \
        {row["document_id"] for row in chunks}
    # …and the one that did not is named, with why and how short it was.
    with gzip.open(tmp_path / "preservation" / "chmedia" /
                   "extraction-quarantine.jsonl.gz", "rt",
                   encoding="utf-8") as handle:
        held = [json.loads(line) for line in handle if line.strip()]
    assert [row["reason"] for row in held] == ["no_chunk_emitted_under_chunk_policy"]
    assert held[0]["body_words"] == len("Body 0".split())
    assert held[0]["chunk_policy"]["rule_legend"] == "legacy-180w-o30@1"
    # Its response bytes are still sealed as an original — nothing is dropped.
    originals = rows_of(tmp_path, result["stages"]["original"])
    assert held[0]["document_id"] in {row["document_id"] for row in originals}


def test_the_vectors_stage_is_read_back_and_the_drop_is_quarantined(tmp_path):
    spool = spool_with(tmp_path, default_pages())
    staged = sorted(emitted_points(spool, "chmedia"))
    result = run(tmp_path, spool, staged=set(staged[:1]))

    assert result["counts"]["chunk_records"] == 2
    assert result["counts"]["vector_records"] == 1
    assert result["counts"]["unstaged_chunks"] == 1
    edge = result["lineage"]["chunks_to_vectors"]
    assert edge == {"input_records": 2, "emitted_parent_records": 1,
                    "quarantined_records": 1, "output_records": 1,
                    "quarantine_receipt": edge["quarantine_receipt"]}
    _validate_edge(edge, "chunks_to_vectors", 2, 1,
                   artifact_root=tmp_path, deep=True)
    vectors = rows_of(tmp_path, result["stages"]["vectors"])
    assert [row["point_id"] for row in vectors] == staged[:1]


def test_a_staged_point_no_gated_row_accounts_for_is_refused(tmp_path):
    spool = spool_with(tmp_path, default_pages())
    staged = emitted_points(spool, "chmedia") | {
        "deadbeef-0000-0000-0000-000000000000"}
    with pytest.raises(subject.NewsLaneSealError,
                       match="not produced by this arm"):
        run(tmp_path, spool, staged=staged)


def test_the_served_identity_is_the_deployed_projector_not_the_chunk_id(
        tmp_path):
    """The collection's ids are minted by aw_b1_projection, not by the chunker."""
    import cytir.core.projection as PROJ
    spool = spool_with(tmp_path, default_pages())
    result = run(tmp_path, spool)
    vectors = rows_of(tmp_path, result["stages"]["vectors"])
    chunks = rows_of(tmp_path, result["stages"]["chunks"])
    assert {row["point_id"] for row in vectors} & \
        {row["point_id"] for row in chunks} == set()
    gated = [json.loads(line) for line
             in (spool / "gated_rows.jsonl").read_text().splitlines() if line]
    by_chunk = {row["point_id"]: row for row in gated}
    for row in vectors:
        chunk_id, = row["chunk_point_ids"]
        assert row["point_id"] == PROJ.project_record(by_chunk[chunk_id])[0]


def test_a_policy_that_is_not_the_legacy_descriptor_is_refused(tmp_path):
    spool = spool_with(tmp_path, default_pages())
    other = (Path(__file__).resolve().parents[1] / "data" /
             "chunk-policy-news-article-768w-o0-adjacent-sections-tail-1152-"
             "tok2560.v1.json")
    with pytest.raises(subject.NewsLaneSealError, match="legacy chunker"):
        subject.seal_arm(
            spool=spool, artifact_root=tmp_path, arm="chmedia",
            out_root=tmp_path / "preservation" / "chmedia", day=DAY,
            producer_version="task452.test", chunk_policy_path=other,
            staged_point_ids=emitted_points(spool, "chmedia"),
            gated_rows=spool / "gated_rows.jsonl")


def test_non_admitted_responses_are_preserved_evidence_not_originals(tmp_path):
    pages = default_pages() + [
        ("https://www.srf.ch/gone", "gone", 404, "text/html", False, []),
        ("https://www.srf.ch/feed.json", "{}", 200, "application/json",
         False, []),
    ]
    result = run(tmp_path, spool_with(tmp_path, pages))
    assert result["counts"]["preserved_responses"] == 4
    assert result["counts"]["admitted_originals"] == 2
    assert result["counts"]["non_admitted_responses"] == 2
    reasons = {row["reason"] for row in
               rows_of(tmp_path, result["stages"]["non_admitted_responses"])}
    assert reasons == {"http_status", "non_html"}
    # They are not silently dropped: the preserved bytes are still readable.
    assert result["lineage"]["original_to_markdown"]["input_records"] == 2


def test_an_original_that_extracted_nothing_is_explicitly_quarantined(tmp_path):
    pages = default_pages() + [
        ("https://www.srf.ch/navshell", "<html>nav</html>", 200,
         "text/html; charset=utf-8", False, []),
    ]
    result = run(tmp_path, spool_with(tmp_path, pages))
    edge = result["lineage"]["original_to_markdown"]
    assert edge == {"input_records": 3, "emitted_parent_records": 2,
                    "quarantined_records": 1, "output_records": 2,
                    "quarantine_receipt": edge["quarantine_receipt"]}
    _validate_edge(edge, "original_to_markdown", 3, 2,
                   artifact_root=tmp_path, deep=True)
    assert result["counts"]["quarantine_records"] == 1


def test_a_parent_without_an_admitted_original_is_refused(tmp_path):
    spool = spool_with(tmp_path, default_pages())
    path = spool / "news_documents.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    rows[0]["url"] = "https://www.srf.ch/never-fetched"
    rows[0]["document_id"] = document_id(rows[0]["url"])
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(subject.NewsLaneSealError, match="no admitted original"):
        run(tmp_path, spool)


def test_a_parent_whose_digest_drifted_is_refused(tmp_path):
    spool = spool_with(tmp_path, default_pages())
    path = spool / "news_documents.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    rows[0]["lineage"]["original_sha256"] = "f" * 64
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(subject.NewsLaneSealError, match="does not bind"):
        run(tmp_path, spool)


def test_a_rewritten_markdown_body_is_refused(tmp_path):
    spool = spool_with(tmp_path, default_pages())
    path = spool / "news_documents.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    rows[0]["markdown"] = rows[0]["markdown"] + "tampered"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(subject.NewsLaneSealError, match="does not bind"):
        run(tmp_path, spool)


def test_a_cited_target_without_anchor_context_is_refused(tmp_path):
    pages = default_pages()
    pages[0] = (*pages[0][:5],
                [{"ordinal": 0, "target_url": "https://ref.test/law",
                  "anchor_or_context": "", "nofollow": False}])
    with pytest.raises(subject.NewsLaneSealError, match="anchor/context"):
        run(tmp_path, spool_with(tmp_path, pages))


def test_a_tampered_preserved_body_is_not_evidence(tmp_path):
    spool = spool_with(tmp_path, default_pages())
    body = next((spool / "pages" / "original" / "publisher_pages").rglob("*.gz"))
    body.write_bytes(b"not gzip at all")
    with pytest.raises(Exception):
        run(tmp_path, spool)


def test_sealed_artifacts_are_immutable_and_the_run_is_reproducible(tmp_path):
    spool = spool_with(tmp_path, default_pages())
    first = run(tmp_path, spool)
    with pytest.raises(subject.NewsLaneSealError, match="refusing to replace"):
        run(tmp_path, spool)
    # The run descriptor itself is content-stable.
    run_path = tmp_path / "preservation" / "chmedia" / "RUN.json"
    assert json.loads(run_path.read_text()) == first


def test_each_arm_seals_only_its_own_publishers(tmp_path):
    spool = spool_with(tmp_path, default_pages())
    swissinfo = run(tmp_path, spool, arm="swissinfo")
    assert swissinfo["counts"]["article_parents"] == 1
    urls = {row["url"] for row in rows_of(tmp_path, swissinfo["stages"]["original"])}
    assert urls == {"https://www.swissinfo.ch/z"}


# --- several queued URLs redirecting onto one page (task 450, live rts.ch) ---

def duplicate_pages(bodies):
    """One URL fetched N times with the given bodies (a rotating hub page)."""
    return [("https://www.srf.ch/hub", body, 200, "text/html; charset=utf-8",
             True, LINK) for body in bodies]


def spool_with_repeats(tmp_path: Path, bodies) -> Path:
    """`spool_with` keys receipts by URL, so build the repeats by hand."""
    spool = spool_with(tmp_path, default_pages())
    url = "https://www.srf.ch/hub"
    rows = [json.loads(line)
            for line in (spool / "news_documents.jsonl").read_text().splitlines()
            if line.strip()]
    for index, body in enumerate(bodies):
        fetched_at = f"2026-08-28T1{index}:00:00Z"
        receipt = PF.seal_page_fetch(
            spool / "pages", f"{url}?q={index}",
            Response(content=body.encode(), url=url,
                     requested_url=f"{url}?q={index}"),
            fetched_at, UA)
        PF.publish_receipt(spool / "pages", f"{url}?q={index}", receipt)
        text = envelope("Hub", url, fetched_at, "srf.ch", body)
        rows.append({
            "schema": subject.DOCUMENT_SCHEMA,
            "document_id": document_id(url), "url": url,
            "headline": "Hub", "byline": None,
            "byline_status": "not_declared_by_publisher",
            "publication_time": "2026-08-27",
            "publication_time_status": "source_declared",
            "publisher": "srf.ch", "fetched_at": fetched_at,
            "markdown": text,
            "markdown_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "body_markdown_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "lineage": {"fetched_at": fetched_at,
                        "original_sha256": receipt["body_sha256"],
                        "original_bytes": receipt["body_bytes"],
                        "http_status": 200, "final_url": url,
                        "receipt_object_id": receipt["object_id"],
                        "transport_sha256":
                            receipt["artifacts"]["transport"]["body_sha256"]},
            "chunk_point_ids": ["99999999-9999-9999-9999-999999999999"],
            "outlinks": LINK,
        })
    (spool / "news_documents.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return spool


def test_identical_repeat_fetches_collapse_to_one_parent(tmp_path):
    spool = spool_with_repeats(tmp_path, ["same body"] * 3)
    result = run(tmp_path, spool)
    assert result["counts"]["article_parents"] == 3
    assert result["counts"]["repeat_fetches_collapsed"] == 2
    assert result["counts"]["divergent_parent_documents"] == 0
    markdown = rows_of(tmp_path, result["stages"]["markdown"])
    hub = [row for row in markdown if row["url"].endswith("/hub")]
    assert len(hub) == 1
    # The earliest fetch is the parent, deterministically.
    assert hub[0]["fetched_at"] == "2026-08-28T10:00:00Z"


def test_divergent_repeat_fetches_are_quarantined_not_guessed(tmp_path):
    spool = spool_with_repeats(tmp_path, ["body one", "body two", "body three"])
    result = run(tmp_path, spool)

    assert result["counts"]["divergent_parent_documents"] == 1
    assert result["counts"]["article_parents"] == 2
    markdown = rows_of(tmp_path, result["stages"]["markdown"])
    assert not [row for row in markdown if row["url"].endswith("/hub")]
    # No parent claims its points, so they can never ride a pack.
    joins = rows_of(tmp_path, result["stages"]["chunk_join"])
    assert not [row for row in joins
                if "99999999-9999-9999-9999-999999999999"
                in row["chunk_point_ids"]]
    quarantine = rows_of(
        tmp_path, {**result["stages"]["markdown"],
                   **{"records": 0}}) if False else None
    edge = result["lineage"]["original_to_markdown"]
    assert edge["quarantined_records"] == 1
    _validate_edge(edge, "original_to_markdown",
                   result["stages"]["original"]["records"],
                   result["stages"]["markdown"]["records"],
                   artifact_root=tmp_path, deep=True)


def test_resolve_parents_groups_on_content_not_on_the_rendered_envelope():
    def row(document, sha, fetched, receipt):
        # markdown_sha256 deliberately differs on every row: the envelope
        # carries `Fetched:`, so it always does. Only the body may decide.
        return {"document_id": document, "body_markdown_sha256": sha,
                "markdown_sha256": f"{document}{fetched}{receipt}",
                "fetched_at": fetched,
                "lineage": {"receipt_object_id": receipt}}

    resolved, divergent = subject.resolve_parents([
        row("b", "1", "t2", "r2"), row("b", "1", "t1", "r1"),
        row("a", "1", "t1", "r1"), row("c", "1", "t1", "r1"),
        row("c", "2", "t2", "r2")])
    assert [r["document_id"] for r in resolved] == ["a", "b"]
    assert [r["fetched_at"] for r in resolved] == ["t1", "t1"]
    assert [r["fetch_repeats"] for r in resolved] == [1, 2]
    assert {r["document_id"] for r in divergent} == {"c"}


def test_repeat_fetches_of_one_document_are_one_original(tmp_path):
    spool = spool_with_repeats(tmp_path, ["same body"] * 3)
    result = run(tmp_path, spool)
    # Three admitted responses for the hub, but one original for it.
    assert result["counts"]["admitted_originals"] == 3
    assert result["counts"]["repeat_fetches"] == 2
    repeats = rows_of(tmp_path, result["stages"]["repeat_fetches"])
    assert len(repeats) == 2
    assert {row["document_id"] for row in repeats} == {
        document_id("https://www.srf.ch/hub")}
    originals = rows_of(tmp_path, result["stages"]["original"])
    hub = [row for row in originals if row["url"].endswith("/hub")]
    assert len(hub) == 1
    # The original named is the one the surviving markdown parent binds.
    markdown = rows_of(tmp_path, result["stages"]["markdown"])
    parent, = [row for row in markdown if row["url"].endswith("/hub")]
    assert hub[0]["receipt_object_id"] == \
        parent["parent_response"]["receipt_object_id"]
    assert hub[0]["original_sha256"] == parent["parent_response"]["original_sha256"]


def test_a_parent_naming_an_unadmitted_receipt_is_refused(tmp_path):
    spool = spool_with(tmp_path, default_pages())
    path = spool / "news_documents.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    rows[0]["lineage"]["receipt_object_id"] = "0" * 64
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(subject.NewsLaneSealError,
                       match="names no admitted original"):
        run(tmp_path, spool)


def test_a_page_emitting_one_point_id_twice_names_one_point(tmp_path):
    spool = spool_with(tmp_path, default_pages())
    path = spool / "news_documents.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    rows[0]["chunk_point_ids"] = rows[0]["chunk_point_ids"] * 2
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = run(tmp_path, spool)
    assert result["counts"]["emitted_points"] == 2
    assert result["counts"]["repeated_points_within_documents"] == 1
    joins = rows_of(tmp_path, result["stages"]["chunk_join"])
    row, = [r for r in joins if r["url"].endswith("/a")]
    assert row["chunk_count"] == 1 and row["emitted_chunk_rows"] == 2


def test_a_point_id_claimed_by_two_documents_still_refuses(tmp_path):
    spool = spool_with(tmp_path, default_pages())
    path = spool / "news_documents.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    rows[1]["chunk_point_ids"] = list(rows[0]["chunk_point_ids"])
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(subject.NewsLaneSealError,
                       match="claimed by two documents"):
        run(tmp_path, spool)

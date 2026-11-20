"""The news per-page row block, one function (task 2783).

`finish_page_rows` is the block `cytir.acquire.sitemap_ingest.process_host` ran inline after
extraction: one date decision per page, then every passage becomes a row with `provenance`,
`disc_class`, `published`, the consent stamp and (for a receipted fetch) the
`news_document_id` / `news_page_fetch` binding. `page_rows` adds the in-process extraction in
front of it, for callers that hold only the page (the `news_daily` Protocol stages).
`process_host` keeps its own extraction (sandbox pool or in-process) and calls
`finish_page_rows` at the same point with the same inputs.
"""
from __future__ import annotations

from cytir.news.extract_published import published_for_page


def finish_page_rows(passages, *, extract_input, url, cls, consent_stamp, now_ref,
                     freshness, sitemap_published=None, page_receipt=None,
                     document_id=None):
    """Return `(rows, page_published)` for the passages of ONE page (pure move)."""
    # Task 448: one date decision per PAGE, shared by all its passages (parsing the HTML
    # once per passage would be wasted work). Only the freshness/news route asks for it.
    page_published = None
    if freshness:
        page_published = published_for_page(
            extract_input, url=url,
            sitemap_published=sitemap_published,
            now_ref=now_ref)
    rows = []
    for p in passages:
        row = p.payload(); row["text"] = p.text
        row["point_id"] = p.point_id
        row["provenance"] = "sitemap-ingest"
        row["disc_class"] = cls
        if not row.get("published"):
            # No defensible date -> the field stays ABSENT, never a guess.
            resolved = page_published or sitemap_published
            if resolved:
                row["published"] = resolved
        # The same live decision already admitted this direct fetch.
        row.update(consent_stamp)
        if page_receipt is not None:
            # Compact binding only; the whole receipt is on disk under --preserve-dir.
            row["news_document_id"] = document_id
            row["news_page_fetch"] = {
                "object_id": page_receipt["object_id"],
                "requested_url": page_receipt["requested_url"],
                "final_url": page_receipt["final_url"],
                "http_status": page_receipt["http_status"],
                "fetched_at": page_receipt["fetched_at"],
                "body_sha256": page_receipt["body_sha256"],
                "transport_sha256":
                    page_receipt["artifacts"]["transport"]["body_sha256"],
            }
        rows.append(row)
    return rows, page_published


def page_rows(html, url, fetch_meta, host_meta, ctx=None):
    """Extract + chunk + finish ONE page in process; returns the rows.

    `host_meta`: `passage_common`, `cls`, `consent_stamp`, optional `lang_default`,
    `freshness`. `fetch_meta`: optional `sitemap_published`, `lastmod`, `digest`,
    `page_receipt`, `document_id`. `ctx`: optional `now_ref`.
    """
    from cytir.acquire.sitemap_ingest import page_date_clamp_day
    from cytir.extract.html_extract import process_html
    ctx = ctx or {}
    freshness = bool(host_meta.get("freshness", False))
    emeta = {
        "content": True, "links": True,
        "passage_common": host_meta["passage_common"],
        "lang_default": host_meta.get("lang_default") or "en",
        "digest": fetch_meta.get("digest"),
        "outlinks_column": False,
        "sitemap_lastmod": fetch_meta.get("lastmod"),
        "now_ref": ctx.get("now_ref"),
        "published_htmldate": freshness,
    }
    passages, _links = process_html(url, html, emeta)
    rows, _pub = finish_page_rows(
        passages, extract_input=html, url=url, cls=host_meta["cls"],
        consent_stamp=host_meta["consent_stamp"],
        now_ref=page_date_clamp_day(ctx.get("now_ref")), freshness=freshness,
        sitemap_published=fetch_meta.get("sitemap_published"),
        page_receipt=fetch_meta.get("page_receipt"),
        document_id=fetch_meta.get("document_id"))
    return rows

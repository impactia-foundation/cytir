"""Build embedding records from Hacker News discussions via Firebase only.

The input is JSONL with one positive HN item ID per object (``id``,
``item_id``, or the lane-native ``hn_object_id``).  The linked story URL is
copied from the story item into ``outlink``; it is never requested.  API and
HN item URL locations are required CLI configuration so releases do not hide
deployment endpoints in source.
"""
from __future__ import annotations
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import datetime as dt
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx

from cytir.acquire.artifact_io import atomic_publish, canonical_json
from cytir.acquire import preserved_stage_io as PS
from cytir.index.preservation_contract import _validate_stage
from cytir.acquire import stage_receipt as SS
from cytir.sources.family import HttpSpec
from cytir.sources.http import make_client
from cytir.sources.hn import identity_label as IDL


VERSION = "hn-discussion-ingest.v1"
RECORD_SCHEMA = "aw.hn-discussion-record.v1"
RECEIPT_SCHEMA = "aw.hn-firebase-fetch-receipt.v1"
MANIFEST_SCHEMA = "aw.hn-discussion-manifest.v1"

# Task 601: the `hn_discussion.original` preservation profile
# (data/source-preservation-profiles-hn-discussion.v1.json) is satisfied by RETAINED raw
# Firebase item bytes -- the same official item/<id>.json responses this
# module already fetches to select and render a discussion, kept sha-bound
# via `cytir/acquire/preserved_stage_io.py` (the same content-addressed store
# `acquire_hn_firebase_preserved.py` uses for the sibling structured-feed
# lane) rather than by a second, independently-timed fetch. Passing
# `--preserve-root` makes retention a side effect of the SAME live fetch this
# run already performs, so the retained bytes and the embedded discussion
# text are provably the same response, not two calls that merely usually
# agree.
PRESERVED_ITEM_RECEIPT_SCHEMA = "aw.hn-discussion-item-receipt.v1"
RUN_SCHEMA = "aw.hn-discussion-acquisition-run.v1"
# FROZEN LABEL (task 1884 ruling): sealed into stage receipts as producer_id; not a path.
PRODUCER = "pipeline/hn_discussion_ingest.py"

DEFAULT_WORD_BUDGET = 1_500
DEFAULT_MAX_COMMENTS = 40
DEFAULT_MAX_DEPTH = 4
DEFAULT_COMMENT_CHARS = 600
DEFAULT_WORKERS = 4
LIVE_SMOKE_MAX_ITEMS = 20
Q5_HOLD_REASON = "q5_identity_label_in_prose"
LANG_SCOPE = frozenset({"en", "de", "fr", "it"})

# The task-510 ruling is an access basis, not an open copyright grant from each
# HN commenter.  Stamp the restriction honestly instead of inventing an SPDX
# ID, and keep training disabled.
LICENSE_STAMP = "api-only"
LICENSE_BASIS = "official-hn-firebase-api-only"


class IngestError(RuntimeError):
    """A source/input contract failed; no output may be published."""


class _TextExtractor(HTMLParser):
    """Small deterministic HN-HTML-to-text converter."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() in {"p", "br", "li", "pre", "blockquote"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"p", "li", "pre", "blockquote"}:
            self.parts.append(" ")


def clean_hn_text(value: Any, *, max_chars: int | None = None) -> str:
    """HTML-unescape, strip tags, normalize whitespace, then character-cap."""
    parser = _TextExtractor()
    parser.feed(str(value or ""))
    parser.close()
    text = " ".join("".join(parser.parts).split())
    return text[:max_chars].rstrip() if max_chars is not None else text


def render_discussion(title: str, story_text: str,
                      comments: list[dict[str, Any]]) -> str:
    """Render title/story/comments as separate blank-line Markdown blocks.

    Comment order is supplied by ``select_comments`` (BFS).  No usernames or
    structural bookkeeping enter this embedding/BM25 surface.
    """
    blocks = [clean_hn_text(title)]
    cleaned_story = clean_hn_text(story_text)
    if cleaned_story:
        blocks.append(cleaned_story)
    blocks.extend(str(comment["text"]) for comment in comments)
    return "\n\n".join(block for block in blocks if block).strip()


def _canonical_item_id(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise IngestError(f"{label}: item ID must be a positive integer")
    try:
        item_id = int(value)
    except (TypeError, ValueError) as exc:
        raise IngestError(f"{label}: item ID must be a positive integer") from exc
    if item_id <= 0 or str(value).strip() != str(item_id):
        raise IngestError(f"{label}: item ID is not canonical")
    return item_id


def load_item_ids(path: Path) -> tuple[list[int], str]:
    """Read a strict, non-empty, duplicate-free JSONL item frontier."""
    try:
        body = path.read_bytes()
    except OSError as exc:
        raise IngestError(f"input unavailable: {path}: {exc}") from exc
    if not body:
        raise IngestError("input JSONL is empty")
    try:
        lines = body.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise IngestError("input JSONL is not UTF-8") from exc
    ids: list[int] = []
    seen: set[int] = set()
    fields = ("id", "item_id", "hn_object_id")
    for number, line in enumerate(lines, 1):
        if not line.strip():
            raise IngestError(f"input JSONL has a blank line at {number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise IngestError(f"input JSONL line {number} is invalid JSON") from exc
        if not isinstance(row, dict):
            raise IngestError(f"input JSONL line {number} is not an object")
        present = [field for field in fields if field in row]
        if len(present) != 1:
            raise IngestError(
                f"input JSONL line {number} must contain exactly one of {fields}")
        item_id = _canonical_item_id(row[present[0]], f"input line {number}")
        if item_id in seen:
            raise IngestError(f"duplicate HN item ID {item_id}")
        seen.add(item_id)
        ids.append(item_id)
    return ids, hashlib.sha256(body).hexdigest()


def _kids(item: dict[str, Any] | None, label: str) -> list[int]:
    if item is None:
        return []
    raw = item.get("kids", [])
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise IngestError(f"{label}: kids is not a list")
    out: list[int] = []
    seen: set[int] = set()
    for index, value in enumerate(raw):
        item_id = _canonical_item_id(value, f"{label} kids[{index}]")
        if item_id in seen:
            raise IngestError(f"{label}: duplicate child ID {item_id}")
        seen.add(item_id)
        out.append(item_id)
    return out


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _timestamp(value: dt.datetime) -> str:
    if value.tzinfo is None:
        raise IngestError("receipt clock returned a naive datetime")
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class FetchedItem:
    item: dict[str, Any] | None
    receipt: dict[str, Any]


class FirebaseFetcher:
    """Official-item endpoint fetcher with bounded 5xx/transport retries."""

    def __init__(self, *, client: Any, api_base_url: str, max_retries: int,
                 backoff_seconds: float, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], dt.datetime] = _utc_now,
                 preserve_root: Path | None = None) -> None:
        self.client = client
        self.api_base_url = api_base_url.rstrip("/")
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.sleep = sleep
        self.clock = clock
        self.preserve_root = preserve_root
        self.retained: dict[int, dict[str, Any]] = {}
        self._preserve_lock = threading.Lock()

    def _retain(self, item_id: int, body: bytes, response: Any,
               requested_url: str, fetched_at: str, attempt: int) -> None:
        """Seal the exact response body + transport identity (ADR-026 L0).

        Runs only when the caller opted in via ``preserve_root``; every
        existing call site (and every pre-601 test) leaves it ``None`` and
        is byte-for-byte unaffected.
        """
        root = self.preserve_root
        assert root is not None
        request = getattr(response, "request", None)
        metadata = {
            "schema": "aw.http-api-response.v2",
            "requested_url": requested_url,
            "request_method": str(getattr(request, "method", "GET")),
            "request_url": str(getattr(request, "url", requested_url)),
            "request_headers": sorted(
                (str(k), str(v))
                for k, v in getattr(request, "headers", {}).items()),
            "final_url": str(getattr(response, "url", requested_url)),
            "http_status": int(response.status_code),
            "fetched_at": fetched_at,
            "response_headers": sorted(
                (str(k), str(v))
                for k, v in getattr(response, "headers", {}).items()),
            "body": {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()},
            "attempt": attempt,
        }
        body_artifact = PS.store_artifact(root, "original/items", "json", body)
        metadata_body = (canonical_json(metadata) + "\n").encode("utf-8")
        transport_artifact = PS.store_artifact(
            root, "original/transport", "json", metadata_body)
        receipt = {
            "schema": PRESERVED_ITEM_RECEIPT_SCHEMA,
            "item_id": item_id,
            "requested_url": requested_url,
            "status": "complete",
            "artifacts": {"body": body_artifact, "transport": transport_artifact},
        }
        encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
        receipt_path = root / "receipts" / f"{item_id}.json"
        with self._preserve_lock:
            if receipt_path.exists():
                if receipt_path.read_bytes() != encoded:
                    raise IngestError(
                        f"invalid immutable retained receipt collision: {receipt_path}")
            else:
                atomic_publish(receipt_path, encoded)
            self.retained[item_id] = receipt

    def fetch(self, item_id: int) -> FetchedItem:
        requested_url = f"{self.api_base_url}/item/{item_id}.json"
        last_error: BaseException | str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.get(requested_url)
            except httpx.TransportError as exc:
                last_error = exc
                retryable = True
            else:
                status = int(response.status_code)
                if status == 200:
                    body = bytes(response.content)
                    try:
                        parsed = json.loads(body)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise IngestError(
                            f"HN item {item_id}: Firebase returned invalid JSON") from exc
                    if parsed is not None and not isinstance(parsed, dict):
                        raise IngestError(
                            f"HN item {item_id}: Firebase payload is not an object/null")
                    if isinstance(parsed, dict) and parsed.get("id") != item_id:
                        raise IngestError(
                            f"HN item {item_id}: response ID is {parsed.get('id')!r}")
                    children = _kids(parsed, f"HN item {item_id}")
                    fetched_at = _timestamp(self.clock())
                    receipt = {
                        "schema": RECEIPT_SCHEMA,
                        "id": item_id,
                        "fetched_at": fetched_at,
                        "kids_count": len(children),
                        "api_bytes": len(body),
                        "sha256": hashlib.sha256(body).hexdigest(),
                        "requested_url": requested_url,
                        "http_status": status,
                    }
                    if self.preserve_root is not None:
                        self._retain(item_id, body, response, requested_url,
                                     fetched_at, attempt)
                    return FetchedItem(parsed, receipt)
                if 500 <= status <= 599:
                    last_error = f"HTTP {status}"
                    retryable = True
                else:
                    raise IngestError(
                        f"HN item {item_id}: Firebase returned HTTP {status}")
            if not retryable or attempt >= self.max_retries:
                break
            self.sleep(self.backoff_seconds * (2 ** attempt))
        raise IngestError(
            f"HN item {item_id}: exhausted retries ({last_error!r})")


def select_comments(fetcher: FirebaseFetcher, story: dict[str, Any], *,
                    word_budget: int = DEFAULT_WORD_BUDGET,
                    max_comments: int = DEFAULT_MAX_COMMENTS,
                    max_depth: int = DEFAULT_MAX_DEPTH,
                    comment_chars: int = DEFAULT_COMMENT_CHARS,
                    receipts: list[dict[str, Any]] | None = None,
                    fetched_ids: set[int] | None = None) -> list[dict[str, Any]]:
    """Select valid comments breadth-first in the API's ranked kids order."""
    output_receipts = receipts if receipts is not None else []
    visited = fetched_ids if fetched_ids is not None else set()
    queue = deque((child, 1) for child in _kids(story, "story"))
    selected: list[dict[str, Any]] = []
    discussion_words = 0
    while queue and len(selected) < max_comments:
        comment_id, depth = queue.popleft()
        if depth > max_depth or comment_id in visited:
            continue
        visited.add(comment_id)
        fetched = fetcher.fetch(comment_id)
        output_receipts.append(fetched.receipt)
        comment = fetched.item
        if (not isinstance(comment, dict) or comment.get("type") != "comment" or
                comment.get("deleted") or comment.get("dead")):
            # A skipped parent quarantines its subtree.
            continue
        text = clean_hn_text(comment.get("text"), max_chars=comment_chars)
        if not text:
            continue
        words = len(text.split())
        if discussion_words + words > word_budget:
            break
        selected.append({
            "id": comment_id,
            "text": text,
            "depth": depth,
            "by": comment.get("by") if isinstance(comment.get("by"), str) else None,
            "time": comment.get("time") if isinstance(comment.get("time"), int) else None,
            "word_count": words,
        })
        discussion_words += words
        if depth < max_depth:
            queue.extend((child, depth + 1)
                         for child in _kids(comment, f"comment {comment_id}"))
    return selected


def _default_lang_detector(text: str, url: str) -> tuple[str, bool]:
    # Import lazily so offline unit tests do not probe/load the 126 MB model.
    from cytir.acquire.fetch_process import detect_lang
    return detect_lang(text, url, "en")


def _item_url(template: str, item_id: int) -> str:
    if "{item_id}" not in template:
        raise IngestError("item URL template must contain {item_id}")
    try:
        value = template.format(item_id=item_id)
    except (KeyError, ValueError) as exc:
        raise IngestError(f"invalid item URL template: {exc}") from exc
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise IngestError("item URL template did not produce an HTTP(S) URL")
    return value


def _validate_api_base(api_base_url: str) -> None:
    parsed = urlsplit(api_base_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise IngestError("API base URL must be an absolute HTTPS URL")
    if parsed.query or parsed.fragment:
        raise IngestError("API base URL must not contain a query or fragment")


def _story_record(fetcher: FirebaseFetcher, story_id: int, *,
                  item_url_template: str, word_budget: int, max_comments: int,
                  max_depth: int, comment_chars: int,
                  lang_detector: Callable[[str, str], tuple[str, bool] | str]
                  ) -> tuple[dict[str, Any] | None, list[dict[str, Any]], int, str | None]:
    """Returns ``(record, receipts, comment_count, hold_reason)``.

    ``record`` is None when the story is quarantined; ``hold_reason`` is
    ``"q5_identity_label_in_prose"`` when the Q5 scan held it out (a
    commenter's own prose carries an ``Author:``-style line), else None.
    """
    fetched = fetcher.fetch(story_id)
    receipts = [fetched.receipt]
    story = fetched.item
    if (not isinstance(story, dict) or story.get("type") != "story" or
            story.get("deleted") or story.get("dead")):
        raise IngestError(f"HN item {story_id}: missing/dead/deleted/non-story root")
    title = clean_hn_text(story.get("title"))
    created = story.get("time")
    if not title or not isinstance(created, int) or isinstance(created, bool) or created <= 0:
        raise IngestError(f"HN story {story_id}: title/time is missing or invalid")
    outlink = story.get("url")
    if outlink is not None and not isinstance(outlink, str):
        raise IngestError(f"HN story {story_id}: outlink is not a string")

    visited = {story_id}
    comments = select_comments(
        fetcher, story, word_budget=word_budget, max_comments=max_comments,
        max_depth=max_depth, comment_chars=comment_chars, receipts=receipts,
        fetched_ids=visited)
    discussion_url = _item_url(item_url_template, story_id)
    text = render_discussion(title, story.get("text") or "", comments)
    # The title block the renderer emits is cleaned a second time inside
    # render_discussion; scan that exact string (task 1445 case ii).
    rendered_title = clean_hn_text(title)
    verdict, reason = IDL.classify_with_reason(
        text, [rendered_title, clean_hn_text(story.get("text")),
               *(str(comment["text"]) for comment in comments)])
    if (verdict == IDL.PROSE and IDL.find_label(rendered_title) is not None
            and IDL.find_label(title) is None):
        reason = "q5_identity_label_in_title_after_clean"
    if verdict == IDL.RENDERER:
        raise IngestError(
            f"HN story {story_id}: the renderer itself emitted a labeled "
            "identity line -- the structural leak this profile's Q5 forbids")
    if verdict == IDL.PROSE:
        return None, receipts, len(comments), reason
    detected = lang_detector(text, discussion_url)
    if isinstance(detected, tuple):
        lang, confident = detected
    else:
        lang, confident = detected, True
    if not isinstance(lang, str) or not lang:
        raise IngestError(f"HN story {story_id}: language detector returned no label")
    if confident and lang not in LANG_SCOPE:
        return None, receipts, len(comments), None

    published_date = dt.datetime.fromtimestamp(
        created, dt.timezone.utc).date().isoformat()
    parsed = urlsplit(discussion_url)
    record = {
        "schema": RECORD_SCHEMA,
        "url": discussion_url,
        "outlink": outlink or None,
        "published_date": published_date,
        # ``published`` is the current news-lane projection key.
        "published": published_date,
        "license": LICENSE_STAMP,
        "license_basis": LICENSE_BASIS,
        "title": title,
        "text": text,
        "snippet": " ".join(text.split()[:25]),
        "anchor": title,
        "lang": lang,
        "domain": parsed.hostname or parsed.netloc,
        "tier": "C",
        "trust_tier": 0.6,
        "source_type": "reference",
        "disc_class": "news",
        "provenance": "hn-discussion-firebase",
        "last_seen": fetched.receipt["fetched_at"][:10],
        "hn_object_id": str(story_id),
        "hn_score": story.get("score") if isinstance(story.get("score"), int) else None,
        "hn_content_mode": "discussion_derived" if comments else "title_only",
        "search_ok": True,
        "train_ok": False,
        "_master": {
            "hn_story_provenance": {
                "by": story.get("by") if isinstance(story.get("by"), str) else None,
                "time": created,
            },
            "hn_comment_provenance": [
                {key: comment[key] for key in
                 ("id", "by", "time", "word_count", "depth")}
                for comment in comments
            ],
            "hn_api_route": LICENSE_BASIS,
        },
    }
    return record, receipts, len(comments), None


def _validate_limits(*, item_cap: int, shard_size: int, workers: int,
                     word_budget: int, max_comments: int, max_depth: int,
                     comment_chars: int, max_retries: int,
                     backoff_seconds: float) -> None:
    positive = {
        "item cap": item_cap, "shard size": shard_size, "workers": workers,
        "word budget": word_budget, "max comments": max_comments,
        "max depth": max_depth, "comment chars": comment_chars,
    }
    for label, value in positive.items():
        if value <= 0:
            raise IngestError(f"{label} must be positive")
    if workers > 4:
        raise IngestError("workers must be <= 4")
    if max_retries < 0 or backoff_seconds < 0:
        raise IngestError("retry count/backoff must be non-negative")


def _artifact(path: Path, root: Path, *, records: int | None = None,
              kind: str) -> dict[str, Any]:
    body = path.read_bytes()
    result: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "kind": kind,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }
    if records is not None:
        result["records"] = records
    return result


def _document_id(item_id: int) -> str:
    # Deliberately a different namespace than
    # `acquire_hn_firebase_preserved.py`'s "hn-firebase-item" ids: that lane's
    # document is the story alone; this one's document is the discussion.
    return hashlib.sha256(f"hn-discussion-item\0{item_id}".encode()).hexdigest()


def _logical_rows(rows: list[dict[str, Any]]) -> tuple[bytes, str]:
    body = b"".join((canonical_json(row) + "\n").encode("utf-8") for row in rows)
    return body, hashlib.sha256(body).hexdigest()


def _seal_row_stage(root: Path, name: str, rows: list[dict[str, Any]],
                    fmt: str, *, seal_root: Path | None = None) -> dict[str, Any]:
    seal_root = seal_root if seal_root is not None else root
    body, logical = _logical_rows(rows)
    artifact = PS.store_artifact(root, f"stages/{name}/data", "jsonl", body)
    return SS.seal(
        artifact_root=seal_root,
        artifact_paths=[PS.artifact_path(root, artifact)],
        receipt_path=root / "stages" / name / "receipt.json", records=len(rows),
        fmt=fmt, logical_sha256=logical, producer_id=PRODUCER,
        producer_version=VERSION)


def seal_preservation_run(preserve_root: Path, *, fetcher: "FirebaseFetcher",
                          records: list[dict[str, Any]],
                          quarantined_story_ids: list[int] = (),
                          q5_held_out_story_ids: list[int] = (),
                          artifact_root: Path | None = None
                          ) -> dict[str, Any]:
    """Bind this run's retained Firebase bytes to its discussion documents.

    ``artifact_root``, when given, is a shared ancestor of ``preserve_root``
    (e.g. the whole tier-2 apply's artifact root, which also holds the pack's
    ``chunks``/``vectors`` stages) that every sealed URI is expressed
    relative to, so one content-pack intent can validate every stage under
    ONE root. Files are still physically written under ``preserve_root``;
    only the sealed URI's base changes. Defaults to ``preserve_root`` itself,
    reproducing every existing caller's behaviour byte-for-byte.

    Called only when ``ingest`` was given ``preserve_root``. Produces
    ``RUN.json`` with the ``original``/``markdown``/``outlinks`` stages the
    ``hn_discussion`` preservation profile (data/source-preservation-hn-discussion-
    profiles.v1.json) requires; the pack step contributes ``chunks`` and
    ``vectors`` downstream, exactly as the sibling ``structured_data_feed``
    lane splits parent-run stages from pack stages
    (``cytir/sources/hn/audit_profile.py``).

    The ARTIFACT INVENTORY under ``original`` covers every item this run
    actually fetched -- every selected comment as well as every story, sha-
    bound, regardless of admission. Its ``records`` COUNT is at
    story/document granularity (one story = one candidate discussion), not
    the raw item count: ``content_pack_preservation``'s ``original_to_markdown``
    lineage edge conserves at that granularity (a story is either an emitted
    parent or quarantined; a story's comments are sub-evidence of its own
    parent, never a second candidate document), and counting every retained
    comment as its own "record" would make that edge's conservation
    arithmetic meaningless.
    """
    if not fetcher.retained:
        raise IngestError("preserve_root given but no items were retained")
    seal_root = artifact_root if artifact_root is not None else preserve_root
    quarantined_story_ids = sorted(set(quarantined_story_ids))
    q5_held_out_story_ids = sorted(set(q5_held_out_story_ids))
    if not set(q5_held_out_story_ids) <= set(quarantined_story_ids):
        raise IngestError("a Q5 hold-out must also be a quarantined story")
    original_paths: list[Path] = []
    for item_id in sorted(fetcher.retained):
        receipt = fetcher.retained[item_id]
        original_paths.append(preserve_root / "receipts" / f"{item_id}.json")
        for artifact in receipt["artifacts"].values():
            original_paths.append(PS.artifact_path(preserve_root, artifact))
    original_paths = sorted(set(original_paths))
    inventory = [{"uri": f"artifact://{p.relative_to(seal_root).as_posix()}",
                 "bytes": p.stat().st_size,
                 "sha256": PS.sha256_file(p)} for p in original_paths]
    _, original_logical = _logical_rows(inventory)
    story_count = len(records) + len(quarantined_story_ids)
    original_stage = SS.seal(
        artifact_root=seal_root, artifact_paths=original_paths,
        receipt_path=preserve_root / "stages" / "original" / "receipt.json",
        records=story_count,
        fmt="exact-firebase-json-response+canonical-headers-transport+item-receipts",
        logical_sha256=original_logical, producer_id=PRODUCER,
        producer_version=VERSION)

    markdown_rows, outlink_rows = [], []
    for record in records:
        item_id = int(record["hn_object_id"])
        document_id = _document_id(item_id)
        markdown_rows.append({
            "document_id": document_id, "url": record["url"],
            "lang": record["lang"], "title": record["title"],
            "markdown": record["text"], "item_id": item_id,
        })
        outlink = record.get("outlink")
        outlink_rows.append({
            "document_id": document_id, "source_url": record["url"],
            "has_outlink": outlink is not None,
            "target_url": outlink,
        })
    markdown_rows.sort(key=lambda row: row["item_id"])
    outlink_rows.sort(key=lambda row: row["document_id"])
    markdown_stage = _seal_row_stage(
        preserve_root, "markdown", markdown_rows,
        "canonical-jsonl+deterministic-gzip", seal_root=seal_root)
    outlinks_stage = _seal_row_stage(
        preserve_root, "outlinks", outlink_rows,
        "canonical-jsonl+deterministic-gzip", seal_root=seal_root)

    stages = {"original": original_stage, "markdown": markdown_stage,
              "outlinks": outlinks_stage}

    lineage = {"original_to_markdown": {
        "input_records": story_count,
        "emitted_parent_records": len(markdown_rows),
        "quarantined_records": len(quarantined_story_ids),
        "output_records": len(markdown_rows),
    }}
    if quarantined_story_ids:
        quarantine = {"reason": "out_of_scope_language",
                      "item_ids": quarantined_story_ids}
        if q5_held_out_story_ids:
            quarantine = {
                "reason": "out_of_scope_language_or_q5_held_out",
                "item_ids": quarantined_story_ids,
                "q5_held_out": {
                    "reason": Q5_HOLD_REASON,
                    "item_ids": q5_held_out_story_ids,
                    "document_ids": [_document_id(i) for i in q5_held_out_story_ids]},
                "out_of_scope_language": {"item_ids": sorted(
                    set(quarantined_story_ids) - set(q5_held_out_story_ids))},
            }
        quarantine_body = (canonical_json(quarantine) + "\n").encode("utf-8")
        quarantine_path = preserve_root / "stages" / "original" / "quarantine.json"
        if quarantine_path.exists():
            if quarantine_path.read_bytes() != quarantine_body:
                raise IngestError(f"immutable quarantine collision: {quarantine_path}")
        else:
            atomic_publish(quarantine_path, quarantine_body)
        lineage["original_to_markdown"]["quarantine_receipt"] = {
            "uri": f"artifact://{quarantine_path.relative_to(seal_root).as_posix()}",
            "bytes": len(quarantine_body),
            "sha256": hashlib.sha256(quarantine_body).hexdigest(),
        }

    run = {
        "schema": RUN_SCHEMA, "version": VERSION, "status": "complete",
        "route": "official_firebase_api",
        "counts": {"retained_items": len(fetcher.retained),
                   "discussion_documents": len(markdown_rows),
                   "outlinks": len(outlink_rows),
                   "quarantined_stories": len(quarantined_story_ids)},
        "stages": stages,
        "lineage": lineage,
    }
    if q5_held_out_story_ids:
        run["counts"]["q5_held_out"] = len(q5_held_out_story_ids)
        run["q5_held_out_document_ids"] = [
            _document_id(i) for i in q5_held_out_story_ids]
    encoded = (json.dumps(run, indent=2, sort_keys=True) + "\n").encode("utf-8")
    run_path = preserve_root / "RUN.json"
    if run_path.exists():
        if run_path.read_bytes() != encoded:
            raise IngestError(f"immutable HN discussion RUN collision: {run_path}")
    else:
        atomic_publish(run_path, encoded)
    for name, stage in stages.items():
        _validate_stage(stage, name, artifact_root=seal_root, deep=True)
    return run


def _publish_output(output_dir: Path, *, records: list[dict[str, Any]],
                    receipts: list[dict[str, Any]], input_path: Path,
                    input_sha256: str, input_count: int, selected_count: int,
                    shard_size: int, api_base_url: str,
                    item_url_template: str, limits: dict[str, Any],
                    comments_selected: int, out_of_scope: int,
                    live_smoke: bool, q5_held_out: int = 0) -> dict[str, Any]:
    if output_dir.exists():
        raise IngestError(f"immutable output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.", dir=output_dir.parent))
    artifacts: list[dict[str, Any]] = []
    try:
        receipt_dir = staging / "receipts"
        receipt_dir.mkdir()
        for receipt in sorted(receipts, key=lambda row: row["id"]):
            path = receipt_dir / f"{receipt['id']}.json"
            path.write_bytes((canonical_json(receipt) + "\n").encode("utf-8"))
            artifacts.append(_artifact(path, staging, kind="fetch_receipt"))

        for shard_index, offset in enumerate(range(0, len(records), shard_size)):
            rows = records[offset:offset + shard_size]
            path = staging / f"records-{shard_index:05d}.jsonl"
            body = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
            path.write_bytes(body)
            artifacts.append(_artifact(path, staging, records=len(rows),
                                       kind="record_shard"))

        manifest = {
            "schema": MANIFEST_SCHEMA,
            "version": VERSION,
            "source": {
                "route": "official_firebase_api",
                "api_base_url": api_base_url.rstrip("/"),
                "item_url_template": item_url_template,
                "linked_pages_fetched": 0,
            },
            "input": {
                "name": input_path.name,
                "bytes": input_path.stat().st_size,
                "sha256": input_sha256,
            },
            "limits": limits,
            "counts": {
                "input_item_ids": input_count,
                "selected_item_ids": selected_count,
                "capped_item_ids": input_count - selected_count,
                "api_items_fetched": len(receipts),
                "comments_selected": comments_selected,
                "records": len(records),
                "out_of_scope": out_of_scope,
                **({"q5_held_out": q5_held_out} if q5_held_out else {}),
                "shards": sum(a["kind"] == "record_shard" for a in artifacts),
                "fetch_receipts": len(receipts),
            },
            "license": {"stamp": LICENSE_STAMP, "basis": LICENSE_BASIS},
            "lang_detection": "fetch_process.detect_lang/fastText-lid.176",
            "live_smoke": live_smoke,
            "artifacts": sorted(artifacts, key=lambda row: row["path"]),
        }
        (staging / "MANIFEST.json").write_bytes(
            (canonical_json(manifest) + "\n").encode("utf-8"))
        os.rename(staging, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def ingest(*, input_path: Path, output_dir: Path, api_base_url: str,
           item_url_template: str, item_cap: int, shard_size: int,
           workers: int = DEFAULT_WORKERS,
           word_budget: int = DEFAULT_WORD_BUDGET,
           max_comments: int = DEFAULT_MAX_COMMENTS,
           max_depth: int = DEFAULT_MAX_DEPTH,
           comment_chars: int = DEFAULT_COMMENT_CHARS,
           max_retries: int = 4, backoff_seconds: float = 0.5,
           client: Any | None = None,
           sleep: Callable[[float], None] = time.sleep,
           clock: Callable[[], dt.datetime] = _utc_now,
           lang_detector: Callable[[str, str], tuple[str, bool] | str]
           = _default_lang_detector,
           live_smoke: bool = False,
           preserve_root: Path | None = None,
           preservation_artifact_root: Path | None = None) -> dict[str, Any]:
    """Fetch, render, shard, receipt, and atomically publish one run.

    ``preserve_root``, when given, retains the exact Firebase item bytes for
    every item this run fetches (task 601, ADR-026 L0) and seals the
    ``hn_discussion.original``/``markdown``/``outlinks`` preservation stages
    under it. ``output_dir``'s embed-ready ``MANIFEST.json``/records/receipts
    are byte-identical whether or not this is set.
    """
    _validate_limits(
        item_cap=item_cap, shard_size=shard_size, workers=workers,
        word_budget=word_budget, max_comments=max_comments, max_depth=max_depth,
        comment_chars=comment_chars, max_retries=max_retries,
        backoff_seconds=backoff_seconds)
    if live_smoke and item_cap > LIVE_SMOKE_MAX_ITEMS:
        raise IngestError(
            f"--live-smoke requires --item-cap <= {LIVE_SMOKE_MAX_ITEMS}")
    ids, input_sha256 = load_item_ids(input_path)
    selected_ids = ids[:item_cap]
    if not selected_ids:
        raise IngestError("item cap selected no input items")
    # Validate the template before any source request.
    _item_url(item_url_template, selected_ids[0])
    _validate_api_base(api_base_url)

    owned_client = client is None
    source_client = client or make_client(HttpSpec(
        timeout=30, follow_redirects=False,
        headers={"Accept": "application/json"}))
    if preserve_root is not None:
        preserve_root.mkdir(parents=True, exist_ok=True)
    fetcher = FirebaseFetcher(
        client=source_client, api_base_url=api_base_url,
        max_retries=max_retries, backoff_seconds=backoff_seconds,
        sleep=sleep, clock=clock, preserve_root=preserve_root)
    try:
        def work(story_id: int):
            return _story_record(
                fetcher, story_id, item_url_template=item_url_template,
                word_budget=word_budget, max_comments=max_comments,
                max_depth=max_depth, comment_chars=comment_chars,
                lang_detector=lang_detector)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(work, selected_ids))
    finally:
        if owned_client:
            source_client.close()

    records: list[dict[str, Any]] = []
    receipts_by_id: dict[int, dict[str, Any]] = {}
    comments_selected = 0
    out_of_scope = 0
    quarantined_story_ids: list[int] = []
    q5_held_out_story_ids: list[int] = []
    for story_id, (record, item_receipts, comment_count, hold) in zip(selected_ids, results):
        comments_selected += comment_count
        q5_held_out_story_ids.extend([story_id] if hold else [])
        out_of_scope += int(record is None and not hold)
        if record is not None:
            records.append(record)
        else:
            quarantined_story_ids.append(story_id)
        for receipt in item_receipts:
            prior = receipts_by_id.get(receipt["id"])
            if prior is not None:
                raise IngestError(
                    f"HN item {receipt['id']} was fetched in more than one tree")
            receipts_by_id[receipt["id"]] = receipt
    limits = {
        "item_cap": item_cap,
        "shard_size": shard_size,
        "workers": workers,
        "word_budget_per_story": word_budget,
        "max_comments_per_story": max_comments,
        "max_depth": max_depth,
        "comment_chars": comment_chars,
        "max_retries": max_retries,
        "backoff_seconds": backoff_seconds,
    }
    manifest = _publish_output(
        output_dir, records=records, receipts=list(receipts_by_id.values()),
        input_path=input_path, input_sha256=input_sha256,
        input_count=len(ids), selected_count=len(selected_ids),
        shard_size=shard_size, api_base_url=api_base_url,
        item_url_template=item_url_template, limits=limits,
        comments_selected=comments_selected, out_of_scope=out_of_scope,
        live_smoke=live_smoke, q5_held_out=len(q5_held_out_story_ids))
    if preserve_root is not None:
        seal_preservation_run(preserve_root, fetcher=fetcher, records=records,
                              quarantined_story_ids=quarantined_story_ids,
                              q5_held_out_story_ids=q5_held_out_story_ids,
                              artifact_root=preservation_artifact_root)
    return manifest


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--item-url-template", required=True)
    parser.add_argument("--item-cap", type=int, required=True)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--word-budget", type=int, default=DEFAULT_WORD_BUDGET)
    parser.add_argument("--max-comments", type=int, default=DEFAULT_MAX_COMMENTS)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--comment-chars", type=int, default=DEFAULT_COMMENT_CHARS)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--backoff-seconds", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true",
                        help="validate and print the bounded plan; no network/writes")
    parser.add_argument("--live-smoke", action="store_true",
                        help="mark a live smoke; enforces item-cap <= 20")
    parser.add_argument("--preserve-root", type=Path, default=None,
                        help="retain exact Firebase item bytes here and seal "
                             "the hn_discussion.original/markdown/outlinks "
                             "preservation stages (task 601, ADR-026 L0)")
    parser.add_argument(
        "--preservation-artifact-root", type=Path, default=None,
        help="seal stage URIs relative to this shared ancestor of "
             "--preserve-root instead of --preserve-root itself, so one "
             "content-pack intent can validate this arm's stages against "
             "the SAME root as the pack's chunks/vectors stages")
    return parser


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = build_parser(prog)
    args = parser.parse_args(argv)
    try:
        _validate_limits(
            item_cap=args.item_cap, shard_size=args.shard_size,
            workers=args.workers, word_budget=args.word_budget,
            max_comments=args.max_comments, max_depth=args.max_depth,
            comment_chars=args.comment_chars, max_retries=args.max_retries,
            backoff_seconds=args.backoff_seconds)
        ids, input_sha256 = load_item_ids(args.input)
        _item_url(args.item_url_template, ids[0])
        _validate_api_base(args.api_base_url)
        if args.live_smoke and args.item_cap > LIVE_SMOKE_MAX_ITEMS:
            raise IngestError(
                f"--live-smoke requires --item-cap <= {LIVE_SMOKE_MAX_ITEMS}")
        if args.dry_run:
            print(canonical_json({
                "dry_run": True,
                "input_item_ids": len(ids),
                "selected_item_ids": min(len(ids), args.item_cap),
                "capped_item_ids": max(0, len(ids) - args.item_cap),
                "input_sha256": input_sha256,
                "would_write": str(args.output_dir),
            }))
            return 0
        manifest = ingest(
            input_path=args.input, output_dir=args.output_dir,
            api_base_url=args.api_base_url,
            item_url_template=args.item_url_template,
            item_cap=args.item_cap, shard_size=args.shard_size,
            workers=args.workers, word_budget=args.word_budget,
            max_comments=args.max_comments, max_depth=args.max_depth,
            comment_chars=args.comment_chars, max_retries=args.max_retries,
            backoff_seconds=args.backoff_seconds, live_smoke=args.live_smoke,
            preserve_root=args.preserve_root,
            preservation_artifact_root=args.preservation_artifact_root)
        print(canonical_json(manifest))
        return 0
    except IngestError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
